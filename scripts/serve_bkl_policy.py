#!/usr/bin/env python3

"""ZMQ inference server for the MVP-Generalize BKL policy.

Receives raw camera images + EEF state from the eval client,
extracts MAE features in real-time, runs the transformer policy,
denormalizes actions (quantile scaling), and returns a 16-step
action chunk of 62D body-frame delta EEF + absolute hand targets.

All config and normalization stats are loaded from train_meta.pt.

Usage:
    python scripts/serve_bkl_policy.py \
        --log-dir logs/<run_name> \
        --epoch 200 \
        --port 5678
"""

import argparse
import glob
import io
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import zmq
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mvp.vision_model.vision_encoder import Encoder
from mvp.bimanual_bc.dataset import process_image, BKL_Dataset
from tools.offline_eval_bkl import build_model


def decode_jpeg_to_rgb(jpeg_bytes):
    """Decode JPEG bytes to RGB numpy array (H, W, 3) uint8."""
    return np.array(Image.open(io.BytesIO(jpeg_bytes)))


@torch.no_grad()
def extract_features(mae_encoder, images_rgb, im_size=224):
    """Extract MAE mean-pooled features from RGB images. Returns (N, 768)."""
    batch = np.stack([process_image(img, im_size) for img in images_rgb])
    batch = torch.tensor(batch, dtype=torch.float32).cuda()
    return mae_encoder(batch, mode='mean').cpu().numpy()


class MVPPolicyServer:

    def __init__(self, cfg, policy, mae_encoder, prompt_embedding,
                 action_q01, action_range, device='cuda'):
        self.cfg = cfg
        self.policy = policy
        self.mae = mae_encoder
        self.device = device

        self.num_steps = cfg.actor.num_steps
        self.num_pred = cfg.actor.num_pred
        self.num_heads = cfg.transformer_concat.num_heads
        self.feat_dim = cfg.actor.im_dim
        self.state_dim = cfg.data.state_dim
        self.action_dim = cfg.data.action_dim
        self.num_cams = len(cfg.data.cams)
        self.im_size = 224

        self.use_proprio = getattr(cfg.data, 'use_proprio', True)
        self.side = getattr(cfg.data, 'side', 'both')

        # Quantile denormalization: pred_denorm = (pred + 1) / 2 * range + q01
        self.action_q01 = action_q01    # (62,) or None
        self.action_range = action_range  # (62,) or None

        # Prompt on GPU
        self.prompt = torch.tensor(prompt_embedding, dtype=torch.float32).to(device)

        # Pre-build attention mask
        L = self.num_steps
        causal = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        self.attn_mask = causal.unsqueeze(0).repeat(self.num_heads, 1, 1)

        # Pre-build masks (matching offline eval lines 188-204)
        self._build_masks()

        # Sliding context window: ring buffer of (features, state) per timestep,
        # matching the training-time sliding window of num_steps frames.
        self._history = []  # list of (features_np, state_np), max len num_steps

        # TODO: remove after debugging — accumulate extracted features for saving
        self._feature_log = []  # list of (num_cams, feat_dim) arrays

    def _build_masks(self):
        """Build state mask and camera visibility mask matching training."""
        cfg = self.cfg

        # Camera visibility
        cam_keys = [f"feat_{c}" for c in cfg.data.cams]
        excluded_cam = None
        if self.side != 'both':
            excluded_cam = f"feat_{BKL_Dataset.SIDE_EXCLUDED_CAMS[self.side]}"
        self.cam_visible = [0.0 if cam_keys[c] == excluded_cam else 1.0
                            for c in range(self.num_cams)]

        # State mask
        self.state_mask_np = np.ones(self.state_dim, dtype=np.float32)
        if not self.use_proprio:
            self.state_mask_np[:] = 0.0
        elif self.side != 'both':
            other = 'left' if self.side == 'right' else 'right'
            for idx in BKL_Dataset.SIDE_INDICES[other]:
                self.state_mask_np[idx] = 0.0

        self.state_mask_t = torch.tensor(self.state_mask_np, dtype=torch.float32).to(self.device)

    def reset_history(self):
        """Clear the observation history (call at the start of each trajectory)."""
        self._history.clear()
        # TODO: remove after debugging
        self._feature_log.clear()

    @torch.no_grad()
    def predict(self, features, state):
        """Run inference with sliding context window. Returns (num_pred, action_dim) denormalized.

        Maintains a history buffer of past observations. The context window is
        built by taking the last num_steps entries, padding on the left by
        repeating the earliest observation (matching training-time behaviour).
        """
        # Append current observation to history
        self._history.append((features.copy(), state.copy()))
        if len(self._history) > self.num_steps:
            self._history = self._history[-self.num_steps:]

        B = 1
        L = self.num_steps

        # Build sliding window arrays from history
        hist_len = len(self._history)
        pad_len = L - hist_len

        # States: (L, state_dim)
        hist_states = np.stack([s for _, s in self._history])  # (hist_len, state_dim)
        if pad_len > 0:
            hist_states = np.concatenate(
                [np.tile(hist_states[0:1], (pad_len, 1)), hist_states], axis=0
            )
        state_t = torch.tensor(hist_states, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, L, state_dim)

        # Features: per-camera (L, feat_dim)
        im_windows = []
        for c in range(self.num_cams):
            hist_feats_c = np.stack([f[c] for f, _ in self._history])  # (hist_len, feat_dim)
            if pad_len > 0:
                hist_feats_c = np.concatenate(
                    [np.tile(hist_feats_c[0:1], (pad_len, 1)), hist_feats_c], axis=0
                )
            im_windows.append(
                torch.tensor(hist_feats_c, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, L, feat_dim)
            )

        prompt_t = self.prompt.unsqueeze(0).repeat(L, 1).unsqueeze(0)  # (1, L, 768)

        obs_each_mod = [prompt_t] + im_windows + [state_t]

        # Build masks (matching offline eval exactly)
        prompt_mask = torch.ones_like(prompt_t)
        img_masks = [torch.full_like(im, self.cam_visible[c]) for c, im in enumerate(im_windows)]
        s_mask = self.state_mask_t.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        mask_each_mod = [prompt_mask] + img_masks + [s_mask]

        # Forward
        preds = self.policy(obs_each_mod=obs_each_mod, mask_each_mod=mask_each_mod, attn_mask=self.attn_mask)
        chunk = preds[-1][0].cpu().numpy()  # (num_pred, action_dim)

        # Denormalize: [-1, 1] → [q01, q99]
        if self.action_q01 is not None:
            chunk = (chunk + 1) / 2 * self.action_range + self.action_q01

        return chunk

    def handle_request(self, payload):
        msg_type = payload.get("type", "obs")
        if msg_type == "reset":
            self.reset_history()
            return {"status": "success", "type": "reset"}
        if msg_type == "handshake":
            return {
                "status": "success",
                "type": "handshake",
                "side": self.side,
                "action_dim": self.action_dim,
                "num_pred": self.num_pred,
                "use_proprio": self.use_proprio,
            }
        # TODO: remove after debugging — save accumulated features to h5
        if msg_type == "save_features":
            save_path = payload.get("path", "server_features.h5")
            if not self._feature_log:
                return {"status": "error", "message": "No features to save"}
            import h5py
            features = np.stack(self._feature_log)  # (T, num_cams, feat_dim)
            with h5py.File(save_path, 'w') as hf:
                hf.create_dataset('features', data=features)
            n = len(self._feature_log)
            print(f"[Server] Saved {n} feature frames to {save_path}")
            return {"status": "success", "type": "save_features", "path": save_path, "num_frames": n}

        try:
            img_left = decode_jpeg_to_rgb(payload["image_wrist_left"])
            img_right = decode_jpeg_to_rgb(payload["image_wrist_right"])
            img_head = decode_jpeg_to_rgb(payload["image_head"])
        except Exception as e:
            return {"status": "error", "message": f"Image decode failed: {e}"}

        state = np.array(payload["state"], dtype=np.float32)
        if state.shape != (self.state_dim,):
            return {"status": "error", "message": f"Expected state ({self.state_dim},), got {state.shape}"}

        features = extract_features(self.mae, [img_left, img_right, img_head], self.im_size)

        # TODO: remove after debugging
        self._feature_log.append(features.copy())

        chunk = self.predict(features, state)

        return {"status": "success", "actions": chunk.tolist()}


def main():
    parser = argparse.ArgumentParser(description="MVP-Generalize BKL policy server")
    parser.add_argument("--log-dir",
                        default="logs/bkl_pick-place-egg_vitb-mae_skip1_pred16_obs1_noise1x_bodyframe-6drot_actonly_proprio_right__041426_1119")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--mae-weights",
                        default="/mnt/amlfs-01/home/dniu/Project/lego/mvp_weights")
    parser.add_argument("--port", type=int, default=5679)
    parser.add_argument("--cuda", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    project_root = os.path.join(os.path.dirname(__file__), '..')

    # Load train metadata
    meta_path = os.path.join(args.log_dir, "train_meta.pt")
    print(f"[Server] Loading: {meta_path}")
    meta = torch.load(meta_path, map_location='cpu')
    cfg = OmegaConf.create(meta['cfg'])

    # Quantile action stats
    action_q01, action_range = None, None
    if 'action_q01' in meta:
        action_q01 = np.array(meta['action_q01'], dtype=np.float32)
        action_q99 = np.array(meta['action_q99'], dtype=np.float32)
        action_range = np.maximum(action_q99 - action_q01, 1e-6)
        print(f"[Server] Quantile stats loaded (q01 shape={action_q01.shape})")

    # Find checkpoint
    if args.epoch is not None:
        ckpt_path = os.path.join(args.log_dir, f"model_ep{args.epoch:04d}.pt")
    else:
        ckpts = sorted(glob.glob(os.path.join(args.log_dir, "model_ep*.pt")))
        assert ckpts, f"No checkpoints in {args.log_dir}"
        ckpt_path = ckpts[-1]
    print(f"[Server] Checkpoint: {ckpt_path}")

    # MAE encoder
    mae_dir = args.mae_weights or project_root
    mae = Encoder("vitb-mae-egosoup", mae_dir, freeze=True)
    mae.cuda().eval()

    # Policy model
    policy = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    mks, uks = policy.load_state_dict(ckpt['model_state'], strict=False)
    if mks: print(f"  Missing: {mks}")
    if uks: print(f"  Unexpected: {uks}")
    policy.cuda().eval()

    # Prompt embedding
    prompt_path = cfg.data.prompt_embedding_path
    if not os.path.isabs(prompt_path):
        prompt_path = os.path.join(project_root, prompt_path)
    prompt = np.load(prompt_path).astype(np.float32)

    server = MVPPolicyServer(cfg, policy, mae, prompt, action_q01, action_range)

    print(f"[Server] state_dim={cfg.data.state_dim}, action_dim={cfg.data.action_dim}, "
          f"num_steps={cfg.actor.num_steps}, num_pred={cfg.actor.num_pred}, "
          f"use_proprio={server.use_proprio}, side={server.side}")

    # Warm up
    _ = server.predict(np.zeros((server.num_cams, server.feat_dim), dtype=np.float32),
                       np.zeros(server.state_dim, dtype=np.float32))
    server.reset_history()
    print("[Server] Warm-up done.")

    # ZMQ
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[Server] Listening on port {args.port}. Ready.")

    step = 0
    while True:
        try:
            payload = pickle.loads(sock.recv())
            t0 = time.time()
            resp = server.handle_request(payload)
            dt = time.time() - t0
            sock.send(pickle.dumps(resp))
            if resp.get("status") == "success" and resp.get("type") != "reset":
                step += 1
                if step % 10 == 0:
                    print(f"[Server] Step {step}, {dt*1000:.1f}ms")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Server] Error: {e}")
            import traceback; traceback.print_exc()
            try: sock.send(pickle.dumps({"status": "error", "message": str(e)}))
            except: pass

    sock.close(); ctx.term()


if __name__ == "__main__":
    main()
