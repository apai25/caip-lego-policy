#!/usr/bin/env python3

"""Online eval of a BKL policy via the ZMQ server (serve_bkl_policy.py).

Replays raw episode data (MP4 videos + H5 states) through the live server,
applies ACT-style temporal ensemble smoothing to the returned action chunks,
converts predicted deltas to absolute EEF targets, and compares against
ground-truth next-states — matching the offline eval pipeline exactly.

Usage:
    # First start the server:
    python scripts/serve_bkl_policy.py --log-dir <log> --epoch 200

    # Then run this script:
    python tools/online_eval_bkl.py \
        --episode-dir /path/to/episode_XXXX \
        --num-exec 4 --smooth-lambda 0.1 \
        --output-dir /path/to/save/plots

    # Then compare with offline eval (separate script):
    python tools/compare_online_offline.py \
        --online-npz /path/to/online_eval_data.npz \
        --offline-npz /path/to/eval_data.npz
"""

import argparse
import os
import pickle
import sys

import cv2
import h5py
import numpy as np
import zmq

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mvp.bimanual_bc.dataset import pose_to_xyz_rot6d, BKL_Dataset
from tools.offline_eval_bkl import (
    DOF_GROUPS,
    apply_temporal_smoothing, action_to_absolute_target, plot_dof_group,
)
from tools.extract_bkl_features import CAM_SUFFIXES, decode_video


# ---------------------------------------------------------------------------
# ZMQ Client
# ---------------------------------------------------------------------------

class PolicyClient:
    """ZMQ REQ client for the MVP-Generalize BKL policy server."""

    def __init__(self, host="localhost", port=5678, timeout_ms=30000):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def _send_recv(self, payload):
        self.sock.send(pickle.dumps(payload))
        return pickle.loads(self.sock.recv())

    def handshake(self):
        return self._send_recv({"type": "handshake"})

    def reset(self):
        return self._send_recv({"type": "reset"})

    def infer(self, jpeg_left, jpeg_right, jpeg_head, state):
        """Send observation, receive (num_pred, action_dim) denormalized actions."""
        resp = self._send_recv({
            "image_wrist_left": jpeg_left,
            "image_wrist_right": jpeg_right,
            "image_head": jpeg_head,
            "state": state,
        })
        if resp.get("status") != "success":
            raise RuntimeError(f"Server error: {resp}")
        return np.array(resp["actions"], dtype=np.float32)

    def save_features(self, path):
        return self._send_recv({"type": "save_features", "path": path})

    def close(self):
        self.sock.close()
        self.ctx.term()


# ---------------------------------------------------------------------------
# Episode data loading (raw — no features.h5)
# ---------------------------------------------------------------------------

def load_episode_raw(episode_dir):
    """Load states, poses, and decode video frames from an episode directory.

    Returns:
        states: (T, 62) absolute state
        left_arm_pose: (T, 4, 4) homogeneous transforms
        right_arm_pose: (T, 4, 4) homogeneous transforms
        video_frames: dict mapping cam name -> list of (H, W, 3) RGB uint8
        T: total raw frames
    """
    ep_name = os.path.basename(episode_dir)
    h5_path = os.path.join(episode_dir, f"{ep_name}.h5")

    with h5py.File(h5_path, 'r') as f:
        left_arm_pose = f['left_arm_current_pose'][:].astype(np.float64)
        right_arm_pose = f['right_arm_current_pose'][:].astype(np.float64)
        left_hand_pos = f['left_hand_joint_positions'][:].astype(np.float32)
        right_hand_pos = f['right_hand_joint_positions'][:].astype(np.float32)

    left_eef = pose_to_xyz_rot6d(left_arm_pose)    # (T, 9)
    right_eef = pose_to_xyz_rot6d(right_arm_pose)  # (T, 9)
    states = np.concatenate([left_eef, right_eef, left_hand_pos, right_hand_pos], axis=1)
    T = len(states)

    # Decode video frames for each camera
    cams = ["left_wrist", "right_wrist", "head"]
    video_frames = {}
    for cam in cams:
        suffix = CAM_SUFFIXES[cam]
        video_path = os.path.join(episode_dir, f"{ep_name}{suffix}")
        frames = decode_video(video_path)
        if len(frames) != T:
            print(f"Warning: {cam} video has {len(frames)} frames but H5 has {T}, truncating")
            frames = frames[:T]
        video_frames[cam] = frames

    return states, left_arm_pose, right_arm_pose, video_frames, T


def frame_to_jpeg_bytes(frame_rgb, quality=100):
    """Encode an RGB frame to JPEG bytes."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, "JPEG encoding failed"
    return buf.tobytes()


def frame_to_png_bytes(frame_rgb):
    """Encode an RGB frame to lossless PNG bytes."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    assert ok, "PNG encoding failed"
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Online eval loop
# ---------------------------------------------------------------------------

def run_online_eval(client, states, video_frames, frame_skip, num_exec,
                    num_pred, jpeg_quality=100, lossless=False):
    """Query the server at model-timestep cadence, collecting action chunks.

    Args:
        client: PolicyClient connected to the server
        states: (T, 62) episode states
        video_frames: dict cam_name -> list of RGB frames
        frame_skip: training frame skip (stride = frame_skip + 1)
        num_exec: model-steps between re-queries
        num_pred: actions per chunk (from server handshake)
        jpeg_quality: JPEG encoding quality (ignored if lossless=True)
        lossless: if True, send PNG instead of JPEG

    Returns:
        all_chunks: (Q, num_pred, 62) denormalized action chunks
        query_model_steps: list of Q model-step indices
        M: total model-timesteps
    """
    T = len(states)
    step = frame_skip + 1
    M = (T - 1) // step
    action_dim = states.shape[1]

    encode_fn = frame_to_png_bytes if lossless else (lambda f: frame_to_jpeg_bytes(f, jpeg_quality))

    query_model_steps = list(range(0, M, num_exec))
    Q = len(query_model_steps)

    all_chunks = np.zeros((Q, num_pred, action_dim), dtype=np.float32)

    client.reset()

    for qi, q in enumerate(query_model_steps):
        raw_t = q * step

        # Encode current camera frames
        img_left = encode_fn(video_frames["left_wrist"][raw_t])
        img_right = encode_fn(video_frames["right_wrist"][raw_t])
        img_head = encode_fn(video_frames["head"][raw_t])

        chunk = client.infer(img_left, img_right, img_head, states[raw_t])
        all_chunks[qi] = chunk

        if qi % 20 == 0:
            print(f"  Query {qi}/{Q} (model-step {q}, raw frame {raw_t})")

    return all_chunks, query_model_steps, M


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Online eval of BKL policy via ZMQ server. "
                    "Validates consistency with offline eval.")
    parser.add_argument("--episode-dir",
                        default="/mnt/amlfs-02/shared/human_egocentric/dniu/datasets/caip_proc/pick_place_egg/success/episode_0001",
                        help="Path to episode directory (H5 + videos)")
    parser.add_argument("--log-dir",
                        default="logs/bkl_pick-place-egg_vitb-mae_skip1_pred16_obs1_noise1x_bodyframe-6drot_actonly_proprio_right__041426_1119",
                        help="Training log directory (output saved here)")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5679, help="Server ZMQ port")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Frame skip matching training config (stride = frame_skip + 1)")
    parser.add_argument("--num-exec", type=int, default=4,
                        help="Model-steps between server re-queries")
    parser.add_argument("--smooth-lambda", type=float, default=0.2,
                        help="Exponential decay for ACT temporal ensemble (0 = uniform avg)")
    parser.add_argument("--jpeg-quality", type=int, default=100,
                        help="JPEG encoding quality for images sent to server")
    parser.add_argument("--lossless", action="store_true", default=True,
                        help="Send PNG (lossless) instead of JPEG")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save plots and data")
    args = parser.parse_args()

    frame_skip = args.frame_skip
    num_exec = args.num_exec
    step = frame_skip + 1

    # Connect to server and handshake
    print(f"Connecting to server at {args.host}:{args.port}...")
    client = PolicyClient(args.host, args.port)
    hs = client.handshake()
    assert hs["status"] == "success", f"Handshake failed: {hs}"

    side = hs["side"]
    action_dim = hs["action_dim"]
    num_pred = hs["num_pred"]
    use_proprio = hs["use_proprio"]
    print(f"Handshake OK: side={side}, action_dim={action_dim}, "
          f"num_pred={num_pred}, use_proprio={use_proprio}")

    if num_pred % num_exec != 0:
        raise ValueError(f"num_pred ({num_pred}) must be divisible by num_exec ({num_exec})")

    # Output directory (in log dir, matching offline eval pattern)
    if args.output_dir is None:
        ep_name = os.path.basename(args.episode_dir)
        args.output_dir = os.path.join(
            args.log_dir,
            f"online_eval_{ep_name}_ne{num_exec}_sl{args.smooth_lambda}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Load episode raw data
    print(f"Loading episode: {args.episode_dir}")
    states, left_arm_pose, right_arm_pose, video_frames, T = load_episode_raw(args.episode_dir)
    M = (T - 1) // step
    print(f"  Raw frames: {T}, Model timesteps: {M} (frame_skip={frame_skip}, step={step})")

    # Run online eval
    img_mode = "PNG (lossless)" if args.lossless else f"JPEG q={args.jpeg_quality}"
    print(f"Running online eval (num_exec={num_exec}, smooth_lambda={args.smooth_lambda}, "
          f"images={img_mode})...")
    all_chunks, query_model_steps, M = run_online_eval(
        client, states, video_frames, frame_skip, num_exec, num_pred,
        jpeg_quality=args.jpeg_quality, lossless=args.lossless
    )
    Q = len(query_model_steps)
    print(f"  Collected {Q} chunks over {M} model-timesteps")

    # Apply temporal smoothing (on denormalized chunks — equivalent to offline)
    smoothed_actions = apply_temporal_smoothing(
        all_chunks, query_model_steps, num_pred, num_exec, args.smooth_lambda, M
    )
    print(f"  Smoothed predictions: {smoothed_actions.shape}")

    # Convert to absolute targets and compute GT comparison
    pred_targets = np.zeros((M, action_dim), dtype=np.float32)
    gt_targets = np.zeros((M, action_dim), dtype=np.float32)

    for t in range(M):
        raw_t = t * step
        raw_t1 = min(raw_t + step, T - 1)

        pred_targets[t] = action_to_absolute_target(
            smoothed_actions[t], left_arm_pose[raw_t], right_arm_pose[raw_t]
        )
        gt_targets[t] = states[raw_t1]

    # Compute per-group L1 errors
    active_indices = BKL_Dataset.SIDE_INDICES[side] if side != 'both' else list(range(action_dim))
    l1_errors = np.abs(pred_targets - gt_targets).mean(axis=0)

    model_steps = np.arange(M)

    print("\nMean L1 error per DOF group (absolute target space):")
    for group_name, (start, end) in DOF_GROUPS.items():
        group_indices = [i for i in range(start, end) if i in active_indices]
        if not group_indices:
            print(f"  {group_name}: -- (masked)")
            continue
        group_error = l1_errors[group_indices].mean()
        print(f"  {group_name}: {group_error:.6f}")
    overall = l1_errors[active_indices].mean()
    print(f"  overall ({side}): {overall:.6f}")

    # Generate plots
    print(f"\nSaving plots to: {args.output_dir}")
    for group_name, (start, end) in DOF_GROUPS.items():
        plot_dof_group(model_steps, pred_targets, gt_targets, group_name,
                       start, end, args.output_dir)

    # Save data
    np.savez(
        os.path.join(args.output_dir, "online_eval_data.npz"),
        pred_targets=pred_targets,
        gt_targets=gt_targets,
        model_steps=model_steps,
        l1_errors=l1_errors,
        num_exec=num_exec,
        num_pred=num_pred,
        smooth_lambda=args.smooth_lambda,
        frame_skip=frame_skip,
        query_model_steps=np.array(query_model_steps),
        smoothed_actions=smoothed_actions,
        raw_chunks=all_chunks,
    )
    print("Saved online_eval_data.npz")

    # Save server-extracted features for comparison with features.h5
    server_feat_path = os.path.join(args.output_dir, "server_features.h5")
    resp = client.save_features(server_feat_path)
    if resp.get("status") == "success":
        print(f"Saved server features ({resp['num_frames']} frames) to {server_feat_path}")
    else:
        print(f"Warning: could not save server features: {resp}")

    client.close()
    print("Done.")


if __name__ == "__main__":
    main()
