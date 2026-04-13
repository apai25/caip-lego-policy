#!/usr/bin/env python3

"""Offline evaluation of a trained BKL policy on a single episode.

Loads model weights and runs sliding-window prediction across an entire episode,
then generates per-DOF plots comparing predicted vs ground-truth actions.

Usage:
    python tools/offline_eval_bkl.py \
        --episode-dir /path/to/episode_XXXX \
        --checkpoint /path/to/model_ep0100.pt \
        --config-path configs/bimanual_bc/config_bkl.yaml \
        --output-dir /path/to/save/plots
"""

import argparse
import os
import sys

import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mvp.bimanual_bc.actor import ActorTransformerConcat_Bimanual, \
    ActorTransformerConcatAttnPooling_Bimanual, \
    ActorTransformerConcatMeanPooling_Bimanual, \
    ActorTransformerConcatAttnPoolingDiffusion_Bimanual


# DOF labels for the 62-dim action space (body-frame delta_eef + 6D rotation)
DOF_LABELS = (
    ["left_dx", "left_dy", "left_dz"] + [f"left_r6d_{i}" for i in range(6)] +
    ["right_dx", "right_dy", "right_dz"] + [f"right_r6d_{i}" for i in range(6)] +
    [f"left_hand_j{i}" for i in range(22)] +
    [f"right_hand_j{i}" for i in range(22)]
)

DOF_GROUPS = {
    "left_arm_eef": (0, 9),
    "right_arm_eef": (9, 18),
    "left_hand": (18, 40),
    "right_hand": (40, 62),
}


def load_episode_data(episode_dir, frame_skip=1):
    """Load state, action, and features from an episode directory."""
    from mvp.bimanual_bc.dataset import pose_to_xyz_rot6d

    ep_name = os.path.basename(episode_dir)
    h5_path = os.path.join(episode_dir, f"{ep_name}.h5")
    feat_path = os.path.join(episode_dir, "features.h5")

    with h5py.File(h5_path, 'r') as f:
        left_arm_pose = f['left_arm_current_pose'][:].astype(np.float64)
        right_arm_pose = f['right_arm_current_pose'][:].astype(np.float64)
        left_hand_pos = f['left_hand_joint_positions'][:].astype(np.float32)
        right_hand_pos = f['right_hand_joint_positions'][:].astype(np.float32)
        left_arm_target_pose = f['left_arm_target_pose'][:].astype(np.float64)
        right_arm_target_pose = f['right_arm_target_pose'][:].astype(np.float64)
        left_hand_cmd = f['left_hand_target_joint_positions'][:].astype(np.float32)
        right_hand_cmd = f['right_hand_target_joint_positions'][:].astype(np.float32)

    # State: absolute EEF xyz+rot6d + hand joints
    left_eef = pose_to_xyz_rot6d(left_arm_pose)    # (T, 9)
    right_eef = pose_to_xyz_rot6d(right_arm_pose)  # (T, 9)
    states = np.concatenate([left_eef, right_eef, left_hand_pos, right_hand_pos], axis=1)  # (T, 62)

    # Compute per-timestep body-frame ground-truth actions with 6D rotation
    T = len(states)
    actions = np.zeros((T, 62), dtype=np.float32)
    for k in range(T - 1):
        t1 = min(k + frame_skip + 1, T - 1)
        # Left arm: body-frame delta
        left_R_curr = left_arm_target_pose[k, :3, :3]
        actions[k, 0:3] = left_R_curr.T @ (left_arm_target_pose[t1, :3, 3] - left_arm_target_pose[k, :3, 3])
        left_R_delta = left_R_curr.T @ left_arm_target_pose[t1, :3, :3]
        actions[k, 3:9] = np.concatenate([left_R_delta[:, 0], left_R_delta[:, 1]])
        # Right arm: body-frame delta
        right_R_curr = right_arm_target_pose[k, :3, :3]
        actions[k, 9:12] = right_R_curr.T @ (right_arm_target_pose[t1, :3, 3] - right_arm_target_pose[k, :3, 3])
        right_R_delta = right_R_curr.T @ right_arm_target_pose[t1, :3, :3]
        actions[k, 12:18] = np.concatenate([right_R_delta[:, 0], right_R_delta[:, 1]])
        # Hands
        actions[k, 18:40] = left_hand_cmd[t1] - left_hand_cmd[k]
        actions[k, 40:62] = right_hand_cmd[t1] - right_hand_cmd[k]

    with h5py.File(feat_path, 'r') as f:
        features = f['features'][:].astype(np.float32)

    return states, actions, features


def build_model(cfg):
    """Build the model from config (same logic as bc.py).

    cfg can be an OmegaConf object or a plain dict (from train_meta.pt).
    """
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    prompt_output_dim = [cfg.actor.prompt_dim]
    img_output_dim = [cfg.actor.im_dim for _ in range(len(cfg.data.cams))]
    state_dim = getattr(cfg.data, 'state_dim', 62)
    action_dim = getattr(cfg.data, 'action_dim', 62)
    state_output_dim = [state_dim]
    action_output_dim = [action_dim]
    output_dims = prompt_output_dim + img_output_dim + state_output_dim + action_output_dim

    if cfg.actor.type == "transformer_concat":
        model = ActorTransformerConcat_Bimanual(
            context_length=cfg.actor.num_steps,
            obs_shape=cfg.actor.obs_dim,
            output_dims=output_dims,
            num_pred=cfg.actor.num_pred,
            policy_cfg=cfg.transformer_concat,
            normalize_input=cfg.transformer_concat.normalize_input,
            normalize_bn_cls=nn.BatchNorm1d,
        )
    elif cfg.actor.type == "transformer_concat_attnpool":
        model = ActorTransformerConcatAttnPooling_Bimanual(
            context_length=cfg.actor.num_steps,
            obs_shape=cfg.actor.obs_dim,
            output_dims=output_dims,
            num_pred=cfg.actor.num_pred,
            policy_cfg=cfg.transformer_concat,
            normalize_input=cfg.transformer_concat.normalize_input,
            normalize_bn_cls=nn.BatchNorm1d,
            num_cam=len(cfg.data.cams),
            im_dim=cfg.actor.im_dim,
            prompt_dim=cfg.actor.prompt_dim,
        )
    else:
        raise ValueError(f"Unsupported actor type for eval: {cfg.actor.type}")

    return model


@torch.no_grad()
def run_offline_eval(model, states, actions, features, prompt_embedding, cfg):
    """Run sliding-window prediction across an entire episode.

    The model predicts num_pred future steps per input. We use the first
    predicted step (next-step prediction) for comparison with GT.

    Returns:
        pred_actions: (T-1, action_dim) first-step predicted actions
        gt_actions: (T-1, action_dim) ground-truth actions
        timesteps: (T-1,) timestep indices
    """
    model.eval()
    T = len(states)
    num_cams = features.shape[1]
    num_steps = cfg.actor.num_steps
    n_heads = cfg.transformer_concat.num_heads

    prompt = torch.tensor(prompt_embedding, dtype=torch.float32).cuda()

    pred_actions = []
    gt_actions = []
    timesteps = []

    for t in range(T - 1):
        # State: just current frame (num_steps=1 typical)
        start = max(0, t - num_steps + 1)
        end = t + 1
        pad_len = num_steps - (end - start)

        state_window = torch.tensor(states[start:end], dtype=torch.float32).cuda()
        feat_window = torch.tensor(features[start:end], dtype=torch.float32).cuda()

        # Pad from the left if needed
        if pad_len > 0:
            state_window = torch.cat([state_window[:1].repeat(pad_len, 1), state_window], dim=0)
            feat_window = torch.cat([feat_window[:1].repeat(pad_len, 1, 1), feat_window], dim=0)

        # Add batch dimension
        prompt_window = prompt.unsqueeze(0).repeat(num_steps, 1).unsqueeze(0)  # (1, L, 768)
        state_window = state_window.unsqueeze(0)   # (1, L, state_dim)
        im_windows = [feat_window[:, c, :].unsqueeze(0) for c in range(num_cams)]

        obs_each_mod = [prompt_window] + im_windows + [state_window]

        # Build masks matching training: respect use_proprio, side, and cam visibility
        prompt_mask = torch.ones_like(prompt_window)
        img_masks = []
        side = getattr(cfg.data, 'side', 'both')
        excluded_cam = None
        if side != 'both':
            from mvp.bimanual_bc.dataset import BKL_Dataset
            excluded_cam = f"feat_{BKL_Dataset.SIDE_EXCLUDED_CAMS[side]}"
        cam_keys = [f"feat_{c}" for c in cfg.data.cams]
        for c_idx in range(num_cams):
            m = torch.ones_like(im_windows[c_idx])
            if cam_keys[c_idx] == excluded_cam:
                m = torch.zeros_like(m)
            img_masks.append(m)
        state_mask = torch.ones_like(state_window)
        if not getattr(cfg.data, 'use_proprio', True):
            state_mask = torch.zeros_like(state_mask)
        elif side != 'both':
            from mvp.bimanual_bc.dataset import BKL_Dataset
            other = 'left' if side == 'right' else 'right'
            for idx in BKL_Dataset.SIDE_INDICES[other]:
                state_mask[:, :, idx] = 0.0
        mask_each_mod = [prompt_mask] + img_masks + [state_mask]

        # Causal attention mask
        L = num_steps
        causal_mask = torch.triu(torch.ones(L, L, device='cuda'), diagonal=1).bool()
        att_mask = causal_mask.unsqueeze(0).repeat(n_heads, 1, 1)

        preds = model(obs_each_mod=obs_each_mod, mask_each_mod=mask_each_mod, attn_mask=att_mask)

        # preds[-1] is action predictions: (1, num_pred, action_dim)
        # Take first predicted step (next-step action)
        pred_action = preds[-1][0, 0].cpu().numpy()

        pred_actions.append(pred_action)
        gt_actions.append(actions[t])
        timesteps.append(t)

    return np.array(pred_actions), np.array(gt_actions), np.array(timesteps)


def plot_dof_group(timesteps, pred_actions, gt_actions, group_name, dof_start, dof_end, output_dir):
    """Plot predicted vs ground-truth for a group of DOFs."""
    n_dofs = dof_end - dof_start
    cols = min(4, n_dofs)
    rows = (n_dofs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    fig.suptitle(f"{group_name} — Predicted vs Ground Truth", fontsize=14)

    for i in range(n_dofs):
        dof_idx = dof_start + i
        ax = axes[i // cols][i % cols]
        ax.plot(timesteps, gt_actions[:, dof_idx], label='GT', alpha=0.7, linewidth=0.8)
        ax.plot(timesteps, pred_actions[:, dof_idx], label='Pred', alpha=0.7, linewidth=0.8)
        ax.set_title(DOF_LABELS[dof_idx], fontsize=9)
        ax.set_xlabel('timestep', fontsize=8)
        ax.set_ylabel('rad', fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for i in range(n_dofs, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{group_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline eval of trained BKL policy")
    parser.add_argument("--episode-dir", required=True, help="Path to episode directory")
    parser.add_argument("--log-dir", required=True,
                        help="Path to training log directory (contains train_meta.pt and model checkpoints)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Checkpoint epoch to load (default: latest)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save plots (default: log-dir/eval_<episode>)")
    args = parser.parse_args()

    # Load train metadata (config + action stats + noise stats)
    meta = torch.load(os.path.join(args.log_dir, "train_meta.pt"), map_location='cpu')
    cfg = OmegaConf.create(meta['cfg'])

    # Find checkpoint
    if args.epoch is not None:
        ckpt_path = os.path.join(args.log_dir, f"model_ep{args.epoch:04d}.pt")
    else:
        import glob
        ckpts = sorted(glob.glob(os.path.join(args.log_dir, "model_ep*.pt")))
        assert ckpts, f"No checkpoints found in {args.log_dir}"
        ckpt_path = ckpts[-1]
    print(f"Checkpoint: {ckpt_path}")

    # Action normalization stats from meta
    action_mean, action_std = None, None
    if 'action_mean' in meta:
        action_mean = np.array(meta['action_mean'], dtype=np.float32)
        action_std = np.maximum(np.array(meta['action_std'], dtype=np.float32), 1e-6)

    # Output directory
    if args.output_dir is None:
        ep_name = os.path.basename(args.episode_dir)
        epoch_str = os.path.basename(ckpt_path).replace('.pt', '')
        args.output_dir = os.path.join(args.log_dir, f"eval_{ep_name}_{epoch_str}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load episode data
    frame_skip = getattr(cfg.data, 'frame_skip', 1)
    print(f"Loading episode: {args.episode_dir}")
    states, actions, features = load_episode_data(args.episode_dir, frame_skip)
    print(f"  States: {states.shape}, Actions: {actions.shape}, Features: {features.shape}")

    # Normalize GT actions (to match what model was trained on)
    if action_mean is not None:
        actions_normed = (actions - action_mean) / action_std
    else:
        actions_normed = actions

    # Load prompt embedding
    prompt_embedding = np.load(cfg.data.prompt_embedding_path).astype(np.float32)

    # Build and load model
    print(f"Loading model from: {ckpt_path}")
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    mks, uks = model.load_state_dict(ckpt['model_state'], strict=False)
    if mks:
        print(f"  Missing keys: {mks}")
    if uks:
        print(f"  Unexpected keys: {uks}")
    model.cuda()

    # Run prediction (model outputs normalized actions)
    print(f"Running offline eval (num_steps={cfg.actor.num_steps}, num_pred={cfg.actor.num_pred}, episode length={len(states)})...")
    pred_actions_normed, gt_actions_normed, timesteps = run_offline_eval(
        model, states, actions_normed, features, prompt_embedding, cfg
    )
    print(f"  Predictions: {pred_actions_normed.shape}")

    # Unnormalize for plotting
    if action_mean is not None:
        pred_actions = pred_actions_normed * action_std + action_mean
        gt_actions_plot = gt_actions_normed * action_std + action_mean
    else:
        pred_actions = pred_actions_normed
        gt_actions_plot = gt_actions_normed

    # Compute per-group L1 errors (in unnormalized space)
    l1_errors = np.abs(pred_actions - gt_actions_plot).mean(axis=0)
    print("\nMean L1 error per DOF group:")
    for group_name, (start, end) in DOF_GROUPS.items():
        group_error = l1_errors[start:end].mean()
        print(f"  {group_name}: {group_error:.6f}")
    print(f"  overall: {l1_errors.mean():.6f}")

    # Generate plots (unnormalized)
    print(f"\nSaving plots to: {args.output_dir}")
    for group_name, (start, end) in DOF_GROUPS.items():
        plot_dof_group(timesteps, pred_actions, gt_actions_plot, group_name, start, end, args.output_dir)

    # Also save raw data for further analysis
    np.savez(
        os.path.join(args.output_dir, "eval_data.npz"),
        pred_actions=pred_actions,
        gt_actions=gt_actions_plot,
        timesteps=timesteps,
        l1_errors=l1_errors,
    )
    print(f"Saved eval_data.npz")


if __name__ == "__main__":
    main()
