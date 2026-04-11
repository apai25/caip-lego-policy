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


# DOF labels for the 58-dim action space (delta_eef mode)
DOF_LABELS = (
    ["left_dx", "left_dy", "left_dz", "left_qx", "left_qy", "left_qz", "left_qw"] +
    ["right_dx", "right_dy", "right_dz", "right_qx", "right_qy", "right_qz", "right_qw"] +
    [f"left_hand_j{i}" for i in range(22)] +
    [f"right_hand_j{i}" for i in range(22)]
)

DOF_GROUPS = {
    "left_arm_eef": (0, 7),
    "right_arm_eef": (7, 14),
    "left_hand": (14, 36),
    "right_hand": (36, 58),
}


def load_episode_data(episode_dir, action_type="delta_eef", frame_skip=1):
    """Load state, action, and features from an episode directory."""
    from mvp.bimanual_bc.dataset import pose_to_xyz_quat
    from scipy.spatial.transform import Rotation as R

    ep_name = os.path.basename(episode_dir)
    h5_path = os.path.join(episode_dir, f"{ep_name}.h5")
    feat_path = os.path.join(episode_dir, "features.h5")

    with h5py.File(h5_path, 'r') as f:
        left_arm_pose = f['left_arm_current_pose'][:].astype(np.float64)
        right_arm_pose = f['right_arm_current_pose'][:].astype(np.float64)
        left_hand_pos = f['left_hand_joint_positions'][:].astype(np.float32)
        right_hand_pos = f['right_hand_joint_positions'][:].astype(np.float32)

        if action_type == "delta_eef":
            left_arm_target_pose = f['left_arm_target_pose'][:].astype(np.float64)
            right_arm_target_pose = f['right_arm_target_pose'][:].astype(np.float64)
            left_hand_cmd = f['left_hand_target_joint_positions'][:].astype(np.float32)
            right_hand_cmd = f['right_hand_target_joint_positions'][:].astype(np.float32)
        else:
            left_arm_jpos = f['left_arm_joint_positions'][:].astype(np.float32)
            right_arm_jpos = f['right_arm_joint_positions'][:].astype(np.float32)
            left_arm_cmd = f['left_arm_target_dofs'][:].astype(np.float32)
            right_arm_cmd = f['right_arm_target_dofs'][:].astype(np.float32)
            left_hand_cmd = f['left_hand_target_joint_positions'][:].astype(np.float32)
            right_hand_cmd = f['right_hand_target_joint_positions'][:].astype(np.float32)

    # State: absolute EEF xyz+quat + hand joints
    left_eef = pose_to_xyz_quat(left_arm_pose)
    right_eef = pose_to_xyz_quat(right_arm_pose)

    if action_type == "delta_eef":
        states = np.concatenate([left_eef, right_eef, left_hand_pos, right_hand_pos], axis=1)
        # Compute per-timestep ground-truth actions with the given frame_skip
        T = len(states)
        actions = np.zeros((T, 58), dtype=np.float32)
        for k in range(T - 1):
            t1 = min(k + frame_skip + 1, T - 1)
            left_rel = np.linalg.inv(left_arm_target_pose[k]) @ left_arm_target_pose[t1]
            right_rel = np.linalg.inv(right_arm_target_pose[k]) @ right_arm_target_pose[t1]
            actions[k, :3] = left_rel[:3, 3]
            actions[k, 3:7] = R.from_matrix(left_rel[:3, :3]).as_quat()
            actions[k, 7:10] = right_rel[:3, 3]
            actions[k, 10:14] = R.from_matrix(right_rel[:3, :3]).as_quat()
            actions[k, 14:36] = left_hand_cmd[t1] - left_hand_cmd[k]
            actions[k, 36:58] = right_hand_cmd[t1] - right_hand_cmd[k]
    else:
        states = np.concatenate([left_arm_jpos, right_arm_jpos, left_hand_pos, right_hand_pos], axis=1)
        actions = np.concatenate([left_arm_cmd, right_arm_cmd, left_hand_cmd, right_hand_cmd], axis=1)

    with h5py.File(feat_path, 'r') as f:
        features = f['features'][:].astype(np.float32)

    return states, actions, features


def build_model(cfg):
    """Build the model from config (same logic as bc.py)."""
    prompt_output_dim = [cfg.actor.prompt_dim]
    img_output_dim = [cfg.actor.im_dim for _ in range(len(cfg.data.cams))]
    state_dim = getattr(cfg.data, 'state_dim', 24)
    action_dim = getattr(cfg.data, 'action_dim', 24)
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
def run_offline_eval(model, states, actions, features, prompt_embedding, num_steps):
    """Run sliding-window prediction across an entire episode.

    Returns:
        pred_actions: (T - num_steps, 58) predicted actions
        gt_actions: (T - num_steps, 58) ground-truth actions
        timesteps: (T - num_steps,) timestep indices
    """
    model.eval()
    T = len(states)
    num_cams = features.shape[1]
    feat_dim = features.shape[2]
    state_dim = states.shape[1]

    # Prepad: repeat first frame num_steps-1 times
    pad_len = num_steps - 1
    states_padded = np.concatenate([np.tile(states[0:1], (pad_len, 1)), states], axis=0)
    features_padded = np.concatenate([np.tile(features[0:1], (pad_len, 1, 1)), features], axis=0)

    prompt = torch.tensor(prompt_embedding, dtype=torch.float32).cuda()  # (768,)

    pred_actions = []
    gt_actions = []
    timesteps = []

    for t in range(T - 1):
        # Window: [t, t + num_steps) in padded arrays = timesteps [t - pad_len, t] in original
        window_start = t  # in padded indexing
        window_end = t + num_steps

        # State window: (num_steps, state_dim)
        state_window = torch.tensor(states_padded[window_start:window_end], dtype=torch.float32).cuda()

        # Feature window: (num_steps, num_cams, feat_dim)
        feat_window = torch.tensor(features_padded[window_start:window_end], dtype=torch.float32).cuda()

        # Prompt: repeat across timesteps (num_steps, prompt_dim)
        prompt_window = prompt.unsqueeze(0).repeat(num_steps, 1)

        # Add batch dimension
        state_window = state_window.unsqueeze(0)       # (1, T, 58)
        prompt_window = prompt_window.unsqueeze(0)      # (1, T, 768)

        # Split features per camera: list of (1, T, feat_dim)
        im_windows = [feat_window[:, cam_id, :].unsqueeze(0) for cam_id in range(num_cams)]

        # Build observation
        obs_each_mod = [prompt_window] + im_windows + [state_window]

        # Masks: all ones (everything visible)
        mask_each_mod = [torch.ones_like(m) for m in obs_each_mod]

        # Attention mask: no padding, just causal
        B = 1
        L = num_steps
        n_heads = 8  # from config
        att_mask = torch.zeros(B, L, L, device='cuda')
        causal_mask = torch.triu(torch.ones(L, L, device='cuda'), diagonal=1).bool()
        att_mask = causal_mask.unsqueeze(0).repeat(B * n_heads, 1, 1)

        # Forward
        preds = model(obs_each_mod=obs_each_mod, mask_each_mod=mask_each_mod, attn_mask=att_mask)

        # Extract action prediction (last element, shape (1, num_pred, action_dim))
        pred_action = preds[-1].squeeze(0).squeeze(0).cpu().numpy()  # (58,)

        # Ground truth: action at timestep t+1 (the target the model was trained to predict)
        gt_action = actions[t + 1]

        pred_actions.append(pred_action)
        gt_actions.append(gt_action)
        timesteps.append(t + 1)

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
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config-path", default="configs/bimanual_bc/config_bkl.yaml",
                        help="Path to config YAML")
    parser.add_argument("--output-dir", default=None, help="Directory to save plots (default: next to checkpoint)")
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config_path)

    # Output directory
    if args.output_dir is None:
        ep_name = os.path.basename(args.episode_dir)
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint), f"eval_{ep_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load action normalization stats
    action_stats_path = getattr(cfg.data, 'action_stats_path', None)
    action_mean, action_std = None, None
    if action_stats_path and os.path.exists(action_stats_path):
        stats = np.load(action_stats_path)
        action_mean = stats['mean'].astype(np.float32)
        action_std = np.maximum(stats['std'].astype(np.float32), 1e-6)
        print(f"Loaded action stats from {action_stats_path}")

    # Load episode data
    action_type = getattr(cfg.data, 'action_type', 'delta_eef')
    frame_skip = getattr(cfg.data, 'frame_skip', 1)
    print(f"Loading episode: {args.episode_dir}")
    states, actions, features = load_episode_data(args.episode_dir, action_type, frame_skip)
    print(f"  States: {states.shape}, Actions: {actions.shape}, Features: {features.shape}")

    # Normalize GT actions (to match what model was trained on)
    if action_mean is not None:
        actions_normed = (actions - action_mean) / action_std
    else:
        actions_normed = actions

    # Load prompt embedding
    prompt_embedding = np.load(cfg.data.prompt_embedding_path).astype(np.float32)

    # Build and load model
    print(f"Loading model from: {args.checkpoint}")
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    mks, uks = model.load_state_dict(ckpt['model_state'], strict=False)
    if mks:
        print(f"  Missing keys: {mks}")
    if uks:
        print(f"  Unexpected keys: {uks}")
    model.cuda()

    # Run prediction (model outputs normalized actions)
    num_steps = cfg.actor.num_steps
    print(f"Running offline eval (num_steps={num_steps}, episode length={len(states)})...")
    pred_actions_normed, gt_actions_normed, timesteps = run_offline_eval(
        model, states, actions_normed, features, prompt_embedding, num_steps
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
