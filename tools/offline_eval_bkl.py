#!/usr/bin/env python3

"""Offline evaluation of a trained BKL policy on a single episode.

Simulates real-time execution: queries the model at model-timestep cadence
(every num_exec * (frame_skip+1) raw frames), applies ACT-style temporal
ensemble smoothing, converts predicted deltas to absolute EEF targets,
and compares against ground-truth next-states.

Usage:
    python tools/offline_eval_bkl.py \
        --episode-dir /path/to/episode_XXXX \
        --log-dir /path/to/training/log \
        --epoch 200 \
        --num-exec 4 --smooth-lambda 0.1 \
        --output-dir /path/to/save/plots
"""

import argparse
import os
import sys

import h5py
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


# DOF labels for absolute EEF targets (xyz + rot6d)
TARGET_DOF_LABELS = (
    ["left_x", "left_y", "left_z"] + [f"left_r6d_{i}" for i in range(6)] +
    ["right_x", "right_y", "right_z"] + [f"right_r6d_{i}" for i in range(6)] +
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
    """Load state, action, features, and raw poses from an episode directory.

    Returns:
        states: (T, 62) absolute state [left_eef(9), right_eef(9), left_hand(22), right_hand(22)]
        actions: (T, 62) body-frame delta actions (computed with frame_skip)
        features: (T, num_cams, feat_dim) pre-extracted MAE features
        left_arm_pose: (T, 4, 4) raw homogeneous transforms
        right_arm_pose: (T, 4, 4) raw homogeneous transforms
    """
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

    # Compute per-timestep actions: body-frame delta + absolute hand targets
    T = len(states)
    actions = np.zeros((T, 62), dtype=np.float32)
    for k in range(T - 1):
        t1 = min(k + frame_skip + 1, T - 1)
        left_R = left_arm_pose[k, :3, :3]
        actions[k, 0:3] = left_R.T @ (left_arm_target_pose[t1, :3, 3] - left_arm_pose[k, :3, 3])
        left_R_delta = left_R.T @ left_arm_target_pose[t1, :3, :3]
        actions[k, 3:9] = np.concatenate([left_R_delta[:, 0], left_R_delta[:, 1]])
        right_R = right_arm_pose[k, :3, :3]
        actions[k, 9:12] = right_R.T @ (right_arm_target_pose[t1, :3, 3] - right_arm_pose[k, :3, 3])
        right_R_delta = right_R.T @ right_arm_target_pose[t1, :3, :3]
        actions[k, 12:18] = np.concatenate([right_R_delta[:, 0], right_R_delta[:, 1]])
        actions[k, 18:40] = left_hand_cmd[t1]
        actions[k, 40:62] = right_hand_cmd[t1]

    with h5py.File(feat_path, 'r') as f:
        features = f['features'][:].astype(np.float32)

    return states, actions, features, left_arm_pose, right_arm_pose


def build_model(cfg):
    """Build the model from config (same logic as bc.py)."""
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


def rot6d_to_matrix(rot6d):
    """Convert 6D rotation representation to 3x3 rotation matrix (Gram-Schmidt)."""
    v1, v2 = rot6d[0:3], rot6d[3:6]
    e1 = v1 / np.linalg.norm(v1)
    u2 = v2 - np.dot(e1, v2) * e1
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1, e2)
    return np.column_stack((e1, e2, e3))


def action_to_absolute_target(action, left_pose_4x4, right_pose_4x4):
    """Convert a body-frame delta action to a 62D absolute target.

    Args:
        action: (62,) denormalized action [left_delta(9), right_delta(9), left_hand(22), right_hand(22)]
        left_pose_4x4: (4, 4) current left arm homogeneous transform
        right_pose_4x4: (4, 4) current right arm homogeneous transform

    Returns:
        target: (62,) absolute target in the same format as state
    """
    target = np.zeros(62, dtype=np.float32)

    # Left arm: body-frame delta → world-frame target
    left_R = left_pose_4x4[:3, :3]
    left_pos = left_pose_4x4[:3, 3]
    target[0:3] = left_pos + left_R @ action[0:3]
    left_R_delta = rot6d_to_matrix(action[3:9])
    left_R_target = left_R @ left_R_delta
    target[3:9] = np.concatenate([left_R_target[:, 0], left_R_target[:, 1]])

    # Right arm: body-frame delta → world-frame target
    right_R = right_pose_4x4[:3, :3]
    right_pos = right_pose_4x4[:3, 3]
    target[9:12] = right_pos + right_R @ action[9:12]
    right_R_delta = rot6d_to_matrix(action[12:18])
    right_R_target = right_R @ right_R_delta
    target[12:18] = np.concatenate([right_R_target[:, 0], right_R_target[:, 1]])

    # Hands: already absolute targets
    target[18:62] = action[18:62]

    return target


@torch.no_grad()
def run_offline_eval(model, states, features, prompt_embedding, cfg,
                     frame_skip=1, batch_size=256, num_exec=1):
    """Run batched inference at model-timestep cadence.

    Model-timestep t corresponds to raw frame t * (frame_skip + 1).
    Queries happen every num_exec model-timesteps.

    Returns:
        all_preds: (Q, num_pred, action_dim) predicted action chunks per query (normalized)
        query_model_steps: list of Q query model-step indices
        M: total number of model-timesteps
    """
    from mvp.bimanual_bc.dataset import BKL_Dataset

    model.eval()
    T = len(states)
    step = frame_skip + 1
    M = (T - 1) // step
    num_cams = features.shape[1]
    num_steps = cfg.actor.num_steps
    num_pred = cfg.actor.num_pred
    n_heads = cfg.transformer_concat.num_heads
    action_dim = getattr(cfg.data, 'action_dim', 62)

    # Query at model-timesteps 0, num_exec, 2*num_exec, ...
    query_model_steps = list(range(0, M, num_exec))
    Q = len(query_model_steps)

    # Precompute context windows at model-timestep cadence
    all_states = []   # (Q, num_steps, state_dim)
    all_feats = []    # (Q, num_steps, num_cams, feat_dim)
    for q in query_model_steps:
        indices = []
        for i in range(num_steps):
            model_t = q - (num_steps - 1 - i)
            raw_idx = max(0, model_t) * step
            raw_idx = min(raw_idx, T - 1)
            indices.append(raw_idx)
        all_states.append(states[indices])
        all_feats.append(features[indices])
    all_states = np.stack(all_states)  # (Q, num_steps, state_dim)
    all_feats = np.stack(all_feats)    # (Q, num_steps, num_cams, feat_dim)

    # Precompute masks
    side = getattr(cfg.data, 'side', 'both')
    cam_keys = [f"feat_{c}" for c in cfg.data.cams]
    excluded_cam = None
    if side != 'both':
        excluded_cam = f"feat_{BKL_Dataset.SIDE_EXCLUDED_CAMS[side]}"
    cam_visible = [0.0 if cam_keys[c] == excluded_cam else 1.0 for c in range(num_cams)]

    use_proprio = getattr(cfg.data, 'use_proprio', True)
    state_mask_vec = np.ones(states.shape[1], dtype=np.float32)
    if not use_proprio:
        state_mask_vec[:] = 0.0
    elif side != 'both':
        other = 'left' if side == 'right' else 'right'
        for idx in BKL_Dataset.SIDE_INDICES[other]:
            state_mask_vec[idx] = 0.0

    L = num_steps
    causal_mask = torch.triu(torch.ones(L, L, device='cuda'), diagonal=1).bool()
    att_mask = causal_mask.unsqueeze(0).repeat(n_heads, 1, 1)

    prompt_t = torch.tensor(prompt_embedding, dtype=torch.float32).cuda()
    prompt_window = prompt_t.unsqueeze(0).repeat(num_steps, 1)  # (L, 768)
    state_mask_t = torch.tensor(state_mask_vec, dtype=torch.float32).cuda()

    # Run batched inference
    all_preds = np.zeros((Q, num_pred, action_dim), dtype=np.float32)

    for b_start in range(0, Q, batch_size):
        b_end = min(b_start + batch_size, Q)
        B = b_end - b_start

        states_b = torch.tensor(all_states[b_start:b_end], dtype=torch.float32).cuda()
        feats_b = torch.tensor(all_feats[b_start:b_end], dtype=torch.float32).cuda()

        prompt_b = prompt_window.unsqueeze(0).expand(B, -1, -1)
        im_list = [feats_b[:, :, c, :] for c in range(num_cams)]

        obs_each_mod = [prompt_b] + im_list + [states_b]

        prompt_mask = torch.ones_like(prompt_b)
        img_masks = [torch.full_like(im, cam_visible[c]) for c, im in enumerate(im_list)]
        s_mask = state_mask_t.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        mask_each_mod = [prompt_mask] + img_masks + [s_mask]

        att_mask_b = att_mask.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_heads, L, L)

        preds = model(obs_each_mod=obs_each_mod, mask_each_mod=mask_each_mod, attn_mask=att_mask_b)
        all_preds[b_start:b_end] = preds[-1].cpu().numpy()

    return all_preds, query_model_steps, M


def apply_temporal_smoothing(all_preds, query_steps, num_pred, num_exec, smooth_lambda, M):
    """Apply ACT-style temporal ensemble smoothing in model-timestep space.

    Args:
        all_preds: (Q, num_pred, action_dim) predicted action chunks per query
        query_steps: list of Q query model-step indices (0, num_exec, 2*num_exec, ...)
        num_pred: number of actions predicted per query
        num_exec: model-steps between consecutive queries
        smooth_lambda: exponential decay (0 = uniform average)
        M: total number of model-timesteps

    Returns:
        smoothed: (M, action_dim) temporally smoothed predicted actions
    """
    action_dim = all_preds.shape[2]
    K = num_pred // num_exec

    raw_weights = np.exp(-smooth_lambda * np.arange(K))

    smoothed = np.zeros((M, action_dim), dtype=np.float32)

    for t in range(M):
        latest_qi = t // num_exec
        latest_qi = min(latest_qi, len(query_steps) - 1)

        preds = []
        for k in range(K):
            qi = latest_qi - k
            if qi < 0:
                break
            q = query_steps[qi]
            offset = t - q
            if offset < 0 or offset >= num_pred:
                break
            preds.append(all_preds[qi, offset])

        if not preds:
            continue

        n = len(preds)
        w = raw_weights[:n]
        w = w / w.sum()
        smoothed[t] = (np.stack(preds) * w[:, None]).sum(axis=0)

    return smoothed


def plot_dof_group(model_steps, pred_targets, gt_targets, group_name, dof_start, dof_end, output_dir):
    """Plot predicted vs ground-truth absolute targets for a group of DOFs."""
    n_dofs = dof_end - dof_start
    cols = min(4, n_dofs)
    rows = (n_dofs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    fig.suptitle(f"{group_name} — Absolute Target (Predicted vs GT)", fontsize=14)

    for i in range(n_dofs):
        dof_idx = dof_start + i
        ax = axes[i // cols][i % cols]
        ax.plot(model_steps, gt_targets[:, dof_idx], label='GT', alpha=0.7, linewidth=0.8)
        ax.plot(model_steps, pred_targets[:, dof_idx], label='Pred', alpha=0.7, linewidth=0.8)
        ax.set_title(TARGET_DOF_LABELS[dof_idx], fontsize=9)
        ax.set_xlabel('model step', fontsize=8)
        ax.set_ylabel('value', fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

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
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--num-exec", type=int, default=1,
                        help="Model-steps between model re-queries (default: 1)")
    parser.add_argument("--smooth-lambda", type=float, default=0.0,
                        help="Exponential decay for ACT temporal ensemble (0 = uniform avg)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save plots (default: log-dir/eval_<episode>)")
    args = parser.parse_args()

    # Load train metadata
    meta = torch.load(os.path.join(args.log_dir, "train_meta.pt"), map_location='cpu')
    cfg = OmegaConf.create(meta['cfg'])

    num_exec = args.num_exec
    num_pred = cfg.actor.num_pred
    frame_skip = getattr(cfg.data, 'frame_skip', 1)
    step = frame_skip + 1

    if num_exec < 1:
        raise ValueError(f"num_exec must be >= 1, got {num_exec}")
    if num_pred % num_exec != 0:
        raise ValueError(f"num_pred ({num_pred}) must be divisible by num_exec ({num_exec})")

    # Find checkpoint
    if args.epoch is not None:
        ckpt_path = os.path.join(args.log_dir, f"model_ep{args.epoch:04d}.pt")
    else:
        import glob
        ckpts = sorted(glob.glob(os.path.join(args.log_dir, "model_ep*.pt")))
        assert ckpts, f"No checkpoints found in {args.log_dir}"
        ckpt_path = ckpts[-1]
    print(f"Checkpoint: {ckpt_path}")

    # Action normalization stats
    action_q01, action_range = None, None
    if 'action_q01' in meta:
        action_q01 = np.array(meta['action_q01'], dtype=np.float32)
        action_q99 = np.array(meta['action_q99'], dtype=np.float32)
        action_range = np.maximum(action_q99 - action_q01, 1e-6)

    # Output directory
    if args.output_dir is None:
        ep_name = os.path.basename(args.episode_dir)
        epoch_str = os.path.basename(ckpt_path).replace('.pt', '')
        args.output_dir = os.path.join(args.log_dir, f"eval_{ep_name}_{epoch_str}_ne{num_exec}_sl{args.smooth_lambda}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load episode data
    print(f"Loading episode: {args.episode_dir}")
    states, _actions, features, left_arm_pose, right_arm_pose = load_episode_data(args.episode_dir, frame_skip)
    T = len(states)
    M = (T - 1) // step
    print(f"  Raw frames: {T}, Model timesteps: {M} (frame_skip={frame_skip}, step={step})")
    print(f"  States: {states.shape}, Features: {features.shape}")

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

    # Run prediction at model-timestep cadence (outputs normalized actions)
    print(f"Running offline eval (num_steps={cfg.actor.num_steps}, num_pred={num_pred}, "
          f"num_exec={num_exec}, smooth_lambda={args.smooth_lambda})...")
    all_preds_normed, query_model_steps, M = run_offline_eval(
        model, states, features, prompt_embedding, cfg,
        frame_skip=frame_skip, batch_size=args.batch_size, num_exec=num_exec
    )
    print(f"  Queries: {len(query_model_steps)} (over {M} model-timesteps)")

    # Apply temporal smoothing in model-timestep space (still normalized)
    smoothed_normed = apply_temporal_smoothing(
        all_preds_normed, query_model_steps, num_pred, num_exec, args.smooth_lambda, M
    )
    print(f"  Smoothed predictions: {smoothed_normed.shape}")

    # Denormalize: [-1, 1] → [q01, q99]
    if action_q01 is not None:
        smoothed_actions = (smoothed_normed + 1) / 2 * action_range + action_q01
    else:
        smoothed_actions = smoothed_normed

    # Convert smoothed deltas to absolute targets (open-loop, using GT state)
    # Also compute GT absolute targets for comparison
    pred_targets = np.zeros((M, 62), dtype=np.float32)
    gt_targets = np.zeros((M, 62), dtype=np.float32)

    for t in range(M):
        raw_t = t * step
        raw_t1 = min(raw_t + step, T - 1)

        # Predicted: apply smoothed delta to GT current state
        pred_targets[t] = action_to_absolute_target(
            smoothed_actions[t], left_arm_pose[raw_t], right_arm_pose[raw_t]
        )

        # GT: the actual next state at the target frame
        gt_targets[t] = states[raw_t1]

    model_steps = np.arange(M)

    # Compute per-group L1 errors in absolute target space
    side = getattr(cfg.data, 'side', 'both')
    from mvp.bimanual_bc.dataset import BKL_Dataset
    active_indices = BKL_Dataset.SIDE_INDICES[side] if side != 'both' else list(range(62))

    l1_errors = np.abs(pred_targets - gt_targets).mean(axis=0)
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
        plot_dof_group(model_steps, pred_targets, gt_targets, group_name, start, end, args.output_dir)

    # Save data for analysis
    np.savez(
        os.path.join(args.output_dir, "eval_data.npz"),
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
    )
    print(f"Saved eval_data.npz")


if __name__ == "__main__":
    main()
