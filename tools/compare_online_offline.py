#!/usr/bin/env python3

"""Compare online eval (via server) with offline eval results.

Loads the .npz files saved by online_eval_bkl.py and offline_eval_bkl.py,
compares raw action chunks, smoothed actions, absolute targets, and
optionally compares server-extracted features with pre-extracted features.h5.

Usage:
    python tools/compare_online_offline.py \
        --online-npz /path/to/online_eval_data.npz \
        --offline-npz /path/to/eval_data.npz \
        --output-dir /path/to/save/comparison_plots

    # With feature comparison (requires episode dir with features.h5
    # and server features saved via the server's save_features endpoint):
    python tools/compare_online_offline.py \
        --online-npz /path/to/online_eval_data.npz \
        --offline-npz /path/to/eval_data.npz \
        --episode-dir /path/to/episode_XXXX \
        --server-features /path/to/server_features.h5
"""

import argparse
import os
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.offline_eval_bkl import DOF_GROUPS, TARGET_DOF_LABELS


def compare_chunks(online_chunks, offline_chunks):
    """Compare raw action chunks and print per-DOF-group discrepancies."""
    Q_on, Q_off = len(online_chunks), len(offline_chunks)
    Q = min(Q_on, Q_off)
    if Q_on != Q_off:
        print(f"  Warning: online has {Q_on} queries, offline has {Q_off}. "
              f"Comparing first {Q}.")

    diff = np.abs(online_chunks[:Q] - offline_chunks[:Q])

    print(f"\nRaw chunk comparison ({Q} queries, {online_chunks.shape[1]} preds each):")
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Max abs diff:  {diff.max():.6f}")
    for group_name, (start, end) in DOF_GROUPS.items():
        print(f"  {group_name}: mean={diff[:, :, start:end].mean():.6f}, "
              f"max={diff[:, :, start:end].max():.6f}")

    return diff


def compare_smoothed(online_smoothed, offline_smoothed):
    """Compare smoothed (temporally ensembled) actions."""
    M_on, M_off = len(online_smoothed), len(offline_smoothed)
    M = min(M_on, M_off)
    if M_on != M_off:
        print(f"  Warning: online has {M_on} steps, offline has {M_off}. "
              f"Comparing first {M}.")

    diff = np.abs(online_smoothed[:M] - offline_smoothed[:M])

    print(f"\nSmoothed action comparison ({M} model-steps):")
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Max abs diff:  {diff.max():.6f}")
    for group_name, (start, end) in DOF_GROUPS.items():
        print(f"  {group_name}: mean={diff[:, start:end].mean():.6f}, "
              f"max={diff[:, start:end].max():.6f}")

    return diff


def compare_targets(online_targets, offline_targets):
    """Compare absolute EEF targets."""
    M_on, M_off = len(online_targets), len(offline_targets)
    M = min(M_on, M_off)
    if M_on != M_off:
        print(f"  Warning: online has {M_on} steps, offline has {M_off}. "
              f"Comparing first {M}.")

    diff = np.abs(online_targets[:M] - offline_targets[:M])

    print(f"\nAbsolute target comparison ({M} model-steps):")
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Max abs diff:  {diff.max():.6f}")
    for group_name, (start, end) in DOF_GROUPS.items():
        print(f"  {group_name}: mean={diff[:, start:end].mean():.6f}, "
              f"max={diff[:, start:end].max():.6f}")

    return diff


def compare_features(server_feat_path, episode_dir, query_model_steps, frame_skip):
    """Compare server-extracted features with pre-extracted features.h5."""
    preextracted_path = os.path.join(episode_dir, "features.h5")
    if not os.path.exists(preextracted_path):
        print("\nNo pre-extracted features.h5 found, skipping feature comparison")
        return

    with h5py.File(server_feat_path, 'r') as f:
        feat_srv = f['features'][:].astype(np.float32)
    with h5py.File(preextracted_path, 'r') as f:
        feat_pre = f['features'][:].astype(np.float32)

    step = frame_skip + 1
    raw_indices = [q * step for q in query_model_steps]
    Q = min(len(feat_srv), len(raw_indices))
    feat_pre_at_queries = feat_pre[raw_indices[:Q]]

    diff = np.abs(feat_srv[:Q] - feat_pre_at_queries)
    rel_diff = diff / (np.abs(feat_pre_at_queries) + 1e-8)

    cam_names = ["left_wrist", "right_wrist", "head"]
    print(f"\nFeature comparison (server vs pre-extracted, {Q} frames):")
    print(f"  Overall mean abs diff: {diff.mean():.6f}")
    print(f"  Overall mean rel diff: {rel_diff.mean():.6f}")
    for c, cam in enumerate(cam_names):
        print(f"  {cam}: abs={diff[:, c].mean():.6f}, rel={rel_diff[:, c].mean():.6f}")


def plot_overlay(model_steps, online_targets, offline_targets,
                 group_name, dof_start, dof_end, output_dir):
    """Plot online vs offline absolute targets for a DOF group."""
    n_dofs = dof_end - dof_start
    cols = min(4, n_dofs)
    rows = (n_dofs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    fig.suptitle(f"{group_name} — Online vs Offline", fontsize=14)

    M = min(len(online_targets), len(offline_targets))
    steps = model_steps[:M]

    for i in range(n_dofs):
        dof_idx = dof_start + i
        ax = axes[i // cols][i % cols]
        ax.plot(steps, offline_targets[:M, dof_idx], label='Offline',
                color='#1f77b4', alpha=0.9, linewidth=1.0)
        ax.plot(steps, online_targets[:M, dof_idx], label='Online',
                color='#d62728', alpha=0.7, linewidth=1.0, linestyle='--')
        ax.set_title(TARGET_DOF_LABELS[dof_idx], fontsize=9)
        ax.set_xlabel('model step', fontsize=8)
        ax.set_ylabel('value', fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for i in range(n_dofs, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"cmp_{group_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare online eval (server) with offline eval results")
    _LOG = "logs/bkl_pick-place-egg_vitb-mae_skip1_pred16_obs1_noise1x_bodyframe-6drot_actonly_proprio_right__041426_1119"

    parser.add_argument("--online-npz",
                        default=os.path.join(_LOG, "online_eval_episode_0001_ne4_sl0.2/online_eval_data.npz"),

                        help="Path to online_eval_data.npz from online_eval_bkl.py")
    parser.add_argument("--offline-npz",
                        default=os.path.join(_LOG, "eval_episode_0001_model_ep0200_ne4_sl0.2/eval_data.npz"),
                        help="Path to eval_data.npz from offline_eval_bkl.py")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save comparison plots (default: alongside online npz)")
    parser.add_argument("--episode-dir", default=None,
                        help="Episode directory with features.h5 for feature comparison")
    parser.add_argument("--server-features", default=None,
                        help="Path to server_features.h5 saved via save_features endpoint")
    args = parser.parse_args()

    online = np.load(args.online_npz)
    offline = np.load(args.offline_npz)

    # Print params
    print("Online eval params:")
    for key in ['num_exec', 'num_pred', 'smooth_lambda', 'frame_skip']:
        if key in online:
            print(f"  {key}: {online[key]}")
    print("Offline eval params:")
    for key in ['num_exec', 'num_pred', 'smooth_lambda', 'frame_skip']:
        if key in offline:
            print(f"  {key}: {offline[key]}")

    # Compare raw chunks
    compare_chunks(online['raw_chunks'], offline['raw_chunks'])

    # Compare smoothed actions
    if 'smoothed_actions' in online and 'smoothed_actions' in offline:
        compare_smoothed(online['smoothed_actions'], offline['smoothed_actions'])

    # Compare absolute targets
    if 'pred_targets' in online and 'pred_targets' in offline:
        compare_targets(online['pred_targets'], offline['pred_targets'])

    # Feature comparison
    if args.server_features and args.episode_dir:
        frame_skip = int(online['frame_skip']) if 'frame_skip' in online else 1
        query_steps = online['query_model_steps'].tolist()
        compare_features(args.server_features, args.episode_dir, query_steps, frame_skip)

    # Overlay plots
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.online_npz), "comparison")
    os.makedirs(args.output_dir, exist_ok=True)

    if 'pred_targets' in online and 'pred_targets' in offline:
        on_tgt = online['pred_targets']
        off_tgt = offline['pred_targets']
        M = min(len(on_tgt), len(off_tgt))
        model_steps = np.arange(M)

        print(f"\nSaving overlay plots to: {args.output_dir}")
        for group_name, (start, end) in DOF_GROUPS.items():
            plot_overlay(model_steps, on_tgt, off_tgt,
                         group_name, start, end, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
