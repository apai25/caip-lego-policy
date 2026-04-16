#!/usr/bin/env python3

"""Compare server-extracted MAE features with pre-extracted features.h5.

The server extracts features from images sent during online eval (at query
frames only). features.h5 contains features extracted offline from raw MP4
frames (all T frames). This script aligns them by frame index and compares.

Usage:
    python tools/compare_features.py

    # Or with custom paths:
    python tools/compare_features.py \
        --server-features <online_eval_dir>/server_features.h5 \
        --episode-dir /path/to/episode_0001 \
        --online-npz <online_eval_dir>/online_eval_data.npz
"""

import argparse
import os
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_LOG = "logs/bkl_pick-place-egg_vitb-mae_skip1_pred16_obs1_noise1x_bodyframe-6drot_actonly_proprio_right__041426_1119"
_ONLINE_DIR = os.path.join(_LOG, "online_eval_episode_0001_ne4_sl0.2")

CAM_NAMES = ["left_wrist", "right_wrist", "head"]


def main():
    parser = argparse.ArgumentParser(
        description="Compare server-extracted features with pre-extracted features.h5")
    parser.add_argument("--server-features",
                        default=os.path.join(_ONLINE_DIR, "server_features.h5"),
                        help="Path to server_features.h5 saved by online eval")
    parser.add_argument("--episode-dir",
                        default="/mnt/amlfs-02/shared/human_egocentric/dniu/datasets/caip_proc/pick_place_egg/success/episode_0001",
                        help="Episode directory containing features.h5")
    parser.add_argument("--online-npz",
                        default=os.path.join(_ONLINE_DIR, "online_eval_data.npz"),
                        help="online_eval_data.npz (for query_model_steps and frame_skip)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save plots (default: alongside server features)")
    args = parser.parse_args()

    # Load online eval metadata for frame index mapping
    online = np.load(args.online_npz)
    query_model_steps = online['query_model_steps'].tolist()
    frame_skip = int(online['frame_skip'])
    step = frame_skip + 1
    raw_indices = [q * step for q in query_model_steps]

    # Load features
    with h5py.File(args.server_features, 'r') as f:
        feat_srv = f['features'][:].astype(np.float32)  # (Q, num_cams, 768)

    preextracted_path = os.path.join(args.episode_dir, "features.h5")
    with h5py.File(preextracted_path, 'r') as f:
        feat_all = f['features'][:].astype(np.float32)  # (T, num_cams, 768)

    Q = min(len(feat_srv), len(raw_indices))
    feat_pre = feat_all[raw_indices[:Q]]

    print(f"Server features:    {feat_srv.shape} (dtype={feat_srv.dtype})")
    print(f"Pre-extracted (T):  {feat_all.shape} (dtype={feat_all.dtype})")
    print(f"Pre-extracted at query frames: {feat_pre.shape}")
    print(f"Frames compared: {Q}")
    print(f"frame_skip={frame_skip}, step={step}")

    # Overall stats
    diff = np.abs(feat_srv[:Q] - feat_pre)
    rel_diff = diff / (np.abs(feat_pre) + 1e-8)

    print(f"\nOverall:")
    print(f"  Mean abs diff:  {diff.mean():.8f}")
    print(f"  Max abs diff:   {diff.max():.8f}")
    print(f"  Mean rel diff:  {rel_diff.mean():.8f}")
    print(f"  Exactly equal:  {np.array_equal(feat_srv[:Q], feat_pre)}")
    print(f"  Close (1e-5):   {np.allclose(feat_srv[:Q], feat_pre, atol=1e-5)}")
    print(f"  Close (1e-4):   {np.allclose(feat_srv[:Q], feat_pre, atol=1e-4)}")

    # Per-camera stats
    print(f"\nPer-camera:")
    print(f"  {'camera':<15} {'mean_abs':>10} {'max_abs':>10} {'mean_rel':>10} {'max_rel':>10}")
    print(f"  {'-'*55}")
    for c, cam in enumerate(CAM_NAMES):
        cd = diff[:, c]
        cr = rel_diff[:, c]
        print(f"  {cam:<15} {cd.mean():>10.8f} {cd.max():>10.8f} "
              f"{cr.mean():>10.8f} {cr.max():>10.8f}")

    # Per-frame L2 norm
    per_frame_l2 = np.linalg.norm(diff.reshape(Q, -1), axis=1)
    worst_idx = per_frame_l2.argmax()
    print(f"\nPer-frame L2 norm of feature diff:")
    print(f"  mean: {per_frame_l2.mean():.8f}")
    print(f"  max:  {per_frame_l2.max():.8f}  (query {worst_idx}, raw_t={raw_indices[worst_idx]})")
    print(f"  min:  {per_frame_l2.min():.8f}")

    # Plots
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.server_features)
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot 1: per-frame L2 diff over time
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(Q), per_frame_l2, linewidth=0.8, color='#1f77b4')
    ax.set_xlabel("query index")
    ax.set_ylabel("L2 feature diff")
    ax.set_title("Per-frame feature diff (server vs pre-extracted)")
    plt.tight_layout()
    path = os.path.join(args.output_dir, "feature_diff_over_time.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"\nSaved: {path}")

    # Plot 2: per-camera abs diff histograms
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for c, cam in enumerate(CAM_NAMES):
        vals = diff[:, c].flatten()
        axes[c].hist(vals, bins=100, alpha=0.8, color='#1f77b4')
        axes[c].set_title(cam, fontsize=10)
        axes[c].set_xlabel("abs diff")
        axes[c].set_ylabel("count")
        axes[c].axvline(vals.mean(), color='#d62728', linestyle='--',
                        label=f"mean={vals.mean():.6f}")
        axes[c].legend(fontsize=7)
    fig.suptitle("Feature abs diff distribution per camera", fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "feature_diff_histogram.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

    # Plot 3: scatter (correlation) first 10 frames, 50 dims per camera
    n_show = min(10, Q)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for c, cam in enumerate(CAM_NAMES):
        sv = feat_srv[:n_show, c, :50].flatten()
        pv = feat_pre[:n_show, c, :50].flatten()
        axes[c].scatter(pv, sv, s=1, alpha=0.3, color='#1f77b4')
        lims = [min(pv.min(), sv.min()), max(pv.max(), sv.max())]
        axes[c].plot(lims, lims, 'r-', linewidth=0.5, alpha=0.5)
        axes[c].set_xlabel("pre-extracted")
        axes[c].set_ylabel("server")
        axes[c].set_title(cam, fontsize=10)
        axes[c].set_aspect('equal')
    fig.suptitle("Feature correlation (first 10 frames, 50 dims)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "feature_correlation.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

    print("Done.")


if __name__ == "__main__":
    main()
