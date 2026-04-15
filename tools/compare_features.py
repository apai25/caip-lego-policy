#!/usr/bin/env python3
"""Compare two features.h5 files (pre-extracted vs server-extracted).

Usage:
    python tools/compare_features.py \
        --features-a /path/to/episode_0001/features.h5 \
        --features-b /path/to/server_features.h5 \
        --label-a preextracted --label-b server
"""

import argparse
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare two features.h5 files")
    parser.add_argument("--features-a", required=True)
    parser.add_argument("--features-b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    with h5py.File(args.features_a, 'r') as f:
        feat_a = f['features'][:]
    with h5py.File(args.features_b, 'r') as f:
        feat_b = f['features'][:]

    print(f"{args.label_a}: {args.features_a}  shape={feat_a.shape}  dtype={feat_a.dtype}")
    print(f"{args.label_b}: {args.features_b}  shape={feat_b.shape}  dtype={feat_b.dtype}")

    T = min(len(feat_a), len(feat_b))
    if len(feat_a) != len(feat_b):
        print(f"\n  WARNING: length mismatch ({len(feat_a)} vs {len(feat_b)}), comparing first {T}")
    feat_a, feat_b = feat_a[:T], feat_b[:T]

    diff = np.abs(feat_a - feat_b)
    print(f"\nComparing {T} frames:")
    print(f"  Mean abs diff: {diff.mean():.8f}")
    print(f"  Max abs diff:  {diff.max():.8f}")
    print(f"  Mean feat magnitude ({args.label_a}): {np.abs(feat_a).mean():.6f}")
    print(f"  Relative diff: {diff.mean() / (np.abs(feat_a).mean() + 1e-10):.6f}")

    # Per-camera comparison
    if feat_a.ndim == 3:
        num_cams = feat_a.shape[1]
        cam_names = ["left_wrist", "right_wrist", "head"][:num_cams]
        print(f"\n  Per-camera mean abs diff:")
        for c in range(num_cams):
            cam_diff = np.abs(feat_a[:, c] - feat_b[:, c]).mean()
            cam_mag = np.abs(feat_a[:, c]).mean()
            print(f"    {cam_names[c]}: {cam_diff:.8f} (relative: {cam_diff / (cam_mag + 1e-10):.6f})")

    exact = np.array_equal(feat_a, feat_b)
    close = np.allclose(feat_a, feat_b, atol=1e-6)
    print(f"\n  Exactly equal: {exact}")
    print(f"  Close (atol=1e-6): {close}")


if __name__ == "__main__":
    main()
