#!/usr/bin/env python3

"""Compute proprioceptive noise statistics from BKL tracking errors.

For each timestep t, computes the tracking error:
    observation[t+1] (actual) vs target[t] (commanded)

Errors are computed in:
- Body frame for translation: R_target[t]^T @ (xyz_obs[t+1] - xyz_target[t])
- Axis-angle for rotation: R_target[t]^T @ R_obs[t+1] → rotvec (3D)
- Simple subtraction for hand joints

Output is 56D: left_xyz(3) + left_axisangle(3) + right_xyz(3) + right_axisangle(3)
               + left_hand(22) + right_hand(22)

Only std is used for noise injection (zero-mean); mean is saved for diagnostics.

Usage:
    python tools/compute_bkl_noise_stats.py \
        --data-root /path/to/task_data \
        --demo-name pick_place_egg \
        --num-demos 90
"""

import argparse
import os
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


HAND_JOINT_DIM = 22


def compute_tracking_errors(h5_path):
    """Compute per-timestep tracking errors from a single episode.

    Returns (N, 56) array of errors for N = T-1 timestep pairs,
    or None if the episode is too short.

    Layout: [left_xyz(3), left_axisangle(3), right_xyz(3), right_axisangle(3),
             left_hand(22), right_hand(22)]
    """
    with h5py.File(h5_path, 'r') as f:
        left_target_pose = f['left_arm_target_pose'][:].astype(np.float64)
        left_current_pose = f['left_arm_current_pose'][:].astype(np.float64)
        right_target_pose = f['right_arm_target_pose'][:].astype(np.float64)
        right_current_pose = f['right_arm_current_pose'][:].astype(np.float64)
        left_hand_target = f['left_hand_target_joint_positions'][:].astype(np.float32)
        left_hand_current = f['left_hand_joint_positions'][:].astype(np.float32)
        right_hand_target = f['right_hand_target_joint_positions'][:].astype(np.float32)
        right_hand_current = f['right_hand_joint_positions'][:].astype(np.float32)

    T = left_target_pose.shape[0]
    if T < 2:
        return None

    n = T - 1  # number of (target[t], observation[t+1]) pairs

    # --- Body-frame translation errors ---
    # R_target[t]^T @ (xyz_obs[t+1] - xyz_target[t])
    left_R_tar = left_target_pose[:n, :3, :3]       # (n, 3, 3)
    left_xyz_tar = left_target_pose[:n, :3, 3]      # (n, 3)
    left_xyz_obs = left_current_pose[1:n+1, :3, 3]  # (n, 3)
    left_trans_err = np.einsum('nij,nj->ni', left_R_tar.transpose(0, 2, 1),
                               left_xyz_obs - left_xyz_tar)  # (n, 3)

    right_R_tar = right_target_pose[:n, :3, :3]
    right_xyz_tar = right_target_pose[:n, :3, 3]
    right_xyz_obs = right_current_pose[1:n+1, :3, 3]
    right_trans_err = np.einsum('nij,nj->ni', right_R_tar.transpose(0, 2, 1),
                                right_xyz_obs - right_xyz_tar)  # (n, 3)

    # --- Axis-angle rotation errors ---
    # R_error = R_target[t]^T @ R_obs[t+1] → axis-angle (3D)
    left_R_obs = left_current_pose[1:n+1, :3, :3]
    left_R_err = np.einsum('nij,njk->nik', left_R_tar.transpose(0, 2, 1), left_R_obs)
    left_rot_err = R.from_matrix(left_R_err).as_rotvec().astype(np.float32)  # (n, 3)

    right_R_obs = right_current_pose[1:n+1, :3, :3]
    right_R_err = np.einsum('nij,njk->nik', right_R_tar.transpose(0, 2, 1), right_R_obs)
    right_rot_err = R.from_matrix(right_R_err).as_rotvec().astype(np.float32)  # (n, 3)

    # --- Hand joint errors ---
    left_hand_err = left_hand_current[1:n+1, :HAND_JOINT_DIM] - left_hand_target[:n, :HAND_JOINT_DIM]
    right_hand_err = right_hand_current[1:n+1, :HAND_JOINT_DIM] - right_hand_target[:n, :HAND_JOINT_DIM]

    # Concatenate: left_xyz(3) + left_aa(3) + right_xyz(3) + right_aa(3) + hands(44) = 56
    errors = np.concatenate([
        left_trans_err, left_rot_err,      # 3 + 3 = 6
        right_trans_err, right_rot_err,    # 3 + 3 = 6
        left_hand_err, right_hand_err,     # 22 + 22 = 44
    ], axis=1).astype(np.float32)          # (n, 56)

    return errors


def main():
    parser = argparse.ArgumentParser(description="Compute BKL proprioceptive noise stats from tracking errors")
    parser.add_argument("--data-root", dest="data_root",
                        default="/mnt/amlfs-02/shared/human_egocentric/dniu/datasets/caip_proc")
    parser.add_argument("--demo-name", dest="demo_name", default="pick_place_egg")
    parser.add_argument("--num-demos", dest="num_demos", type=int, default=90,
                        help="Number of training demos (to exclude test set)")
    parser.add_argument("--output", default=None, help="Output path (default: auto-named in data root)")
    args = parser.parse_args()

    # Find episodes
    demo_dir = os.path.join(args.data_root, args.demo_name)
    if os.path.isdir(os.path.join(demo_dir, "success")):
        search_dir = os.path.join(demo_dir, "success")
    else:
        search_dir = demo_dir

    episodes = sorted([
        os.path.join(search_dir, d) for d in os.listdir(search_dir)
        if os.path.isdir(os.path.join(search_dir, d))
    ])[:args.num_demos]

    print(f"Computing noise stats over {len(episodes)} episodes")

    all_errors = []
    for ep_dir in tqdm(episodes, desc="Scanning demos"):
        ep_name = os.path.basename(ep_dir)
        h5_path = os.path.join(ep_dir, f"{ep_name}.h5")
        if not os.path.exists(h5_path):
            print(f"  Skip (no h5): {ep_dir}")
            continue

        errors = compute_tracking_errors(h5_path)
        if errors is not None:
            all_errors.append(errors)

    if not all_errors:
        raise ValueError("No valid error samples from any demo.")

    all_errors = np.concatenate(all_errors, axis=0)
    mean = all_errors.mean(axis=0)
    std = all_errors.std(axis=0)

    print(f"\nNoise stats (shape {std.shape}, {all_errors.shape[0]} samples):")
    print(f"  Left EEF  trans std (body-frame): {std[0:3]}")
    print(f"  Left EEF  rot   std (axis-angle): {std[3:6]}")
    print(f"  Right EEF trans std (body-frame): {std[6:9]}")
    print(f"  Right EEF rot   std (axis-angle): {std[9:12]}")
    print(f"  Left hand  joint std (22): min={std[12:34].min():.6f} max={std[12:34].max():.6f}")
    print(f"  Right hand joint std (22): min={std[34:56].min():.6f} max={std[34:56].max():.6f}")
    print(f"  Mean (diagnostics): max_abs={np.abs(mean).max():.6f}")

    # Save (mean for diagnostics, std for noise injection)
    if args.output is None:
        args.output = os.path.join(args.data_root, args.demo_name, "noise_stats.npz")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, mean=mean, std=std)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
