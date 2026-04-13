#!/usr/bin/env python3

"""Compute action normalization statistics for BKL data.

Iterates over training episodes, computes body-frame delta EEF actions
with 6D rotation + delta hand joints, and computes per-dimension mean
and std for z-score normalization.

Usage:
    python tools/compute_bkl_action_stats.py \
        --data-root /path/to/caip_proc \
        --demo-name pick_place_egg \
        --frame-skip 1 \
        --num-demos 90
"""

import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def compute_action_for_pair(action_data, k, t1):
    """Compute body-frame delta action for a (k, t1) pair.

    Same logic as BKL_Dataset._compute_action.
    Returns 62D: [left_xyz(3), left_rot6d(6), right_xyz(3), right_rot6d(6),
                   left_hand(22), right_hand(22)]
    """
    left_curr = action_data['left_arm_target_pose'][k]
    left_tgt = action_data['left_arm_target_pose'][t1]
    left_R_curr = left_curr[:3, :3]
    left_delta_xyz = (left_R_curr.T @ (left_tgt[:3, 3] - left_curr[:3, 3])).astype(np.float32)
    left_R_delta = left_R_curr.T @ left_tgt[:3, :3]
    left_delta_rot6d = np.concatenate([left_R_delta[:, 0], left_R_delta[:, 1]]).astype(np.float32)

    right_curr = action_data['right_arm_target_pose'][k]
    right_tgt = action_data['right_arm_target_pose'][t1]
    right_R_curr = right_curr[:3, :3]
    right_delta_xyz = (right_R_curr.T @ (right_tgt[:3, 3] - right_curr[:3, 3])).astype(np.float32)
    right_R_delta = right_R_curr.T @ right_tgt[:3, :3]
    right_delta_rot6d = np.concatenate([right_R_delta[:, 0], right_R_delta[:, 1]]).astype(np.float32)

    left_hand_delta = (action_data['left_hand_cmd'][t1] - action_data['left_hand_cmd'][k]).astype(np.float32)
    right_hand_delta = (action_data['right_hand_cmd'][t1] - action_data['right_hand_cmd'][k]).astype(np.float32)
    return np.concatenate([left_delta_xyz, left_delta_rot6d,
                           right_delta_xyz, right_delta_rot6d,
                           left_hand_delta, right_hand_delta])  # (62,)


def load_action_data(h5_path):
    """Load raw action data from episode HDF5."""
    with h5py.File(h5_path, 'r') as f:
        T = f['timestamp'].shape[0]
        action_data = {
            'left_arm_target_pose': f['left_arm_target_pose'][:].astype(np.float64),
            'right_arm_target_pose': f['right_arm_target_pose'][:].astype(np.float64),
            'left_hand_cmd': f['left_hand_target_joint_positions'][:].astype(np.float32),
            'right_hand_cmd': f['right_hand_target_joint_positions'][:].astype(np.float32),
        }
    return T, action_data


def main():
    parser = argparse.ArgumentParser(description="Compute BKL action normalization stats")
    parser.add_argument("--data-root", dest="data_root",
                        default="/mnt/amlfs-02/shared/human_egocentric/dniu/datasets/caip_proc")
    parser.add_argument("--demo-name", dest="demo_name", default="pick_place_egg")
    parser.add_argument("--frame-skip", dest="frame_skip", type=int, default=1)
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

    print(f"Computing action stats over {len(episodes)} episodes")
    print(f"  frame_skip: {args.frame_skip}")

    all_actions = []
    for ep_dir in tqdm(episodes):
        ep_name = os.path.basename(ep_dir)
        h5_path = os.path.join(ep_dir, f"{ep_name}.h5")
        if not os.path.exists(h5_path):
            continue

        T, action_data = load_action_data(h5_path)

        for k in range(T - 1):
            t1 = min(k + args.frame_skip + 1, T - 1)
            act = compute_action_for_pair(action_data, k, t1)
            all_actions.append(act)

    all_actions = np.array(all_actions)
    mean = all_actions.mean(axis=0)
    std = all_actions.std(axis=0)

    print(f"\nAction stats (shape {mean.shape}):")
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")
    print(f"  Min std: {std.min():.6f}, Max std: {std.max():.6f}")

    # Save
    if args.output is None:
        args.output = os.path.join(
            args.data_root, args.demo_name,
            f"action_stats_delta_eef_skip{args.frame_skip}.npz"
        )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, mean=mean, std=std)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
