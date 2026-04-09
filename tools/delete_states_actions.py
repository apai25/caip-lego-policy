#!/usr/bin/env python3

"""Store MAE features."""

from glob import glob
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import argparse
import joblib
import numpy as np
import os
import h5py
import shutil
import torch
import torch.nn as nn



@torch.no_grad()
def delete_states_actions(data_root, demo_name):

    # Source/dest root dirs
    src_root = os.path.join(data_root, demo_name)

    # Go over trajectories
    for traj_dir in tqdm(sorted(os.listdir(src_root))):
        src_traj_path = os.path.join(src_root, traj_dir)
        # Go over observations
        for obs_fname in sorted(os.listdir(src_traj_path)):
            # Metadata
            if not obs_fname.endswith(".pkl"):
                continue
            # Observations
            src_obs_path = os.path.join(src_traj_path, obs_fname)
            with open(src_obs_path, "rb") as f:
                obs = joblib.load(f)
            delete_keys = ['rgb_left', 'rgb_right', 'left_arm_cmd', 'right_arm_cmd', 'left_fingers_cmd', 'right_fingers_cmd',
                           'left_fingers_joint_pos', 'left_fingers_touch', 'right_fingers_joint_pos',
                           'right_fingers_touch', 'left_arm_joint_pos', 'right_arm_joint_pos']
            for k in delete_keys:
                if k in obs:
                    obs.pop(k)
            # Save updated obs
            joblib.dump(obs, src_obs_path, compress=3)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", dest="data_root", default="/home/ilija/data/dexnet-panda")
    parser.add_argument("--demo-name", dest="demo_name", default="pick-yellow-cube_01-23-2023")
    args = parser.parse_args()
    # Compute and store features
    torch.multiprocessing.set_start_method('spawn')
    delete_states_actions(
        args.data_root, args.demo_name
    )
