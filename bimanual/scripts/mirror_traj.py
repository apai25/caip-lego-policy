#!/usr/bin/env python3

"""Mirror bimanual trajectories."""
import os
import click
import joblib
import json
import numpy as np

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

def mirror_arms(j):
    return [-2*np.pi - j[0], -np.pi - j[1], -j[2], -np.pi - j[3], -j[4], -j[5]]

def mirror_img(im):
    return np.flip(im, axis=1)

def mirror_frame(d):
    d['right_arm_joint_pos'], d['left_arm_joint_pos'] = mirror_arms(d['left_arm_joint_pos']), mirror_arms(d['right_arm_joint_pos'])
    d['right_arm_cmd'], d['left_arm_cmd'] = mirror_arms(d['left_arm_cmd']), mirror_arms(d['right_arm_cmd'])

    d['rgb_left'], d['rgb_right'] = mirror_img(d['rgb_right']), mirror_img(d['rgb_left'])
    d['rgb_head'] = mirror_img(d['rgb_head'])

    d['left_fingers_joint_pos'], d['right_fingers_joint_pos'] = d['right_fingers_joint_pos'], d['left_fingers_joint_pos']
    d['left_fingers_touch'], d['right_fingers_touch'] = d['right_fingers_touch'], d['left_fingers_touch']
    d['left_fingers_cmd'], d['right_fingers_cmd'] = d['right_fingers_cmd'], d['left_fingers_cmd']

def swap_left_right(s):
    temp_placeholder = "<TEMP>"  # Make sure this placeholder does not occur in the original string
    s = s.replace("left", temp_placeholder)
    s = s.replace("right", "left")
    s = s.replace(temp_placeholder, "right")
    return s

def mirror_traj(args):
    original, target = args
    print('creating', target)
    os.makedirs(target, exist_ok=True)
    pkls = sorted(glob(os.path.join(original, "*.pkl")))
    for p in pkls:
        data = joblib.load(p)
        mirror_frame(data)
        if 'task' in data:
            data['task'] = swap_left_right(data['task'])
        target_p = os.path.join(target, os.path.basename(p)) 
        joblib.dump(data, target_p, compress=3)

    if os.path.exists(os.path.join(original, 'success.txt')):
        open(os.path.join(target, 'success.txt'), 'w').close()
    elif os.path.exists(os.path.join(original, 'failure.txt')):
        open(os.path.join(target, 'failure.txt'), 'w').close()
    else:
        print("success / failure file not detected! skipped.")
        return

    with open(os.path.join(original, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    
    if 'task' in meta:
        meta['task'] = swap_left_right(meta['task'])

    with open(os.path.join(target, 'metadata.json'), 'w') as f:
        json.dump(meta, f)

@click.command()
@click.option('--data-dir', prompt='Data directory', help='Directory containing the trajectories to mirror.')
def main(data_dir):
    path = os.path.normpath(data_dir)
    data_root, dataset_name = os.path.split(path)
    target = os.path.join(data_root, 'mirror-' + dataset_name)

    print(f'creating directory at {target}')
    os.makedirs(target, exist_ok=True)

    dirs = sorted(os.listdir(path))
    args = [(os.path.join(path, d), os.path.join(target, d)) for d in dirs]

    num_workers = 8
    with Pool(num_workers) as p:
        for _ in tqdm(p.imap(mirror_traj, args), total=len(dirs)):
            pass

if __name__ == '__main__':
    main()
