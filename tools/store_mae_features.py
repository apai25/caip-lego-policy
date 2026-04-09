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

from mvp.vision_model.vision_encoder import Encoder
from mvp.bimanual_bc.dataset import process_image


@torch.no_grad()
def store_mae_features(data_root, save_root, demo_name, model_name, model_dir, save_mode, start, end, fp16, keys):

    # Load the model
    model = Encoder(model_name, model_dir, freeze=True)
    model.cuda()
    model.eval()

    im_size = 256 if ('vitl' in model_name or 'vith' in model_name) else 224

    # Source/dest root dirs
    src_root = os.path.join(data_root, demo_name)
    dst_root = os.path.join(save_root, model_name, demo_name)
    os.makedirs(dst_root, exist_ok=True)

    if start == -1 or end == -1:
        start = 0
        end = len(os.listdir(src_root))
    # Go over trajectories
    for traj_dir in tqdm(sorted(os.listdir(src_root))[start:end]):
        src_traj_path = os.path.join(src_root, traj_dir)
        dst_traj_path = os.path.join(dst_root, traj_dir)
        os.makedirs(dst_traj_path, exist_ok=True)
        # Go over observations
        features = []
        for obs_fname in sorted(os.listdir(src_traj_path)):
            src_obs_path = os.path.join(src_traj_path, obs_fname)
            dst_obs_path = os.path.join(dst_traj_path, obs_fname)
            # Metadata
            if not obs_fname.endswith(".pkl"):
                shutil.copy(src_obs_path, dst_traj_path)
                continue
            # Observations
            with open(src_obs_path, "rb") as f:
                obs = joblib.load(f)
            # Compute the features
            for cam in keys:
                # Prepare the image
                im = obs[f"rgb_{cam}"]
                im = process_image(im, im_size=im_size)
                im = torch.tensor(im).cuda().unsqueeze(0)
                # Compute the embedding
                im_emb = model(im, save_mode)
                im_emb = im_emb.squeeze(0).cpu().numpy()
                if fp16:
                    im_emb = im_emb.astype(np.float16)
                # Update the obs
                obs[f"feat_{cam}"] = im_emb
                del obs[f"rgb_{cam}"]
            # if save_mode=='all', put all features into one h5 file
            if 'all' in save_mode:
                merged_features = np.stack([obs[f"feat_{cam}"] for cam in keys], axis=0)
                features.append(merged_features)
                for cam in keys:
                    del obs[f"feat_{cam}"]
            # Save updated obs
            joblib.dump(obs, dst_obs_path, compress=3)

        if 'all' in save_mode:
            features = np.stack(features, axis=0)
            hdf5_file_path = os.path.join(dst_traj_path, 'features.h5')
            with h5py.File(hdf5_file_path, 'w') as hdf_file:
                hdf_file.create_dataset('features', data=features)

    print("Saved features to: {}".format(dst_root))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", dest="data_root", default="/home/ilija/data/dexnet-panda")
    parser.add_argument("--save-root", dest="save_root", default="/home/ilija/data/features")
    parser.add_argument("--demo-name", dest="demo_name", default="pick-yellow-cube_01-23-2023")
    parser.add_argument("--model-name", dest="model_name", default="vitb-mae-egosoup") 
    parser.add_argument("--model-dir", dest="model_dir", default="/home/ilija/data/pretrained/encoders")
    parser.add_argument("--save_mode", dest="save_mode", type=str, choices=['cls', 'mean', 'all', 'multiscale_mean', 'multiscale_all'], help='choose which kind of feature to save')
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--keys', nargs='+', default=["left", "hand", "right"])
    args = parser.parse_args()
    # Compute and store features
    torch.multiprocessing.set_start_method('spawn')
    store_mae_features(
        args.data_root, args.save_root, args.demo_name,
        args.model_name, args.model_dir, args.save_mode,
        args.start, args.end, args.fp16, args.keys
    )
