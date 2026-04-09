#!/usr/bin/env python3

"""Store MAE features."""

from PIL import Image
import json
from tqdm import tqdm
import argparse
import joblib
import numpy as np
import os
import h5py
import shutil
import torch
from mvp.bimanual_bc.dataset import process_image_no_normalize
from visual_feature_selector_yoloworld_sam2 import VisualFeatureSelector



@torch.no_grad()
def store_mae_features(visual_feature_selector_path, data_root, save_root, demo_name, model_name, prompt_text, prompt_key, save_all_features,
                       start, end, fp16, keys):

    # Load the model
    model = VisualFeatureSelector(visual_feature_selector_path=visual_feature_selector_path, prompt_text = prompt_text)

    # im_size = 224

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

        # Get text prompt for attention pooling
        with open(os.path.join(src_traj_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        if prompt_text is not None:
            prompt = prompt_text
        elif prompt_key in metadata:
            prompt = metadata[prompt_key]
        else:
            prompt = None  # prompt is probably in each pkl file

        # Go over observations
        features = []
        for cam_idx, cam in enumerate(keys):
            model.reset()
            current_feature = []
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
                # check if there's any prompt in the pkl file
                if prompt is None and prompt_key in obs:
                    prompt = obs[prompt_key]
                # Compute the features
                # # Prepare the image
                # im = Image.fromarray(obs[f"rgb_{cam}"])
                # im = process_image_no_normalize(im, im_size)
                # im = im.astype(np.uint8)  # Convert to unsigned byte type
                # # Convert the array to an image
                # im = Image.fromarray(im)

                # # save as tmp
                tmp_path = '/home/niudt/tmp/{}/{}/{}/{}.png'.format(dst_obs_path.split('/')[-3], dst_obs_path.split('/')[-2], cam, obs_fname.split('.')[0])
                # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                # im.save(tmp_path)

                # Compute the embedding
                if save_all_features:
                    assert False
                else:
                    im_emb = model.inference_frame(img_path=tmp_path, raw_image_array = obs[f"rgb_{cam}"])
                im_emb = im_emb.squeeze(0).cpu().numpy()
                if fp16:
                    im_emb = im_emb.astype(np.float16)
                current_feature.append(im_emb)
                if cam_idx == (len(keys) - 1):
                    for _cam in keys:
                        del obs[f"rgb_{_cam}"]
                    # Save updated obs
                    joblib.dump(obs, dst_obs_path, compress=3)

            merged_features = np.stack(current_feature, axis=0)
            features.append(merged_features)

        # put all features into one h5 file
        features = np.stack(features, axis=0)
        features = np.transpose(features, (1, 0, 2))
        hdf5_file_path = os.path.join(dst_traj_path, 'features.h5')
        with h5py.File(hdf5_file_path, 'w') as hdf_file:
            hdf_file.create_dataset('features', data=features)

        # save the prompt text into the new metadata file if not yet
        with open(os.path.join(dst_traj_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        if prompt_text is not None:
            metadata[prompt_key] = prompt_text
            with open(os.path.join(dst_traj_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f)

    print("Saved features to: {}".format(dst_root))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", dest="data_root", default="/home/ilija/data/bimanual/")
    parser.add_argument("--save-root", dest="save_root", default="/home/niudt/data/bimanual_1001/features/features_attn_pooled_visual_feature_selector")
    parser.add_argument("--demo-name", dest="demo_name", default="pick-yellow-right_04-29-2024")
    parser.add_argument("--model-name", dest="model_name", default="dinov2-b")
    parser.add_argument("--prompt", dest="prompt", type=str, default='yellow cube')
    parser.add_argument("--prompt-key", dest="prompt_key", default="attn_pool_prompt")
    parser.add_argument("--visual-feature-selector-path", dest="visual_feature_selector_path", default="/home/niudt/project/combine/YoloWorld-SAM2")
    parser.add_argument('--save-all-features', dest="save_all_features", action='store_true', default=False)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--keys', nargs='+', default=["left", "head", "right"])
    args = parser.parse_args()
    # Compute and store features
    torch.multiprocessing.set_start_method('spawn')
    store_mae_features(args.visual_feature_selector_path,
        args.data_root, args.save_root, args.demo_name,
        args.model_name, args.prompt, args.prompt_key, args.save_all_features,
        args.start, args.end, args.fp16, args.keys
    )
