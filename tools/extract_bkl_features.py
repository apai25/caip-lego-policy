#!/usr/bin/env python3

"""Extract vision features from BKL MP4 videos using MAE-pretrained ViT.

Each BKL episode directory contains:
  - {episode_name}_left_wrist.mp4
  - {episode_name}_right_wrist.mp4
  - {episode_name}_head_left_rgb.mp4
  - {episode_name}.h5

This script decodes the MP4 frames, runs them through a frozen ViT encoder,
and saves the features to features.h5 with shape (T, num_cams, embed_dim).
"""

import argparse
import os
import sys
import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mvp.vision_model.vision_encoder import Encoder
from mvp.bimanual_bc.dataset import process_image


# Camera suffixes in the MP4 filenames, in the order they appear in features.h5
CAM_SUFFIXES = {
    "left_wrist": "_left_wrist.mp4",
    "right_wrist": "_right_wrist.mp4",
    "head": "_head_left_rgb.mp4",
}


def decode_video(video_path):
    """Decode all frames from an MP4 file. Returns (T, H, W, 3) uint8 numpy array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


@torch.no_grad()
def extract_features(model, frames, im_size, batch_size=64, mode='mean'):
    """Extract features from a list of RGB frames.

    Args:
        model: Vision encoder model (on GPU)
        frames: list of (H, W, 3) uint8 numpy arrays
        im_size: target image size for the encoder
        batch_size: number of frames to process at once
        mode: 'mean' for mean-pooled, 'cls' for CLS token, 'all' for all patches

    Returns:
        (T, embed_dim) numpy array for mean/cls, or (T, N, embed_dim) for all
    """
    all_features = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        # Preprocess: center crop, resize, normalize, HWC->CHW
        batch = np.stack([process_image(f, im_size) for f in batch_frames])
        batch = torch.tensor(batch, dtype=torch.float32).cuda()
        # Forward through encoder
        feats = model(batch, mode=mode)  # (B, embed_dim) or (B, N, embed_dim)
        all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def process_episode(model, ep_dir, cams, im_size, batch_size, mode, fp16):
    """Process a single episode: decode videos, extract features, save to HDF5."""
    output_path = os.path.join(ep_dir, 'features.h5')

    # Skip if already processed
    if os.path.exists(output_path):
        return True

    ep_name = os.path.basename(ep_dir)
    h5_path = os.path.join(ep_dir, f"{ep_name}.h5")
    if not os.path.exists(h5_path):
        print(f"Warning: skipping {ep_dir}, no {ep_name}.h5 found")
        return False

    with h5py.File(h5_path, 'r') as f:
        expected_T = f['timestamp'].shape[0]

    # Extract features per camera
    cam_features = []
    for cam_name in cams:
        suffix = CAM_SUFFIXES[cam_name]
        video_path = os.path.join(ep_dir, f"{ep_name}{suffix}")
        if not os.path.exists(video_path):
            print(f"Warning: missing video {video_path}")
            return False

        frames = decode_video(video_path)
        if len(frames) != expected_T:
            print(f"Warning: {video_path} has {len(frames)} frames but H5 has {expected_T} timesteps")
            # Truncate to shorter length
            frames = frames[:expected_T]

        feats = extract_features(model, frames, im_size, batch_size, mode)  # (T, C) or (T, N, C)
        cam_features.append(feats)

    # Stack cameras: (T, num_cams, C) or (T, num_cams, N, C)
    features = np.stack(cam_features, axis=1)
    if fp16:
        features = features.astype(np.float16)

    # Save
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('features', data=features)

    return True


def main():
    parser = argparse.ArgumentParser(description="Extract BKL vision features from MP4 videos")
    parser.add_argument("--data-root", dest="data_root",
                        default="/mnt/amlfs-02/shared/human_egocentric/dniu/datasets/bkl_inlab/raw/task_data")
    parser.add_argument("--demo-name", dest="demo_name",
                        default="sugar_pour_merged_04-07-2026_100")
    parser.add_argument("--model-name", dest="model_name", default="vitb-mae-egosoup")
    parser.add_argument("--model-dir", dest="model_dir",
                        default="/mnt/amlfs-01/home/dniu/Project/lego/mvp_weights")
    parser.add_argument("--mode", default="mean", choices=["cls", "mean", "all"],
                        help="Feature pooling mode")
    parser.add_argument("--cams", nargs='+', default=["left_wrist", "right_wrist", "head"])
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()

    # Load encoder
    print(f"Loading encoder: {args.model_name}")
    model = Encoder(args.model_name, args.model_dir, freeze=True)
    model.cuda()
    model.eval()

    im_size = 256 if ('vitl' in args.model_name or 'vith' in args.model_name) else 224

    # Find episodes
    demo_dir = os.path.join(args.data_root, args.demo_name)
    if os.path.isdir(os.path.join(demo_dir, "success")):
        search_dir = os.path.join(demo_dir, "success")
    else:
        search_dir = demo_dir

    episodes = sorted([
        os.path.join(search_dir, d) for d in os.listdir(search_dir)
        if os.path.isdir(os.path.join(search_dir, d))
    ])

    end = args.end if args.end != -1 else len(episodes)
    episodes = episodes[args.start:end]
    print(f"Processing {len(episodes)} episodes from {search_dir}")

    # Process each episode
    success = 0
    for ep_dir in tqdm(episodes):
        if process_episode(model, ep_dir, args.cams, im_size, args.batch_size, args.mode, args.fp16):
            success += 1

    print(f"Done. Processed {success}/{len(episodes)} episodes.")


if __name__ == "__main__":
    main()
