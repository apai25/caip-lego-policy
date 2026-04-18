#!/usr/bin/env python3

"""Extract vision features from BKL MP4 videos using MAE-pretrained ViT.

Each BKL episode directory contains:
  - {episode_name}_left_wrist.mp4
  - {episode_name}_right_wrist.mp4
  - {episode_name}_head_left_rgb.mp4
  - {episode_name}.h5

This script decodes the MP4 frames, runs them through a frozen ViT encoder in
all-patches mode (CLS token + 196 patch tokens — no extraction-time pooling),
and saves the resulting (K, T, num_cams, 197, 768) feature tensor to features.h5.
Pooling is applied at training / inference time inside mvp/bimanual_bc/bc.py
(see mean_pool_patches), so the patch axis is preserved on disk for future
learned-pool variants (attention pooling, etc).

With --num-variants K > 1, writes K feature variants per frame for image-space
augmentation. Slot 0 is ALWAYS the identity (plain process_image); slots 1..K-1
apply whichever of --color-jitter / --crop-jitter are enabled, with independent
RNG draws per (variant, camera, frame). Variant 0 identity is the invariant the
training-time validation path reads — do not change it.

Output HDF5 layout:
  dataset 'features': shape (K, T, num_cams, 197, 768)   (always — K >= 1)
  attributes on 'features':
    num_variants: int   (K)
    color_jitter: bool
    crop_jitter:  bool
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mvp.vision_model.vision_encoder import Encoder
from mvp.bimanual_bc.dataset import process_image, process_image_augmented


# Camera suffixes in the MP4 filenames, in the order they appear in features.h5
CAM_SUFFIXES = {
    "left_wrist": "_left_wrist.mp4",
    "right_wrist": "_right_wrist.mp4",
    "head": "_head_left_rgb.mp4",
}


def decode_video(video_path):
    """Decode all frames from an MP4 file. Returns list of (H, W, 3) uint8 arrays."""
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


def _preprocess_one(args):
    """Preprocess a single (variant_idx, frame_idx) task.

    variant_idx == 0 -> plain process_image (identity).
    variant_idx >= 1 -> process_image_augmented with a seed-derived RNG so the
    result is reproducible regardless of thread scheduling.
    """
    (variant_idx, frame_idx, frame, im_size, aug_cfg, base_seed) = args
    if variant_idx == 0:
        return variant_idx, frame_idx, process_image(frame, im_size)
    # Deterministic per-(variant, frame) RNG so output is invariant to worker order.
    rng = np.random.default_rng(np.array([base_seed, variant_idx, frame_idx], dtype=np.uint64))
    return variant_idx, frame_idx, process_image_augmented(
        frame, im_size, rng,
        use_color=aug_cfg['use_color'],
        use_crop=aug_cfg['use_crop'],
        brightness=aug_cfg['brightness'],
        contrast=aug_cfg['contrast'],
        saturation=aug_cfg['saturation'],
        hue=aug_cfg['hue'],
        crop_jitter_pixels=aug_cfg['crop_jitter_pixels'],
    )


def preprocess_camera(frames, im_size, num_variants, aug_cfg, base_seed, pool, timing=None):
    """Phase 1 (CPU, parallel): produce (K*T, 3, H, W) preprocessed stack.

    Uses the shared ThreadPoolExecutor so multiple episodes can compete for
    CPU cycles concurrently. cv2/numpy ops release the GIL, so threads scale.
    """
    T = len(frames)
    total = num_variants * T
    stack = np.empty((total, 3, im_size, im_size), dtype=np.float32)
    tasks = (
        (k, t, frames[t], im_size, aug_cfg, base_seed)
        for k in range(num_variants) for t in range(T)
    )
    t0 = time.perf_counter()
    for (k, t, arr) in pool.map(_preprocess_one, tasks):
        stack[k * T + t] = arr
    if timing is not None:
        timing['cpu'] += time.perf_counter() - t0
    return stack


@torch.no_grad()
def gpu_infer(model, stack, gpu_batch, gpu_lock, timing=None):
    """Phase 2 (GPU, serialized via gpu_lock):
    (K*T, 3, H, W) -> (K*T, 197, 768)   # 1 CLS + 196 patch tokens per image.

    gpu_lock ensures only one episode/camera is on the GPU at a time even when
    multiple episodes are in flight; CPU workers from other episodes can keep
    running while this one holds the lock.
    """
    total = stack.shape[0]
    out = np.empty((total, 197, 768), dtype=np.float32)
    t0 = time.perf_counter()
    with gpu_lock:
        for i in range(0, total, gpu_batch):
            chunk = stack[i:i + gpu_batch]
            batch_t = torch.from_numpy(chunk).cuda(non_blocking=True)
            feats = model(batch_t, mode='all')  # (B, 197, 768)
            out[i:i + gpu_batch] = feats.cpu().numpy()
        torch.cuda.synchronize()
    if timing is not None:
        timing['gpu'] += time.perf_counter() - t0
    return out


def process_episode(model, ep_dir, cams, im_size, batch_size,
                    fp16, num_variants, aug_cfg, force,
                    cpu_pool, gpu_lock, timing, timing_lock):
    """Decode videos, extract features, save to features.h5.

    When multiple episodes run concurrently (via the episode pool in main()),
    they share cpu_pool and are serialized on gpu_lock. Each episode processes
    its 3 cameras in sequence: CPU preprocess for cam i overlaps with GPU
    inference for cam i-1 of *other* episodes in flight.
    """
    output_path = os.path.join(ep_dir, 'features.h5')

    if os.path.exists(output_path) and not force:
        return True

    ep_name = os.path.basename(ep_dir)
    h5_path = os.path.join(ep_dir, f"{ep_name}.h5")
    if not os.path.exists(h5_path):
        print(f"Warning: skipping {ep_dir}, no {ep_name}.h5 found")
        return False

    with h5py.File(h5_path, 'r') as f:
        expected_T = f['timestamp'].shape[0]

    local_timing = {'decode': 0.0, 'cpu': 0.0, 'gpu': 0.0}
    cam_features = []
    for cam_name in cams:
        suffix = CAM_SUFFIXES[cam_name]
        video_path = os.path.join(ep_dir, f"{ep_name}{suffix}")
        if not os.path.exists(video_path):
            print(f"Warning: missing video {video_path}")
            return False

        t0 = time.perf_counter()
        frames = decode_video(video_path)
        local_timing['decode'] += time.perf_counter() - t0
        if len(frames) != expected_T:
            print(f"Warning: {video_path} has {len(frames)} frames but H5 has {expected_T} timesteps")
            frames = frames[:expected_T]

        base_seed = abs(hash((ep_name, cam_name))) % (2 ** 31)
        stack = preprocess_camera(frames, im_size, num_variants, aug_cfg,
                                  base_seed, cpu_pool, local_timing)
        feats_flat = gpu_infer(model, stack, batch_size, gpu_lock, local_timing)
        del stack  # free ~4.3GB before stacking next camera
        # feats_flat: (K*T, 197, 768) -> (K, T, 197, 768)
        cam_features.append(feats_flat.reshape(num_variants, len(frames), 197, -1))

    features = np.stack(cam_features, axis=2)  # (K, T, num_cams, 197, 768)
    if fp16:
        features = features.astype(np.float16)

    with h5py.File(output_path, 'w') as hf:
        dset = hf.create_dataset('features', data=features)
        dset.attrs['num_variants'] = num_variants
        dset.attrs['color_jitter'] = bool(aug_cfg['use_color'])
        dset.attrs['crop_jitter'] = bool(aug_cfg['use_crop'])

    with timing_lock:
        for k, v in local_timing.items():
            timing[k] += v
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
    parser.add_argument("--cams", nargs='+', default=["left_wrist", "right_wrist", "head"])
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256,
                        help="GPU batch size for ViT forward (ViT-B saturates ~32-64 frames but "
                             "larger batches amortize Python overhead; 256 is a good default)")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=16,
                        help="CPU threads for preprocess + augmentation (cv2/numpy release the GIL)")
    parser.add_argument("--episode-concurrency", dest="episode_concurrency", type=int, default=2,
                        help="Max episodes in flight at once per GPU. Each holds ~4.3GB of "
                             "preprocessed fp32 + ~13GB of output features (K*T*3cams*197*768*4) "
                             "in RAM. Use --fp16 to halve the output size; lower this when RAM-limited.")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to shard episodes across. When > 1, the parent forks "
                             "one subprocess per GPU (CUDA_VISIBLE_DEVICES pinned) and each worker "
                             "handles episodes[rank::gpus]. Episodes per GPU share a CPU pool but "
                             "GPUs run independently — no cross-GPU sync.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--fp16", action="store_true", default=False)

    # K-variant augmentation
    parser.add_argument("--num-variants", dest="num_variants", type=int, default=1,
                        help="Number of feature variants per frame. Slot 0 is always the "
                             "identity (plain process_image). K=1 means no augmentation.")
    parser.add_argument("--color-jitter", dest="color_jitter", action="store_true", default=False,
                        help="Enable color jitter on variants 1..K-1")
    parser.add_argument("--color-jitter-brightness", type=float, default=0.3)
    parser.add_argument("--color-jitter-contrast", type=float, default=0.3)
    parser.add_argument("--color-jitter-saturation", type=float, default=0.3)
    parser.add_argument("--color-jitter-hue", type=float, default=0.05)
    parser.add_argument("--crop-jitter", dest="crop_jitter", action="store_true", default=False,
                        help="Enable random crop offset on variants 1..K-1")
    parser.add_argument("--crop-jitter-pixels", type=int, default=16)
    parser.add_argument("--force", action="store_true", default=False,
                        help="Re-extract and overwrite features.h5 even if it exists")
    args = parser.parse_args()

    if args.num_variants < 1:
        parser.error("--num-variants must be >= 1")
    if args.num_variants == 1 and (args.color_jitter or args.crop_jitter):
        print("[warn] K=1 means the single variant is the identity slot; "
              "--color-jitter/--crop-jitter will be ignored.")

    # Multi-GPU orchestration: parent forks one subprocess per GPU, each pinned to
    # one device via CUDA_VISIBLE_DEVICES and given a shard of episodes via
    # EXTRACT_RANK / EXTRACT_WORLD. Children run the same script with --gpus=1 so
    # they don't recurse.
    if args.gpus > 1 and os.environ.get('EXTRACT_RANK') is None:
        # Respect a pre-set CUDA_VISIBLE_DEVICES: rank r gets the r-th device
        # listed. If unset, rank r gets physical GPU r.
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible:
            gpu_ids = [d.strip() for d in visible.split(',') if d.strip()]
            if len(gpu_ids) < args.gpus:
                parser.error(f"--gpus={args.gpus} but CUDA_VISIBLE_DEVICES only exposes "
                             f"{len(gpu_ids)} devices ({visible})")
            gpu_ids = gpu_ids[:args.gpus]
        else:
            gpu_ids = [str(r) for r in range(args.gpus)]
        print(f"Forking {args.gpus} GPU workers on GPUs [{','.join(gpu_ids)}] "
              f"(stride sharding: episodes[rank::{args.gpus}])")
        # Drop any --gpus flag from argv so children don't recurse; force --gpus=1.
        cleaned = []
        skip = False
        for a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if a == '--gpus':
                skip = True
                continue
            if a.startswith('--gpus='):
                continue
            cleaned.append(a)
        child_argv = cleaned + ['--gpus', '1']
        procs = []
        for rank in range(args.gpus):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu_ids[rank]
            env['EXTRACT_RANK'] = str(rank)
            env['EXTRACT_WORLD'] = str(args.gpus)
            procs.append(subprocess.Popen([sys.executable, sys.argv[0]] + child_argv, env=env))
        exit_codes = [p.wait() for p in procs]
        if any(c != 0 for c in exit_codes):
            print(f"[error] GPU workers returned: {exit_codes}")
            sys.exit(1)
        print("All GPU workers finished.")
        sys.exit(0)

    rank = int(os.environ.get('EXTRACT_RANK', 0))
    world = int(os.environ.get('EXTRACT_WORLD', 1))

    aug_cfg = {
        'use_color': args.color_jitter and args.num_variants > 1,
        'use_crop': args.crop_jitter and args.num_variants > 1,
        'brightness': args.color_jitter_brightness,
        'contrast': args.color_jitter_contrast,
        'saturation': args.color_jitter_saturation,
        'hue': args.color_jitter_hue,
        'crop_jitter_pixels': args.crop_jitter_pixels,
    }

    # Load encoder
    print(f"Loading encoder: {args.model_name} (mode=mean, K={args.num_variants})")
    print(f"  augmentations: color_jitter={aug_cfg['use_color']}, crop_jitter={aug_cfg['use_crop']}")
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
    # Shard across GPU workers: rank r handles episodes[r::world].
    if world > 1:
        episodes = episodes[rank::world]
    tag = f"[rank {rank}/{world}] " if world > 1 else ""
    print(f"{tag}Processing {len(episodes)} episodes from {search_dir}  "
          f"(gpu_batch={args.batch_size}, workers={args.num_workers}, "
          f"episode_concurrency={args.episode_concurrency})")

    timing = {'decode': 0.0, 'cpu': 0.0, 'gpu': 0.0}
    timing_lock = threading.Lock()
    gpu_lock = threading.Lock()
    success_counter = [0]
    success_lock = threading.Lock()

    wall_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.num_workers) as cpu_pool, \
         ThreadPoolExecutor(max_workers=args.episode_concurrency) as ep_pool:

        def run_one(ep_dir):
            ok = process_episode(model, ep_dir, args.cams, im_size, args.batch_size,
                                 args.fp16, args.num_variants, aug_cfg, args.force,
                                 cpu_pool, gpu_lock, timing, timing_lock)
            if ok:
                with success_lock:
                    success_counter[0] += 1
            return ok

        futures = [ep_pool.submit(run_one, ep) for ep in episodes]
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc=f"rank {rank}" if world > 1 else "extract",
                      position=rank if world > 1 else None):
            pass

    wall = time.perf_counter() - wall_t0
    success = success_counter[0]

    # CPU and GPU timings sum across concurrent episodes — they can exceed wall time.
    print(f"\n{tag}Done. Processed {success}/{len(episodes)} episodes in {wall:.1f}s "
          f"({wall / max(success, 1):.1f}s/ep wall-clock).")
    print(f"{tag}  decode        : {timing['decode']:7.1f}s (summed across episodes)")
    print(f"{tag}  CPU preprocess: {timing['cpu']:7.1f}s (summed; parallel across {args.num_workers} workers × {args.episode_concurrency} episodes)")
    print(f"{tag}  GPU inference : {timing['gpu']:7.1f}s (summed; serialized on one GPU)")


if __name__ == "__main__":
    main()
