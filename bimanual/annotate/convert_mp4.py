"""Convert trajectories (head image only) from pkl format into mp4s"""

import os
import click
import joblib
import moviepy.editor as mpy

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

def convert_traj(args):
    traj_name, traj_dir, save_dir = args
    imgs = []
    for pkl in sorted(glob(os.path.join(traj_dir, '*.pkl'))):
        data = joblib.load(pkl)
        imgs.append(data['rgb_head'])
    
    clip = mpy.ImageSequenceClip(imgs, fps=30)
    clip.write_videofile(os.path.join(save_dir, f"{traj_name}.mp4"))


@click.command()
@click.option("--traj-dir", required=True, type=str, help="Directory for trajectories stored in pkl format")
@click.option("--video-dir", required=True, type=str, help="Directory to save mp4 head camera videos")
def main(traj_dir, video_dir):
    os.makedirs(video_dir, exist_ok=True)

    args = []
    for d in os.listdir(traj_dir):
        args.append((d, os.path.join(traj_dir, d), video_dir))
    
    num_workers = 8
    with Pool(num_workers) as p:
        for _ in tqdm(p.imap(convert_traj, args), total=len(args)):
            pass

if __name__ == "__main__":
    main()
