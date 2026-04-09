"""Script to visualize many trajectories in the rerun browser app."""
import os
import argparse
import joblib
import json
import numpy as np
import random
import rerun as rr
import signal
import subprocess
import time
import cv2

from glob import glob
from pathlib import Path

def log_traj(data_dir):
    print(f"Loading data from {data_dir}")
    traj = []
    for pkl in sorted(glob(f"{data_dir}/*.pkl")):
        traj.append(joblib.load(pkl))
    assert len(traj) > 0, "No data found"

    with open(f'{data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    finger_joint_names = [
        "index",
        "middle",
        "ring",
        "pinky",
        "lthumb",
        "uthumb",
    ]

    # ['time_stamps', 'right_arm_joint_pos', 'left_arm_joint_pos', 'left_fingers_joint_pos', 'right_fingers_joint_pos', 'left_fingers_touch', 'right_fingers_touch', 'rgb_head', 'rgb_left', 'rgb_right']
    for i, obs in enumerate(traj):
        keys = list(obs.keys())
        rr.set_time_sequence("step", i)
        if 'time_stamps' in obs: rr.set_time_seconds("wall_time", np.mean(obs["time_stamps"]))

        rr.log("metadata", rr.TextDocument(f'''```json\n{json.dumps(metadata, indent=4)}\n```''', media_type=rr.MediaType.MARKDOWN))

        # arm joints
        for side in ["left", "right"]:
            for i in range(len(arm_joint_names)):
                if f"{side}_arm_joint_pos" not in obs:
                    continue
                rr.log(
                    f"ur3e_joints/{side}_{arm_joint_names[i]}",
                    rr.Scalar(obs[f"{side}_arm_joint_pos"][i]),
                )
                rr.log(
                    f'ur3e_joints/{side}_{arm_joint_names[i]}_action',
                    rr.Scalar(obs[f'{side}_arm_cmd'][i]),
                )

        # finger joints
        for side in ["left", "right"]:
            for i in range(len(finger_joint_names)):
                if f"{side}_fingers_joint_pos" not in obs:
                    continue
                rr.log(
                    f"hand_joints/{side}_{finger_joint_names[i]}",
                    rr.Scalar(obs[f"{side}_fingers_joint_pos"][i]),
                )
                rr.log(
                    f'hand_joints/{side}_{finger_joint_names[i]}_action',
                    rr.Scalar(obs[f'{side}_fingers_cmd'][i]),
                )

        for side in ["left", "right"]:
            for i in range(30):
                if f'{side}_fingers_touch' not in obs:
                    continue
                rr.log(
                    f"touch/{side}_{i}",
                    rr.Scalar(obs[f'{side}_fingers_touch'][i]),
                )

        cam_names = ["rgb_left", "rgb_head", "rgb_right"]
        for i, cam_name in enumerate(cam_names):
            if cam_name in obs:
                if obs[cam_name].shape == (480, 640, 3):
                    obs[cam_name] = obs[cam_name][:, 80:-80, :]
                obs[cam_name] = cv2.resize(obs[cam_name], (224, 224), interpolation=cv2.INTER_LINEAR)
                rr.log(f"{i}_{cam_name}", rr.Image(obs[cam_name]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-traj', type=int, default=10, help='Number of demos to process')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--port', type=int, required=True, help='TCP port')
    parser.add_argument('--sort', action='store_true', help='Sort trajectories')
    args = parser.parse_args()

    app_id = os.path.split(os.path.split(args.data_dir)[0])[1]
    print(f'Starting {app_id=}')

    process = subprocess.Popen(f"rerun --serve --port {args.port}", shell=True)

    traj_dirs = sorted(os.listdir(args.data_dir))
    assert len(traj_dirs) > 0, "No data found"

    traj_lens = [len(os.listdir(os.path.join(args.data_dir, tr))) for tr in traj_dirs]
    trajs = sorted([(tl, td) for td, tl in zip(traj_dirs, traj_lens)])
    if not args.sort:
        random.shuffle(trajs)

    traj_loaded = 0
    for _, td in trajs:
        traj = os.path.join(args.data_dir, td)
        success = os.path.exists(os.path.join(traj, "success.txt"))
        if not success: continue

        suffix = "_success" if success else "_failure"
        rid = str(Path(traj).name + suffix)

        print(f"Loading new recording with {rid=}")
        rr.init(app_id, recording_id=rid, spawn=True, exp_add_to_app_default_blueprint=False, port=args.port)
        log_traj(traj)

        traj_loaded += 1
        if traj_loaded >= args.num_traj:
            break

    print("Finished loading trajectories")
    print("Press Ctrl+C to stop the server")

    try:
        while True:
            time.sleep(1/10)
    except:
        process.send_signal(signal.SIGINT)

    process.wait()
    print("Server stopped")

if __name__ == "__main__":
    main()
