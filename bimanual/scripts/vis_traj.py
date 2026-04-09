"""Script to visualize a single trajectory in rerun local app."""
import argparse
import joblib
import json
import numpy as np
import os
import rerun as rr

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

        metadata_str = f'''\n{json.dumps(metadata, indent=4)}\n'''
        if 'task' in obs:
            metadata_str += f'''\nFrame-level Task: {obs['task']}\n'''
        rr.log("metadata", rr.TextDocument(f'''```{metadata_str}```''', media_type=rr.MediaType.MARKDOWN))

        # arm joints
        for side in ["left", "right"]:
            # arm joints
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
            
            # arm force
            for i in range(6):
                if f"{side}_arm_tcp_force" not in obs:
                    continue
                rr.log(
                    f"ur3e_force/{side}_{i}",
                    rr.Scalar(obs[f"{side}_arm_tcp_force"][i]),
                )

            # finger joints
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

            # finger touch
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
                rr.log(f"{i}_{cam_name}", rr.Image(obs[cam_name]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=str, required=True, help="Path to the traj")
    args = parser.parse_args()

    app_id = os.path.split(args.traj)[1]
    print(f"Starting {app_id=}")

    success = os.path.exists(os.path.join(args.traj, "success.txt"))
    suffix = "_success" if success else "_failure"
    rid = str(Path(args.traj).name + suffix)

    print(f"Loading new recording with {rid=}")
    rr.init(app_id, recording_id=rid, spawn=True)
    log_traj(args.traj)


if __name__ == "__main__":
    main()
