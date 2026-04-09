import datetime
import os
from pathlib import Path
import joblib
from loguru import logger
from multiprocessing import Pool
from tqdm import tqdm
import json

class TrajectoryWriter:
    def __init__(self, traj_dir):
        self.traj_dir = Path(traj_dir)
        os.makedirs(self.traj_dir, exist_ok=True)
        self.traj = []

    def add_obs(self, obs):
        # check no obs are empty
        for k in obs:
            if len(obs[k]) == 0:
                logger.warning("[TrajectoryWriter] Empty obs")
                raise Exception("Empty obs")
        self.traj.append(obs)

    def add_action(self, action):
        for k in action:
            if len(action[k]) == 0:
                logger.warning("[TrajectoryWriter] Empty action")
                raise Exception("Empty action")
        self.traj.append(action)

    def add_action_obs(self, action, obs):
        for k in action:
            if len(action[k]) == 0:
                logger.warning("[TrajectoryWriter] Empty action")
                raise Exception("Empty action")
        for k in obs:
            if len(obs[k]) == 0:
                logger.warning("[TrajectoryWriter] Empty obs")
                raise Exception("Empty obs")

        # check frame is not the same as the past 10 frames
        if len(self.traj) > 10:
            assert not all(
                [f["time_stamps"] == obs["time_stamps"] for f in self.traj[-10:]]
            ), f"Too many constant frames: {[f['time_stamps'] for f in self.traj[-10:]]}"

        frame = {**action, **obs}
        self.traj.append(frame)

    @staticmethod
    def save_obs(arg):
        path, obs = arg
        joblib.dump(obs, path, compress=3)

    def save(self, success=True):
        if len(self.traj) == 0:
            logger.info("[TrajectoryWriter] Trajectory is empty. Aborting traj save")
            return

        # Check that traj is not constant by verifying first and last obs are not equivalent
        first_obs = self.traj[0]
        last_obs = self.traj[-1]
        assert not (
            first_obs["right_fingers_joint_pos"] == last_obs["right_fingers_joint_pos"]
        ).all(), f'Right finger joints equivalent {first_obs["right_fingers_joint_pos"]=}, {last_obs["right_fingers_joint_pos"]=}'
        assert not (
            first_obs["left_fingers_joint_pos"] == last_obs["left_fingers_joint_pos"]
        ).all(), f'Left finger joints equivalent {first_obs["left_fingers_joint_pos"]=}, {last_obs["left_fingers_joint_pos"]=}'

        # Create dir using current datetime
        current_datetime = datetime.datetime.now()
        datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = self.traj_dir / datetime_str
        os.makedirs(dir_name)

        # Save metadata
        metadata = {
            "success": success,
            "task": "pick right arm",
        }
        with open(dir_name / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Save traj pkls
        num_workers = 12
        args = [
            (dir_name / "{:04d}.pkl".format(i), obs) for i, obs in enumerate(self.traj)
        ]
        with Pool(num_workers) as p:
            for _ in tqdm(p.imap(TrajectoryWriter.save_obs, args), total=len(args)):
                pass

        # Save success/failure txt
        if success:
            open(dir_name / "success.txt", "w").close()
        else:
            open(dir_name / "failure.txt", "w").close()

        # Reset traj
        self.traj = []

        logger.info("[TrajectoryWriter] Trajectory written to {}".format(dir_name))
    
