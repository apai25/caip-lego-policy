#!/usr/bin/env python3

"""Run policy on bimanual robot."""
import warnings
import os
import joblib

import math
import hydra
import omegaconf
import redis
import time
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from mvp.bimanual_bc.vision_encoder import Encoder
from mvp.bc.actor import ActorTransformerConcat, ActorTransformerConcatAttnPooling
from mvp.bimanual_bc.dataset import compute_state, process_image

from bimanual.env.real import DualUR3ERealEnv


def compute_commanded_state(obs_t1, default_pos_left_arm, default_pos_right_arm):
    # use commanded action instead of next-step state
    left_arm_action = np.array(obs_t1["left_arm_cmd"]) - default_pos_left_arm
    right_arm_action = np.array(obs_t1["right_arm_cmd"]) - default_pos_right_arm
    left_fingers_action = np.array(obs_t1["left_fingers_cmd"])
    right_fingers_action = np.array(obs_t1["right_fingers_cmd"])
    action = np.concatenate(
        [left_arm_action, right_arm_action, left_fingers_action, right_fingers_action]
    )

    return action


def format_actions(actions, default_pos_left_arm, default_pos_right_arm):
    return {
        "left_arm_cmd": actions[:6] + np.array(default_pos_left_arm),
        "right_arm_cmd": actions[6:12] + np.array(default_pos_right_arm),
        "left_fingers_cmd": actions[12:18],
        "right_fingers_cmd": actions[18:24],
    }


@hydra.main(
    version_base=None, config_name="config", config_path="../../configs/bimanual_bc"
)
def run_robot(cfg: omegaconf.DictConfig):
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushall()

    with DualUR3ERealEnv(window=False, show_cams=True) as env:
        control_freq = 15
        num_steps = cfg.test.num_steps

        env.init(random=False)
        obs = env.get_obs()
        # for i in range(10):
        #     obs = env.get_obs()
        #     print(obs)
        #     time.sleep(1)

        obs_0 = obs
        prev_action = None

        replay_path = (
            "/home/ilija/data/pick-yellow-right_03-28-2024/2024-03-21_22-32-39/"
        )
        pkl_files = [
            file for file in sorted(os.listdir(replay_path)) if file.endswith(".pkl")
        ]
        print(pkl_files)
        obs_all = [joblib.load(os.path.join(replay_path, file)) for file in pkl_files]
        obs_all = obs_all[::2]

        target_states = []
        reached_states = []

        for step in range(len(obs_all) - 1):
            begin_t = time.time()

            state_next = compute_state(
                obs_all[step + 1],
                cfg.data.default_pos_left_arm,
                cfg.data.default_pos_right_arm,
            )
            commanded_state_next = compute_commanded_state(
                obs_all[step + 1],
                cfg.data.default_pos_left_arm,
                cfg.data.default_pos_right_arm,
            )
            action = format_actions(
                commanded_state_next,
                cfg.data.default_pos_left_arm,
                cfg.data.default_pos_right_arm,
            )

            # step env
            env.step(action)

            # sleep
            time.sleep(max(0, 1.0 / control_freq - (time.time() - begin_t)))
            obs = env.get_obs()

            reached_state = compute_state(
                obs, cfg.data.default_pos_left_arm, cfg.data.default_pos_right_arm
            )

            target_states.append(state_next)
            reached_states.append(reached_state)

        target_states = np.stack(target_states, axis=0)
        reached_states = np.stack(reached_states, axis=0)

        for i in range(6, 12):
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(obs_all) - 1), target_states[:, i], "-r")
            plt.plot(np.arange(len(obs_all) - 1), reached_states[:, i], "-b")
            plt.savefig(f"plot_commanded_state_{i}.png")


if __name__ == "__main__":
    run_robot()
