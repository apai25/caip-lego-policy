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
from mvp.bimanual_bc.tokenizer import ActionTokenizerBimanual
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

    # create a tokenizer
    num_pred = cfg.actor.num_pred
    tokenizer = ActionTokenizerBimanual(num_steps=cfg.actor.num_pred,
                                        model_type=cfg.tokenizer.model_type)

    # Load a checkpoint
    if cfg.test.weights:
        state_dict = torch.load(cfg.test.weights, map_location="cpu")["model_state"]
        mks, uks = tokenizer.load_state_dict(state_dict, strict=False)
        print("loaded checkpoint from: {}".format(cfg.test.weights))
        print("missing key: {}".format(mks))
        print("unexpected keys: {}".format(uks))
    
    tokenizer = tokenizer.cuda()
    tokenizer.eval()

    with DualUR3ERealEnv(window=False, show_cams=True) as env:
        control_freq = 15

        env.init(random=False)
        obs = env.get_obs()
        # for i in range(10):
        #     obs = env.get_obs()
        #     print(obs)
        #     time.sleep(1)

        obs_0 = obs
        prev_action = None

        replay_path = (
            "/home/ilija/data/pick-yellow-right_04-07-2024/2024-04-07_23-34-41/"
        )
        pkl_files = [
            file for file in sorted(os.listdir(replay_path)) if file.endswith(".pkl")
        ]
        print(pkl_files)
        obs_all = [joblib.load(os.path.join(replay_path, file)) for file in pkl_files]
        obs_all = obs_all[::2]

        target_states = []
        reached_states = []

        for iter in range((len(obs_all) - 1) // num_pred):

            states_next = [compute_state(
                obs_all[step + 1],
                cfg.data.default_pos_left_arm,
                cfg.data.default_pos_right_arm,
            ) for step in range(iter * num_pred, (iter + 1) * num_pred)]
            states_next = np.stack(states_next, axis=0)

            commanded_states_next = [compute_commanded_state(
                obs_all[step + 1],
                cfg.data.default_pos_left_arm,
                cfg.data.default_pos_right_arm,
            ) for step in range(iter * num_pred, (iter + 1) * num_pred)]
            commanded_states_next = np.stack(commanded_states_next, axis=0)

            current_state = compute_state(
                obs_all[iter * num_pred], cfg.data.default_pos_left_arm, cfg.data.default_pos_right_arm
            )

            tokenized_commanded_states_next, _, _ = tokenizer.encode(torch.Tensor(commanded_states_next[None] - current_state[None, None]).cuda())
            tokenized_commanded_states_next = tokenizer.decode(tokenized_commanded_states_next)
            tokenized_commanded_states_next = (tokenized_commanded_states_next.detach().cpu().numpy() + current_state[None, None]).squeeze(0)
            # tokenized_commanded_states_next = commanded_states_next + np.random.randn(num_pred, 24) * 0.01

            print(((tokenized_commanded_states_next - commanded_states_next)**2).mean())

            for step in range(num_pred):

                begin_t = time.time()

                state_next = states_next[step]
                commanded_state_next = commanded_states_next[step]
                tokenized_commanded_state_next = tokenized_commanded_states_next[step]
                action = format_actions(
                    # commanded_state_next,
                    tokenized_commanded_state_next,
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


        # plot the difference between desired states and reached states
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for i, ax in enumerate(axes.flat):
            ax.plot(np.arange((len(obs_all) - 1) // num_pred * num_pred), target_states[:, i + 6], "-r")
            ax.plot(np.arange((len(obs_all) - 1) // num_pred * num_pred), reached_states[:, i + 6], "-b")
            ax.set_title(f"Plot for state {i + 6}")

        plt.tight_layout()
        plt.savefig("desired_states_vs_reached_states.png")


if __name__ == "__main__":
    run_robot()

    
