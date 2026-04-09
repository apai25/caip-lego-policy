#!/usr/bin/env python3

"""Run policy on bimanual robot."""
import hydra
import omegaconf
import redis
import time
import numpy as np

from loguru import logger

from mvp.bimanual_bc.dataset import compute_state
from bimanual.models.transformer_agent import format_actions, TransformerAgent
from bimanual.env.real import DualUR3ERealEnv
from bimanual.util.keyboard import KeyboardListener


@hydra.main(version_base=None, config_name="config", config_path="../../configs/bimanual_bc")
def run_robot(cfg: omegaconf.DictConfig):
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushall()

    with DualUR3ERealEnv(window=False, show_cams=True) as env:
        control_freq = 15 
        num_steps = cfg.test.num_steps
        agent = TransformerAgent(cfg)
        env.init(random=False)
        listener = KeyboardListener()

        assert cfg.actor.look_ahead == 0
        print('Enter a prompt. Type "quit" to exit. Type "reset" to reset to default initial position.')
        while True:
            # Do not log warning statements over prompt cursor
            env.warn_skip(False)

            # Get prompt
            prompt = input("bimanual>>> ")
            if prompt.lower() == 'quit':
                break
            if prompt.lower() == 'reset':
                env.init(random=False)
                continue
            if len(prompt) < 10:
                continue
            logger.info(f"Setting {prompt=}")
            agent.set_prompt(prompt)

            # Reset agent and listener
            obs = env.get_obs()
            agent.reset_buffers(obs)
            all_actions = np.zeros((num_steps, num_steps + cfg.actor.num_exec - 1, 24))
            listener.reset()

            # Run policy. Catch keyboard interrupt to exit execution early
            for step in range(num_steps):
                begin_t = time.time()

                # get action
                actions = agent.act(obs, process_img=(step % agent.process_img_every == 0))
                all_actions[step, step:step+cfg.actor.num_exec] = actions[:cfg.actor.num_exec]

                if cfg.actor.num_agg == -1:
                    # no temporal aggregation
                    action = all_actions[step - step % cfg.actor.num_exec, step]
                else:
                    start_idx = max(step - cfg.actor.num_exec + 1, 0)
                    end_idx = min(start_idx + cfg.actor.num_agg, step + 1)
                    selected_actions = all_actions[start_idx: end_idx, step]
                    weights = np.exp(-0.2 * np.array(list(range(selected_actions.shape[0]))))
                    action = (selected_actions * weights[..., None]).sum(axis=0) / weights.sum()

                # Safety measure. If the movement in any joint is larger than 1 rad, then abort immediately.
                assert np.abs(action[:12] - compute_state(obs, cfg.data.default_pos_left_arm, cfg.data.default_pos_right_arm)[:12]).max() < 2.0, \
                        "Large joint movement detected! Abort now..."

                action = format_actions(action, cfg.data.default_pos_left_arm, cfg.data.default_pos_right_arm)

                # step env
                env.step(action)

                # sleep
                logger.info(f"[Step {step}/{num_steps}] Took {time.time() - begin_t}")
                time.sleep(max(0, 1.0 / control_freq - (time.time() - begin_t)))
                obs = env.get_obs()

                if listener.ctrlq_pressed:
                    logger.info("Quiting policy execution")
                    break
                    

if __name__ == "__main__":
    run_robot()
