#!/usr/bin/env python3

"""Evaluate pick policy with grid of cube positions on bimanual robot."""
import hydra
import omegaconf
import redis
import time
import numpy as np
from omegaconf import OmegaConf
from loguru import logger

from mvp.bimanual_bc.dataset import compute_state
from bimanual.models.transformer_agent import format_actions, TransformerAgent
from bimanual.env.real import DualUR3ERealEnv
from bimanual.util.keyboard import KeyboardListener


GRID_RIGHT_QPOS = [
    # [-1.8276403586017054, -1.4510453504375, -1.5231715440750122, -2.486485620538229, 0.510415256023407, 0.7963149547576904], fail
    # [-1.9588235060321253, -1.9284621677794398, -1.1332306861877441, -2.417659421960348, 0.355668306350708, 0.760219931602478], fail
    [-2.033132855092184, -2.1809588871397914, -0.9020600318908691, -1.8210278950133265, 0.26560717821121216, 0.310488224029541], 
    [-2.1855037848102015, -1.7029744587340296, -1.4863970279693604, -1.9353586635985316, 0.6059000492095947, 0.4187048077583313], 
    [-2.1809051672564905, -2.0839997730650843, -1.2121742963790894, -1.9498540363707484, 0.37738290429115295, 0.4398021697998047], 
    [-2.192183319722311, -2.3841072521605433, -0.8551125526428223, -1.5558914107135315, 0.31705477833747864, 0.1835971623659134], 
    [-2.4073341528521937, -1.8677698574461878, -1.9048186540603638, -0.8582750123790284, 0.5067670345306396, -0.143195931111471], 
    [-2.35602313676943, -2.2641097507872523, -1.5308632850646973, -0.8165646356395264, 0.33510875701904297, -0.11374551454652959], 
    [-2.358760658894674, -2.5639435253539027, -0.9880634546279907, -0.8156109613231202, 0.4131850302219391, -0.41574031511415654], 
    [-2.577033821736471, -2.4602557621397914, -1.713452696800232, -0.19240696847949224, 0.3742654025554657, -0.3991630713092249], 
    [-2.529761616383688, -2.6535941563048304, -1.3120880126953125, -0.19690020502124028, 0.4218795597553253, -0.47761470476259404], 
    [-2.5334137121783655, -2.9316617451109828, -0.7679498791694641, -0.19783909738574224, 0.5794112682342529, -0.8869956175433558], 
]


# @hydra.main(version_base=None, config_name="config", config_path="../../configs/bimanual_bc")
def run_robot():
    cfg = OmegaConf.load('/home/ilija/code/mvp_generalize_v2/mvp-generalize/bimanual/test_config/config.yaml')
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushall()

    with DualUR3ERealEnv(window=False, show_cams=False) as env:
        control_freq = 15 
        num_steps = cfg.test.num_steps
        agent = TransformerAgent(cfg)
        env.init(random=False)

        listener = KeyboardListener()

        assert cfg.actor.look_ahead == 0
        print('Evaluating cube pick on 3x4 grid. Press Ctrl+Q in the middle of policy execution to exit early.')

        results = []

        for i in range(12):
            # Do not log warning statements over prompt cursor
            env.warn_skip(False)

            # Move to grid position
            env.init(random=False, qpos={'right': GRID_RIGHT_QPOS[i]})

            # Place cube at grid position
            input(f"Press Enter when cube is placed at grid position {i+1} ")

            # Move to initial position
            env.init(random=False)
            time.sleep(0.5)

            # Execute policy
            logger.info(f"Executing policy for cube at grid position {i+1}")
            obs = env.get_obs()
            agent.reset_buffers(obs)
            all_actions = np.zeros((num_steps, num_steps + cfg.actor.num_exec - 1, 24))
            listener.reset()

            for step in range(num_steps):
                begin_t = time.time()

                # get action
                actions = agent.act(obs, process_img=(step % agent.process_img_every == 0), step=step)
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
                logger.info(f'[Step {step}/{num_steps}] Took {time.time() - begin_t}')
                time.sleep(max(0, 1.0 / control_freq - (time.time() - begin_t)))
                obs = env.get_obs()

                if listener.ctrlq_pressed:
                    logger.info("Quiting policy execution")
                    break

            while True:
                result = input("Success? (y/n): ")
                if result in ['y', 'n']:
                    results.append(result)
                    break
    
        logger.info(f'Results: {results=}')
        logger.info(f'Number of successes: {results.count("y")}/{len(results)}')

if __name__ == "__main__":
    run_robot()
