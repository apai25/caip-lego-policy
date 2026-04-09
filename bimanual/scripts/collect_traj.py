"""Trajectory collection script for bimanual robot using Quest3 VR / MANUS glove Teleoperation."""
import click
import redis
import time
from loguru import logger

from bimanual.control.quest3 import VRPolicy
from bimanual.control.manus import MocapPolicy
from bimanual.env.real import DualUR3ERealEnv
from bimanual.env.sim import TermStatus

@click.command()
@click.option("--num-traj", type=int, default=100, help="Number of demos to collect")
@click.option("--data-dir", required=True, type=str, help="Directory to save data")
@click.option("--save-hz", default=30.0, help="Save frequency")
@click.option("--step-hz", default=300.0, help="Step frequency")
@click.option('--mode', type=click.Choice(['manus', 'handtrack']), default='manus', help='Use MANUS mocap gloves or Quest3 vision-based handtracking')
def collect_traj(num_traj, data_dir, save_hz, step_hz, mode):
    r = redis.Redis(host="localhost", port=6379, db=0)

    if mode == 'manus':
        TeleopPolicy = MocapPolicy
    elif mode == 'handtrack':
        TeleopPolicy = VRPolicy
    else:
        raise ValueError(f'Mode {mode} not supported')

    with DualUR3ERealEnv(pedals=True, window=False, save_dir=data_dir, save_freq=save_hz) as env, TeleopPolicy(env) as policy:
        for i in range(num_traj):
            logger.info(f"[CollectTrajectory] Starting new trajectory ({i}/{num_traj})")

            env.init(random=False)
            policy.reset_state()
            policy.warmup()
            r.flushall()
            time.sleep(0.5)

            obs = env.get_obs()
            while True:
                term_status = env.get_term_status()
                if term_status != TermStatus.ACTIVE:
                    break

                if not env.is_active():
                    time.sleep(1.0 / step_hz)
                    continue

                begin_t = time.time()
                action = policy.forward(obs)
                obs = env.step(action)
                end_t = time.time()

                sleep_time = max(0, 1.0 / step_hz - (end_t - begin_t))
                time.sleep(sleep_time)

            logger.info(f"[CollectTrajectory] Trajectory ended with termination status: {term_status}")

            # Don't save if quit
            if term_status == TermStatus.QUIT:
                break

            # Save trajectory
            success = term_status == TermStatus.SUCCESS
            env.save_traj(success)

    logger.info("[CollectTrajectory] Done collecting trajectories")


if __name__ == "__main__":
    collect_traj()
