#!/usr/bin/env python3

"""Train a policy with BC."""

import hydra
import omegaconf
import os
import wandb

from omegaconf import OmegaConf

from mvp.utils.sys_utils import omegaconf_to_dict, print_dict, dump_cfg
from mvp.utils.sys_utils import set_np_formatting, set_seed
from mvp.utils.utils import suppress_print, suppress_wandb

import mvp.bimanual_bc.bc as bc

import torch
import torch.distributed as dist


@hydra.main(version_base=None, config_name="config_bkl", config_path="../configs/bimanual_bc")
def train(cfg: omegaconf.DictConfig):

    # Set up distributed env
    if cfg.num_gpus > 1:
        dist.init_process_group("nccl")
        local_rank = dist.get_rank() % cfg.num_gpus
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    # Set up logging
    if local_rank == 0:
        print_dict(omegaconf_to_dict(cfg))
        os.makedirs(cfg.logdir, exist_ok=True)
        dump_cfg(cfg, cfg.logdir)
        wandb.init(
            dir=cfg.logdir,
            name=cfg.wandb.name,
            project=cfg.wandb.project,
            # entity=cfg.wandb.entity,
            config=omegaconf.OmegaConf.to_container(cfg),
            mode=cfg.wandb.mode,
        )
    else:
        suppress_print()
        suppress_wandb()

    # Set rng seed
    #seed = cfg.seed * cfg.num_gpus + local_rank
    seed = cfg.seed
    set_np_formatting()
    set_seed(seed)

    # Perform training
    bc.train(cfg)

    # Clean up
    if cfg.num_gpus > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    train()
