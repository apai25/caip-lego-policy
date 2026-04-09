#!/usr/bin/env python3

"""Logging."""

import builtins
import logging
import os
import sys
import wandb

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

import mvp.utils.distributed as dist


# Global TensorBoard writer object
_TB_WRITER_SINGLETON = None


def _suppress_print():
    """Suppresses printing from the current process."""
    def ignore(*_objects, _sep=" ", _end="\n", _file=sys.stdout, _flush=False):
        pass
    builtins.print = ignore


def setup_tb_and_wandb_logging(cfg):
    """Sets up the TensorBoard and Wandb logging."""
    global _TB_WRITER_SINGLETON
    # Initialize wandb before construct summary writer
    wandb.init(
        dir=cfg.logdir,
        name=cfg.wandb.name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg),
        mode=cfg.wandb.mode,
        resume=True,
        sync_tensorboard=True,
    )
    tb_events_dir = os.path.join(cfg.logdir, "events")
    os.makedirs(tb_events_dir, exist_ok=True)
    _TB_WRITER_SINGLETON = SummaryWriter(tb_events_dir)


def setup_logging(cfg):
    """Sets up the logging."""
    # Enable logging only for the main process
    if dist.is_main_proc():
        # Configure logging
        logging.root.handlers = []
        FORMAT = "[%(filename)s: %(lineno)3d]: %(message)s"
        logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
        # Setup TensorBoard and Wandb logging
        setup_tb_and_wandb_logging(cfg)
    else:
        _suppress_print()


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)
