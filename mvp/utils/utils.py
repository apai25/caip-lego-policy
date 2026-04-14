#!/usr/bin/env python3

"""Utils."""

import builtins
import copy
import json
import math
import numpy as np
import os
import sys
import termcolor
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F


def get_optimizer_groups(model, default_wd):
    param_group_names, param_group_vars = dict(), dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # ks = [k for (k, x) in enumerate(["bn", "ln", "norm", "bias", ""]) if x in n]
        if "token" in n:
            name_apx = "t"
            wd_val = 0.0
        elif 'pos_embed' in n:
            name_apx = "p"
            wd_val = 0.0
        elif "bn" in n or "ln" in n or "norm" in n:
            name_apx = "n"
            wd_val = 0.0
        elif "bias" in n:
            name_apx = "b"
            wd_val = 0.0
        else:
            name_apx = 'w'
            wd_val = default_wd

        param_group = f"wd:{name_apx}"
        if param_group not in param_group_names:
            item = {"params": [], "weight_decay": wd_val}
            param_group_names[param_group] = copy.deepcopy(item)
            param_group_vars[param_group] = copy.deepcopy(item)
        param_group_names[param_group]["params"].append(n)
        param_group_vars[param_group]["params"].append(p)

    param_list = list(param_group_vars.values())

    param_group_str = termcolor.colored(
        json.dumps(param_group_names, sort_keys=True, indent=2), "blue"
    )
    print("Parameter groups:\n" + param_group_str)

    return param_list


def adjust_lr(optimizer, base_lr, cur_epoch, warmup_epoch, num_epoch):
    if cur_epoch < warmup_epoch:
        lr = base_lr * cur_epoch / warmup_epoch
    else:
        lr = base_lr * 0.5 * \
            (1. + math.cos(math.pi * (cur_epoch - warmup_epoch) / (num_epoch - warmup_epoch)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def save_checkpoint(out_dir, model, cur_epoch):
    fname = "model_ep{:04d}.pt".format(cur_epoch)
    out_f = os.path.join(out_dir, fname)
    torch.save({
        "epoch": cur_epoch,
        "model_state": unwrap_model(model).state_dict()
    }, out_f)
    print("Saved checkpoint to: {}".format(out_f))


def save_best_checkpoint(out_dir, model, cur_epoch, best_loss):
    fname = "model_best.pt"
    out_f = os.path.join(out_dir, fname)
    torch.save({
        "epoch": cur_epoch,
        "model_state": unwrap_model(model).state_dict(),
        "loss": best_loss
    }, out_f)
    print("Saved best checkpoint to: {}".format(out_f))


def load_checkpoint(file_path, model):
    assert os.path.exists(file_path)
    checkpoint = torch.load(file_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print("Loaded checkpoint from: {}".format(file_path))


def scaled_all_reduce(tensors):
    """
    Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group.
    """
    # There is no need for reduction in the single-proc case
    if not dist.is_initialized():
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / dist.get_world_size())
    return tensors


def suppress_print():
    """Suppresses printing from the current process."""
    def ignore(*_objects, _sep=" ", _end="\n", _file=sys.stdout, _flush=False):
        pass
    builtins.print = ignore


def suppress_wandb():
    """Suppresses wandb logging from the current_process."""
    def ignore(data, step=None, commit=None, sync=None):
        pass
    wandb.log = ignore


def masked_mse_loss(pred, target, mask):
    """
    Calculate the mean squared loss on only mask=1 positions for each dimension,
    and then take the average mean loss across different dimensions.
    """
    assert (pred.dim() == 3 or pred.dim() == 4) and (target.dim() == 3 or target.dim() == 4) and mask.dim() == 3  # B*T*C or B*T*N*C
    if pred.dim() == 4:
        mask = mask.unsqueeze(-2).repeat(1, 1, pred.shape[-2], 1)
    diff = (pred - target).pow(2)
    loss_per_dim = (diff * mask).sum(dim=tuple(range(pred.dim() - 1))) / (mask.sum(dim=tuple(range(pred.dim() - 1))) + 1e-6)
    loss = loss_per_dim.mean()
    return loss


def masked_l1_loss(pred, target, mask):
    """
    Calculate the mean squared loss on only mask=1 positions for each dimension,
    and then take the average mean loss across different dimensions.
    """
    assert (pred.dim() == 3 or pred.dim() == 4) and (target.dim() == 3 or target.dim() == 4) and mask.dim() == 3  # B*T*C or B*T*N*C
    if pred.dim() == 4:
        mask = mask.unsqueeze(-2).repeat(1, 1, pred.shape[-2], 1)
    diff = (pred - target).abs()
    mask_count = mask.sum(dim=tuple(range(pred.dim() - 1)))
    loss_per_dim = (diff * mask).sum(dim=tuple(range(pred.dim() - 1))) / (mask_count + 1e-6)
    active_dims = mask_count > 0
    loss = loss_per_dim[active_dims].mean() if active_dims.any() else loss_per_dim.mean()
    return loss


def masked_ce_loss(pred, target, mask):
    """
    Calculate cross entropy loss only for entries that has mask=1 for every dimension
    """
    assert pred.dim() == 3 and target.dim() == 2 and mask.dim() == 3  # B*T*C, target is B*T
    mask = torch.all(mask, dim=-1)
    pred, target, mask = pred.reshape(-1, pred.shape[-1]), target.reshape(-1), mask.reshape(-1)
    loss = F.cross_entropy(pred[mask], target[mask])
    return loss


def check_data_filter(metadata, data_filter):
    filtered = False
    for k, v in data_filter.items():
        if k in metadata.keys() and metadata[k] != v:
            filtered = True
            break
    return filtered