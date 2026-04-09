#!/usr/bin/env python3

"""Meters (adapted from: https://github.com/facebookresearch/pycls/blob/main/pycls/core/meters.py)."""

import decimal
import simplejson
import time
import wandb

from collections import deque

import numpy as np
import torch


def time_string(seconds):
    """Converts time in seconds to a fixed-width string format."""
    days, rem = divmod(int(seconds), 24 * 3600)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024


def float_to_decimal(data, prec=4):
    """Convert floats to decimals which allows for fixed width json."""
    if prec and isinstance(data, dict):
        return {k: float_to_decimal(v, prec) for k, v in data.items()}
    if prec and isinstance(data, float):
        return decimal.Decimal(("{:." + str(prec) + "f}").format(data))
    else:
        return data


def dump_json_stats(stats, stats_type):
    """Covert stats dict into json string for logging."""
    stats = float_to_decimal(stats)
    stats_json = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    return "{:s}: {:s}".format(stats_type, stats_json)


class Timer(object):
    """A simple timer (adapted from Detectron)."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0
        self.reset()

    def tic(self):
        # using time.time as time.clock does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        if not isinstance(value, list):
            value = [value]
        self.deque.extend(value)
        self.count += len(value)
        self.total += sum(value)

    def get_value(self, mode="mean"):
        if not self.deque:
            return 0.0
        if mode == "median":
            return self.get_win_median()
        elif mode == "mean":
            return self.get_win_mean()
        elif mode == "global_mean":
            return self.get_global_mean()
        elif mode == "latest":
            return self.get_latest()
        else:
            raise NotImplementedError

    def get_win_median(self):
        return np.median(list(self.deque))

    def get_win_mean(self):
        return np.mean(list(self.deque))

    def get_global_mean(self):
        return self.total / self.count

    def get_latest(self):
        return self.deque[-1]


class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, num_ep, epoch_iters, log_period=10, phase="train"):
        self.num_ep = num_ep
        self.epoch_iters = epoch_iters
        self.max_iter = num_ep * epoch_iters
        self.log_period = log_period
        self.phase = phase
        self.iter_timer = Timer()
        self.loss = ScalarMeter(log_period)
        self.loss_total = 0.0
        self.lr = None
        self.num_samples = 0

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, lr, mb_size):
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = cur_epoch * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.num_ep),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % self.log_period == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            print(dump_json_stats(stats, self.phase + "_iter"))
            cur_iter_abs = cur_epoch * self.epoch_iters + cur_iter + 1
            wandb.log({
                "train/iter": cur_iter_abs,
                "train/iter_loss": stats["loss"],
                "train/lr": stats["lr"],
            })

    def get_epoch_stats(self, cur_epoch):
        cur_iter_total = (cur_epoch + 1) * self.epoch_iters
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.num_ep),
            "time_avg": self.iter_timer.average_time,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "eta": time_string(eta_sec),
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        print(dump_json_stats(stats, self.phase + "_epoch"))
        wandb.log({
            "train/ep": cur_epoch + 1,
            "train/ep_loss": stats["loss"]
        })


class TestMeter(object):
    """Measures testing stats."""

    def __init__(self, num_ep, epoch_iters, log_period=10, phase="test"):
        self.num_ep = num_ep
        self.epoch_iters = epoch_iters
        self.log_period = log_period
        self.phase = phase
        self.iter_timer = Timer()
        self.loss = ScalarMeter(log_period)
        self.loss_total = 0.0
        self.num_samples = 0

    def reset(self, min_errs=False):
        self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, mb_size):
        self.loss.add_value(loss)
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = gpu_mem_usage()
        iter_stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.num_ep),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "loss": self.loss.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % self.log_period == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            print(dump_json_stats(stats, self.phase + "_iter"))

    def get_epoch_stats(self, cur_epoch):
        mem_usage = gpu_mem_usage()
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.num_ep),
            "time_avg": self.iter_timer.average_time,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "loss": avg_loss,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        print(dump_json_stats(stats, self.phase + "_epoch"))
        wandb.log({
            "test/ep": cur_epoch + 1,
            "test/ep_loss": stats["loss"]
        })
