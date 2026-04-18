#!/usr/bin/env python3

"""Demo dataset."""

import copy
import cv2
import joblib
import math
import numpy as np
import os
import random
import json
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from PIL import Image, ImageFilter
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image

from mvp.utils.utils import check_data_filter

# ImageNet per-channel mean and standard deviation (in RGB order)
_IN_MEAN = [0.485, 0.456, 0.406]
_IN_STD = [0.229, 0.224, 0.225]


def color_norm(im, mean, std):
    """Performs per-channel normalization."""
    for i in range(3):
        im[:, :, i] = (im[:, :, i] - mean[i]) / std[i]
    return im


def color_unnorm(im, mean, std):
    """Performs per-channel unnormalization."""
    for i in range(3):
        im[:, :, i] = im[:, :, i] * std[i] + mean[i]
    return im


def normalize(im):
    """Performs image normalization."""
    # [0, 255] -> [0, 1]
    im = im.astype(np.float32) / 255.0
    # Color norm
    im = color_norm(im, _IN_MEAN, _IN_STD)
    # HWC -> CHW
    im = im.transpose([2, 0, 1])
    return im


def unnormalize(im):
    """Performs image unnormalization."""
    # CHW -> HWC
    im = im.transpose([1, 2, 0])
    # Color unnorm
    im = color_unnorm(im, _IN_MEAN, _IN_STD)
    # [0, 1] -> [0, 255]
    im = (im.astype(np.float32) * 255.0).astype(np.uint8)
    return im


def center_crop(im, crop_size):
    """Performs center cropping."""
    h, w = im.shape[:2]
    x = math.ceil((w - crop_size) / 2)
    y = math.ceil((h - crop_size) / 2)
    return im[y:(y + crop_size), x:(x + crop_size), :]


def process_image(im, im_size):
    """Processes an image for network input."""
    im = np.array(im).astype(np.float32)
    im = center_crop(im, im.shape[0])
    im = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_LINEAR)
    im = normalize(im)
    return im


_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _apply_color_jitter(im_u8, rng, brightness, contrast, saturation, hue):
    """torchvision-style ColorJitter on a uint8 HWC RGB array.

    numpy + cv2 implementation (no PIL) — ~30x faster than the PIL version
    while matching torchvision's semantics:
      brightness: im * b_fac
      contrast:   (im - gray_mean) * c_fac + gray_mean   (gray_mean = ITU-R BT.601 luma mean)
      saturation: gray + s_fac * (im - gray)             (per-pixel luma grayscale)
      hue:        shift H in HSV by h_fac * 180 degrees  (cv2 hue is 0..180)
    Ops applied in a random order.
    """
    b_fac = float(rng.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness))
    c_fac = float(rng.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast))
    s_fac = float(rng.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation))
    h_fac = float(rng.uniform(-hue, hue))
    fn_idx = rng.permutation(4)

    im = im_u8.astype(np.float32)
    for i in fn_idx:
        if i == 0:
            im = im * b_fac
        elif i == 1:
            gray_mean = float((im @ _LUMA).mean())
            im = (im - gray_mean) * c_fac + gray_mean
        elif i == 2:
            gray = (im @ _LUMA)[..., None]
            im = gray + s_fac * (im - gray)
        elif i == 3:
            hsv = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(np.int32) + int(round(h_fac * 180))) % 180
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
    return np.clip(im, 0, 255).astype(np.uint8)


def _jittered_crop(im, rng, crop_jitter_pixels):
    """Square crop with origin randomly offset by +/- crop_jitter_pixels from center."""
    h, w = im.shape[:2]
    base_size = min(h, w)
    crop_size = base_size - 2 * crop_jitter_pixels
    if crop_size <= 0:
        return center_crop(im, base_size)
    cx = (w - crop_size) // 2
    cy = (h - crop_size) // 2
    dx = int(rng.integers(-crop_jitter_pixels, crop_jitter_pixels + 1))
    dy = int(rng.integers(-crop_jitter_pixels, crop_jitter_pixels + 1))
    x = max(0, min(w - crop_size, cx + dx))
    y = max(0, min(h - crop_size, cy + dy))
    return im[y:y + crop_size, x:x + crop_size, :]


def process_image_augmented(im, im_size, rng,
                            use_color=False, use_crop=False,
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05,
                            crop_jitter_pixels=16):
    """process_image variant with optional color/crop jitter.

    With use_color=False and use_crop=False, the output is bit-identical to
    process_image(). Extraction relies on this to guarantee variant 0 is the
    clean baseline that validation reads.
    """
    im = np.array(im)
    if use_color:
        im_u8 = im if im.dtype == np.uint8 else np.clip(im, 0, 255).astype(np.uint8)
        im_u8 = _apply_color_jitter(im_u8, rng, brightness, contrast, saturation, hue)
        im = im_u8.astype(np.float32)
    else:
        im = im.astype(np.float32)
    if use_crop:
        im = _jittered_crop(im, rng, crop_jitter_pixels)
    else:
        im = center_crop(im, im.shape[0])
    im = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_LINEAR)
    im = normalize(im)
    return im


def process_image_no_normalize(im, im_size):
    """Processes an image for network input."""
    im = np.array(im).astype(np.float32)
    im = center_crop(im, im.shape[0])
    im = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_LINEAR)
    return im


def process_image_for_autosam(im, im_size):
    """Processes an image for network input."""
    im = np.array(im)
    im = center_crop(im, im.shape[0])
    im = np.array(resize(to_pil_image(im), (im_size, im_size)))
    return im


def color_norm_batch(ims, mean, std):
    """Performs per-channel normalization (batch version)."""
    t_mean = torch.tensor(mean, dtype=torch.float, device=ims.device).view(1, 3, 1, 1)
    t_std = torch.tensor(std, dtype=torch.float, device=ims.device).view(1, 3, 1, 1)
    return (ims - t_mean) / t_std


def center_crop_batch(ims, crop_size):
    """Performs center cropping (batch version)."""
    h, w = ims.shape[1:3]
    x = math.ceil((w - crop_size) / 2)
    y = math.ceil((h - crop_size) / 2)
    return ims[:, y:(y + crop_size), x:(x + crop_size), :]


def process_image_batch(ims, im_size):
    """Processes images for network input (betch version)."""
    ims = center_crop_batch(ims, ims.shape[1])
    ims = ims.permute(0, 3, 1, 2).float() / 255.0
    ims = F.interpolate(ims, size=(im_size, im_size), mode="bilinear")
    ims = color_norm_batch(ims, _IN_MEAN, _IN_STD)
    return ims


def compute_state(obs_t, default_pos_left_arm, default_pos_right_arm, use_touch=False):
    if 'left_arm_joint_pos' in obs_t and \
        'right_arm_joint_pos' in obs_t and \
        'left_fingers_joint_pos' in obs_t and \
        'right_fingers_joint_pos' in obs_t and \
        'left_fingers_touch' in obs_t and \
        'right_fingers_touch' in obs_t:

        left_arm_joint_pos = np.array(obs_t['left_arm_joint_pos']) - np.array(default_pos_left_arm)
        right_arm_joint_pos = np.array(obs_t['right_arm_joint_pos']) - np.array(default_pos_right_arm)
        left_fingers_joint_pos = np.array(obs_t['left_fingers_joint_pos'])
        right_fingers_joint_pos = np.array(obs_t['right_fingers_joint_pos'])
        left_fingers_touch = np.array(obs_t['left_fingers_touch']) / 4096
        right_fingers_touch = np.array(obs_t['right_fingers_touch']) / 4096

        if use_touch:
            state = np.concatenate([left_arm_joint_pos, right_arm_joint_pos,
                                    left_fingers_joint_pos, right_fingers_joint_pos,
                                    left_fingers_touch, right_fingers_touch])
        else:
            state = np.concatenate([left_arm_joint_pos, right_arm_joint_pos,
                                    left_fingers_joint_pos, right_fingers_joint_pos])
        return state

    else:
        return None


def compute_action(obs_t, obs_t1, default_pos_left_arm, default_pos_right_arm):
    # left_arm_joint_pos_t1 = np.array(obs_t1['left_arm_joint_pos']) - default_pos_left_arm
    # right_arm_joint_pos_t1 = np.array(obs_t1['right_arm_joint_pos']) - default_pos_right_arm
    # left_fingers_joint_pos_t1 = np.array(obs_t1['left_fingers_joint_pos'])
    # right_fingers_joint_pos_t1 = np.array(obs_t1['right_fingers_joint_pos'])
    # state_t1 = np.concatenate([left_arm_joint_pos_t1, right_arm_joint_pos_t1,
    #                            left_fingers_joint_pos_t1, right_fingers_joint_pos_t1])
    #
    # return state_t1


    if 'left_arm_cmd' in obs_t1 and \
        'right_arm_cmd' in obs_t1 and \
        'left_fingers_cmd' in obs_t1 and \
        'right_fingers_cmd' in obs_t1:

        # use commanded action instead of next-step state
        left_arm_action = np.array(obs_t1['left_arm_cmd']) - default_pos_left_arm
        right_arm_action = np.array(obs_t1['right_arm_cmd']) - default_pos_right_arm
        left_fingers_action = np.array(obs_t1['left_fingers_cmd'])
        right_fingers_action = np.array(obs_t1['right_fingers_cmd'])
        action = np.concatenate([left_arm_action, right_arm_action,
                                   left_fingers_action, right_fingers_action])

        return action

    else:
        return None


class Bimanual_Dataset(torch.utils.data.Dataset):
    """Dataset."""

    def __init__(
            self, features, demo_root, demo_dirs, inmem,
            start_ind=0, num_demos=1000000,
            im_size=224, cams=["hand"], num_steps=1, num_pred=1, look_ahead=0,
            noisy_skip=False, frame_skip=0, default_pos_left_arm=[0., 0., 0., 0., 0., 0.],
            default_pos_right_arm=[0., 0., 0., 0., 0., 0.],
            joint_noise_mean=0.0, joint_noise_std=0.0, joint_noise_std_scale=1.0,
            data_filter={}, history_repeating=0.0, img_sample_num=-1,
            use_all_features=False, action_data_ratio=None, use_touch=False, skip_failure=True,
    ):
        self._features = features
        self._demo_root = demo_root
        self._demo_dirs = demo_dirs
        self._inmem = inmem
        self._l_ind = start_ind
        self._r_ind = start_ind + num_demos
        self._im_size = im_size
        self._cams = cams
        self._cam_keys = [f"feat_{cam}" if self._features else f"rgb_{cam}" for cam in cams]
        self._num_steps = num_steps
        self._num_pred = num_pred
        self._look_ahead = look_ahead
        self._noisy_skip = noisy_skip
        self._frame_skip = frame_skip
        self._default_pos_left_arm = default_pos_left_arm
        self._default_pos_right_arm = default_pos_right_arm
        self._joint_noise_mean = np.array(joint_noise_mean)
        self._joint_noise_std = np.array(joint_noise_std)
        self._joint_noise_std_scale = joint_noise_std_scale
        self._data_filter = data_filter
        self._history_repeating = history_repeating
        self._img_sample_num = img_sample_num
        self._use_all_features = use_all_features
        self._action_data_ratio = action_data_ratio if action_data_ratio is not None else [1] * len(
            self._demo_dirs)  # how much data has action labels for each demo_dir
        self._use_touch = use_touch
        self._skip_failure = skip_failure
        self._feature_files = []
        self._all_prompts_text = []
        self._dataset = self._construct()
        self._feature_files = np.array(self._feature_files)

    def recalibrate_wrist_joint(self, demo_obs):
        '''
        Since the wrist joint (last value in left/right_arm_joint_pos) has infinite range,
        we recalibrate it back to [-2pi, 2pi].
        This is done for each demo separately. It makes sure the wrist joint pos in each demo is
        within the range and continuous.
        '''
        left_wrist_pos = [obs['left_arm_joint_pos'][-1] for obs in demo_obs]
        right_wrist_pos = [obs['right_arm_joint_pos'][-1] for obs in demo_obs]
        min_left_wrist_pos = min(left_wrist_pos)
        max_left_wrist_pos = max(left_wrist_pos)
        min_right_wrist_pos = min(right_wrist_pos)
        max_right_wrist_pos = max(right_wrist_pos)

        left_offset = 0
        if max_left_wrist_pos > 2 * math.pi:
            left_offset = max_left_wrist_pos // (2 * math.pi) * (2 * math.pi)
        elif min_left_wrist_pos < -2 * math.pi:
            left_offset = min_left_wrist_pos // (-2 * math.pi) * (-2 * math.pi)

        right_offset = 0
        if max_right_wrist_pos > 2 * math.pi:
            right_offset = max_right_wrist_pos // (2 * math.pi) * (2 * math.pi)
        elif min_right_wrist_pos < -2 * math.pi:
            right_offset = min_right_wrist_pos // (-2 * math.pi) * (-2 * math.pi)

        for obs in demo_obs:
            obs['left_arm_joint_pos'][-1] -= left_offset
            obs['right_arm_joint_pos'][-1] -= right_offset

        return demo_obs

    def _construct(self):
        print("Loading demos from: {}".format(self._demo_root))
        print("Loading demo dirs: {}".format(self._demo_dirs))
        print("Num demos per dir: {}".format(self._r_ind - self._l_ind))
        dataset = []
        demo_lens = []
        # Collect all demo paths
        demo_paths = []
        for demo_dir_name, action_ratio in sorted(zip(self._demo_dirs, self._action_data_ratio)):
            demo_dir_path = os.path.join(self._demo_root, demo_dir_name)
            demo_dir_demo_paths = []
            # Collect demos from demo dirs
            for i, demo_name in enumerate(sorted(os.listdir(demo_dir_path))):
                demo_path = os.path.join(demo_dir_path, demo_name)
                # Filter out demos that failed
                if self._skip_failure and not os.path.exists(os.path.join(demo_path, "success.txt")):
                    continue
                # if self._data_filter != {}:
                #     metadata = json.load(open(os.path.join(demo_path, "metadata.json"), "r"))
                #     if check_data_filter(metadata, self._data_filter):
                #         continue
                # # Filter out data that does not have enough cameras
                # cam_miss = False
                # obs_data = joblib.load(os.path.join(demo_path, '0000.pkl'))
                # for cam_key in self._cam_keys:
                #     if cam_key not in obs_data.keys():
                #         cam_miss = True
                # if cam_miss:
                #     continue

                demo_dir_demo_paths.append(demo_path)
            # Take the desired demo range
            demo_dir_demo_paths = demo_dir_demo_paths[self._l_ind:self._r_ind]
            demo_paths.extend(demo_dir_demo_paths)
        # Extract observations from demos
        for i, demo_path in enumerate(tqdm(demo_paths)):
            prompt = None
            prompt_text = None
            if os.path.exists(os.path.join(demo_path, "metadata.json")):
                metadata = json.load(open(os.path.join(demo_path, "metadata.json"), 'r'))
                if 'instruction_embedding' in metadata:
                    prompt = np.array(metadata['instruction_embedding'])
                    prompt_text = metadata['instruction'] if 'instruction' in metadata else metadata['task']
                if 'visible_cams' in metadata:
                    visible_cam_keys = [f"feat_{cam}" if self._features else f"rgb_{cam}" for cam in metadata['visible_cams']]
                else:
                    visible_cam_keys = self._cam_keys
            demo_obs = []
            for j, obs_file in enumerate(sorted(os.listdir(demo_path))):
                obs_path = os.path.join(demo_path, obs_file)
                # Skip success/fail indicator files
                if not obs_file.endswith(".pkl"):
                    continue
                with open(obs_path, "rb") as f:
                    obs_j = joblib.load(f)
                if not self._inmem:
                    for cam_key in self._cam_keys:
                        if cam_key in obs_j:
                            obs_j.pop(cam_key)
                obs_j["step_ind"] = j
                obs_j['prompt'] = prompt
                obs_j['prompt_text'] = prompt_text
                demo_obs.append(obs_j)
            # demo_obs = self.recalibrate_wrist_joint(demo_obs)
            self._feature_files.append(os.path.join(demo_path, "features.h5"))
            demo_lens.append(len(demo_obs))
            # Compute actions
            for k in range(0, len(demo_obs) - 1):
                # Noisy frame skip
                frame_skip = np.random.randint(self._frame_skip + 1) if self._noisy_skip else self._frame_skip
                t1 = min(k + frame_skip + 1, len(demo_obs) - 1)
                obs_t, obs_t1 = demo_obs[k], demo_obs[t1]
                state_t = compute_state(obs_t, self._default_pos_left_arm, self._default_pos_right_arm, self._use_touch)
                act_t = compute_action(obs_t, obs_t1, self._default_pos_left_arm, self._default_pos_right_arm)
                mod_mask = np.ones((84 + 24)) if self._use_touch else np.ones((24 + 24))
                prompt = np.array(obs_t["instruction_embedding"]) if "instruction_embedding" in obs_t else obs_t["prompt"]
                prompt_text = obs_t["instruction"] if "instruction" in obs_t else obs_t["task"] if "task" in obs_t else obs_t["prompt_text"]
                if prompt_text not in self._all_prompts_text:
                    self._all_prompts_text.append(prompt_text)
                if state_t is None:
                    state_t = np.zeros((84)) if self._use_touch else np.zeros((24))
                    mod_mask[:-24] = 0
                if act_t is None:
                    act_t = np.zeros((24))
                    mod_mask[-24:] = 0
                # inject noise in state
                element = {
                    "demo_ind": i,
                    "step_ind": obs_t["step_ind"],
                    "state": state_t,
                    "action": act_t,
                    "frame_skip": t1 - k,
                    "process_state": k != 0,
                    "mod_mask": mod_mask,
                    "prompt": prompt,
                    "prompt_text": prompt_text,
                    "padded": False,
                    "visible_cam_keys": visible_cam_keys,
                }
                keys = visible_cam_keys if self._inmem else []
                for key in keys:
                    element[key] = obs_t[key]
                if k == 0:
                    # prepad the dataset with stationary states
                    for _ in range(self._num_steps - self._num_pred + self._look_ahead):
                        e_pad = copy.deepcopy(element)
                        e_pad["action"] = e_pad["state"][:24].copy()
                        e_pad["frame_skip"] = 1
                        e_pad["process_state"] = True
                        e_pad["padded"] = True
                        e_pad["prompt"] = np.zeros_like(e_pad["prompt"]) if e_pad["prompt"] is not None else None
                        dataset.append(e_pad)
                dataset.append(element)
        print("Total num demos: {:,}".format(len(demo_lens)))
        print("Total num steps: {:,}".format(len(dataset)))
        print("Mean demo len: {:.3f}".format(np.mean(demo_lens)))
        return dataset

    def process_state(self, state):
        # TEMP: joint noise based on eval time scale
        noise = np.random.normal(self._joint_noise_mean, self._joint_noise_std * self._joint_noise_std_scale).astype(
            np.float32)
        state[:24] += noise
        return state

    def __getitem__(self, ind):
        # Retrieve dataset entries
        demo_ind = self._dataset[ind]["demo_ind"]
        entries = []
        j = ind
        for _ in range(self._num_steps + self._look_ahead):
            cur_entry = self._dataset[j]
            if cur_entry["demo_ind"] != demo_ind:
                break
            entries.append(cur_entry)
            j += cur_entry["frame_skip"]
            if j >= len(self):
                break
        pad_num = self._num_steps + self._look_ahead - len(entries)
        if pad_num > 0:
            entries = entries + [entries[-1] for _ in range(pad_num)]
        len_entries = len(entries)
        # Retrieve images/features
        visible_cam_keys = entries[0]['visible_cam_keys']
        if self._img_sample_num == -1:
            img_selected_ids = list(range(self._num_steps))
            entries_w_img = entries[:len_entries - self._look_ahead]
        else:
            img_selected_ids = random.sample(range(self._num_steps - self._num_pred), self._img_sample_num) + \
                               random.sample(range(self._num_steps - self._num_pred, self._num_steps), self._img_sample_num)
            img_selected_ids = sorted(img_selected_ids)
            entries_w_img = [entries[id] for id in img_selected_ids]
        if self._inmem:
            im_data_all = entries_w_img
        else:
            step_inds = [entry['step_ind'] for entry in entries_w_img]
            unique_step_inds = sorted(list(
                set(step_inds)))  # using unique inds because h5 file does not support batch indexing with repeating indices
            # with h5py.File(entries[0]["path"], 'r') as hf:
            with h5py.File(self._feature_files[demo_ind], 'r') as hf:
                features = hf['features'][
                    unique_step_inds].copy()  # T * 3 * C for mean feature / T * 3 * N * C for all features
            ind_to_feature = {ind: features[i] for i, ind in enumerate(unique_step_inds)}
            im_data_all = [{cam_key: ind_to_feature[ind][self._cam_keys.index(cam_key)]
                           for cam_key in visible_cam_keys}
                           for ind in step_inds]
        # process images
        ims = [[] for cam in self._cams]
        for im_data in im_data_all:
            for cam_id, cam_key in enumerate(self._cam_keys):
                if cam_key not in visible_cam_keys:
                    continue
                if self._features:
                    im = im_data[cam_key]
                else:
                    im = im_data[cam_key]
                    im = process_image(im, self._im_size)
                ims[cam_id].append(im)
        # for non-visible cams, use the faetures from a visible cam as a placeholder
        for cam_id, cam_key in enumerate(self._cam_keys):
            if cam_key not in visible_cam_keys:
                ims[cam_id] = copy.deepcopy(ims[self._cam_keys.index(visible_cam_keys[0])])
        # Retrieve states/actions
        state_noiseless = [
            entry["state"].astype(np.float32)[None, ...] for entry in entries[:len_entries - self._look_ahead]
        ]
        state = [
            self.process_state(entry["state"]).astype(np.float32)[None, ...] if entry["process_state"] else
            entry["state"].astype(np.float32)[None, ...] for entry in entries[:len_entries - self._look_ahead]
        ]
        action = [entry["action"].astype(np.float32)[None, ...] for entry in entries[self._look_ahead:]]
        if entries[0]["prompt"] is not None:
            prompts = [entry["prompt"].astype(np.float32)[None, ...] for entry in entries[:len_entries - self._look_ahead]]
            prompts_text = [entry["prompt_text"] for entry in entries[:len_entries - self._look_ahead]]
        else:
            prompts = [np.zeros((1,)) for entry in entries[:len_entries - self._look_ahead]]
            prompts_text = ["" for entry in entries[:len_entries - self._look_ahead]]
        mod_mask = [entry['mod_mask'].astype(np.float32)[None, ...] for entry in entries[:len_entries - self._look_ahead]]

        # Array shape: (T, data_dim)
        ims = [torch.tensor(np.stack(ims_c, axis=0)) for ims_c in ims]
        state = torch.Tensor(np.concatenate(state, axis=0))
        state_noiseless = torch.Tensor(np.concatenate(state_noiseless, axis=0))
        action = torch.Tensor(np.concatenate(action, axis=0))
        prompts = torch.Tensor(np.concatenate(prompts, axis=0))
        mod_mask = torch.Tensor(np.concatenate(mod_mask, axis=0))

        # using repeated history
        if self._history_repeating != 0 and random.random() < self._history_repeating:
            repeat_steps = random.randrange(1, self._num_steps)
            for i in range(len(ims)):
                ims[i][:-repeat_steps] = ims[i][repeat_steps:].clone()
                ims[i][-repeat_steps:] = ims[i][-repeat_steps - 1].clone()
            state[:-repeat_steps] = state[repeat_steps:].clone()
            state[-repeat_steps:] = state[-repeat_steps - 1].clone()
            state_noiseless[:-repeat_steps] = state_noiseless[repeat_steps:].clone()
            state_noiseless[-repeat_steps:] = state_noiseless[-repeat_steps - 1].clone()
            action[:-repeat_steps] = action[repeat_steps:].clone()
            action[-repeat_steps:] = action[-repeat_steps - 1].clone()
            action[-repeat_steps:, :7] = 0
            prompts[:-repeat_steps] = prompts[repeat_steps:].clone()
            prompts[-repeat_steps:] = prompts[-repeat_steps - 1].clone()
            prompts_text[:-repeat_steps] = prompts[repeat_steps:]
            prompts_text[-repeat_steps:] = [prompts_text[-repeat_steps - 1] for _ in range(repeat_steps)]

        pi_obs = state
        pi_obs_noiseless = state_noiseless
        pi_act = action

        att_mask = [float(entry["padded"]) for entry in entries[:len_entries - self._look_ahead]]
        att_mask = torch.Tensor(np.array(att_mask))  # (T,)

        img_selected_ids = torch.LongTensor(img_selected_ids)

        visible_cam_mask = torch.Tensor([1 if key in visible_cam_keys else 0 for key in self._cam_keys])


        return ims, pi_obs, pi_obs_noiseless, pi_act, prompts, prompts_text, visible_cam_mask, mod_mask, att_mask, img_selected_ids

    def __len__(self):
        return len(self._dataset)



def pose_to_xyz_quat(pose_4x4):
    """Convert (T, 4, 4) homogeneous transforms to (T, 7) xyz + quaternion (xyzw)."""
    xyz = pose_4x4[:, :3, 3]  # (T, 3)
    rot = R.from_matrix(pose_4x4[:, :3, :3])
    quat = rot.as_quat()  # (T, 4) in xyzw convention
    return np.concatenate([xyz, quat], axis=1).astype(np.float32)  # (T, 7)


def pose_to_xyz_rot6d(pose_4x4):
    """Convert (T, 4, 4) homogeneous transforms to (T, 9) xyz + 6D rotation.

    6D rotation = first two columns of rotation matrix, concatenated column-major:
    [r00, r10, r20, r01, r11, r21] where rij = R[i,j].
    """
    xyz = pose_4x4[:, :3, 3]                       # (T, 3)
    col0 = pose_4x4[:, :3, 0]                      # (T, 3) — first column
    col1 = pose_4x4[:, :3, 1]                      # (T, 3) — second column
    rot6d = np.concatenate([col0, col1], axis=1)    # (T, 6)
    return np.concatenate([xyz, rot6d], axis=1).astype(np.float32)  # (T, 9)


class BKL_Dataset(torch.utils.data.Dataset):
    """Dataset for BKL embodiment data.

    Each episode is stored as:
      - A single HDF5 file with state/action trajectories
      - Pre-extracted vision features in a separate features.h5
      - Camera names: left_wrist, right_wrist, head

    State: absolute EEF xyz+rot6d (9+9) + absolute hand joints (22+22) = 62 dims
    Action: body-frame delta EEF (xyz+6D rot) + absolute hand target positions = 62 dims
    """

    STATE_DIM = 62   # xyz(3)+rot6d(6) per arm + hand(22) per hand = 9+9+22+22
    ACTION_DIM = 62  # same structure, body-frame deltas

    # Side index ranges within the 62-dim state/action vectors.
    # Used by the `side` parameter to mask out unused dimensions.
    SIDE_INDICES = {
        'left':  list(range(0, 9)) + list(range(18, 40)),   # left arm (9) + left hand (22)
        'right': list(range(9, 18)) + list(range(40, 62)),   # right arm (9) + right hand (22)
    }
    # Wrist camera to exclude when training on a single side.
    SIDE_EXCLUDED_CAMS = {
        'left':  'right_wrist',
        'right': 'left_wrist',
    }

    def __init__(
            self, features, demo_root, demo_dirs, inmem,
            start_ind=0, num_demos=1000000,
            im_size=224, cams=["left_wrist", "right_wrist", "head"], num_steps=1, num_pred=1, look_ahead=0,
            noisy_skip=False, frame_skip=0,
            joint_noise_std=0.0, joint_noise_std_scale=1.0,
            history_repeating=0.0, img_sample_num=-1,
            prompt_text="pour the sugar", prompt_embedding=None, prompt_embedding_path=None,
            skip_failure=True,
            action_stats_path=None,
            noise_stats_path=None,
            side="both",
            augment_features=False,
            **kwargs,  # absorb unused args (e.g. joint_noise_mean from old configs)
    ):
        # Load prompt embedding from path if provided
        if prompt_embedding is None and prompt_embedding_path is not None:
            prompt_embedding = np.load(prompt_embedding_path).astype(np.float32)

        # Load action normalization stats (quantile scaling: map [q01, q99] → [-1, 1])
        if action_stats_path is not None and os.path.exists(action_stats_path):
            stats = np.load(action_stats_path)
            self._action_q01 = stats['q01'].astype(np.float32)
            self._action_q99 = stats['q99'].astype(np.float32)
            self._action_range = np.maximum(self._action_q99 - self._action_q01, 1e-6)
            print(f"Loaded action stats from {action_stats_path}")
        else:
            self._action_q01 = None
            self._action_q99 = None
            self._action_range = None

        # Load proprioceptive noise stats (56D: axis-angle rotation errors)
        # Always zero-mean — only std is used for noise injection
        if noise_stats_path is not None and os.path.exists(noise_stats_path):
            noise_stats = np.load(noise_stats_path)
            joint_noise_std = noise_stats['std'].astype(np.float32)
            print(f"Loaded noise stats from {noise_stats_path} (zero-mean, {len(joint_noise_std)}D)")

        self._features = features
        self._demo_root = demo_root
        self._demo_dirs = demo_dirs
        self._inmem = inmem
        self._l_ind = start_ind
        self._r_ind = start_ind + num_demos
        self._im_size = im_size
        self._cams = cams
        self._cam_keys = [f"feat_{cam}" for cam in cams]
        self._num_steps = num_steps
        self._num_pred = num_pred
        self._look_ahead = look_ahead
        self._noisy_skip = noisy_skip
        self._frame_skip = frame_skip
        # Noise is 56D: [left_xyz(3), left_axisangle(3), right_xyz(3), right_axisangle(3), left_hand(22), right_hand(22)]
        _NOISE_DIM = 56
        self._joint_noise_std = np.zeros(_NOISE_DIM, dtype=np.float32) if isinstance(joint_noise_std, (int, float)) and joint_noise_std == 0.0 else np.array(joint_noise_std, dtype=np.float32)
        self._joint_noise_std_scale = joint_noise_std_scale
        self._augment_features = augment_features
        self._history_repeating = history_repeating
        self._img_sample_num = img_sample_num
        self._prompt_text = prompt_text
        self._prompt_embedding = np.array(prompt_embedding) if prompt_embedding is not None else None
        self._skip_failure = skip_failure
        assert side in ("both", "left", "right"), f"side must be 'both', 'left', or 'right', got '{side}'"
        self._side = side
        self._feature_files = []
        self._num_feat_variants = []  # populated alongside _feature_files during _construct
        self._all_prompts_text = [prompt_text]
        self._dataset = self._construct()
        self._feature_files = np.array(self._feature_files)
        self._num_feat_variants = np.array(self._num_feat_variants, dtype=np.int32)

    def _load_episode_h5(self, h5_path):
        """Load state and raw action data from a single episode HDF5 file.

        Returns:
            states: (T, 62) — absolute EEF xyz+rot6d + absolute hand joints
            action_data: dict with raw arrays for computing actions with arbitrary frame_skip
        """
        with h5py.File(h5_path, 'r') as f:
            left_arm_pose = f['left_arm_current_pose'][:].astype(np.float64)   # (T, 4, 4)
            right_arm_pose = f['right_arm_current_pose'][:].astype(np.float64) # (T, 4, 4)
            left_hand_pos = f['left_hand_joint_positions'][:].astype(np.float32)  # (T, 22)
            right_hand_pos = f['right_hand_joint_positions'][:].astype(np.float32) # (T, 22)
            left_arm_target_pose = f['left_arm_target_pose'][:].astype(np.float64)   # (T, 4, 4)
            right_arm_target_pose = f['right_arm_target_pose'][:].astype(np.float64) # (T, 4, 4)
            left_hand_cmd = f['left_hand_target_joint_positions'][:].astype(np.float32)  # (T, 22)
            right_hand_cmd = f['right_hand_target_joint_positions'][:].astype(np.float32) # (T, 22)

        # State: absolute EEF xyz+rot6d + absolute hand joints
        left_eef = pose_to_xyz_rot6d(left_arm_pose)    # (T, 9)
        right_eef = pose_to_xyz_rot6d(right_arm_pose)  # (T, 9)
        states = np.concatenate([left_eef, right_eef, left_hand_pos, right_hand_pos], axis=1)  # (T, 62)

        action_data = {
            'left_arm_current_pose': left_arm_pose,
            'right_arm_current_pose': right_arm_pose,
            'left_arm_target_pose': left_arm_target_pose,
            'right_arm_target_pose': right_arm_target_pose,
            'left_hand_cmd': left_hand_cmd,
            'right_hand_cmd': right_hand_cmd,
        }

        return states, action_data

    def _compute_action(self, action_data, k, t1):
        """Compute action from timestep k targeting timestep t1.

        Arm: body-frame delta relative to current state pose (not target pose).
            delta_xyz = R_curr_state^T @ (t_target_future - t_curr_state)
            R_delta = R_curr_state^T @ R_target_future → first 2 columns (6D)
        Hand: absolute target joint positions at t1.

        Returns (62,): [left_xyz(3), left_rot6d(6), right_xyz(3), right_rot6d(6),
                        left_hand(22), right_hand(22)]
        """
        # Left arm: body-frame delta relative to current state
        left_state = action_data['left_arm_current_pose'][k]
        left_tgt = action_data['left_arm_target_pose'][t1]
        left_R_state = left_state[:3, :3]
        left_delta_xyz = (left_R_state.T @ (left_tgt[:3, 3] - left_state[:3, 3])).astype(np.float32)
        left_R_delta = left_R_state.T @ left_tgt[:3, :3]
        left_delta_rot6d = np.concatenate([left_R_delta[:, 0], left_R_delta[:, 1]]).astype(np.float32)

        # Right arm: body-frame delta relative to current state
        right_state = action_data['right_arm_current_pose'][k]
        right_tgt = action_data['right_arm_target_pose'][t1]
        right_R_state = right_state[:3, :3]
        right_delta_xyz = (right_R_state.T @ (right_tgt[:3, 3] - right_state[:3, 3])).astype(np.float32)
        right_R_delta = right_R_state.T @ right_tgt[:3, :3]
        right_delta_rot6d = np.concatenate([right_R_delta[:, 0], right_R_delta[:, 1]]).astype(np.float32)

        # Hands: absolute target joint positions
        left_hand_target = action_data['left_hand_cmd'][t1].astype(np.float32)
        right_hand_target = action_data['right_hand_cmd'][t1].astype(np.float32)
        return np.concatenate([left_delta_xyz, left_delta_rot6d,
                               right_delta_xyz, right_delta_rot6d,
                               left_hand_target, right_hand_target])  # (62,)

    def _construct(self):
        print("Loading BKL demos from: {}".format(self._demo_root))
        print("Loading demo dirs: {}".format(self._demo_dirs))
        dataset = []
        demo_lens = []

        # Collect episode paths
        episode_paths = []
        for demo_dir_name in sorted(self._demo_dirs):
            demo_dir_path = os.path.join(self._demo_root, demo_dir_name)
            # Check for success/failure subdirectory structure
            if os.path.isdir(os.path.join(demo_dir_path, "success")):
                search_dir = os.path.join(demo_dir_path, "success")
            else:
                search_dir = demo_dir_path
            for ep_name in sorted(os.listdir(search_dir)):
                ep_path = os.path.join(search_dir, ep_name)
                if not os.path.isdir(ep_path):
                    continue
                episode_paths.append(ep_path)
        # Take the desired range
        episode_paths = episode_paths[self._l_ind:self._r_ind]

        prompt = self._prompt_embedding
        prompt_text = self._prompt_text

        for i, ep_path in enumerate(tqdm(episode_paths)):
            ep_name = os.path.basename(ep_path)
            h5_path = os.path.join(ep_path, f"{ep_name}.h5")
            if not os.path.exists(h5_path):
                print(f"Warning: skipping {ep_path}, no h5 file found")
                continue

            states, action_data = self._load_episode_h5(h5_path)
            T = len(states)
            demo_lens.append(T)

            # Feature file (pre-extracted vision features)
            feature_file = os.path.join(ep_path, "features.h5")
            self._feature_files.append(feature_file)
            # Probe num_variants from the HDF5 attr. Missing attr == legacy
            # pre-variant-axis file; fail loudly rather than silently mis-read.
            if os.path.exists(feature_file):
                with h5py.File(feature_file, 'r') as hf:
                    if 'num_variants' not in hf['features'].attrs:
                        raise RuntimeError(
                            f"{feature_file} is missing the 'num_variants' HDF5 attr. "
                            f"This is a legacy pre-variant-axis file. Re-run "
                            f"tools/extract_bkl_features.py to regenerate."
                        )
                    nv = int(hf['features'].attrs['num_variants'])
            else:
                nv = 1  # file not yet extracted — dataset can still enumerate
            self._num_feat_variants.append(nv)

            visible_cam_keys = self._cam_keys
            if self._side in self.SIDE_EXCLUDED_CAMS:
                excluded = f"feat_{self.SIDE_EXCLUDED_CAMS[self._side]}"
                visible_cam_keys = [k for k in visible_cam_keys if k != excluded]

            for k in range(T - 1):
                frame_skip = np.random.randint(self._frame_skip + 1) if self._noisy_skip else self._frame_skip
                t1 = min(k + frame_skip + 1, T - 1)

                state_t = states[k]
                act_t = self._compute_action(action_data, k, t1)

                # Normalize action: map [q01, q99] → [-1, 1], clip outliers
                if self._action_q01 is not None:
                    act_t = (act_t - self._action_q01) / self._action_range * 2 - 1
                    act_t = np.clip(act_t, -1, 1)

                mod_mask = np.ones(self.STATE_DIM + self.ACTION_DIM, dtype=np.float32)
                if self._side != "both":
                    # Zero out the unused side in both state and action portions
                    other = "left" if self._side == "right" else "right"
                    for idx in self.SIDE_INDICES[other]:
                        mod_mask[idx] = 0.0                       # state portion
                        mod_mask[self.STATE_DIM + idx] = 0.0      # action portion

                element = {
                    "demo_ind": i,
                    "step_ind": k,
                    "state": state_t,
                    "action": act_t,
                    "frame_skip": t1 - k,
                    "process_state": k != 0,
                    "mod_mask": mod_mask,
                    "prompt": prompt,
                    "prompt_text": prompt_text,
                    "padded": False,
                    "visible_cam_keys": visible_cam_keys,
                }
                if k == 0:
                    # Prepad: stationary action = no arm movement + current hand position
                    stationary_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
                    stationary_action[3:6] = [1, 0, 0]   # left rot col0 (identity)
                    stationary_action[6:9] = [0, 1, 0]   # left rot col1 (identity)
                    stationary_action[12:15] = [1, 0, 0]  # right rot col0 (identity)
                    stationary_action[15:18] = [0, 1, 0]  # right rot col1 (identity)
                    stationary_action[18:40] = action_data['left_hand_cmd'][0]   # initial hand pos
                    stationary_action[40:62] = action_data['right_hand_cmd'][0]  # initial hand pos
                    if self._action_q01 is not None:
                        zero_action = np.clip((stationary_action - self._action_q01) / self._action_range * 2 - 1, -1, 1)
                    else:
                        zero_action = stationary_action
                    for _ in range(self._num_steps - self._num_pred + self._look_ahead):
                        e_pad = copy.deepcopy(element)
                        e_pad["action"] = zero_action.copy()
                        e_pad["frame_skip"] = 1
                        e_pad["process_state"] = True
                        e_pad["padded"] = True
                        e_pad["prompt"] = np.zeros_like(e_pad["prompt"]) if e_pad["prompt"] is not None else None
                        dataset.append(e_pad)
                dataset.append(element)

        print("Total num demos: {:,}".format(len(demo_lens)))
        print("Total num steps: {:,}".format(len(dataset)))
        print("Mean demo len: {:.3f}".format(np.mean(demo_lens)))
        return dataset

    def process_state(self, state):
        """Add noise to proprioceptive state (62D).

        Noise stats are 56D: [left_xyz(3), left_axisangle(3), right_xyz(3),
        right_axisangle(3), left_hand(22), right_hand(22)].

        - Translation/hand joints: uniform noise in [-std*scale, +std*scale]
        - Rotation: axis-angle perturbation → rotation matrix → applied to 6D cols
        """
        std = self._joint_noise_std * self._joint_noise_std_scale
        if std.sum() == 0:
            return state
        state = state.copy()

        # Left arm: xyz uniform noise
        state[0:3] += np.random.uniform(-std[0:3], std[0:3]).astype(np.float32)
        # Left arm: rotation noise via axis-angle perturbation
        left_aa = np.random.uniform(-std[3:6], std[3:6])
        if np.linalg.norm(left_aa) > 1e-8:
            R_noise = R.from_rotvec(left_aa).as_matrix()
            col0, col1 = state[3:6].copy(), state[6:9].copy()
            R_orig = np.column_stack([col0, col1, np.cross(col0, col1)])
            R_pert = (R_noise @ R_orig).astype(np.float32)
            state[3:6] = R_pert[:, 0]
            state[6:9] = R_pert[:, 1]

        # Right arm: xyz uniform noise
        state[9:12] += np.random.uniform(-std[6:9], std[6:9]).astype(np.float32)
        # Right arm: rotation noise via axis-angle perturbation
        right_aa = np.random.uniform(-std[9:12], std[9:12])
        if np.linalg.norm(right_aa) > 1e-8:
            R_noise = R.from_rotvec(right_aa).as_matrix()
            col0, col1 = state[12:15].copy(), state[15:18].copy()
            R_orig = np.column_stack([col0, col1, np.cross(col0, col1)])
            R_pert = (R_noise @ R_orig).astype(np.float32)
            state[12:15] = R_pert[:, 0]
            state[15:18] = R_pert[:, 1]

        # Hand joints: uniform noise
        state[18:40] += np.random.uniform(-std[12:34], std[12:34]).astype(np.float32)
        state[40:62] += np.random.uniform(-std[34:56], std[34:56]).astype(np.float32)

        return state

    def __getitem__(self, ind):
        # Retrieve dataset entries
        demo_ind = self._dataset[ind]["demo_ind"]
        entries = []
        j = ind
        for _ in range(self._num_steps + self._look_ahead):
            cur_entry = self._dataset[j]
            if cur_entry["demo_ind"] != demo_ind:
                break
            entries.append(cur_entry)
            j += cur_entry["frame_skip"]
            if j >= len(self):
                break
        pad_num = self._num_steps + self._look_ahead - len(entries)
        if pad_num > 0:
            entries = entries + [entries[-1] for _ in range(pad_num)]
        len_entries = len(entries)

        # Retrieve images/features
        visible_cam_keys = entries[0]['visible_cam_keys']
        if self._img_sample_num == -1:
            img_selected_ids = list(range(self._num_steps))
            entries_w_img = entries[:len_entries - self._look_ahead]
        else:
            img_selected_ids = random.sample(range(self._num_steps - self._num_pred), self._img_sample_num) + \
                               random.sample(range(self._num_steps - self._num_pred, self._num_steps), self._img_sample_num)
            img_selected_ids = sorted(img_selected_ids)
            entries_w_img = [entries[id] for id in img_selected_ids]

        step_inds = [entry['step_ind'] for entry in entries_w_img]
        unique_step_inds = sorted(list(set(step_inds)))
        feature_file = self._feature_files[demo_ind]
        num_variants = int(self._num_feat_variants[demo_ind])
        if os.path.exists(feature_file):
            with h5py.File(feature_file, 'r') as hf:
                dset = hf['features']  # always (K, T, cams, 197, C), K >= 1
                if not self._augment_features:
                    # Validation (or training with aug disabled): always variant 0.
                    features = dset[0, unique_step_inds].copy()
                else:
                    # Training: pick an independent variant per camera so that each
                    # __getitem__ sees K^num_cams possible combinations. With K==1
                    # this collapses to variant 0 for every cam — no special case.
                    k_per_cam = [random.randrange(num_variants) for _ in self._cam_keys]
                    per_cam = [dset[k, unique_step_inds, c:c+1].copy()
                               for c, k in enumerate(k_per_cam)]
                    features = np.concatenate(per_cam, axis=1)
            ind_to_feature = {ind: features[i] for i, ind in enumerate(unique_step_inds)}
            im_data_all = [{cam_key: ind_to_feature[ind][self._cam_keys.index(cam_key)]
                           for cam_key in visible_cam_keys}
                           for ind in step_inds]
        else:
            # No features yet — return zeros (for testing data pipeline without vision).
            # Shape matches one (variant, cam, frame)'s all-patches feature: (197, 768).
            im_data_all = [{cam_key: np.zeros((197, 768), dtype=np.float32) for cam_key in visible_cam_keys}
                           for _ in step_inds]

        # Process images/features
        ims = [[] for _ in self._cams]
        for im_data in im_data_all:
            for cam_id, cam_key in enumerate(self._cam_keys):
                if cam_key not in visible_cam_keys:
                    continue
                ims[cam_id].append(im_data[cam_key])
        for cam_id, cam_key in enumerate(self._cam_keys):
            if cam_key not in visible_cam_keys:
                ims[cam_id] = copy.deepcopy(ims[self._cam_keys.index(visible_cam_keys[0])])

        # Retrieve states/actions
        state_noiseless = [
            entry["state"].astype(np.float32)[None, ...] for entry in entries[:len_entries - self._look_ahead]
        ]
        state = [
            self.process_state(entry["state"]).astype(np.float32)[None, ...] if entry["process_state"] else
            entry["state"].astype(np.float32)[None, ...] for entry in entries[:len_entries - self._look_ahead]
        ]
        action = [entry["action"].astype(np.float32)[None, ...] for entry in entries[self._look_ahead:]]
        if entries[0]["prompt"] is not None:
            prompts = [entry["prompt"].astype(np.float32)[None, ...] for entry in entries[:len_entries - self._look_ahead]]
            prompts_text = [entry["prompt_text"] for entry in entries[:len_entries - self._look_ahead]]
        else:
            prompts = [np.zeros((1,), dtype=np.float32) for entry in entries[:len_entries - self._look_ahead]]
            prompts_text = ["" for entry in entries[:len_entries - self._look_ahead]]
        mod_mask = [entry['mod_mask'].astype(np.float32)[None, ...] for entry in entries[:len_entries - self._look_ahead]]

        # Stack to tensors
        ims = [torch.tensor(np.stack(ims_c, axis=0)) for ims_c in ims]
        state = torch.Tensor(np.concatenate(state, axis=0))
        state_noiseless = torch.Tensor(np.concatenate(state_noiseless, axis=0))
        action = torch.Tensor(np.concatenate(action, axis=0))
        prompts = torch.Tensor(np.concatenate(prompts, axis=0))
        mod_mask = torch.Tensor(np.concatenate(mod_mask, axis=0))

        pi_obs = state
        pi_obs_noiseless = state_noiseless
        pi_act = action

        att_mask = [float(entry["padded"]) for entry in entries[:len_entries - self._look_ahead]]
        att_mask = torch.Tensor(np.array(att_mask))

        img_selected_ids = torch.LongTensor(img_selected_ids)
        visible_cam_mask = torch.Tensor([1 if key in visible_cam_keys else 0 for key in self._cam_keys])

        return ims, pi_obs, pi_obs_noiseless, pi_act, prompts, prompts_text, visible_cam_mask, mod_mask, att_mask, img_selected_ids

    def __len__(self):
        return len(self._dataset)


class Bimanual_Dataset_NoImage(Bimanual_Dataset):
    """
    Dataset class for training tokenizers only. Only return states and actions.
    """

    def __getitem__(self, ind):
        # Retrieve dataset entries
        demo_ind = self._dataset[ind]["demo_ind"]
        entries = []
        j = ind
        for _ in range(self._num_steps):
            cur_entry = self._dataset[j]
            if cur_entry["demo_ind"] != demo_ind:
                break
            entries.append(cur_entry)
            j += cur_entry["frame_skip"]
            if j >= len(self):
                break
        pad_num = self._num_steps - len(entries)
        if pad_num > 0:
            entries = entries + [entries[-1] for _ in range(self._num_steps - len(entries))]

        # Retrieve stattes/actions
        state = [
            self.process_state(entry["state"]).astype(np.float32)[None, ...] if entry["process_state"] else
            entry["state"].astype(np.float32)[None, ...] for entry in entries
        ]
        # state = [entry["state"].astype(np.float32)[None, ...] for entry in entries]
        action = [entry["action"].astype(np.float32)[None, ...] for entry in entries]

        # Array shape: (T, data_dim)
        state = torch.Tensor(np.concatenate(state, axis=0))
        action = torch.Tensor(np.concatenate(action, axis=0))

        # using repeated history
        if self._history_repeating != 0 and random.random() < self._history_repeating:
            repeat_steps = random.randrange(1, self._num_steps)
            state[:-repeat_steps] = state[repeat_steps:].clone()
            state[-repeat_steps:] = state[-repeat_steps - 1].clone()
            action[:-repeat_steps] = action[repeat_steps:].clone()
            action[-repeat_steps:] = action[-repeat_steps - 1].clone()

        pi_obs = state
        pi_act = action

        return pi_obs, pi_act