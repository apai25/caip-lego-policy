import os
import time
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import redis

from copy import deepcopy
from loguru import logger
from pynput.keyboard import Key, KeyCode, Listener
from threading import Thread

from bimanual.control.robot import RobotController
from bimanual.env.sim import DualUR3ESimEnv, TermStatus
from bimanual.util.datastream import WebcamSensor, AbilityHandDS, UR3EArmDS, DataSync
from bimanual.util.traj import TrajectoryWriter


def get_cam_ids(serial_nums):
    cam_ids = []
    for sn in serial_nums:
        v4l2_path = f"/dev/v4l/by-id/usb-046d_Logitech_BRIO_{sn}-video-index0"
        vid_path = os.readlink(v4l2_path)
        cam_id = int(os.path.basename(vid_path).split("video")[-1])
        cam_ids.append(cam_id)
    return cam_ids


class DualUR3ERealEnv:
    def __init__(
        self,
        left_arm_ip="10.42.0.100",
        right_arm_ip="10.42.1.100",
        left_hand_port="/dev/ttyACM0",
        right_hand_port="/dev/ttyACM1",
        cam_left_sn="EF834C70",
        cam_head_sn="7F90B826",
        cam_right_sn="2C7356D1",
        cam_fps=60,
        cam_res=(640, 480),
        window=False,
        show_cams=False,
        save_cams=False,
        pedals=False,
        save_dir=None,
        save_freq=30.0,
    ):
        self.sim = DualUR3ESimEnv(window=window)
        self.robot = RobotController(
            left_arm_ip=left_arm_ip,
            right_arm_ip=right_arm_ip,
            left_hand_port=left_hand_port,
            right_hand_port=right_hand_port,
        )
  
        self.deadman_pedal = False
        if pedals:
            self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
#             self.deadman_pedal = False
            self.success_pedal = False
            self.failure_pedal = False
            self.quit = False
        else:
            self.listener = None

        self.show_cams = show_cams
        self.save_cams = save_cams
        self.save_cams_freq = 30
        assert not save_cams or save_dir, "Must specify a directory to save the camera images!"
        if self.show_cams:
            self.show_cams_active = False
            self.show_cams_thread = Thread(target=self.vis_cams)
            self.show_cams_thread.setDaemon(True)
            self.display_im = None
            self.markers = []
        
        self.save_dir = save_dir
        if self.save_dir:
            self.writer_active = False
            self.save_freq = save_freq
            self.latest_action = None
            self.latest_obs = None
            self.save_traj_result = None
            self.traj_writer = TrajectoryWriter(self.save_dir)
            self.writer_thread = Thread(target=self.record_action_obs)
            self.writer_thread.setDaemon(True)

        cam_names = ["left", "head", "right"]
        cam_ids = get_cam_ids([cam_left_sn, cam_head_sn, cam_right_sn])

        self.datastreams = [
            WebcamSensor(cam_id, cam_names[i], cam_res, cam_fps)
            for i, cam_id in enumerate(cam_ids)
        ] + [
            AbilityHandDS("left"),
            AbilityHandDS("right"),
            UR3EArmDS("left"),
            UR3EArmDS("right"),
        ]
        self.datasync = DataSync(self.datastreams, frequency=1/30)
        self.r = redis.Redis(host="localhost", port=6379, db=0)
        self.r.flushall()

    # ====== context manager =======
    def __enter__(self):
        self.start(wait=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ====== launch methods ======
    def start(self, wait=True):
        if self.listener:
            self.listener.start()
        self.robot.start(wait=True)
        for ds in self.datastreams:
            ds.start()
        self.datasync.start()
        if self.show_cams:
            self.show_cams_active = True
            self.show_cams_thread.start()
        if self.save_dir:
            self.writer_active = True
            self.writer_thread.start()

    def stop(self):
        logger.info(f"[DualUR3ERealEnv] Stopping real env")
        if self.show_cams:
            self.show_cams_active = False
        self.robot.stop()
        self.datasync.stop()
        for ds in self.datastreams:
            ds.end_read()
        self.sim.stop()
        if self.listener:
            self.listener.stop()

        logger.info("[DualUR3ERealEnv] Stopped real env")

    # ====== init methods ======
    def init(self, random=True, qpos={}):
        logger.info(f"[DualUR3ERealEnv] Initializing robot {random=}")
        self.r.flushall()
        self.latest_action = None
        self.latest_obs = None
        self.datasync.warn_skip = False
        state = self.robot.init(random=random, qpos=qpos)
        self.sim.init_state(state)
        self.sim.update_pedal_states()
        self.datasync.reset()

    # ====== camera methods ======
    def overlaps_square(self, x, y):
        overlaps = False
        for markerx, markery, _ in self.markers:
            dist = np.sqrt((markerx - x) ** 2 + (markery - y) ** 2)
            logger.info(f'{dist}')
            if dist < 50:
                overlaps = True
                break
        return overlaps
            
    def add_cube_marker(self, color='r', task='stack'):
        if not self.show_cams:
            return

        while True:
            if task == 'stack':
                x,y = np.random.uniform(730, 900), np.random.uniform(210, 320)
            elif task == 'sort':
                x,y = np.random.uniform(680, 800), np.random.uniform(220, 330)
            else:
                raise Exception(f'Task {task} not supported')
            size = 25
            angle = np.random.uniform(0, 360)
            square = patches.Rectangle((-size/2, -size/2), size*2, size*2, linewidth=2, edgecolor=color, facecolor='none')
            t = transforms.Affine2D().rotate_deg_around(0, 0, angle) + transforms.Affine2D().translate(x, y) + self.ax.transData
            square.set_transform(t)
            if not self.overlaps_square(x, y):
                break
        self.ax.add_patch(square)
        self.markers.append([x, y, square])
    
    def clear_markers(self):
        if not self.show_cams:
            return
        for _, _, marker in self.markers:
            marker.remove()
        self.markers = []

    def vis_cams(self):
        logger.info(f"[DualUR3ERealEnv] Starting cams visualizer")
        fig = plt.figure(frameon=False, figsize=(15, 5))
        self.ax = plt.Axes(fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        fig.add_axes(self.ax)
        iter_num = 0
        while self.show_cams_active:
            data = self.datasync.get_last()
            if data is None:
                time.sleep(1 / 30)
                continue
            (
                rgb_left,
                rgb_head,
                rgb_right,
                left_finger_joints,
                right_finger_joints,
                left_arm_joints,
                right_arm_joints,
            ) = data
            rgb_left = np.rot90(rgb_left[1], 2)[:, 80:-80, :]
            rgb_right = np.rot90(rgb_right[1], 2)[:, 80:-80, :]
            uncropped_rgb_head = rgb_head[1]
            rgb_head = rgb_head[1][:, 80:-80, :]
            im = np.concatenate([rgb_left, rgb_head, rgb_right], axis=1)
            if self.display_im is None:
                self.display_im = self.ax.imshow(im, aspect='auto')
            else:
                self.display_im.set_data(im)
            
            # saving cam images
            if self.save_cams and iter_num % int(1 / self.save_cams_freq * 30) == 0:
                img = Image.fromarray(uncropped_rgb_head)
                img.save(os.path.join(self.save_dir, f"frame_{iter_num // int(1 / self.save_cams_freq * 30):04d}.png"))
            iter_num += 1
            
            plt.pause(0.001)
            plt.draw()
            time.sleep(1 / 100)
        plt.close()
        logger.info(f"[DualUR3ERealEnv] Closing cams visualizer")
    
    # ====== traj writer methods ======
    def record_action_obs(self):
        while self.writer_active:
            if self.save_traj_result is not None:
                self.datasync.warn_skip = False
                self.traj_writer.save(success=self.save_traj_result)
                self.datasync.warn_skip = True
                self.save_traj_result = None
            if not self.is_active() or self.latest_action is None or self.latest_obs is None:
                time.sleep(1/60)
                continue
            self.traj_writer.add_action_obs(self.latest_action, self.latest_obs)
            time.sleep(1/self.save_freq)
    
    def save_traj(self, result):
        self.save_traj_result = result
        # Wait for saving to finish
        while self.save_traj_result is not None:
            time.sleep(1/30)

    # ====== step methods ======
    def step(self, action):
        self.latest_action = action
        self.robot.update_targets(action)
        self.sim.step(action)
        return self.get_obs()

    def get_term_status(self):
        if self.quit:
            return TermStatus.QUIT
        elif self.failure_pedal:
            return TermStatus.FAILURE
        elif self.success_pedal:
            return TermStatus.SUCCESS
        return TermStatus.ACTIVE

    def is_active(self):
        return self.deadman_pedal

    def get_obs(self):
        data = self.datasync.get_last()
        (
            rgb_left,
            rgb_head,
            rgb_right,
            left_finger_data,
            right_finger_data,
            left_arm_data,
            right_arm_data,
        ) = data

        rgb_left_t, rgb_left = rgb_left
        rgb_head_t, rgb_head = rgb_head
        rgb_right_t, rgb_right = rgb_right
        left_finger_joints_t, (left_finger_joints, left_finger_touch) = left_finger_data
        right_finger_joints_t, (right_finger_joints, right_finger_touch) = right_finger_data

        left_arm_joints_t, (left_qpos, left_qvel, left_tcppos, left_tcpvel, left_tcpforce) = left_arm_data
        right_arm_joints_t, (right_qpos, right_qvel, right_tcppos, right_tcpvel, right_tcpforce) = right_arm_data

        ts = [
            rgb_left_t,
            rgb_head_t,
            rgb_right_t,
            left_finger_joints_t,
            right_finger_joints_t,
            left_arm_joints_t,
            right_arm_joints_t,
        ]

        # Rotate left and right wrist images by 180 degrees
        # since the cameras are mounted upside down.
        rgb_left = np.rot90(rgb_left, 2)
        rgb_right = np.rot90(rgb_right, 2)
        obs = {
            "time_stamps": ts,
            "rgb_left": rgb_left,
            "rgb_head": rgb_head,
            "rgb_right": rgb_right,
            "left_fingers_joint_pos": left_finger_joints,
            "left_fingers_touch": left_finger_touch,
            "right_fingers_joint_pos": right_finger_joints,
            "right_fingers_touch": right_finger_touch,
            "left_arm_joint_pos": left_qpos,
            "left_arm_joint_vel": left_qvel,
            "left_arm_tcp_pose": left_tcppos,
            "left_arm_tcp_vel": left_tcpvel,
            "left_arm_tcp_force": left_tcpforce,
            "right_arm_joint_pos": right_qpos,
            "right_arm_joint_vel": right_qvel,
            "right_arm_tcp_pose": right_tcppos,
            "right_arm_tcp_vel": right_tcpvel,
            "right_arm_tcp_force": right_tcpforce,
        }

        # Verify all obs values are non-empty
        for key in obs:
            assert len(obs[key]) > 0, f"Received empty obs for {key}"

        # Enable datasync warnings
        self.datasync.warn_skip = True

        self.latest_obs = obs

        return obs

    def warn_skip(self, skip):
        self.datasync.warn_skip = skip

    # ======= keyboard presses ========
    def on_press(self, key):
        if key == Key.alt_l:
            self.deadman_pedal = True
        elif key == KeyCode.from_char("c"):
            self.success_pedal = True
        elif key == KeyCode.from_char("q"):
            self.failure_pedal = True
        elif key == Key.esc:
            self.quit = True

    def on_release(self, key):
        if key == Key.alt_l:
            self.deadman_pedal = False
        elif key == KeyCode.from_char("c"):
            self.success_pedal = False
        elif key == KeyCode.from_char("q"):
            self.failure_pedal = False
