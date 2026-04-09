import json
import numpy as np
import os
import pickle
import redis
import subprocess
import threading
import time

from loguru import logger
from ppadb.client import Client as AdbClient
from scipy.spatial.transform import Rotation as R

from bimanual.util.ik import FingerIKSolver, ArmIKSolver
from bimanual.util.rot_utils import scipy_to_quat, quat_to_scipy


class VRTarget:
    def __init__(self, sim=None):
        # (x,y,z) for root, thumb, index, middle, ring, pinky
        self.left_pos = np.zeros((6, 3))
        self.right_pos = np.zeros((6, 3))

        self.left_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.right_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # only used for vr controller index trigger
        self.left_fingers = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -50.0])
        self.right_fingers = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -50.0])

        self.hand_tracking = True

        self.right_index_trigger = 0.0
        self.right_hand_trigger = 0.0
        self.left_index_trigger = 0.0
        self.left_hand_trigger = 0.0

        self.right_origin_pos = None
        self.right_origin_quat = None
        self.right_vr_origin_pos = None
        self.right_vr_origin_quat = None

        self.left_origin_pos = None
        self.left_origin_quat = None
        self.left_vr_origin_pos = None
        self.left_vr_origin_quat = None

        self.left_active = False
        self.right_active = False
        self.right_enabled = False
        self.left_enabled = False

        self.offset = np.zeros(3)

        self.sim = sim

    def is_active(self, side):
        if side == "left":
            return self.left_active
        else:
            return self.right_active

    def get_hands(self):
        wl, xl, yl, zl = self.left_quat
        wr, xr, yr, zr = self.right_quat
        return {
            "left": (self.left_pos + self.offset, R.from_quat([xl, yl, zl, wl])),
            "right": (self.right_pos + self.offset, R.from_quat([xr, yr, zr, wr])),
        }

    def set_fingers(self, index_trigger_left, index_trigger_right):
        high = np.array([70.0, 70.0, 70.0, 70.0, 40.0, -100.0])
        low = np.array([20.0, 20.0, 20.0, 20.0, 20.0, -100.0])

        self.left_fingers[:] = low + (high - low) * index_trigger_left
        self.right_fingers[:] = low + (high - low) * index_trigger_right

    def recenter(self):
        desired_center = np.array([-0.3, 0, 1.2])
        cur_center = (self.left_pos[0] + self.right_pos[0]) / 2.0
        self.offset = desired_center - cur_center


class Quest3Reader:
    def __init__(self, device, sim, freq=300):
        self.thread = None

        # Check if quest3 is properly connected
        dev = subprocess.check_output('adb devices'.split()).decode('utf-8').split('\n')[1]
        if len(dev) == 0 or 'no permissions' in dev:
            raise Exception("Quest3 is not properly connected!")

        self.client = AdbClient(host="127.0.0.1", port=5037)
        self.device = self.client.device(device)
        self.freq = freq
        self.target = VRTarget()
        self.running = False
        self.sim = sim
        self.rlabels = [
            "rightRoot",
            "rightThumb",
            "rightIndex",
            "rightMiddle",
            "rightRing",
            "rightPinky",
            "rightRootQuat",
        ]
        self.llabels = [
            "leftRoot",
            "leftThumb",
            "leftIndex",
            "leftMiddle",
            "leftRing",
            "leftPinky",
            "leftRootQuat",
        ]

        self.right_offset = np.zeros(3)
        self.left_offset = np.zeros(3)
        self.right_adj = np.zeros(3)
        self.left_adj = np.zeros(3)
        self.right_collision = False
        self.left_collision = False

        self.offset = np.array([-0.5, 0, 1.5])
        self.start_time = time.time()
        self.updates = 0
        self.pedal = False

        self.r = redis.Redis(host="localhost", port=6379, db=0)

    def run(self):
        self.running = True
        self.thread = threading.Thread(target=self.connect)
        self.thread.start()

    def get_target(self):
        return self.target

    def connect(self):
        self.device.shell("logcat", handler=self.process_logcat)

    def process_logcat(self, connection):
        # Flush old Quest data from logcat
        for _ in range(200):
            data = connection.read(4096*10)
            if not data:
                break
        
        logger.info("[Quest3Reader] Flushed old quest3 data. Starting quest3 reader.")
        while self.running:
            data = connection.read(1024 * 4)
            if not data:
                break
            data = data.decode("utf-8")
            j = None
            if "Unity" in data:
                data = data.split(" ")
                for d in data:
                    if "rightRoot" in d:
                        j = d.split("\n")[0]
                        break
                if j:
                    try:
                        j = json.loads(j)
                        for k in j:
                            if isinstance(j[k], str) and len(j[k]) > 0:
                                j[k] = json.loads(j[k])
                    except:
                        pass

                    if isinstance(j, dict):
                        self.update_target(j)
            time.sleep(1 / 200)

    def update_target(self, j):
        assert self.sim is not None

        if not j["tracked"]:
            return

        # ==== hand tracking ====
        if j["handTracking"]:
            self.target.hand_tracking = True
            rotZ = R.from_euler("z", 180, degrees=True)

            right_root = np.array(
                [
                    -j[self.rlabels[0]]["z"],
                    j[self.rlabels[0]]["x"],
                    j[self.rlabels[0]]["y"],
                ]
            )
            left_root = np.array(
                [
                    -j[self.llabels[0]]["z"],
                    j[self.llabels[0]]["x"],
                    j[self.llabels[0]]["y"],
                ]
            )

            right_tracked = (
                self.pedal and (right_root != np.array([0.0, 0.0, 0.0])).all()
            )
            left_tracked = self.pedal and (left_root != np.array([0.0, 0.0, 0.0])).all()
            
            rq = j[self.rlabels[-1]]
            lq = j[self.llabels[-1]]

            of = R.from_euler("ZYX", [90, 180, -90], degrees=True)
            of1 = R.from_euler("ZYX", [90, 180, -90], degrees=True)
            of2 = R.from_euler("YZX", [90, 90, 0], degrees=True)
            of3 = R.from_euler("YZX", [90, -90, 180], degrees=True)

            ofz = R.from_euler("z", 0, degrees=True)
            ofzr = R.from_euler("z", 180, degrees=True)

            rq = of * R.from_quat([rq["x"], rq["y"], -rq["z"], -rq["w"]]) * of2 * ofzr
            lq = of1 * R.from_quat([lq["x"], lq["y"], -lq["z"], -lq["w"]]) * of3 * ofz

            rq = rq.as_quat()
            lq = lq.as_quat()
            
            # Calculate diff between actual robot and controller
            _, actual_right_quat = self.sim.get_root_pos("right")
            cur_right_vr_quat = R.from_quat([rq[0], rq[1], rq[2], rq[3]])
            actual_quat = R.from_quat([actual_right_quat[1], actual_right_quat[2], actual_right_quat[3], actual_right_quat[0]])
            diff_quat = cur_right_vr_quat.inv() * actual_quat
            right_diff_mag = np.linalg.norm(diff_quat.as_rotvec())
            
            _, actual_left_quat = self.sim.get_root_pos("left")
            cur_left_vr_quat = R.from_quat([lq[0], lq[1], lq[2], lq[3]])
            actual_quat = R.from_quat([actual_left_quat[1], actual_left_quat[2], actual_left_quat[3], actual_left_quat[0]])
            diff_quat = cur_left_vr_quat.inv() * actual_quat
            left_diff_mag = np.linalg.norm(diff_quat.as_rotvec())
            

            if right_tracked and (self.target.right_enabled or right_diff_mag < 0.4):
                self.target.right_enabled = True
                right_pos = (
                    rotZ.apply(np.array([-j[self.rlabels[0]]["z"], j[self.rlabels[0]]["x"], j[self.rlabels[0]]["y"]])) + self.offset
                )

                if self.target.right_origin_pos is None:
                    self.target.right_origin_pos = self.target.right_pos[0].copy()
                    self.target.right_vr_origin_pos = right_pos.copy()

                for i in range(6):
                    pos = (
                        rotZ.apply(
                            np.array(
                                [
                                    -j[self.rlabels[i]]["z"],
                                    j[self.rlabels[i]]["x"],
                                    j[self.rlabels[i]]["y"],
                                ]
                            )
                        )
                        + self.offset
                    )
                    self.target.right_pos[i, :] = self.target.right_origin_pos + (
                        pos - self.target.right_vr_origin_pos
                    ) + self.right_offset + self.right_adj
                
                self.target.right_quat[0] = rq[3]
                self.target.right_quat[1] = rq[0]
                self.target.right_quat[2] = rq[1]
                self.target.right_quat[3] = rq[2]

            else:
                self.target.right_enabled = False
                if self.target.right_origin_pos is not None:
                    self.target.right_origin_pos = None
                    self.target.right_vr_origin_pos = None
                    right_pos, right_quat = self.sim.get_root_pos("right")
                    self.target.right_pos[0, :] = right_pos
                    self.target.right_quat[:] = right_quat
                    self.right_offset = np.zeros(3)
                    self.right_adj = np.zeros(3)



            if left_tracked and (self.target.left_enabled or left_diff_mag < 0.4):
                self.target.left_enabled = True
                left_pos = (
                    rotZ.apply(
                        np.array(
                            [
                                -j[self.llabels[0]]["z"],
                                j[self.llabels[0]]["x"],
                                j[self.llabels[0]]["y"],
                            ]
                        )
                    )
                    + self.offset
                )

                if self.target.left_origin_pos is None:
                    self.target.left_origin_pos = self.target.left_pos[0].copy()
                    self.target.left_vr_origin_pos = left_pos.copy()

                for i in range(6):
                    pos = (
                        rotZ.apply(
                            np.array(
                                [
                                    -j[self.llabels[i]]["z"],
                                    j[self.llabels[i]]["x"],
                                    j[self.llabels[i]]["y"],
                                ]
                            )
                        )
                        + self.offset
                    )
                    self.target.left_pos[i, :] = self.target.left_origin_pos + (
                        pos - self.target.left_vr_origin_pos
                    ) + self.left_offset + self.left_adj
                
                self.target.left_quat[0] = lq[3]
                self.target.left_quat[1] = lq[0]
                self.target.left_quat[2] = lq[1]
                self.target.left_quat[3] = lq[2]
                
            else:
                self.target.left_enabled = False
                if self.target.left_origin_pos is not None:
                    self.target.left_origin_pos = None
                    self.target.left_vr_origin_pos = None
                    left_pos, left_quat = self.sim.get_root_pos("left")
                    self.target.left_pos[0, :] = left_pos
                    self.target.left_quat[:] = left_quat
                    self.left_offset = np.zeros(3)
                    self.left_adj = np.zeros(3)


        # ==== controller tracking ====
        else:
            self.target.hand_tracking = False

            self.target.right_index_trigger = j["rightIndexTrigger"]
            self.target.right_hand_trigger = j["rightHandTrigger"]
            self.target.left_index_trigger = j["leftIndexTrigger"]
            self.target.left_hand_trigger = j["leftHandTrigger"]

            self.target.left_active = self.target.left_hand_trigger > 0.8
            self.target.right_active = self.target.right_hand_trigger > 0.8

            rotZ = R.from_euler("z", 180, degrees=True)
            right_pos = rotZ.apply(
                np.array(
                    [
                        -j[self.rlabels[0]]["z"],
                        j[self.rlabels[0]]["x"],
                        j[self.rlabels[0]]["y"],
                    ]
                )
            )
            left_pos = rotZ.apply(
                np.array(
                    [
                        -j[self.llabels[0]]["z"],
                        j[self.llabels[0]]["x"],
                        j[self.llabels[0]]["y"],
                    ]
                )
            )

            rq = j[self.rlabels[-1]]
            lq = j[self.llabels[-1]]
            of = R.from_euler("ZYX", [90, 180, 90], degrees=True)
            of1 = R.from_euler("ZYX", [90, 180, 90], degrees=True)
            of2 = R.from_euler("ZYX", [90, 0, 0], degrees=True)
            of3 = R.from_euler("ZYX", [-90, 0, 0], degrees=True)

            rq = rotZ * of * R.from_quat([rq["x"], rq["y"], rq["z"], rq["w"]]) * of2
            lq = rotZ * of1 * R.from_quat([lq["x"], lq["y"], lq["z"], lq["w"]]) * of3

            rq = rq.as_quat()
            lq = lq.as_quat()

            if self.target.right_hand_trigger > 0.8:
                if self.target.right_origin_pos is None:
                    self.target.right_origin_pos = self.target.right_pos[0].copy()
                    self.target.right_origin_quat = quat_to_scipy(
                        self.target.right_quat.copy()
                    )

                    self.target.right_vr_origin_pos = right_pos.copy()
                    self.target.right_vr_origin_quat = R.from_quat(
                        [rq[0], rq[1], rq[2], rq[3]]
                    )

                cur_vr_pos = right_pos.copy()
                cur_vr_quat = R.from_quat([rq[0], rq[1], rq[2], rq[3]])
                self.target.right_pos[0, :] = self.target.right_origin_pos + (
                    cur_vr_pos - self.target.right_vr_origin_pos
                )
                self.target.right_quat[:] = scipy_to_quat(
                    cur_vr_quat
                    * self.target.right_vr_origin_quat.inv()
                    * self.target.right_origin_quat
                )
            else:
                if self.target.right_origin_pos is not None:
                    self.target.right_origin_pos = None
                    self.target.right_origin_quat = None
                    self.target.right_vr_origin_pos = None
                    self.target.right_vr_origin_quat = None

                    right_pos, right_quat = self.sim.get_root_pos("right")
                    self.target.right_pos[0, :] = right_pos
                    self.target.right_quat[:] = right_quat

            if self.target.left_hand_trigger > 0.8:
                if self.target.left_origin_pos is None:
                    self.target.left_origin_pos = self.target.left_pos[0].copy()
                    self.target.left_origin_quat = quat_to_scipy(
                        self.target.left_quat.copy()
                    )

                    self.target.left_vr_origin_pos = left_pos.copy()
                    self.target.left_vr_origin_quat = R.from_quat(
                        [lq[0], lq[1], lq[2], lq[3]]
                    )

                cur_vr_pos = left_pos.copy()
                cur_vr_quat = R.from_quat([lq[0], lq[1], lq[2], lq[3]])
                self.target.left_pos[0, :] = self.target.left_origin_pos + (
                    cur_vr_pos - self.target.left_vr_origin_pos
                )
                self.target.left_quat[:] = scipy_to_quat(
                    cur_vr_quat
                    * self.target.left_vr_origin_quat.inv()
                    * self.target.left_origin_quat
                )
            else:
                if self.target.left_origin_pos is not None:
                    self.target.left_origin_pos = None
                    self.target.left_origin_quat = None
                    self.target.left_vr_origin_pos = None
                    self.target.left_vr_origin_quat = None

                    left_pos, left_quat = self.sim.get_root_pos("left")
                    self.target.left_pos[0, :] = left_pos
                    self.target.left_quat[:] = left_quat

            self.target.set_fingers(j["rightIndexTrigger"], j["leftIndexTrigger"])

        self.updates += 1

    def reset_state(self):
        self.target.right_origin_pos = None
        self.target.right_vr_origin_pos = None
        right_pos, right_quat = self.sim.get_root_pos("right")
        self.target.right_pos[0, :] = right_pos
        self.target.right_quat[:] = right_quat

        self.target.left_origin_pos = None
        self.target.left_vr_origin_pos = None
        left_pos, left_quat = self.sim.get_root_pos("left")
        self.target.left_pos[0, :] = left_pos
        self.target.left_quat[:] = left_quat

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("[Quest3Reader] Stopped quest3 reader")

    def __del__(self):
        self.stop()


class VRPolicy:
    def __init__(self, env):
        self.quest3 = Quest3Reader("2G0YC1ZF860PXL", env.sim)
        self.env = env
        self.target = self.quest3.get_target()
        self.env.sim.register_target(self.target)
        self.r = redis.Redis(host="localhost", port=6379, db=0)
        self.thread = None
        self.use_pedal = True

    # ====== context manager ======
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ====== launch methods ======
    def start(self):
        self.quest3.run()

        # launch ik processes
        self.ik_processes = []
        for side in ["left", "right"]:
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                p = FingerIKSolver(side, finger)
                self.ik_processes.append(p)
                p.start(wait=False)
            p = ArmIKSolver(side)
            self.ik_processes.append(p)
            p.start(wait=False)

        self.keep_running = True
        self.thread = threading.Thread(target=self.submit_ik)
        self.thread.start()

        logger.info("[VRPolicy] Started VR policy")

    def stop(self):
        self.keep_running = False
        self.quest3.stop()
        for p in self.ik_processes:
            logger.info(f"Waiting on {p.name}")
            p.stop(wait=False)
        self.thread.join()
        logger.info("[VRPolicy] Stopped VR policy")

    def reset_state(self):
        self.quest3.reset_state()

    # ====== get ik solutions ======
    def get_arm_ik(self, side):
        sol = self.r.get(f"{side}_arm_action")
        if sol is None:
            return self.env.sim.get_arm_qpos(side)
        return pickle.loads(sol)

    def get_finger_ik(self, side, finger):
        sol = self.r.get(f"{side}_{finger}_action")
        if sol is None:
            return self.env.sim.get_finger_qpos(side, finger)
        return pickle.loads(sol)

    def get_arm_action(self, side):
        action = self.get_arm_ik(side)
        return action

    def get_hand_action(self, side):
        index = self.get_finger_ik(side, "index")
        middle = self.get_finger_ik(side, "middle")
        ring = self.get_finger_ik(side, "ring")
        pinky = self.get_finger_ik(side, "pinky")
        thumb = self.get_finger_ik(side, "thumb")

        action = np.array([index[0], middle[0], ring[0], pinky[0], thumb[1], thumb[0]])
        return action


    # ====== main loop ======
    def submit_ik(self):
        while self.keep_running:
            self.quest3.pedal = self.env.is_active()
            if not self.quest3.pedal:
                time.sleep(1.0 / 200)
                continue

            targets = self.env.sim.get_vr_targets()
            if not targets:
                time.sleep(1.0 / 200)
                continue

            for side in ["right", "left"]:
                for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                    target_pos = targets[f"{side}_{finger}"]
                    cur_finger_pos = self.env.sim.get_finger_pos(side, finger)
                    dx = target_pos - cur_finger_pos
                    Jp = self.env.sim.get_finger_jac(side, finger)
                    qpos = self.env.sim.get_finger_qpos(side, finger)
                    self.r.set(f"{side}_{finger}_solve", pickle.dumps((qpos, Jp, dx)))

                qpos = self.env.sim.get_arm_qpos(side)
                other_qpos = self.env.sim.get_arm_qpos(
                    "left" if side == "right" else "right"
                )
                target_pos, target_rot = targets[f"{side}_arm"]
                self.r.set(
                    f"{side}_arm_solve",
                    pickle.dumps((qpos, target_pos, target_rot, other_qpos)),
                )

            time.sleep(1.0 / 200)
    
    def warmup(self, iter=10):
        for i in range(iter):
            targets = self.env.sim.get_vr_targets()
            for side in ["right", "left"]:
                for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                    target_pos = targets[f"{side}_{finger}"]
                    cur_finger_pos = self.env.sim.get_finger_pos(side, finger)
                    dx = target_pos - cur_finger_pos
                    Jp = self.env.sim.get_finger_jac(side, finger)
                    qpos = self.env.sim.get_finger_qpos(side, finger)
                    self.r.set(f"{side}_{finger}_solve", pickle.dumps((qpos, Jp, dx)))

                qpos = self.env.sim.get_arm_qpos(side)
                other_qpos = self.env.sim.get_arm_qpos(
                    "left" if side == "right" else "right"
                )
                target_pos, target_rot = targets[f"{side}_arm"]
                self.r.set(
                    f"{side}_arm_solve",
                    pickle.dumps((qpos, target_pos, target_rot, other_qpos)),
                )

    # ====== policy inference ======
    def forward(self, obs_dict):
        left_arm_action = self.get_arm_action("left")
        right_arm_action = self.get_arm_action("right")
        left_hand_action = self.get_hand_action("left")
        right_hand_action = self.get_hand_action("right")

        return {
            "left_arm_cmd": left_arm_action,
            "right_arm_cmd": right_arm_action,
            "left_fingers_cmd": left_hand_action,
            "right_fingers_cmd": right_hand_action,
        }
