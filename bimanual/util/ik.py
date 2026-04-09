import multiprocessing as mp
import numpy as np
import pickle
import redis
import time

from casadi import *
from copy import deepcopy
from loguru import logger
from scipy.spatial.transform import Rotation as R

from bimanual.control.ur3e import DEFAULT_UR3E_POS
from bimanual.util.rot_utils import rotation_matrix_to_euler


# list of collision spheres with radius r and x,y,z in local joint coord frame from DH transforms
# [ [x,y,z,r], ... ]
COLLISION_CONFIG = {
    "right_shoulder_pan_joint": [],  # [0,-0.125,0.125,0.07], [0,-0.25,0.125,0.07]
    "right_shoulder_lift_joint": [
        [0, 0, 0.025, 0.05],
        [0, 0, 0.125, 0.06],
        [0.1, 0, 0.125, 0.05],
        [0.225, 0, 0.12, 0.07],
    ],
    "right_elbow_joint": [[0, 0, 0.025, 0.05], [0.1, 0, 0.025, 0.05]],
    "right_wrist_1_joint": [[0, 0, 0, 0.05]],
    "right_wrist_2_joint": [[0, 0, 0, 0.05]],
    "right_wrist_3_joint": [
        [0, 0, 0, 0.05],
        [0, 0, 0.1, 0.05],
        # [-0.02, -0.02, 0.15, 0.05],
    ],  # [0,0,0.1,0.03],[0,0,0.05,0.03],[0.025,0,0.075,0.03],[-0.025,0,0.075,0.03],[-0.025,0,0.1,0.03],[-0.025,0,0.125,0.03],[0.025,0,0.125,0.03],[0,0,0.15,0.03]],
    "left_shoulder_pan_joint": [],
    "left_shoulder_lift_joint": [
        [0, 0, 0.025, 0.05],
        [0, 0, 0.125, 0.06],
        [0.1, 0, 0.125, 0.05],
        [0.225, 0, 0.12, 0.07],
    ],
    "left_elbow_joint": [[0, 0, 0.025, 0.05], [0.1, 0, 0.025, 0.05]],
    "left_wrist_1_joint": [[0, 0, 0, 0.05]],
    "left_wrist_2_joint": [[0, 0, 0, 0.05]],
    "left_wrist_3_joint": [
        [0, 0, 0, 0.05],
        [0, 0, 0.1, 0.05],
        # [-0.02, -0.02, 0.15, 0.05],
    ],  # [0,0,0.1,0.03], [0,0,0.05,0.03],[0.025,0,0.075,0.03],[-0.025,0,0.075,0.03],[-0.025,0,0.1,0.03],[-0.025,0,0.125,0.03],[0.025,0,0.125,0.03],[0,0,0.15,0.03]],
    "stand": [
        [-0.6, 0, 1.45, 0.1],
        [-0.6, 0, 1.35, 0.1],
        [-0.6, 0, 1.25, 0.1],
        [-0.6, 0, 1.15, 0.1],
        [-0.6, 0, 1.05, 0.1],
        [-0.6, 0.15, 1.45, 0.1],
        [-0.6, -0.15, 1.45, 0.1],
    ],
    "table": [
        [i, j, 0.85, 0.1]
        for i in np.arange(-0.5, 0.6, 0.1)
        for j in np.arange(-0.5, 0.6, 0.1)
    ],
}

COLLISION_EXCLUSION = {
    "right_shoulder_pan_joint": ["right_shoulder_lift_joint"],
    "right_shoulder_lift_joint": ["right_shoulder_pan_joint", "right_elbow_joint"],
    "right_elbow_joint": ["right_shoulder_lift_joint", "right_wrist_1_joint"],
    "right_wrist_1_joint": ["right_elbow_joint", "right_wrist_2_joint"],
    "right_wrist_2_joint": ["right_wrist_1_joint", "right_wrist_3_joint"],
    "right_wrist_3_joint": ["right_wrist_2_joint"],
    "left_shoulder_pan_joint": ["left_shoulder_lift_joint"],
    "left_shoulder_lift_joint": ["left_shoulder_pan_joint", "left_elbow_joint"],
    "left_elbow_joint": ["left_shoulder_lift_joint", "left_wrist_1_joint"],
    "left_wrist_1_joint": ["left_elbow_joint", "left_wrist_2_joint"],
    "left_wrist_2_joint": ["left_wrist_1_joint", "left_wrist_3_joint"],
    "left_wrist_3_joint": ["left_wrist_2_joint"],
    "stand": [],
    "table": [],
}


class ArmIKSolver(mp.Process):
    def __init__(self, side, freq=300):
        super(ArmIKSolver, self).__init__(name="ArmIKSolver")
        self.ready_event = mp.Event()
        self.finish_event = mp.Event()
        self.freq = freq
        self.side = side

        self.arm_joint_names = [
            f"{side}_shoulder_pan_joint",
            f"{side}_shoulder_lift_joint",
            f"{side}_elbow_joint",
            f"{side}_wrist_1_joint",
            f"{side}_wrist_2_joint",
            f"{side}_wrist_3_joint",
        ]

        self.default_pos = DEFAULT_UR3E_POS[side]
        joint_lrange = np.array(
            [np.pi / 10, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 2, np.pi]
        )
        joint_urange = np.array(
            [np.pi / 10, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 2, np.pi]
        )
        self.upper = self.default_pos + joint_urange
        self.lower = self.default_pos - joint_lrange

        self.r = redis.Redis(host="localhost", port=6379, db=0)

    # ===== launch functions =====
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        logger.info(f"[ArmIKSolver] Solver process spawned at {self.pid}")

    def start_wait(self):
        self.ready_event.wait()
        logger.info(f"[ArmIKSolver] Solver process ready")

    def stop(self, wait=True):
        self.finish_event.set()
        if wait:
            self.stop_wait()
        logger.info(f"[ArmIKSolver] Solver process stopped")

    def stop_wait(self):
        self.join()

    # ===== helpers =====
    @staticmethod
    def fkmap(qpos, right=True, joint_idx=-1, trans=False):
        theta = qpos
        if right:
            offset = np.array([-np.pi / 4 - np.pi / 2, np.pi, 0, 0, 0, 0])
        else:
            offset = np.array([-np.pi / 4, np.pi, 0, 0, 0, 0])

        theta -= offset

        a = [0, -0.24355, -0.2132, 0, 0, 0]
        d = [0.15185, 0, 0, 0.13105, 0.08535, 0.0921]
        alpha = [-np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]

        def transformation_matrix(theta, d, a, alpha):
            return np.array(
                [
                    [
                        np.cos(theta),
                        -np.sin(theta) * np.cos(alpha),
                        np.sin(theta) * np.sin(alpha),
                        a * np.cos(theta),
                    ],
                    [
                        np.sin(theta),
                        np.cos(theta) * np.cos(alpha),
                        -np.cos(theta) * np.sin(alpha),
                        a * np.sin(theta),
                    ],
                    [0, np.sin(alpha), np.cos(alpha), d],
                    [0, 0, 0, 1],
                ]
            )

        # euler="2.356 3.14 0" pos="-0.6 -0.21 1.45"
        if right:
            rot = (
                R.from_euler("XYZ", [2.356, 3.14, 0], degrees=False)
                * R.from_euler("X", 180, degrees=True)
                * R.from_euler("Z", 0, degrees=True)
            ).as_matrix()

            base = np.array(
                [
                    [rot[0, 0], rot[0, 1], rot[0, 2], -0.6],
                    [rot[1, 0], rot[1, 1], rot[1, 2], -0.21],
                    [rot[2, 0], rot[2, 1], rot[2, 2], 1.45],
                    [0, 0, 0, 1],
                ]
            )
        else:
            # -2.356 3.14 0
            rot = (
                R.from_euler("XYZ", [-2.356, 3.14, 0], degrees=False)
                * R.from_euler("X", 180, degrees=True)
                * R.from_euler("Z", 0, degrees=True)
            ).as_matrix()
            base = np.array(
                [
                    [rot[0, 0], rot[0, 1], rot[0, 2], -0.6],
                    [rot[1, 0], rot[1, 1], rot[1, 2], 0.22],
                    [rot[2, 0], rot[2, 1], rot[2, 2], 1.45],
                    [0, 0, 0, 1],
                ]
            )

        translate = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.077],
                [0, 0, 0, 1],
            ]
        )

        if right:
            rotate = R.from_euler("y", 90).as_matrix()
        else:
            rotate = R.from_euler("y", -90).as_matrix()
        rotation = np.array(
            [
                [rotate[0, 0], rotate[0, 1], rotate[0, 2], 0],
                [rotate[1, 0], rotate[1, 1], rotate[1, 2], 0],
                [rotate[2, 0], rotate[2, 1], rotate[2, 2], 0],
                [0, 0, 0, 1],
            ]
        )

        A = []
        for i in range(6):
            A.append(transformation_matrix(theta[i], d[i], a[i], alpha[i]))
        # self.vis_transform(base @ A[0])
        # self.vis_transform(base @ A[0] @ A[1])
        # self.vis_transform(base @ A[0] @ A[1] @ A[2])
        # self.vis_transform(base @ A[0] @ A[1] @ A[2] @ A[3])
        # self.vis_transform(base @ A[0] @ A[1] @ A[2] @ A[3] @ A[4])
        # self.vis_transform(base @ A[0] @ A[1] @ A[2] @ A[3] @ A[4] @ A[5])
        # self.vis_transform(base @ A[0] @ A[1] @ A[2] @ A[3] @ A[4] @ A[5])
        if joint_idx == -1:
            H = base @ A[0] @ A[1] @ A[2] @ A[3] @ A[4] @ A[5] @ translate  # @ rotation
        else:
            H = base
            for j in range(joint_idx + 1):
                H = H @ A[j]

        theta += offset

        if trans:
            return H
        return H[:3, 3].reshape(-1), H[:3, :3]

    def collision_cost(self, opt_qpos, other_qpos):
        extra_cost = 0
        cfg = deepcopy(COLLISION_CONFIG)
        for qpos in [opt_qpos, other_qpos]:
            for i, joint in enumerate(self.arm_joint_names):
                H = ArmIKSolver.fkmap(
                    qpos, right=(self.side == "right"), joint_idx=i, trans=True
                )
                for i in range(len(cfg[joint])):
                    x, y, z, r = cfg[joint][
                        i
                    ]  # x y z in local joint coord frame, sphere has radius r
                    Hp = H @ np.array(
                        [
                            [1, 0, 0, x],
                            [0, 1, 0, y],
                            [0, 0, 1, z],
                            [0, 0, 0, 1],
                        ]
                    )
                    global_pos = Hp[:3, 3]
                    # print(global_pos)
                    # sphere[:3] = global_pos
                    cfg[joint][i] = global_pos.tolist() + [r]

        constraints = []
        joint_names = list(cfg.keys())
        for i, ijoint in enumerate(joint_names):
            for j, jjoint in enumerate(joint_names[i + 1 :]):
                if jjoint in COLLISION_EXCLUSION[ijoint]:
                    continue
                for isphere in cfg[ijoint]:
                    for jsphere in cfg[jjoint]:
                        cond = (isphere[0] - jsphere[0]) ** 2 + (
                            isphere[1] - jsphere[1]
                        ) ** 2 + (isphere[2] - jsphere[2]) ** 2 >= (
                            isphere[3] + jsphere[3]
                        ) ** 2
                        if not isinstance(cond, bool):
                            constraints.append(cond)
                        extra_cost += if_else(
                            (isphere[0] - jsphere[0]) ** 2
                            + (isphere[1] - jsphere[1]) ** 2
                            + (isphere[2] - jsphere[2]) ** 2
                            >= (isphere[3] + jsphere[3]) ** 2,
                            0,
                            10000000000,
                        )
        return extra_cost

    def collisions(self, opt_qpos, other_qpos):
        arm_joint_names = lambda x: [
            f"{x}_shoulder_pan_joint",
            f"{x}_shoulder_lift_joint",
            f"{x}_elbow_joint",
            f"{x}_wrist_1_joint",
            f"{x}_wrist_2_joint",
            f"{x}_wrist_3_joint",
        ]
        other_hand = lambda x: "left" if x == "right" else "right"
        collisions = []
        cfg = deepcopy(COLLISION_CONFIG)
        for qpos, hand in [(opt_qpos, self.side), (other_qpos, other_hand(self.side))]:
            for i, joint in enumerate(arm_joint_names(hand)):
                H = ArmIKSolver.fkmap(
                    qpos, right=(hand == "right"), joint_idx=i, trans=True
                )
                for i in range(len(cfg[joint])):
                    x, y, z, r = cfg[joint][
                        i
                    ]  # x y z in local joint coord frame, sphere has radius r
                    Hp = H @ np.array(
                        [
                            [1, 0, 0, x],
                            [0, 1, 0, y],
                            [0, 0, 1, z],
                            [0, 0, 0, 1],
                        ]
                    )
                    global_pos = Hp[:3, 3]
                    # print(global_pos)
                    # sphere[:3] = global_pos
                    cfg[joint][i] = global_pos.tolist() + [r]

        joint_names = list(cfg.keys())
        for i, ijoint in enumerate(joint_names):
            for j, jjoint in enumerate(joint_names[i + 1 :]):
                if jjoint in COLLISION_EXCLUSION[ijoint]:
                    continue
                for ii, isphere in enumerate(cfg[ijoint]):
                    for jj, jsphere in enumerate(cfg[jjoint]):
                        cond = (isphere[0] - jsphere[0]) ** 2 + (
                            isphere[1] - jsphere[1]
                        ) ** 2 + (isphere[2] - jsphere[2]) ** 2 >= (
                            isphere[3] + jsphere[3]
                        ) ** 2
                        collisions.append(cond)
                        # if not cond:
                        #     print('collision', ijoint, jjoint, ii, jj, isphere, jsphere)
        return len(collisions) - sum(collisions)

    # ===== solving functions =====
    def solve(self, qpos, target_pos, target_rot, other_qpos):
        # logger.info(f'{qpos=}, {target_pos=}, {target_rot=}, {other_qpos=}')
        n = 6
        try:
            opti = Opti()
            x = opti.variable(n)
            opti.set_initial(x, np.zeros(n))

            fkpos, fkr = ArmIKSolver.fkmap(qpos + x, right=self.side == "right")
            target_r = target_rot @ fkr.T
            rerr = rotation_matrix_to_euler(target_r)
            fkelbowpos, _ = ArmIKSolver.fkmap(
                qpos + x, right=self.side == "right", joint_idx=1
            )

            obj = (
                3 * np.sum((fkpos - target_pos) ** 2)
                + 0.1 * np.sum(rerr**2)
                + 0.1 * sumsqr(x)
                - 0.05 * fkelbowpos[1] ** 2
            )
            # obj += collision_cost(qpos + x, other_qpos)
            constraints = [
                self.lower[:5] <= (x + qpos)[:5],
                (x + qpos)[:5] <= self.upper[:5],
            ]  # + collision_cost(qpos + x, other_qpos)

            opti.minimize(obj)
            opti.subject_to(constraints)
            opti.solver("ipopt", {"ipopt": {"print_level": 0}, "print_time": 0})
            sol = opti.solve()
            sol = sol.value(x)

            # max_ang_vel = 0.2
            # dq_abs_max = np.abs(sol).max()
            # if dq_abs_max > max_ang_vel:
            #     sol *= max_ang_vel / dq_abs_max

            num_collisions = self.collisions(qpos + sol, other_qpos)
            if num_collisions > 0:
                logger.info(f"[ArmIKSolver] Collision detected")
                sol = np.zeros(n)

        except Exception as e:
            logger.info(f"{e}")
            sol = np.zeros(n)

        return sol + qpos

    # ===== main loop =====
    def run(self):
        while not self.finish_event.is_set():
            try:
                # receive 3d target positions
                payload = self.r.get(f"{self.side}_arm_solve")
                if payload is None:
                    time.sleep(1 / self.freq)
                    continue

                ret = pickle.loads(payload)
                if ret is None:
                    break
                qpos, target_pos, target_rot, other_qpos = ret

                # solve IK
                start_t = time.time()
                sol = self.solve(qpos, target_pos, target_rot, other_qpos)

                # send joint angles
                self.r.set(f"{self.side}_arm_action", pickle.dumps(sol))

                # sleep
                end_t = time.time()
                if 1 / self.freq - (end_t - start_t) > 0:
                    time.sleep(1 / self.freq - (end_t - start_t))
            except Exception as e:
                logger.info(f"{e}")
                time.sleep(1 / self.freq)

        logger.info("[ArmIKSolver] Finished loop")


class FingerIKSolver(mp.Process):
    def __init__(self, side, finger, freq=300):
        super(FingerIKSolver, self).__init__(name="FingerIKSolver")
        self.ready_event = mp.Event()
        self.finish_event = mp.Event()
        self.freq = freq
        self.side = side
        self.finger = finger

        self.r = redis.Redis(host="localhost", port=6379, db=0)

    # ===== launch functions =====
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        logger.info(f"[FingerIKSolver] Solver process spawned at {self.pid}")

    def start_wait(self):
        self.ready_event.wait()
        logger.info(f"[FingerIKSolver] Solver process ready")

    def stop(self, wait=True):
        logger.info(f"[FingerIKSolver] Solver process stopping")
        self.finish_event.set()
        if wait:
            self.stop_wait()
        logger.info(f"[FingerIKSolver] Solver process stopped")

    def stop_wait(self):
        self.join()

    # ===== solving functions =====
    def solve(self, qpos, finger_J, finger_error):
        q1_range = [0, 1.7453]
        q2_range = [0, 2.6586]
        reg = 0.005 if self.finger == "thumb" else 0.005
        c = np.array([1.05851325, 0.72349796])

        if self.finger == "thumb":
            q1_range = [-1.7453, 0]
            q2_range = [0, 1.7453]
        n = 2
        opti = Opti()
        x = opti.variable(n)
        opti.set_initial(x, np.zeros(n))
        opti.minimize(sumsqr(mtimes(finger_J, x) - finger_error) + reg * sumsqr(x))
        constraints = [
            q1_range[0] <= qpos[0] + x[0],
            qpos[0] + x[0] <= q1_range[1],
            q2_range[0] <= qpos[1] + x[1],
            qpos[1] + x[1] <= q2_range[1],
        ]
        if self.finger != "thumb":
            constraints.append(x[1] == c[0] * x[0])
        opti.subject_to(constraints)
        opti.solver("ipopt", {"ipopt": {"print_level": 0}, "print_time": 0})
        sol = opti.solve()
        return sol.value(x) + qpos

    # ===== main loop =====
    def run(self):
        while not self.finish_event.is_set():
            try:
                # receive 3d target positions
                payload = self.r.get(f"{self.side}_{self.finger}_solve")
                if payload is None:
                    time.sleep(1 / self.freq)
                    continue
                ret = pickle.loads(payload)
                qpos, finger_J, finger_error = ret

                # solve IK
                start_t = time.time()
                sol = self.solve(qpos, finger_J, finger_error)

                # send joint angles
                self.r.set(f"{self.side}_{self.finger}_action", pickle.dumps(sol))

                # sleep
                end_t = time.time()
                if 1 / self.freq - (end_t - start_t) > 0:
                    time.sleep(1 / self.freq - (end_t - start_t))
            except Exception as e:
                logger.info(f"{e}")
                time.sleep(1 / self.freq)

        logger.info("[FingerIKSolver] Finished loop")
