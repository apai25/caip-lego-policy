import mujoco as mj
import mujoco_viewer as mjv
import numpy as np
import glfw

from copy import deepcopy
from loguru import logger
from scipy.spatial.transform import Rotation as R

from bimanual.control.quest3 import VRTarget
from bimanual.util.render import SimpleRenderer
from bimanual.util.ik import *


class TermStatus:
    SUCCESS = "success"
    FAILURE = "failure"
    ACTIVE = "active"
    QUIT = "quit"


class DualUR3ESimEnv:

    def __init__(self, window=True):
        self.window = window
        self.model = mj.MjModel.from_xml_path("/home/ilija/code/mvp_generalize_v2/mvp-generalize/assets/mjcf/dual_ur3e_45/scene.xml")
        self.data = mj.MjData(self.model)

        if self.window:
            self.viewer = mjv.MujocoViewer(
                self.model,
                self.data,
                height=720*2,
                width=720*2,
                hide_menus=True,
                mode="window",
            )
        else:
            # self.viewer = SimpleRenderer(self.model, self.data, height=720, width=720)
            self.viewer = None

        # joint/site names for arm and hand
        self.arm_joint_names = lambda name: [
            f"{name}_shoulder_pan_joint",
            f"{name}_shoulder_lift_joint",
            f"{name}_elbow_joint",
            f"{name}_wrist_1_joint",
            f"{name}_wrist_2_joint",
            f"{name}_wrist_3_joint",
        ]
        finger_site_names = lambda name: dict(
            index=f"{name}_index_L1_site",
            middle=f"{name}_middle_L1_site",
            ring=f"{name}_ring_L1_site",
            pinky=f"{name}_pinky_L1_site",
            thumb=f"{name}_thumb_L1_site",
        )
        finger_joint_names = lambda name: dict(
            index=[f"ability_{name}:index_q1", f"ability_{name}:index_q2"],
            middle=[f"ability_{name}:middle_q1", f"ability_{name}:middle_q2"],
            ring=[f"ability_{name}:ring_q1", f"ability_{name}:ring_q2"],
            pinky=[f"ability_{name}:pinky_q1", f"ability_{name}:pinky_q2"],
            thumb=[f"ability_{name}:thumb_q1", f"ability_{name}:thumb_q2"],
        )

        # robot metadata
        self.robot_meta = {}
        for hand in ["right", "left"]:
            _arm_joint_names = self.arm_joint_names(hand)
            _finger_joint_names = finger_joint_names(hand)
            _finger_site_names = finger_site_names(hand)

            self.robot_meta[hand] = dict(
                arm_joints=[self.model.joint(i).qposadr[0] for i in _arm_joint_names],
                arm_dofs=[self.model.joint(i).dofadr[0] for i in _arm_joint_names],
                arm_mocap_idx=0,
                finger_joints={
                    finger: [self.model.joint(i).qposadr[0] for i in joint_names]
                    for finger, joint_names in _finger_joint_names.items()
                },
                finger_dofs={
                    finger: [self.model.joint(i).dofadr[0] for i in joint_names]
                    for finger, joint_names in _finger_joint_names.items()
                },
                finger_site_ids={
                    finger: self.model.site(site_name).id
                    for finger, site_name in _finger_site_names.items()
                },
                finger_mocap_idxs=dict(thumb=15, index=3, middle=6, ring=12, pinky=9),
                finger_range={
                    finger: [self.model.joint(i).range for i in joint_names]
                    for finger, joint_names in _finger_joint_names.items()
                },
            )

        # video capture
        self.cam = mj.MjvCamera()
        self.cam.lookat = np.array([0, 0, 1.2])
        self.cam.distance = 1.5
        self.cam.azimuth = 180
        self.cam.elevation = -25
        self.video_frames = []

        # pedal states
        self.right_pedal = False
        self.middle_pedal = False
        self.left_pedal = False

        self.target = None

        logger.info("[DualUR3ESimEnv] Initialized mujoco simulator")

    # ====== launch methods ======
    def stop(self):
        if self.window:
            self.viewer.close()
        logger.info("[DualUR3ESimEnv] Stopped mujoco simulator")

    # ====== vr target methods =======
    def register_target(self, target):
        self.target = target

    def clear_pedals(self):
        self.right_pedal = False
        self.middle_pedal = False
        self.left_pedal = False

    def get_pedal_states(self):
        return (self.left_pedal, self.middle_pedal, self.right_pedal)

    def update_pedal_states(self):
        if not self.window:
            return
        self.right_pedal = (
            glfw.get_key(self.viewer.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
        )
        self.middle_pedal = glfw.get_key(self.viewer.window, glfw.KEY_C) == glfw.PRESS
        self.left_pedal = glfw.get_key(self.viewer.window, glfw.KEY_Q) == glfw.PRESS

    def get_deadman_switch(self):
        return self.right_pedal

    def get_termination_status(self):
        if self.middle_pedal:
            return TerminationStatus.SUCCESS
        elif self.left_pedal:
            return TerminationStatus.FAILURE
        else:
            return TerminationStatus.NOT_TERMINATED

    def get_vr_targets(self):
        targets = {}

        for side in ["left", "right"]:
            hand_pos, hand_rot = self.target.get_hands()[side]
            cur_rot = R.from_matrix(
                self.data.site_xmat[
                    self.model.site(f"{side}_attachment_site").id
                ].reshape(3, 3)
            ) * R.from_euler("y", -90 if side == "right" else 90, degrees=True)
            target_rot = hand_rot
            cur_root_pos = self.data.site_xpos[
                self.model.site(f"{side}_attachment_site").id
            ] + cur_rot.apply(np.array([0, 0, -0.05 if side == "left" else -0.05]))
            target_root_pos = hand_pos[0, :]

            trans = cur_root_pos - target_root_pos
            cur_hand_pos = hand_pos + trans

            diff = cur_rot * target_rot.inv()
            cur_hand_pos = diff.apply(cur_hand_pos - cur_root_pos) + cur_root_pos

            if self.target.hand_tracking:
                for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                    target_idx = {
                        "thumb": 0,
                        "index": 1,
                        "middle": 2,
                        "ring": 3,
                        "pinky": 4,
                    }
                    target_pos = cur_hand_pos[target_idx[finger] + 1, :]
                    targets[f"{side}_{finger}"] = target_pos.copy()
            elif not self.target.hand_tracking and self.target.is_active(side):
                pass

            targets[f"{side}_arm"] = [target_root_pos, target_rot.as_matrix()]

        return targets

    def get_mocap_targets(self):
        targets = {}

        for side in ["left", "right"]:
            hand_pos, hand_rot = self.target.get_hands()[side]
            cur_rot = R.from_matrix(
                self.data.site_xmat[
                    self.model.site(f"{side}_attachment_site").id
                ].reshape(3, 3)
            ) * R.from_euler("y", -90 if side == "right" else 90, degrees=True)
            target_rot = hand_rot
            offset = -0.05 if isinstance(self.target, VRTarget) else 0.0
            cur_root_pos = self.data.site_xpos[
                self.model.site(f"{side}_attachment_site").id
            ] + cur_rot.apply(np.array([0, 0, offset]))
            target_root_pos = hand_pos[0, :]

            trans = cur_root_pos - target_root_pos
            cur_hand_pos = hand_pos + trans

            diff = cur_rot * target_rot.inv()
            cur_hand_pos = diff.apply(cur_hand_pos - cur_root_pos) + cur_root_pos

            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                target_idx = {
                    "thumb": 0,
                    "index": 1,
                    "middle": 2,
                    "ring": 3,
                    "pinky": 4,
                }
                target_pos = cur_hand_pos[target_idx[finger] + 1, :]
                targets[f"{side}_{finger}"] = target_pos.copy()


            targets[f"{side}_arm"] = [target_root_pos, target_rot.as_matrix()]

        return targets

    # ====== init methods =======
    def init_state(self, state):
        for side in ["left", "right"]:
            self.data.qpos[self.robot_meta[side]["arm_joints"]] = state[f"{side}_arm"]

            c = np.array([1.05851325, 0.72349796])
            index, middle, ring, pinky, lthumb, uthumb = np.deg2rad(
                [10.0, 10.0, 10.0, 10.0, 10.0, -40.0]
            )
            self.data.qpos[self.robot_meta[side]["finger_joints"]["index"]] = np.array(
                [index, c[0] * index + c[1]]
            )
            self.data.qpos[self.robot_meta[side]["finger_joints"]["middle"]] = np.array(
                [middle, c[0] * middle + c[1]]
            )
            self.data.qpos[self.robot_meta[side]["finger_joints"]["ring"]] = np.array(
                [ring, c[0] * ring + c[1]]
            )
            self.data.qpos[self.robot_meta[side]["finger_joints"]["pinky"]] = np.array(
                [pinky, c[0] * pinky + c[1]]
            )
            self.data.qpos[self.robot_meta[side]["finger_joints"]["thumb"]] = np.array(
                [uthumb, lthumb]
            )

        mj.mj_forward(self.model, self.data)
        self.render()

    # ====== step methods =======
    def apply_action(self, action):
        if not action:
            return

        self.data.qpos[self.robot_meta["right"]["arm_joints"]] = action["right_arm_cmd"]
        self.data.qpos[self.robot_meta["left"]["arm_joints"]] = action["left_arm_cmd"]

        c = np.array([1.05851325, 0.72349796])
        for side in ["right", "left"]:
            index, middle, ring, pinky, lthumb, uthumb = action[f"{side}_fingers_cmd"]

            self.data.qpos[self.robot_meta[side]["finger_joints"]["index"]] = np.array(
                [index, c[0] * index + c[1]]
            )
            self.data.qpos[self.robot_meta[side]["finger_joints"]["middle"]] = np.array(
                [middle, c[0] * middle + c[1]]
            )
            self.data.qpos[self.robot_meta[side]["finger_joints"]["ring"]] = np.array(
                [ring, c[0] * ring + c[1]]
            )
            self.data.qpos[self.robot_meta[side]["finger_joints"]["pinky"]] = np.array(
                [pinky, c[0] * pinky + c[1]]
            )
            self.data.qpos[self.robot_meta[side]["finger_joints"]["thumb"]] = np.array(
                [uthumb, lthumb]
            )

    def step(self, action):
        self.apply_action(action)
        self.update_pedal_states()
        self.add_vr_markers()
        # self.add_collision_markers()
        mj.mj_forward(self.model, self.data)
        self.render()

    # ====== finger and arm methods =======
    def get_finger_jac(self, side, finger):
        J = np.zeros((3, self.model.nv))
        mj.mj_jacSite(
            self.model,
            self.data,
            J,
            None,
            self.robot_meta[side]["finger_site_ids"][finger],
        )
        return J[:, self.robot_meta[side]["finger_dofs"][finger]]

    def get_finger_pos(self, side, finger):
        finger_site_ids = self.robot_meta[side]["finger_site_ids"]
        return self.data.site_xpos[finger_site_ids[finger]]

    def get_finger_qpos(self, side, finger):
        finger_joints = self.robot_meta[side]["finger_joints"][finger]
        return self.data.qpos[finger_joints]

    def get_arm_qpos(self, side):
        arm_joints = self.robot_meta[side]["arm_joints"]
        return self.data.qpos[arm_joints]

    def get_root_pos(self, hand_name):
        cur_rot = R.from_matrix(
            self.data.site_xmat[
                self.model.site(f"{hand_name}_attachment_site").id
            ].reshape(3, 3)
        ) * R.from_euler("y", -90 if hand_name == "right" else 90, degrees=True)
        cur_root_pos = self.data.site_xpos[
            self.model.site(f"{hand_name}_attachment_site").id
        ] + cur_rot.apply(np.array([0, 0, -0.02 if hand_name == "left" else -0.02]))

        x, y, z, w = cur_rot.as_quat()
        cur_quat = np.array([w, x, y, z])

        return cur_root_pos, cur_quat

    # ====== render methods =======
    def add_vr_markers(self):
        if not self.window or self.target is None:
            return

        for hand_name, (hand_pos, hand_rot) in self.target.get_hands().items():
            num_joints = hand_pos.shape[0]
            self.viewer.add_marker(
                pos=hand_pos[0, :].reshape(-1),
                size=[0.04, 0.04, 0.04],
                rgba=[1, 0, 0, 1],
                type=mj.mjtGeom.mjGEOM_SPHERE,
                label=f"",
            )

            cur_rot = R.from_matrix(
                self.data.site_xmat[
                    self.model.site(f"{hand_name}_attachment_site").id
                ].reshape(3, 3)
            ) * R.from_euler("y", -90 if hand_name == "right" else 90, degrees=True)
            target_rot = hand_rot
            offset = -0.05 if isinstance(self.target, VRTarget) else 0.0
            cur_root_pos = self.data.site_xpos[
                self.model.site(f"{hand_name}_attachment_site").id
            ] + cur_rot.apply(np.array([0, 0, offset]))
            target_root_pos = hand_pos[0, :]

            trans = cur_root_pos - target_root_pos
            cur_hand_pos = hand_pos + trans

            diff = cur_rot * target_rot.inv()
            adjusted_hand_pos = diff.apply(cur_hand_pos - cur_root_pos) + cur_root_pos

            if self.target.hand_tracking:
                for i in range(num_joints):
                    self.viewer.add_marker(
                        pos=adjusted_hand_pos[i, :].reshape(-1),
                        size=[0.01, 0.01, 0.01],
                        rgba=[0, 1, 0, 1] if i != 0 else [1, 0, 0, 1],
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        label=f"",
                    )

    def add_collision_markers(self, local_axes=False):
        if not self.window:
            return

        cfg = deepcopy(COLLISION_CONFIG)
        for hand in ["left", "right"]:
            for i, joint in enumerate(self.arm_joint_names(hand)):
                H = ArmIKSolver.fkmap(
                    self.data.qpos[self.robot_meta[hand]["arm_joints"]],
                    right=(hand == "right"),
                    joint_idx=i,
                    trans=True,
                )

                if local_axes:
                    origin = H[:3, 3]
                    rot = H[:3, :3]
                    self.viewer.add_marker(
                        pos=origin,
                        mat=rot,
                        size=np.array([0.005, 0.005, 0.005]) * 30,
                        rgba=[1, 0, 0, 1],
                        type=mj.mjtGeom.mjGEOM_LINE,
                        label=f"",
                    )
                    self.viewer.add_marker(
                        pos=origin,
                        mat=rot @ R.from_euler("y", 90, degrees=True).as_matrix(),
                        size=np.array([0.005, 0.005, 0.005]) * 30,
                        rgba=[0, 1, 0, 1],
                        type=mj.mjtGeom.mjGEOM_LINE,
                        label=f"",
                    )
                    self.viewer.add_marker(
                        pos=origin,
                        mat=rot @ R.from_euler("x", 90, degrees=True).as_matrix(),
                        size=np.array([0.005, 0.005, 0.005]) * 30,
                        rgba=[0, 0, 1, 1],
                        type=mj.mjtGeom.mjGEOM_LINE,
                        label=f"",
                    )

                for sphere in cfg[joint]:
                    x, y, z, r = (
                        sphere  # x y z in local joint coord frame, sphere has radius r
                    )
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
                    self.viewer.add_marker(
                        pos=global_pos,
                        size=[r, r, r],
                        rgba=[1, 1, 1, 1],
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        label=f"",
                    )

        for name in cfg:
            if name in self.arm_joint_names("right") or name in self.arm_joint_names(
                "left"
            ):
                continue
            for sphere in cfg[name]:
                x, y, z, r = sphere
                self.viewer.add_marker(
                    pos=np.array([x, y, z]),
                    size=[r, r, r],
                    rgba=[1, 1, 1, 1],
                    type=mj.mjtGeom.mjGEOM_SPHERE,
                    label=f"",
                )

    def render(self):
        if self.window:
            self.viewer.render()

    def capture_frame(self):
        frame = self.viewer.read_pixels(self.cam, depth=False)
        self.video_frames.append(frame)
