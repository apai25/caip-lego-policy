"""Client for MANUS mocap gloves."""
import json
import matplotlib.pyplot as plt
import numpy as np
import redis
import pickle
import time
import transforms3d as t3d
import zmq
import subprocess
import traceback

from loguru import logger
from ppadb.client import Client as AdbClient
from scipy.spatial.transform import Rotation as R
from threading import Thread, Lock

from bimanual.util.ik import FingerIKSolver, ArmIKSolver
from bimanual.util.rot_utils import scipy_to_quat, quat_to_scipy


class MocapTarget:
    def __init__(self, sim=None):
        self.left_root = np.zeros(3)
        self.right_root = np.zeros(3)

        # (x,y,z) for thumb, index, middle, ring, pinky
        self.left_pos = np.zeros((5, 3))
        self.right_pos = np.zeros((5, 3))

        self.left_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.right_quat = np.array([1.0, 0.0, 0.0, 0.0])

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

        self.offset = np.zeros(3)
        self.sim = sim
        self.hand_tracking = True
        self.enabled = False

    def is_active(self, side):
        if side == "left":
            return self.left_active
        else:
            return self.right_active

    def get_hands(self):
        wl, xl, yl, zl = self.left_quat
        wr, xr, yr, zr = self.right_quat
        left_rot = R.from_quat([xl, yl, zl, wl])
        right_rot = R.from_quat([xr, yr, zr, wr])

        rotz = R.from_euler("z", -90, degrees=True)

        adjusted_left_pos = (left_rot * rotz).apply(self.left_pos)
        adjusted_right_pos = (right_rot * rotz).apply(self.right_pos)
        return {
            "left": (np.vstack([self.left_root, adjusted_left_pos + self.left_root]) + self.offset, R.from_quat([xl, yl, zl, wl])),
            "right": (np.vstack([self.right_root, adjusted_right_pos + self.right_root]) + self.offset, R.from_quat([xr, yr, zr, wr])),
        }

    def recenter(self):
        desired_center = np.array([-0.3, 0, 1.2])
        cur_center = (self.left_root + self.right_root) / 2.0
        self.offset = desired_center - cur_center

class ManusReader:
    '''Client for MANUS mocap gloves for finger tracking.'''
    def __init__(self, target, read_freq=100):
        self.target = target
        self.read_freq = read_freq
        self.pedal = False
        self.data = None
    
    def run(self, wait=True):
        self.running = True
        self.thread = Thread(target=self.read_data_thread)
        self.thread.setDaemon(True)
        self.thread.start()

        if wait:
            logger.info(f'[ManusReader] Waiting for manus data...')
            while self.data is None:
                time.sleep(1/10)
    
    def stop(self):
        self.running = False
        self.thread.join()
    
    def read_data_thread(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.bind("tcp://localhost:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        message = self.socket.recv_string()
        self.data = json.loads(message)

        while self.running:
            if self.pedal:
                message = self.socket.recv_string()
                self.data = json.loads(message)
                self.update_target(self.data)
            time.sleep(1 / self.read_freq)
        
        self.socket.close()
        self.context.term()
    
    def update_target(self, data):
        assert 'left' in data and 'right' in data, "Manus data does not contain both hands."
        global_pos_left, global_rot_left = self.get_global_data(data['left'])
        global_pos_right, global_rot_right = self.get_global_data(data['right'])
        if not global_pos_left or not global_pos_right:
            return
        for i in range(5):
            offset = 2 if i == 0 else 3
            self.target.left_pos[i, :] = global_pos_left[1+i*4+offset]
            self.target.right_pos[i, :] = global_pos_right[1+i*4+offset]
            
    def get_global_data(self, data):
        if not data:
            return [], []
        
        local_pos, local_rot, scales = [], [], []
        for d in data:
            local_pos.append(np.array(d['pos']))
            local_rot.append(np.array(d['rot']))
            scales.append(np.array(d['scale']))
        
        # Solve for global positions
        # for trainsform chains of 4 nodes per 5 fingers
        # Each transform chain starts at the hand root
        global_pos = [np.zeros(3)]
        global_rot = [t3d.quaternions.quat2mat(np.array([1., 0., 0., 0.]))]
        for finger_idx in range(5):
            cum_pos = global_pos[0].copy()
            cum_rot = global_rot[0].copy()
            for node_idx in range(4):
                idx = 1 + finger_idx*4 + node_idx
                cur_pos = local_pos[idx] * scales[idx]
                cur_rot = t3d.quaternions.quat2mat(local_rot[idx])

                cum_pos += np.dot(cum_rot @ cur_rot if finger_idx == 0 else cum_rot, cur_pos)
                cum_rot = cum_rot @ cur_rot

                global_pos.append(cum_pos.copy())
                global_rot.append(cum_rot.copy())
        
        for i in range(len(global_pos)):
            global_pos[i][0] *= -1
        
        return global_pos, global_rot

class Quest3ControllerReader:
    RIGHT_LABELS = [
        "rightRoot",
        "rightThumb",
        "rightIndex",
        "rightMiddle",
        "rightRing",
        "rightPinky",
        "rightRootQuat",
    ]

    LEFT_LABELS = [
        "leftRoot",
        "leftThumb",
        "leftIndex",
        "leftMiddle",
        "leftRing",
        "leftPinky",
        "leftRootQuat",
    ]

    def __init__(self, target, sim, device='2G0YC1ZF860PXL', read_freq=200):
        dev = subprocess.check_output('adb devices'.split()).decode('utf-8').split('\n')[1]
        if len(dev) == 0 or 'no permissions' in dev:
            raise Exception("Quest3 is not properly connected!")

        self.client = AdbClient(host="127.0.0.1", port=5037)
        self.device = self.client.device(device)
        self.target = target
        self.read_freq = read_freq
        self.lock = Lock()
        self.data = {}
        self.pedal = False
        self.sim = sim
    
    def run(self):
        self.running = True
        self.thread = Thread(target=self.read_data_thread)
        self.thread.setDaemon(True)
        self.thread.start()
        logger.info(f'[Quest3ControllerReader] Started reading Quest3 controller data')
    
    def stop(self):
        self.running = False
        self.thread.join()
        logger.info(f'[Quest3ControllerReader] Stopped reading Quest3 controller data')
    
    def read_data_thread(self):
        self.device.shell('logcat', handler=self.process_logcat)
    
    def process_logcat(self, connection):
        
        # Flush old Quest data from logcat
        for _ in range(200):
            data = connection.read(4096*10)
            if not data:
                break
        logger.info(f'[Quest3Controller] Finished flushing old quest3 data.')
        
        while self.running:
            data = connection.read(1024*4)
            if not data:
                break

            data = data.decode('utf-8')
            unity_data = None
            if 'Unity' in data:
                data = data.split(' ')
                for d in data:
                    if 'rightRoot' in d:
                        unity_data = d.split('\n')[0]
                        break
            
            if unity_data:
                try:
                    unity_data = json.loads(unity_data)
                    for key in unity_data:
                        if isinstance(unity_data[key], str) and len(unity_data[key]) > 0:
                            unity_data[key] = json.loads(unity_data[key])
                    with self.lock:
                        data = unity_data
                        self.update_target(data)
                except:
                    pass

            time.sleep(1 / self.read_freq)
        
    def update_target(self, data):
        if not data:
            return
        
        if data['handTracking']:
            return
        
        rotZ = R.from_euler('z', 180, degrees=True)
        left_pos = rotZ.apply(np.array([
            -data[self.LEFT_LABELS[0]]["z"],
            data[self.LEFT_LABELS[0]]["x"],
            data[self.LEFT_LABELS[0]]["y"],
        ]))
        lq = data[self.LEFT_LABELS[-1]]

        right_pos = rotZ.apply(np.array([
            -data[self.RIGHT_LABELS[0]]["z"],
            data[self.RIGHT_LABELS[0]]["x"],
            data[self.RIGHT_LABELS[0]]["y"],
        ]))
        rq = data[self.RIGHT_LABELS[-1]]
        
        of = R.from_euler("ZYX", [90, 180, 90], degrees=True)
        of1 = R.from_euler("ZYX", [90, 180, 90], degrees=True)
        of2 = R.from_euler("ZYX", [90, 0, 0], degrees=True)
        of3 = R.from_euler("ZYX", [-90, 0, 0], degrees=True)

        ofx = R.from_euler('x', 45, degrees=True)
        ofy = R.from_euler('y', 0, degrees=True)
        ofz = R.from_euler('z', 180, degrees=True)
        of4neg = R.from_euler('z', -90, degrees=True)
        of4 = R.from_euler('z', 90, degrees=True)
        ofup = R.from_euler('y', -45, degrees=True)
        ofcenter = R.from_euler('z', 45, degrees=True)
        ofupneg = R.from_euler('y', 45, degrees=True)
        ofcenterneg = R.from_euler('z', -45, degrees=True)


        lq = rotZ * of1 * R.from_quat([lq["x"], lq["y"], lq["z"], lq["w"]]) * of3 * ofx * ofy * ofz * of4 * ofupneg * ofcenterneg
        rq = rotZ * of * R.from_quat([rq["x"], rq["y"], rq["z"], rq["w"]]) * of2 * ofx * ofy * ofz * of4neg * ofup * ofcenter

        lq = lq.as_quat()
        rq = rq.as_quat()
        
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

        #print(left_diff_mag, right_diff_mag)

        # Only update target if controller orientation is close to ee orientation        
        if self.pedal and (self.target.enabled or right_diff_mag < 0.5 and left_diff_mag < 0.5):
            self.target.enabled = True
            if self.target.right_origin_pos is None:
                self.target.right_origin_pos = self.target.right_root.copy()
                self.target.right_vr_origin_pos = right_pos.copy()
            cur_vr_pos = right_pos.copy()
            
            self.target.right_root = self.target.right_origin_pos + (
                cur_vr_pos - self.target.right_vr_origin_pos
            )
            self.target.right_quat[:] = scipy_to_quat(
                cur_right_vr_quat
            )

            if self.target.left_origin_pos is None:
                self.target.left_origin_pos = self.target.left_root.copy()
                self.target.left_vr_origin_pos = left_pos.copy()
            cur_vr_pos = left_pos.copy()
            self.target.left_root = self.target.left_origin_pos + (
                cur_vr_pos - self.target.left_vr_origin_pos
            )
            self.target.left_quat[:] = scipy_to_quat(
                cur_left_vr_quat
            )            
        else:
            self.target.enabled = False
            if self.target.right_origin_pos is not None:
                self.target.right_origin_pos = None
                self.target.right_origin_quat = None
                self.target.right_vr_origin_pos = None
                self.target.right_vr_origin_quat = None

                right_pos, right_quat = self.sim.get_root_pos("right")
                self.target.right_root[:] = right_pos
                self.target.right_quat[:] = right_quat
            if self.target.left_origin_pos is not None:
                self.target.left_origin_pos = None
                self.target.left_origin_quat = None
                self.target.left_vr_origin_pos = None
                self.target.left_vr_origin_quat = None

                left_pos, left_quat = self.sim.get_root_pos("left")
                self.target.left_root[:] = left_pos
                self.target.left_quat[:] = left_quat
    
    def reset_state(self):
        self.target.right_origin_pos = None
        self.target.right_vr_origin_pos = None
        self.target.right_origin_quat = None
        self.target.right_vr_origin_quat = None
        right_pos, right_quat = self.sim.get_root_pos("right")
        self.target.right_root[:] = right_pos
        self.target.right_quat[:] = right_quat

        self.target.left_origin_pos = None
        self.target.left_vr_origin_pos = None
        self.target.left_origin_quat = None
        self.target.left_vr_origin_quat = None
        left_pos, left_quat = self.sim.get_root_pos("left")
        self.target.left_root[:] = left_pos
        self.target.left_quat[:] = left_quat


class MocapPolicy:
    def __init__(self, env):
        self.target = MocapTarget(sim=env.sim)
        self.manus = ManusReader(self.target)
        self.quest3 = Quest3ControllerReader(self.target, env.sim)

        self.env = env
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
        self.manus.run()

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
        self.thread = Thread(target=self.submit_ik)
        self.thread.start()

        logger.info("[VRPolicy] Started VR policy")

    def stop(self):
        self.keep_running = False
        self.manus.stop()
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
        ret = pickle.loads(sol)
        return ret

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
            self.manus.pedal = self.env.is_active()
            self.quest3.pedal = self.env.is_active()

            if not (self.quest3.pedal and self.target.enabled):
                time.sleep(1.0 / 200)
                continue

            targets = self.env.sim.get_mocap_targets()
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
            targets = self.env.sim.get_mocap_targets()
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

    # ====== inference ======
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
    
def plot_points_and_frames(ax, global_pos, global_rot):
    ax.cla()
    if not global_pos:
        return

    for i in range(len(global_pos)):
        point = global_pos[i]
        rotation_matrix = global_rot[i]

        # Generate local coordinate frame axes
        s = 0.0
        local_x = s*rotation_matrix[:, 0]
        local_y = s*rotation_matrix[:, 1]
        local_z = s*rotation_matrix[:, 2]

        # Plot point
        ax.scatter(*point, color='r', s=50)

        # Plot local coordinate frame axes
        ax.quiver(*point, *local_x, color='g')
        ax.quiver(*point, *local_y, color='b')
        ax.quiver(*point, *local_z, color='k')
    
    for finger_idx in range(5):
        prev = 0
        for node_idx in range(4):
            idx = 1 + finger_idx*4 + node_idx
            x = [global_pos[prev][0], global_pos[idx][0]]
            y = [global_pos[prev][1], global_pos[idx][1]]
            z = [global_pos[prev][2], global_pos[idx][2]]
            ax.plot(x, y, z, color='k')
            prev = idx

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('Global Positions and Local Coordinate Frames')
    ax.set_xlim([-.1, .1])
    ax.set_ylim([-.1, .1])
    ax.set_zlim([-.1, .1])

    plt.pause(0.001)
    plt.draw()

if __name__ == '__main__':
    client = ManusReader()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    start = time.time()
    count = 0
    try:
        while True:
            global_pos, global_rot = client.get_global_pos()
            plot_points_and_frames(ax, global_pos, global_rot)
            time.sleep(0.001)
            count += 1
    except KeyboardInterrupt:
        pass
    end = time.time()
    print(f"FPS: {count / (end - start)}")
