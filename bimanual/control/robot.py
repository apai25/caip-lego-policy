import time
import redis
import pickle
import numpy as np
import multiprocessing as mp

from loguru import logger
from threading import Thread

from bimanual.control.ur3e import UR3EArm
from bimanual.control.abh import AbilityHand


class RobotController(mp.Process):
    """To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """

    def __init__(
        self, left_arm_ip, right_arm_ip, left_hand_port, right_hand_port, freq=300
    ):
        super(RobotController, self).__init__(name="RobotController")

        self.left_arm_ip = left_arm_ip
        self.right_arm_ip = right_arm_ip
        self.left_hand_port = left_hand_port
        self.right_hand_port = right_hand_port

        self.ready_event = mp.Event()
        self.finish_event = mp.Event()
        self.init_event = mp.Event()

        self.freq = freq

        self.r = redis.Redis(host="localhost", port=6379, db=0)

    ## ====== launch methods ======
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        logger.info(f"[RobotController] Controller process spawned at {self.pid}")

    def start_wait(self):
        self.ready_event.wait()
        logger.info(f"[RobotController] Controller process ready")

    def stop(self, wait=False):
        self.finish_event.set()
        if wait:
            self.stop_wait()

    def stop_wait(self):
        self.join()
        logger.info(f"[RobotController] Controller process stopped")

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ===== context manager =====
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ===== update methods ======
    def update_targets(self, action):
        self.r.set("action", pickle.dumps(action))

    def execute_update_targets(self):
        payload = self.r.get("action")
        if payload is None:
            return

        action = pickle.loads(payload)

        self.right_hand.update_target(np.rad2deg(action["right_fingers_cmd"]))
        self.left_hand.update_target(np.rad2deg(action["left_fingers_cmd"]))
        self.left_arm.update_target(action["left_arm_cmd"])
        self.right_arm.update_target(action["right_arm_cmd"])

    # ===== init methods =====
    def init(self, random=True, qpos={}):
        self.r.set("random_init", pickle.dumps((random, qpos)))
        self.init_event.set()
        state = pickle.loads(self.r.blpop("init_state", timeout=0)[1])
        logger.info(f"[RobotController] Initializing {state=}")
        return state

    def run_init(self):
        self.left_arm.servo_stop()
        self.right_arm.servo_stop()

        init_args = self.r.get("random_init")

        arms = [self.left_arm, self.right_arm]
        for arm in arms:
            if init_args is not None:
                random, qpos = pickle.loads(init_args)
                if arm.name in qpos:
                    # goto initial target qpos
                    arm.goto_init_target(qpos[arm.name], blocking=False)
                elif random:
                    # randomly initialize arms
                    arm.goto_random_target(blocking=False)
                else:
                    # default initialize arms
                    arm.reset(blocking=False)
            else:
                # default initialize arms
                arm.reset(blocking=False)

        self.right_hand.reset()
        self.left_hand.reset()

        self.left_arm.wait_until_stopped()
        self.right_arm.wait_until_stopped()
        time.sleep(1)

        self.left_arm.zero_force_sensor()
        self.right_arm.zero_force_sensor()

        state = {
            "left_arm": self.left_arm.get_joint_pos()[1],
            "right_arm": self.right_arm.get_joint_pos()[1],
        }
        self.init_event.clear()
        self.r.lpush("init_state", pickle.dumps(state))

    ## ===== main loop =====
    def run(self):
        self.left_arm = UR3EArm(self.left_arm_ip, "left")
        self.right_arm = UR3EArm(self.right_arm_ip, "right")
        self.left_hand = AbilityHand(self.left_hand_port, "left")
        self.right_hand = AbilityHand(self.right_hand_port, "right")
        self.r = redis.Redis(host="localhost", port=6379, db=0)
        self.ready_event.set()

        while not self.finish_event.is_set():

            # update targets
            self.execute_update_targets()

            # retarget
            if self.init_event.is_set():
                self.run_init()

            # command targets for arms
            # hands are already commanded by their own threads after updating targets
            self.left_arm.goto_target()
            self.right_arm.goto_target()

            # read sensors
            right_finger_sensor = self.right_hand.get_sensors()
            left_finger_sensor = self.left_hand.get_sensors()
            left_arm_sensor = self.left_arm.get_sensors()
            right_arm_sensor = self.right_arm.get_sensors()

            # send to redis for ipc
            self.r.set("right_finger_sensor", pickle.dumps(right_finger_sensor))
            self.r.set("left_finger_sensor", pickle.dumps(left_finger_sensor))
            self.r.set("left_arm_sensor", pickle.dumps(left_arm_sensor))
            self.r.set("right_arm_sensor", pickle.dumps(right_arm_sensor))

            time.sleep(1 / self.freq)

        self.left_hand.stop()
        self.right_hand.stop()
