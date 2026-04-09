import threading
import redis
import time
import pickle
import itertools
import cv2
from loguru import logger


class DataSync:
    """
    A class for synchronizing data based on timestamps
    each of the object it takes in has a timestamp and a buffer attribute
    we record data at a fixed frequency
    """

    def __init__(self, datasources: list, frequency=1 / 30):
        self.datasources = datasources
        self.f = frequency
        self.names = [str(d) for d in datasources]
        self.buffers = [d.buffer for d in datasources]
        self.timestamps = [d.timestamp for d in datasources]
        self.locks = [d.lock for d in datasources]
        self.indices_last = [d.index_last for d in datasources]
        self.active = False
        self.data = None
        self.data_lock = threading.Lock()
        self.thread = None
        self.warn_skip = False

    def start(self):
        self.thread = threading.Thread(target=self.start_ros)
        self.thread.setDaemon(True)
        self.thread.start()
        self.wait_until_ready()

    def start_ros(self):
        """
        select timestamp the same way as ros does
        """

        self.active = True
        while self.active:
            time.sleep(self.f / 2)
            stamp = time.time()
            [l.acquire() for l in self.locks]
            stamps, skip_one = [], False
            for qi, queue in enumerate(self.timestamps):
                topic_stamps = []
                for s in queue:
                    stamp_delta = abs(s - stamp)
                    if stamp_delta > self.f:
                        continue  # far over the slop
                    topic_stamps.append((s, stamp_delta))
                if not topic_stamps:
                    [l.release() for l in self.locks]
                    if self.warn_skip:
                        logger.warning(f"[DataSync] Skipping {self.names[qi]}")
                    skip_one = True
                    break
                topic_stamps = sorted(topic_stamps, key=lambda x: x[1])
                stamps.append(topic_stamps)
            if skip_one:
                continue
            for vv in itertools.product(*[next(iter(zip(*s))) for s in stamps]):
                vv = list(vv)
                # insert the new message
                qt = list(zip(self.buffers, self.timestamps, vv))
                if ((max(vv) - min(vv)) < self.f) and (
                    len([1 for b, ts, t in qt if t not in ts]) == 0
                ):
                    msgs = [(t, b[ts.index(t)]) for b, ts, t in qt]
                    self.data_lock.acquire()
                    self.data = msgs
                    self.data_lock.release()
                    break  # fast finish after the synchronization
            [l.release() for l in self.locks]

    def stop(self):
        self.active = False
        self.data = None

    def get_last(self):
        self.data_lock.acquire()
        last = self.data
        self.data_lock.release()
        return last

    def wait_until_ready(self):
        logger.info(f"[DataSync] Waiting for data read buffers to fill up... {[len(ds.buffer) for ds in self.datasources]}")
        while not all([len(ds.buffer) < ds.max_buffer_len for ds in self.datasources]):
            time.sleep(0.1)
        logger.info(f"[DataSync] Data buffers filled!")

        logger.info(f"[DataSync] Waiting for data to sync...")
        while self.data is None:
            time.sleep(0.1)
        logger.info(f"[DataSync] Data synced!")

    def reset(self):
        for ds in self.datasources:
            ds.clear_buffer()
        self.data = None
        self.wait_until_ready()


class WebcamSensor:
    """
    A webcam sensor that supports multithreading
    """

    def __init__(
        self, cam_id, name, cam_res=[640, 480], set_fps=60, max_buffer_len=10
    ) -> None:
        """
        cam_id: read from v4l2-ctl --list-devices
        cam_res: (width, height)
        set_fps: set the fps of the camera
        max_buffer_len: the maximum number of frames to store in the buffer
        """
        self.cam_id = cam_id
        self.name = name
        self.cap = cv2.VideoCapture(cam_id)
        width, height = cam_res
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, height)
        self.cap.set(cv2.CAP_PROP_FPS, set_fps)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.active = False
        self.buffer = []
        self.timestamp = []
        self.index_last = [0]
        self.max_buffer_len = max_buffer_len
        self.lock = threading.Lock()
        self.fps = set_fps
        self.thread = None

    def __str__(self):
        return f"Webcam{self.name.capitalize()}"

    def start(self):
        self.thread = threading.Thread(target=self.start_read)
        self.thread.setDaemon(True)
        self.thread.start()

    def start_read(self):
        self.flush()
        self.active = True
        prev = time.time()
        while self.active:
            ret, frame = self.cap.read()
            if frame is None:
                logger.warning(f'[WebcamDatastream] Camera {self.name} has empty frame!')
                time.sleep(1/120)
                continue
            self.buffer.append(frame[:, :, ::-1])
            self.fps = 1.0 / (time.time() - prev)
            prev = time.time()
            self.timestamp.append(time.time())
            if self.lock.acquire(blocking=False):
                if len(self.buffer) > self.max_buffer_len:
                    del (
                        self.buffer[: -self.max_buffer_len],
                        self.timestamp[: -self.max_buffer_len],
                    )
                self.index_last[0] = len(self.buffer) - 1
                self.lock.release()
            prev = time.time()
            time.sleep(1 / 120)

    def end_read(self):
        self.active = False
        self.buffer = []
        self.timestamp = []

    def clear_buffer(self):
        self.buffer.clear()
        self.timestamp.clear()

    def flush(self, num_ims=5):
        for i in range(num_ims):
            _, _ = self.cap.read()

    def __del__(self):
        self.cap.release()


class AbilityHandDS:
    """
    A thread for reading robot joint states and velocity data at a fixed frequency
    """

    def __init__(self, name, max_buffer_len=10, freq=1 / 60):
        self.name = name
        self.buffer, self.timestamp = [], []
        self.max_buffer_len = max_buffer_len
        self.lock = threading.Lock()
        self.active = False
        self.freq = freq
        self.index_last = [0]
        self.r = redis.Redis(host="localhost", port=6379, db=0)
        self.thread = None

    def __str__(self):
        return f"AbilityHand{self.name.capitalize()}"

    def start(self):
        self.thread = threading.Thread(target=self.start_read)
        self.thread.setDaemon(True)
        self.thread.start()

    def current_joint_positions(self):
        value = self.r.get(f"{self.name}_finger_sensor")
        while True:
            value = self.r.get(f"{self.name}_finger_sensor")
            if value is not None:
                ct, joints, touch = pickle.loads(value)
                if len(self.timestamp) > 0 and ct == self.timestamp[-1]:
                    pass
                elif len(joints) != 0:
                    break
            time.sleep(1 / 120)
        return ct, joints, touch

    def start_read(self):
        self.active = True
        while self.active:
            if self.freq is not None:
                time.sleep(self.freq)
            ct, joints, touch = self.current_joint_positions()
            self.buffer.append((joints, touch))
            self.timestamp.append(ct)
            if self.lock.acquire(blocking=False):
                if len(self.buffer) > self.max_buffer_len:
                    del (
                        self.buffer[: -self.max_buffer_len],
                        self.timestamp[: -self.max_buffer_len],
                    )
                self.index_last[0] = len(self.buffer) - 1
                self.lock.release()

    def clear_buffer(self):
        self.buffer.clear()
        self.timestamp.clear()

    def end_read(self):
        self.active = False
        self.buffer = []
        self.timestamp = []


class UR3EArmDS:
    """
    A thread for reading robot joint states and velocity data at a fixed frequency
    """

    def __init__(self, name, max_buffer_len=10, freq=1 / 60):
        self.name = name
        self.buffer, self.timestamp = [], []
        self.max_buffer_len = max_buffer_len
        self.lock = threading.Lock()
        self.active = False
        self.freq = freq
        self.index_last = [0]
        self.r = redis.Redis(host="localhost", port=6379, db=0)
        self.thread = None

    def __str__(self):
        return f"UR3EArm{self.name.capitalize()}"

    def start(self):
        self.thread = threading.Thread(target=self.start_read)
        self.thread.setDaemon(True)
        self.thread.start()

    def current_joint_positions(self):
        value = self.r.get(f"{self.name}_arm_sensor")
        while value is None:
            value = self.r.get(f"{self.name}_arm_sensor")
            time.sleep(1.0 / 60)
        ct, qpos, qvel, tcppos, tcpvel, tcpforce = pickle.loads(value)
        return ct, qpos, qvel, tcppos, tcpvel, tcpforce

    def start_read(self):
        self.active = True
        while self.active:
            if self.freq is not None:
                time.sleep(self.freq)
            ct, qpos, qvel, tcppos, tcpvel, tcpforce = self.current_joint_positions()
            self.buffer.append((qpos, qvel, tcppos, tcpvel, tcpforce))
            self.timestamp.append(ct)
            if self.lock.acquire(blocking=False):
                if len(self.buffer) > self.max_buffer_len:
                    del (
                        self.buffer[: -self.max_buffer_len],
                        self.timestamp[: -self.max_buffer_len],
                    )
                self.index_last[0] = len(self.buffer) - 1
                self.lock.release()

    def clear_buffer(self):
        self.buffer.clear()
        self.timestamp.clear()

    def end_read(self):
        self.active = False
        self.buffer = []
        self.timestamp = []
