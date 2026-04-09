import numpy as np
import serial
import threading
import time

from loguru import logger
from threading import Thread

from bimanual.control.abh_api import (
    farr_to_dposition,
    parse_hand_data,
    PPP_stuff,
    unstuff_PPP_stream,
)


class AbilityHand:
    def __init__(self, port, name, baud=1000000):
        self.name = name
        self.serial = serial.Serial(port, baud, timeout=0, write_timeout=0)
        self.rPos, self.rI, self.rV, self.rFSR = [], [], [], []
        self.bytebuffer = bytes([])
        self.stuff_buffer = np.array([])
        self.default_target = np.array([10.0, 10.0, 10.0, 10.0, 10.0, -40.0])
        self.curr_target = self.default_target.copy()
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.num_writes = 0
        self.num_reads = 0
        self.read_time = time.time()
        self.active = False
        self.thread = Thread(target=self.start)
        self.thread.setDaemon(True)
        self.thread.start()

        logger.info(f"[AbilityHand] Initialized {name} hand")

    # ====== read sensors ======
    def get_joint_pos(self):
        with self.read_lock:
            curr_pos = self.rPos.copy()
            read_time = self.read_time
        return read_time, np.deg2rad(curr_pos)

    def get_touch(self):
        with self.read_lock:
            curr_touch = self.rFSR.copy()
            read_time = self.read_time
        return read_time, curr_touch

    def get_sensors(self):
        with self.read_lock:
            curr_pos = self.rPos.copy()
            curr_touch = self.rFSR.copy()
            read_time = self.read_time
        return read_time, np.deg2rad(curr_pos), curr_touch

    # ====== set targets ======
    def update_target(self, action):
        with self.write_lock:
            self.curr_target[:] = action

    def reset(self):
        with self.write_lock:
            self.curr_target[:] = self.default_target

    # ====== low level communication ======
    def read(self):
        try:
            nb = bytes([])
            while len(nb) == 0:
                nb = self.serial.read(512)

            self.bytebuffer = self.bytebuffer + nb
            npbytes = np.frombuffer(self.bytebuffer, np.uint8)
            for b in npbytes:
                payload, self.stuff_buffer = unstuff_PPP_stream(b, self.stuff_buffer)
                if len(payload) != 0:
                    rPos, rI, rV, rFSR = parse_hand_data(payload)
                    if rPos.size > 0:
                        with self.read_lock:
                            self.rPos, self.rI, self.rV, self.rFSR = rPos, rI, rV, rFSR
                            self.read_time = time.time()
                        self.num_reads += 1
                        self.bytebuffer = bytes([])
                        self.stuff_buffer = np.array([])
        except Exception as e:
            logger.info(f"{e}")

    def write(self, fpos):
        try:
            msg = farr_to_dposition(0x50, fpos, 1)
            self.serial.write(PPP_stuff(bytearray(msg)))
            self.num_writes += 1
        except Exception as e:
            logger.info(f"{e}")

    # ====== launch methods ======
    def start(self):
        self.active = True
        while self.active:
            with self.write_lock:
                target = self.curr_target.copy()
            self.write(target)
            self.read()
            time.sleep(1 / 300)

    def stop(self):
        self.active = False
