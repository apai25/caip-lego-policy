#!/usr/bin/env python3

"""Visualize camera streams."""

import copy
import cv2
import datetime
import itertools
import numpy as np
import os
import sys
import threading
import time

from PIL import Image

class WebcamSensor:
    """
    A webcam sensor that supports multithreading
    """

    def __init__(
        self, cam_id, cam_res=[640, 480], set_fps=60, max_buffer_len=10
    ) -> None:
        """
        cam_id: read from v4l2-ctl --list-devices
        cam_res: (width, height)
        set_fps: set the fps of the camera
        max_buffer_len: the maximum number of frames to store in the buffer
        """
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

    def start_read(self):
        self.flush()
        self.active = True
        while self.active:
            ret, frame = self.cap.read()
            if frame is None:
                print("empty frame")
                time.sleep(1/120)
                continue
            self.buffer.append(frame)
            self.timestamp.append(time.time())
            if self.lock.acquire(blocking=False):
                if len(self.buffer) > self.max_buffer_len:
                    del (
                        self.buffer[: -self.max_buffer_len],
                        self.timestamp[: -self.max_buffer_len],
                    )
                self.index_last[0] = len(self.buffer) - 1
                self.lock.release()

    def end_read(self):
        self.active = False
        self.buffer = []
        self.timestamp = []

    def flush(self, num_ims=5):
        for i in range(num_ims):
            _, _ = self.cap.read()

    def __del__(self):
        self.cap.release()


class DataSync:
    """
    A class for synchronizing data based on timestamps
    each of the object it takes in has a timestamp and a buffer attribute
    we record data at a fixed frequency
    """

    def __init__(self, datasources: list, frequency=1 / 30):
        self.datasources = datasources
        self.f = frequency
        self.buffers = [d.buffer for d in datasources]
        self.timestamps = [d.timestamp for d in datasources]
        self.locks = [d.lock for d in datasources]
        self.indices_last = [d.index_last for d in datasources]
        self.active = False
        self.data = None
        self.data_lock = threading.Lock()

    def start_ros(self):
        """
        select timestamp the same way as ros does
        """
        self.active = True
        while self.active:
            time.sleep(self.f)
            stamp = time.time()
            [l.acquire() for l in self.locks]
            stamps, skip_one = [], False
            for queue in self.timestamps:
                topic_stamps = []
                for s in queue:
                    stamp_delta = abs(s - stamp)
                    if stamp_delta > self.f:
                        continue  # far over the slop
                    topic_stamps.append((s, stamp_delta))
                if not topic_stamps:
                    [l.release() for l in self.locks]
                    print("skipping one")
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
                    msgs = [(b[ts.index(t)], t) for b, ts, t in qt]
                    self.data_lock.acquire()
                    self.data = msgs
                    self.data_lock.release()
                    break  # fast finish after the synchronization
            [l.release() for l in self.locks]
            # print the variance in the timestamps from different sources
            # var = np.var([t for b, t in self.data])
            # print(f"Time diff to record (should be close to {self.f}): ", [stamp-t for _, t in self.data])
            # print("Timestamp Variance (in seconds): ", var)
            # print("Time: ", [t for b, t in self.data])
            # print("Data: ", [b for b, t in self.data])

    def stop(self):
        self.active = False
        self.data = None

    def get_last(self):
        self.data_lock.acquire()
        last = self.data
        self.data_lock.release()
        return last


def multi_thread(datastreams):
    print("Starting data streams...")
    data_read_threads = [threading.Thread(target=ds.start_read) for ds in datastreams]
    data_sync = DataSync(datastreams, frequency=1 / 30)
    data_sync_thread = threading.Thread(target=data_sync.start_ros)
    for drt in data_read_threads:
        drt.start()
    while not all([len(ds.buffer) == ds.max_buffer_len for ds in datastreams]):
        time.sleep(0.1)
    data_sync_thread.start()
    while data_sync.get_last() is None:
        time.sleep(0.1)

    time.sleep(1)

    (rgb_left, rgb_head, rgb_right) = data_sync.get_last()
    rgb_left = np.rot90(rgb_left[0], 2)
    rgb_right = np.rot90(rgb_right[0], 2)
    rgb_head = rgb_head[0]
    im = np.concatenate([rgb_left, rgb_head, rgb_right], axis=1)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"cams_{formatted_datetime}.jpeg"

    img = Image.fromarray(im[:, :, ::-1])
    img.save(os.path.join('img', fname))

    print("Viewing images...")
    while True:
        (rgb_left, rgb_head, rgb_right) = data_sync.get_last()
        rgb_left = np.rot90(rgb_left[0], 2)
        rgb_right = np.rot90(rgb_right[0], 2)
        rgb_head = rgb_head[0]
        im = np.concatenate([rgb_left, rgb_head, rgb_right], axis=1)
        cv2.imshow("cams", im)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

    print("Ending data streams...")
    data_sync.stop()
    data_sync_thread.join()
    time.sleep(0.1)
    for ds in datastreams:
        ds.end_read()
    for drt in data_read_threads:
        drt.join()


def get_cam_ids(serial_nums):
    cam_ids = []
    for sn in serial_nums:
        v4l2_path = f"/dev/v4l/by-id/usb-046d_Logitech_BRIO_{sn}-video-index0"
        vid_path = os.readlink(v4l2_path)
        cam_id = int(os.path.basename(vid_path).split("video")[-1])
        cam_ids.append(cam_id)
    return cam_ids


def view_cameras():
    # cam_ids: left, head, right
    serial_nums = ["EF834C70", "7F90B826", "2C7356D1"] 
    cam_ids = get_cam_ids(serial_nums)
    datastreams = [WebcamSensor(cam_id, max_buffer_len=5) for cam_id in cam_ids]
    multi_thread(datastreams)


if __name__ == "__main__":
    view_cameras()
