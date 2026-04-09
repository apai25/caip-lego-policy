"""Client for MANUS mocap gloves."""
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import transforms3d as t3d
import zmq

from threading import Thread, Lock

class ManusClient:

    def __init__(self):
        self.thread = Thread(target=self.read_data_thread)
        self.thread.setDaemon(True)
        self.lock = Lock()
        self.data = []
        self.read_freq = 100
        self.active = True
        self.thread.start()
    
    def __del__(self):
        self.stop()
    
    def stop(self):
        self.active = False
        self.thread.join()
    
    def read_data_thread(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.bind("tcp://localhost:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        while self.active:
            message = self.socket.recv_string()
            with self.lock:
                print(message)
                self.data = json.loads(message)
            time.sleep(1 / self.read_freq)
        
        self.socket.close()
        self.context.term()

    def get_global_pos(self):
        with self.lock:
            data = self.data
        
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
            for node_idx in range(3):
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
        for node_idx in range(3):
            idx = 1 + finger_idx*3 + node_idx
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
    client = ManusClient()

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
