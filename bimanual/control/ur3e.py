import rtde_control
import rtde_receive
import rtde_io
import time
import numpy as np
import time

from loguru import logger
from typing import List


# play data
DEFAULT_UR3E_POS = {
    "right": np.array([
        -2.198981587086813,
        -2.2018891773619593,
        -1.534730076789856,
        -0.1098826688579102,
        1.2620022296905518,
        0,
    ]),
    "left": np.array([
        -4.065179173146383,
        -0.8556114000133057,
        1.419995133076803,
        -3.108495374719137,
        -1.3419583479510706,
        0,
    ]),
}

## kitchen data
#DEFAULT_UR3E_POS = {
#    "right": np.array([
#        -2.425814930592672,
#        -1.7651893101134242,
#        -2.0178232192993164,
#        -0.08198674142871099,
#        1.2157135009765625,
#        -0.39373714128603154
#    ]),
#    "left": np.array([
#        -3.857370376586914,
#        -1.376403343476369,
#        2.0178232192993164,
#        -3.059605912161082,
#        -1.2157135009765625,
#        0.39373714128603154
#    ]),
#}


class UR3EArm:
    def __init__(self, ip, name):
        self.name = name
        self.ur_c = rtde_control.RTDEControlInterface(ip)
        self.ur_r = rtde_receive.RTDEReceiveInterface(ip)
        self.ur_io = rtde_io.RTDEIOInterface(ip)

        self.default_pos = DEFAULT_UR3E_POS[name]
        self.curr_target = self.get_joint_pos()[1]

        self.ur_c.zeroFtSensor()
        logger.info(f"[UR3EArm] Initialized {name} arm")

    # ====== read sensors ======
    def get_joint_pos(self):
        q = self.ur_r.getActualQ()
        return time.time(), q

    def get_tcp_torque(self):
        torque = np.array(self.ur_r.getActualTCPForce())
        return torque

    def get_joint_torques(self):
        torques = np.array(self.ur_c.getJointTorques())
        return torques

    def get_sensors(self):
        qpos = self.ur_r.getActualQ()
        qvel = self.ur_r.getActualQd()
        tcppos = self.ur_r.getActualTCPPose()
        tcpvel = self.ur_r.getActualTCPSpeed()
        tcpforce = self.ur_r.getActualTCPForce()
        return time.time(), qpos, qvel, tcppos, tcpvel, tcpforce

    # ====== set targets ======
    def update_target(self, action):
        self.curr_target[:] = action

    # ====== command targets ======
    def goto_random_target(self, blocking=True):
        self.curr_target[:] = self.default_pos + np.random.uniform(-0.2, 0.2, 6)
        self.move_joint(self.curr_target, asyn=not blocking)

    def goto_init_target(self, target, blocking=True):
        self.curr_target[:] = target
        self.move_joint(self.curr_target, asyn=not blocking)

    def reset(self, blocking=True):
        self.move_joint(self.default_pos, asyn=not blocking)

    def wait_until_stopped(self):
        while not self.is_stopped():
            time.sleep(1 / 30)
        self.curr_target[:] = self.get_joint_pos()[1]

    def goto_target(self):
        self.servo_joint(self.curr_target)
    
    def restart_forcemode(self):
        self.ur_c.forceModeStop()
        self.ur_c.zeroFtSensor()
        self.ur_c.forceMode([0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], 2, [0.1,0.1,0.1,0.1,0.1,0.1])
    
    def zero_force_sensor(self):
        self.ur_c.zeroFtSensor()

    # ====== ur api ======
    def servo_joint(self, target, time=0.002, lookahead_time=0.15, gain=150):
        # def servo_joint(self, target, time=0.002, lookahead_time=0.1, gain=150):
        """
        Servoj can be used for online realtime control of joint positions.
        It is designed for movements over greater distances.
        time: time where the command is controlling the robot. The function is blocking for time t [S]
        lookahead_time: time [S], range [0.03,0.2] smoothens the trajectory with this lookahead time. A low value gives fast reaction, a high value prevents overshoot.
        gain: proportional gain for following target position, range [100,2000]. The higher the gain, the faster reaction the robot will have.
        """
        # cur_torques = np.mean(self.torques, axis=0)
        # correction = self.kp * cur_torques + self.kd * (self.prev_torques - cur_torques)
        # self.prev_torques[:] = cur_torques
        # logger.info(f"[UR3EArm] Servoing to {target} with correction {correction}")
        # self.ur_c.servoJ(target + correction, 0.0, 0.0, time, lookahead_time, gain)
        # self.torques.append(self.get_joint_torques())
        # if len(self.torques) > 50:
        #     del self.torques[:-20]

        self.ur_c.servoJ(target, 0.0, 0.0, time, lookahead_time, gain)

    def move_joint(self, target, interp="joint", vel=1.0, acc=1, asyn=False):
        """
        the movej() command offers you a complete trajectory planning with acceleration, de-acceleration etc.
        It is designed for movements over greater distances.
        """
        if interp == "joint":
            self.ur_c.moveJ(target, vel, acc, asyn)
        elif interp == "tcp":
            self.ur_c.moveL_FK(target, vel, acc, asyn)
        else:
            raise KeyError("interpolation muct be in joint or tcp space")

    def speed_tcp(self, vel, acc=10, t=0):
        """
        Accelerate linearly in tcp space and continue with constant tcp speed.
        """
        self.ur_c.speedL(vel, acc, time=t)

    def speed_joint(self, vel, acc=10):
        """
        Accelerate linearly in joint space and continue with constant joint speed.
        """
        self.ur_c.speedJ(vel, acc)

    # TODO verify the below two are correct
    def move_joint_path(
        self, waypoints: List[np.ndarray], vels, accs, blends, asyn=False
    ):
        """
        waypoints input should be N*6 for joint
        The size of the blend radius is per default a shared value for all the waypoint.
        A smaller value will make the path turn sharper whereas a higher value will make the path smoother.
        """
        assert isinstance(waypoints, np.ndarray), "waypoints must be a numpy array"
        assert waypoints.shape[1] == 6, "dimension of waypoints must be Nx6"
        path = np.hstack(
            (
                waypoints,
                np.array(vels)[..., None],
                np.array(accs)[..., None],
                np.array(blends)[..., None],
            )
        )
        self.ur_c.moveJ(path, asyn)

    def get_joints(self):
        q = self.ur_r.getActualQ()
        return q

    def get_tcp_pose(self):
        # p[:3] is x,y,z
        # p[3:6] is axis angle rotation
        p = self.ur_r.getActualTCPPose()
        return p

    def is_stopped(self):
        return self.ur_c.isSteady()

    def move_until_contact(
        self, vel, thres, acc=0.25, direction=np.array((0, 0, 1, 0, 0, 0))
    ):
        assert len(vel) == 6, "dimension of vel mush be 6"
        self.ur_c.speedL(vel, acceleration=acc)
        time.sleep(0.5)
        startforce = np.array(self.ur_r.getActualTCPForce())
        while True:
            force = np.array(self.ur_r.getActualTCPForce())
            if np.linalg.norm((startforce - force).dot(direction)) > thres:
                break
            time.sleep(0.008)
        self.ur_c.speedStop()

    def move_linear(self, start, end, vel, thres=17, acc=0.25):
        assert (
            len(start) == 3 and len(end) == 3
        ), "dimension of start and goal must be 3"
        assert (
            len(vel) == 3
        ), "vel is for x,y,z direction and the first nonzero one would be used"

        if abs(end[0] - start[0]) > 0.01:
            direc_x = vel[0]
            direc_y = (
                np.clip((end[1] - start[1]) / (end[0] - start[0] + 1e-6), -10, 10)
                * direc_x
            )
            direc_z = (
                np.clip((end[2] - start[2]) / (end[0] - start[0] + 1e-6), -10, 10)
                * direc_x
            )
        elif abs(end[1] - start[1]) > 0.01:
            direc_y = vel[1]
            direc_x = (
                np.clip((end[0] - start[0]) / (end[1] - start[1] + 1e-6), -10, 10)
                * direc_y
            )
            direc_z = (
                np.clip((end[2] - start[2]) / (end[1] - start[1] + 1e-6), -10, 10)
                * direc_y
            )
        else:
            direc_z = vel[2]
            direc_x = (
                np.clip((end[0] - start[0]) / (end[2] - start[2] + 1e-6), -10, 10)
                * direc_z
            )
            direc_y = (
                np.clip((end[1] - start[1]) / (end[2] - start[2] + 1e-6), -10, 10)
                * direc_z
            )

        dire = [direc_x, direc_y, direc_z, 0, 0, 0]
        self.ur_c.speedL(dire, accelaration=acc)
        time.sleep(0.5)
        startforce = np.array(self.ur_r.getActualTCPForce())
        while True:
            force = np.array(self.ur_r.getActualTCPForce())
            if (
                np.linalg.norm((startforce - force).dot(dire)) > thres
                or np.linalg.norm(self.get_pose().translation - end) < 0.01
            ):
                break
            time.sleep(0.008)
        self.ur_c.speedStop()

    def start_freedrive(self):
        self.ur_c.teachMode()

    def stop_freedrive(self):
        self.ur_c.endTeachMode()

    def stop_script(self):
        self.ur_c.stopScript()

    def reupload_script(self):
        self.ur_c.reuploadScript()
    
    def is_connected(self):
        return self.ur_c.isConnected()
    
    def reconnect(self):
        if self.is_connected():
            return

    def disconnect(self):
        self.ur_c.disconnect()
        self.ur_r.disconnect()
        self.ur_io.disconnect()

    def servo_stop(self):
        self.ur_c.servoStop()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    # url = UR3EArm("10.42.1.100", "left")
    urr = UR3EArm("10.42.0.100", "right")
    url = UR3EArm("10.42.1.100", "left")
    
    urr.move_joint(DEFAULT_UR3E_POS['right'])
    url.move_joint(DEFAULT_UR3E_POS['left'])
    
    # print(urr.get_joint_pos())
    # print(url.get_joint_pos())

    # urr.servo_joint([-2.3142762819873255, -2.1950208149352015, -1.6021356582641602, -0.28126056612048345, 1.272944092750549, 0.020581722259521484])
    # url.servo_joint([-4.260195795689718, -0.6665948194316407, 1.3996503988849085, -3.0132209263243617, -1.4481328169452112, 0.08825898170471191])
    
    # while True:
        # time.sleep(1/10)

    # startforce = np.array(urr.ur_r.getActualTCPForce())

    '''
    joints = []
    urr.start_freedrive()
    for i in range(16):
        input(f'Press enter to capture {i}: ')
        joints.append(urr.get_joint_pos()[1])

    print(joints)
    '''

    """
    # Initialize your figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the limits of your 3D plot
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])

    # Initialize a vector, here starting as a zero vector
    vector = ax.quiver(0, 0, 0, 0, 0, 0)

    # Update function for the animation
    def update_vector(num):
        # Simulate reading new vector components, here just rotating for demonstration
        force = np.array(urr.ur_r.getActualTCPForce())
        diff = startforce - force
        new_dx, new_dy, new_dz, _, _, _ = diff
        print(new_dz)

        ax.cla()
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])

        ax.quiver(0, 0, 0, new_dx, new_dy, new_dz, color='blue')

    # Creating animation
    ani = FuncAnimation(fig, update_vector, frames=range(100), interval=100)

    plt.show()
    """

    '''
    pose = np.array(urr.ur_r.getActualTCPPose())

    while True:
        force = np.array(urr.ur_r.getActualTCPForce())
        diff = startforce - force

        target = pose.copy()
        target[:] -= 0.0005 * diff[:]

        urr.ur_c.servoL(target.copy(), 0, 0, 0.002, 0.1, 300)
        print(pose, target - pose)
        print()
        time.sleep(1/500)
    '''

    '''
    qpos = np.array(urr.get_joint_pos()[1])
    startforce = np.array(urr.ur_c.getJointTorques())
    while True:
        force = np.array(urr.ur_c.getJointTorques())
        diff = startforce - force
        target = qpos.copy()
        target -= 0.001 * diff
        urr.ur_c.servoJ(target, 0, 0, 0.002, 0.1, 300)
        time.sleep(1/500)
    '''

    """
    while True:
        urr.servo_joint(DEFAULT_UR3E_POS["right"])

        force = urr.ur_r.getActualTCPForce()
        print(np.linalg.norm((startforce - force).dot(direction)))

        time.sleep(1 / 500)
    """

    # qpos = np.array(urr.get_joint_pos()[1])
    # forces = []
    # for i in range(10):
    #     forces.append(np.array(urr.ur_c.getJointTorques()))
    #     time.sleep(1/500)

    # while True:
    #     force = np.array(urr.ur_c.getJointTorques())
    #     diff = startforce - np.mean(forces, axis=0)
    #     forces.append(force)

    #     print(diff)
    #     # mask = np.abs(diff) > np.array([10,7,4,0.6,0.5,4])
    #     # mask = mask * np.array([0,1,1,1,1,1])
    #     k = np.array([0.001, 0.001, 0.01, 0.1, 0.1, 0.01])
    #     target = qpos.copy()
    #     target -= k * diff # * mask
    #     urr.ur_c.servoJ(target, 0, 0, 0.002, 0.15, 150)

    #     time.sleep(1/100)

    #     if len(forces) > 30:
    #         del forces[:-10]

    # urr.ur_c.zeroFtSensor()
    # # target = np.array(urr.ur_r.getActualQ())
    # target = np.array(urr.ur_r.getActualTCPPose())
    # # urr.ur_c.forceMode(urr.ur_r.getActualTCPPose(), [1,1,1,1,1,1], [0,0,0,0,0,0], 2, [1,1,1,1,1,1])
    # urr.ur_c.forceMode([0,0,0,0,0,0], [1,1,1,0,0,0], [0,0,0,0,0,0], 2, [1,1,1,1,1,1])
    # try:
    #     while True:
    #         urr.ur_c.servoL(target, 0.0, 0.0, 0.002, 0.15, 150)
    #         target[1] -= 0.0001
    #         time.sleep(1/500)
    # except KeyboardInterrupt:
    #     # urr.ur_c.forceModeStop()
    #     pass

