import numpy as np
from scipy.spatial.transform import Rotation as R


def scipy_to_quat(rot):
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def quat_to_scipy(q):
    w, x, y, z = q
    return R.from_quat([x, y, z, w])


def rotation_matrix_to_euler(R):
    """
    Convert a rotation matrix to yaw, pitch, and roll angles.

    Parameters:
    R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
    tuple: A tuple containing yaw, pitch, and roll angles in radians.
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    return np.array(
        [
            np.arctan2(R[2, 1], R[2, 2] + 1e-5),
            np.arctan2(-R[2, 0], sy + 1e-5),
            np.arctan2(R[1, 0], R[0, 0] + 1e-5),
        ]
    )
