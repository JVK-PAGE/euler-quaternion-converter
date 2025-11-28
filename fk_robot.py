# this is for task 2
#!/usr/bin/env python3
"""
fk_robot.py

Forward kinematics for a 4-revolute-joint robot where each joint axis
is perpendicular to the previous. Flexible: provide any axis sequence
(e.g. ['z','y','z','y']) and link lengths.

Convention:
 - Each joint: rotate about axis_i by theta_i (radians), then translate along local +X by L_i.
 - Homogeneous transforms (4x4) are used.
 - Returns end-effector position in base frame: (x, y, z).
"""

import numpy as np
import math

def rot_matrix(axis, theta):
    """Return 3x3 rotation matrix for rotation about axis ('x','y','z') by theta (radians)."""
    c = math.cos(theta)
    s = math.sin(theta)
    if axis == 'x':
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)
    if axis == 'y':
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)
    if axis == 'z':
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
    raise ValueError("axis must be one of 'x','y','z'")

def homogeneous_transform(R, t):
    """Construct 4x4 homogeneous transform from 3x3 R and 3x1 t."""
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T

def trans_x(L):
    """Homogeneous translation along local +X by L."""
    return homogeneous_transform(np.eye(3), [L, 0.0, 0.0])

def fk_forward(joint_angles, joint_axes, link_lengths):
    """
    Compute forward kinematics.
    - joint_angles: list or array of N angles in radians [theta1, theta2, ...]
    - joint_axes: list of N axes, each 'x','y', or 'z'
    - link_lengths: list/array of N link lengths (distance from joint i to next joint along local +X)
    Returns:
      - end_pos: (x, y, z) coordinates of end-effector in base frame
      - T_total: final 4x4 homogeneous transform from base to end-effector
      - intermediate_transforms: list of 4x4 transforms after each joint (useful for visualization)
    """
    assert len(joint_angles) == len(joint_axes) == len(link_lengths), "All inputs must have the same length N"
    N = len(joint_angles)

    T = np.eye(4, dtype=float)  # base frame
    intermediate = [T.copy()]

    for i in range(N):
        axis = joint_axes[i]
        theta = float(joint_angles[i])
        L = float(link_lengths[i])

        R = rot_matrix(axis, theta)   # rotation in current local frame
        T_rot = homogeneous_transform(R, [0,0,0])
        T = T @ T_rot                 # apply rotation about local axis
        T = T @ trans_x(L)            # translate along local +X by link length
        intermediate.append(T.copy())

    # end-effector position (origin of end-effector frame in base coordinates)
    end_pos_h = T @ np.array([0.0, 0.0, 0.0, 1.0])
    end_pos = end_pos_h[:3].tolist()
    return np.array(end_pos), T, intermediate

# ---------------------
# Example / test
# ---------------------
def example():
    # 4 links, each length L = 1.0
    L = 1.0
    link_lengths = [L, L, L, L]

    # Example axis ordering (each axis perpendicular to previous): z, y, z, y
    joint_axes = ['z','y','z','y']

    # Example joint angles in degrees (convert to radians)
    joint_angles_deg = [30, -45, 60, 10]  # you can change these
    joint_angles = [math.radians(a) for a in joint_angles_deg]

    end_pos, T, intermediates = fk_forward(joint_angles, joint_axes, link_lengths)

    print("Joint axes:", joint_axes)
    print("Joint angles (deg):", joint_angles_deg)
    print("Link lengths:", link_lengths)
    print("End-effector position (x,y,z):", end_pos)
    print("Full transform T:\n", T)

    # Optional: simple text check by manual calculation or visualization
    return end_pos, T, intermediates

# Optional small visualization with matplotlib
def visualize(intermediates):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # registers 3d projection
    except Exception as e:
        print("Matplotlib not available:", e)
        return

    pts = []
    for T in intermediates:
        p = T @ np.array([0.0,0.0,0.0,1.0])
        pts.append(p[:3])

    pts = np.array(pts)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pts[:,0], pts[:,1], pts[:,2], '-o', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot chain: joint positions (base -> end-effector)')
    # equal aspect
    max_range = np.array([pts[:,0].max()-pts[:,0].min(),
                          pts[:,1].max()-pts[:,1].min(),
                          pts[:,2].max()-pts[:,2].min()]).max() / 2.0
    mid_x = (pts[:,0].max()+pts[:,0].min()) * 0.5
    mid_y = (pts[:,1].max()+pts[:,1].min()) * 0.5
    mid_z = (pts[:,2].max()+pts[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()

if __name__ == "__main__":
    end_pos, T, intermediates = example()
    try:
        visualize(intermediates)
    except Exception:
        pass
