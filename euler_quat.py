#!/usr/bin/env python3 
# this is the code for the Task 1
"""
euler_quat.py
Utilities to convert Euler angles <-> quaternions with edge-case handling.

Quaternion format: [w, x, y, z] (scalar first).

Supported Euler orders: 'xyz' and 'zyx' (intrinsic Tait-Bryan).
Units: radians by default. Use degrees=True to pass/return degrees.
"""

import numpy as np
import math
import warnings

EPS = 1e-12

def _to_radians(angles, degrees):
    if degrees:
        return np.deg2rad(angles)
    return np.asarray(angles, dtype=float)

def _from_radians(angles, degrees):
    if degrees:
        return np.rad2deg(angles)
    return angles

def normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < EPS:
        raise ValueError("Quaternion norm is zero (invalid quaternion).")
    return q / n

def quat_mul(q1, q2):
    """Multiply quaternions q1 * q2. Format [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=float)

def axis_angle_to_quat(axis, angle):
    """Axis must be length-3. Returns [w,x,y,z]. angle in radians."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])  # identity
    axis = axis / n
    half = angle / 2.0
    w = math.cos(half)
    s = math.sin(half)
    x, y, z = axis * s
    return np.array([w, x, y, z], dtype=float)

def euler_to_quaternion(angles, order='xyz', degrees=False):
    """
    Convert Euler angles to quaternion.
    - angles: iterable of 3 angles (alpha, beta, gamma) applied in given 'order' (intrinsic).
    - order: 'xyz' (default) or 'zyx' supported. Lowercase letters x,y,z.
    - degrees: if True, input angles are in degrees.
    Returns quaternion [w,x,y,z] normalized.
    """
    angles = _to_radians(angles, degrees)
    if len(angles) != 3:
        raise ValueError("angles must be length 3")
    # Build per-axis quaternions for half-angles then multiply in order
    # Rotation about x-axis:
    rx = axis_angle_to_quat([1,0,0], angles[0])
    ry = axis_angle_to_quat([0,1,0], angles[1])
    rz = axis_angle_to_quat([0,0,1], angles[2])

    if order == 'xyz':
        # intrinsic xyz means rotate by roll about X, then pitch about Y, then yaw about Z:
        # The resulting quaternion is q = qz * qy * qx  (note multiplication order)
        q = quat_mul(quat_mul(rz, ry), rx)
    elif order == 'zyx':
        # rotate about Z, then Y, then X -> q = qx * qy * qz
        q = quat_mul(quat_mul(rx, ry), rz)
    else:
        raise NotImplementedError("Only 'xyz' and 'zyx' orders implemented. Extend if needed.")
    return normalize_quaternion(q)

def quaternion_to_euler(q, order='xyz', degrees=False):
    """
    Convert quaternion -> Euler angles (intrinsic order).
    q: array-like [w,x,y,z]
    order: 'xyz' or 'zyx'
    degrees: if True, return angles in degrees.
    Returns angles (3,) in same order as expected by euler_to_quaternion (alpha, beta, gamma).
    Edge cases: detects gimbal lock (singularity) and returns deterministic value.
    """
    q = normalize_quaternion(q)
    w, x, y, z = q

    # Precompute commonly used terms
    # We will derive formulas for both orders.
    if order == 'xyz':
        # Intrinsic rotations X (roll) then Y (pitch) then Z (yaw).
        # Equivalent converted to formulas:
        # Note: Many references present extrinsic/zyx versions; here we adopt the standard derivation.
        # Compute elements of rotation matrix R from quaternion:
        R11 = 1 - 2*(y*y + z*z)
        R12 = 2*(x*y - z*w)
        R13 = 2*(x*z + y*w)
        R21 = 2*(x*y + z*w)
        R22 = 1 - 2*(x*x + z*z)
        R23 = 2*(y*z - x*w)
        R31 = 2*(x*z - y*w)
        R32 = 2*(y*z + x*w)
        R33 = 1 - 2*(x*x + y*y)

        # For intrinsic xyz: roll = atan2(R32, R33), pitch = asin(-R31), yaw = atan2(R21, R11)
        # Note: different sign conventions exist; chosen to be consistent with euler_to_quaternion above.
        # clamp R31 to [-1,1]
        val = -R31
        val_clamped = max(-1.0, min(1.0, val))
        pitch = math.asin(val_clamped)
        # detect gimbal lock
        if abs(abs(val_clamped) - 1.0) < 1e-6:
            # Gimbal lock: pitch is +-90 degrees. roll and yaw are coupled.
            warnings.warn("Gimbal lock detected (pitch ~= +-90 deg). Returning roll=0 and combined yaw.")
            # When pitch = +pi/2 or -pi/2, we can set roll=0 and compute yaw from R12 and R13
            # For pitch = +pi/2:
            #   yaw + roll = atan2(R12, R13)
            # We'll set roll = 0 and compute yaw accordingly.
            roll = 0.0
            yaw = math.atan2(-R12, R13)
        else:
            roll = math.atan2(R32, R33)
            yaw = math.atan2(R21, R11)

        angles = np.array([roll, pitch, yaw], dtype=float)
        return _from_radians(angles, degrees)

    elif order == 'zyx':
        # Many libraries (e.g., aerospace) use intrinsic zyx, which corresponds to yaw(Z), pitch(Y), roll(X).
        # Using standard formulas:
        # pitch = asin( clamp(2*(w*y - z*x)) )
        # roll = atan2( 2*(w*x + y*z), 1 - 2*(x*x + y*y) )
        # yaw = atan2( 2*(w*z + x*y), 1 - 2*(y*y + z*z) )
        t = 2.0*(w*y - z*x)
        t_clamped = max(-1.0, min(1.0, t))
        pitch = math.asin(t_clamped)
        # gimbal lock if |t_clamped| ~ 1
        if abs(abs(t_clamped) - 1.0) < 1e-6:
            warnings.warn("Gimbal lock detected (pitch ~= +-90 deg). Setting yaw=0.")
            # Set yaw = 0, roll derived:
            yaw = 0.0
            roll = math.atan2(-2*(w*x - y*z), 1 - 2*(x*x + y*y))
        else:
            roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        angles = np.array([yaw, pitch, roll], dtype=float) if False else np.array([roll, pitch, yaw], dtype=float)
        # Return same ordering of angles (alpha, beta, gamma) matching euler_to_quaternion expectation
        # For 'zyx', we keep the return as [roll (X), pitch (Y), yaw (Z)]
        angles = np.array([roll, pitch, yaw], dtype=float)
        return _from_radians(angles, degrees)
    else:
        raise NotImplementedError("Only 'xyz' and 'zyx' orders implemented.")

# --------------------
# Small self-tests
# --------------------
def _almost_equal(a, b, tol=1e-6):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.allclose(a, b, atol=tol, rtol=0)

def _quat_equivalent(q1, q2, tol=1e-6):
    """Return True if q1 and q2 represent the same rotation (q ~ ±q)."""
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    # Check both signs (q2 and -q2)
    d1 = np.linalg.norm(q1 - q2)
    d2 = np.linalg.norm(q1 + q2)
    return min(d1, d2) <= tol

def run_unit_tests():
    print("Running improved unit tests (compare rotations, not raw Euler triples)...")

    # Test 1: identity
    q_id = euler_to_quaternion([0, 0, 0], degrees=False)
    assert _almost_equal(q_id, np.array([1.0, 0.0, 0.0, 0.0]))
    e = quaternion_to_euler(q_id)
    # roundtrip back to quaternion and compare
    q_back = euler_to_quaternion(e, degrees=False)
    assert _quat_equivalent(q_id, q_back)

    # Test 2: 90 deg rotation about X
    qx = euler_to_quaternion([math.pi/2, 0, 0], order='xyz')
    qx_back = euler_to_quaternion(quaternion_to_euler(qx, order='xyz'), order='xyz')
    assert _quat_equivalent(qx, qx_back)

    # Test 3: random round-trip verifying quaternion equivalence
    rng = np.random.RandomState(0)
    for i in range(200):
        angs = rng.uniform(-math.pi, math.pi, size=3)
        q = euler_to_quaternion(angs, order='xyz')
        # Convert back to Euler then back to quaternion
        angs_back = quaternion_to_euler(q, order='xyz')
        q_back = euler_to_quaternion(angs_back, order='xyz')
        if not _quat_equivalent(q, q_back, tol=1e-5):
            print("Roundtrip quaternion mismatch example:")
            print("original euler:", angs)
            print("original quat :", q)
            print("euler back    :", angs_back)
            print("quat back     :", q_back)
            raise AssertionError("Roundtrip failed (quaternion mismatch)")
    print("All rotation-based unit tests passed!")


if __name__ == "__main__":
    # Quick CLI demo
    import argparse
    parser = argparse.ArgumentParser(description="Euler <-> Quaternion converter demo")
    parser.add_argument("--mode", choices=["e2q", "q2e", "test"], default="test", help="mode")
    parser.add_argument("--angles", nargs=3, type=float, help="Euler angles (3 numbers). Default radians.")
    parser.add_argument("--degrees", action="store_true", help="Angles are in degrees")
    parser.add_argument("--order", default="xyz", choices=["xyz","zyx"], help="Euler order (intrinsic)")
    parser.add_argument("--quat", nargs=4, type=float, help="Quaternion w x y z")
    args = parser.parse_args()

    if args.mode == "test":
        run_unit_tests()
    elif args.mode == "e2q":
        if args.angles is None:
            parser.error("Provide 3 Euler angles")
        q = euler_to_quaternion(args.angles, order=args.order, degrees=args.degrees)
        print("Quaternion [w x y z]:", q)
    elif args.mode == "q2e":
        if args.quat is None:
            parser.error("Provide quaternion w x y z")
        e = quaternion_to_euler(args.quat, order=args.order, degrees=args.degrees)
        print("Euler angles [roll pitch yaw]:", e)
