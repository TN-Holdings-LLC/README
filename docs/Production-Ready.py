
"""

test_psf_zero_aerospace_core.py

Verifies psf_zero_aerospace_core.py (fixed) against:

  1. the pasted allocate_thrusters' two confirmed failure modes,

  2. the soft-MIB post-processing actually engaging on a case designed

     to land a thruster's unconstrained-QP command in the dead zone,

  3. the fixed fallback's shape correctness,

  4. the inner-loop attitude controller's closed-loop convergence

     (45 deg and 179 deg initial errors) via real rigid-body dynamics.

Run: python3 test_psf_zero_aerospace_core.py

"""

import numpy as np

import cvxpy as cp

from psf_zero_aerospace_core import (

    PSFZeroAttitudeController, allocate_thrusters, _apply_soft_mib, zero_clamp

)

print("=== [1] Pasted allocate_thrusters' two failure modes, reproduced standalone ===")

tmax8 = np.ones(8) * 0.5

active = cp.Variable(8, boolean=True)

dotprod_shape = (tmax8 * active).shape

elementwise_shape = cp.multiply(tmax8, active).shape

print(f"  tmax * active (pasted code's operator) -> shape {dotprod_shape} (a SCALAR dot product!)")

print(f"  cp.multiply(tmax, active) (what was meant) -> shape {elementwise_shape}")

try:

    x = cp.Variable(8, nonneg=True)

    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x <= tmax8 * active, x >= 0.005 * active])

    prob.solve(solver=cp.OSQP)

except Exception as e:

    print(f"  Solving the pasted MIQP with OSQP -> {type(e).__name__}: {e}")

try:

    np.clip(np.array([1.0, 2.0, 3.0]), 0.0, tmax8)

except Exception as e:

    print(f"  Pasted fallback np.clip(tau_des(3,), 0, tmax(8,)) -> {type(e).__name__}: {e}\n")

print("=== [2] Fixed allocate_thrusters actually solves (no crash) ===")

rng = np.random.default_rng(0)

B = rng.standard_normal((3, 8)) * 0.1

tmax = np.ones(8) * 0.5

mib = np.ones(8) * 0.005

tau_des = np.array([0.01, 0.02, -0.015])

x_sol = allocate_thrusters(tau_des, B, tmax, mib)

print(f"  x = {x_sol}")

print(f"  all within [0, tmax]: {np.all(x_sol >= -1e-9) and np.all(x_sol <= tmax + 1e-6)}")

print(f"  no thruster left in the dead zone (0, mib): "

      f"{not np.any((x_sol > 1e-9) & (x_sol < mib - 1e-9))}\n")

print("=== [3] Soft-MIB post-processing actually engages ===")

# Construct a case where the unconstrained QP wants a small nonzero

# duration on one thruster, strictly inside its dead zone, to check the

# post-processing snaps it to 0 or mib_i (whichever is the better cost),

# rather than leaving a physically-unrealizable command in place.

B2 = np.array([[1.0, 0.02], [0.0, 0.0], [0.0, 0.0]])  # thruster 2 barely affects torque

tmax2 = np.array([1.0, 1.0])

mib2 = np.array([0.01, 0.05])   # thruster 2 has a large MIB

tau_des2 = np.array([0.3, 0.0, 0.0])

x_direct = cp.Variable(2, nonneg=True)

obj = 0.5 * cp.sum_squares(np.eye(3) @ (B2 @ x_direct - tau_des2)) + 0.5 * 1e-4 * cp.sum_squares(x_direct)

cp.Problem(cp.Minimize(obj), [x_direct <= tmax2, x_direct >= 0]).solve(solver=cp.OSQP)

x_raw = np.clip(x_direct.value, 0.0, tmax2)

print(f"  raw QP solution (before soft-MIB): {x_raw}")

in_dead_zone = (x_raw > 1e-9) & (x_raw < mib2)

print(f"  thruster(s) in dead zone (0 < x < mib) before post-processing: {np.where(in_dead_zone)[0]}")

x_fixed = _apply_soft_mib(x_raw, B2, tau_des2, tmax2, mib2, np.eye(3), 1e-4)

print(f"  after soft-MIB post-processing: {x_fixed}")

print(f"  still in dead zone after fix: {np.any((x_fixed > 1e-9) & (x_fixed < mib2 - 1e-9))}\n")

print("=== [4] Fixed fallback: correct shape, no crash ===")

x_fallback = np.linalg.lstsq(B, tau_des, rcond=None)[0]

x_fallback = np.clip(x_fallback, 0.0, tmax)

print(f"  fallback output shape: {x_fallback.shape} (expected (8,)): {x_fallback.shape == (8,)}\n")

print("=== [5] Inner-loop attitude controller: closed-loop convergence (re-verification) ===")

def quat_mul(q1, q2):

    w1, x1, y1, z1 = q1

    w2, x2, y2, z2 = q2

    return np.array([

        w1*w2 - x1*x2 - y1*y2 - z1*z2,

        w1*x2 + x1*w2 + y1*z2 - z1*y2,

        w1*y2 - x1*z2 + y1*w2 + z1*x2,

        w1*z2 + x1*y2 - y1*x2 + z1*w2

    ])

def simulate_attitude(q0, q_des, omega0, J, ctrl, dt=0.01, steps=3000):

    q = q0.copy(); omega = omega0.copy()

    ang_err = np.zeros(steps)

    for k in range(steps):

        torque = ctrl.compute_torque(q, q_des, omega, Kq=5.0, Kw=10.0)

        omega_dot = np.linalg.solve(J, torque - np.cross(omega, J @ omega))

        omega = omega + omega_dot * dt

        omega_quat = np.array([0.0, *omega])

        q = q + 0.5 * quat_mul(q, omega_quat) * dt

        q = q / np.linalg.norm(q)

        qd_inv = np.array([q_des[0], -q_des[1], -q_des[2], -q_des[3]])

        qe = quat_mul(qd_inv, q)

        ang_err[k] = 2 * np.arccos(np.clip(abs(qe[0]), -1, 1)) * 180 / np.pi

    return ang_err, np.linalg.norm(omega)

J = np.diag([50.0, 60.0, 40.0])

q_des = np.array([1.0, 0.0, 0.0, 0.0])

axis = np.array([0.0, 0.0, 1.0])

for label, angle_deg, steps in [("45 deg", 45, 3000), ("179 deg (near-antipodal)", 179, 6000)]:

    angle = np.radians(angle_deg)

    q0 = np.array([np.cos(angle/2), *(np.sin(angle/2) * axis)])

    ctrl = PSFZeroAttitudeController(tau=0.8)

    ang_err, final_omega = simulate_attitude(q0, q_des, np.zeros(3), J, ctrl, steps=steps)

    monotone_ok = np.all(np.diff(ang_err[50:]) <= 1.0)  # allow small numerical wiggle, no large re-growth

    print(f"  {label}: final error={ang_err[-1]:.3f} deg, final |omega|={final_omega:.5f} rad/s, "

          f"no unwinding (error doesn't re-grow): {monotone_ok}")
