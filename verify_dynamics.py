"""
Dynamics verification for coffee_pouring_env.py.

Four independent checks:
  1. Zero-torque constant velocity  — double integrator sanity
  2. Constant torque from rest      — parabolic position trajectory
  3. End-effector velocity = J @ theta_dot  (numerical differentiation)
  4. End-effector acceleration = J_dot @ theta_dot + J @ theta_ddot  (numerical diff)
"""

import numpy as np
from coffee_pouring_env import (
    arm_dynamics, position_cup, jacobian, jacobian_dot,
    get_cup_acceleration, A_matrix, B_matrix,
)

PASS = "PASS"
FAIL = "FAIL"

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── Test 1: Zero torque → constant velocity, linear positions ─────────────────
section("Test 1: Zero torque — constant velocity / linear theta")

dt = 0.001
n_steps = 500
K = np.diag([1.0, 2.0, 3.0])

phi0 = np.array([0.1, -0.2, 0.3, 0.5, -0.4, 0.2])   # nonzero initial velocity
u    = np.zeros(3)

phi = phi0.copy()
for _ in range(n_steps):
    phi = arm_dynamics(phi, u, K, dt)

T = n_steps * dt
theta_expected  = phi0[:3] + phi0[3:] * T
dtheta_expected = phi0[3:]             # must be unchanged

theta_err  = np.abs(phi[:3]  - theta_expected).max()
dtheta_err = np.abs(phi[3:]  - dtheta_expected).max()

print(f"  theta error  (expect ~0): {theta_err:.2e}  {PASS if theta_err < 1e-10 else FAIL}")
print(f"  dtheta error (expect ~0): {dtheta_err:.2e}  {PASS if dtheta_err < 1e-10 else FAIL}")


# ── Test 2: Constant torque from rest → parabolic trajectory ──────────────────
section("Test 2: Constant torque from rest — parabolic position")

dt = 0.001
n_steps = 200
K = np.diag([1.0, 2.0, 3.0])

phi0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
u    = np.array([1.0, 0.5, -0.5])

phi = phi0.copy()
for _ in range(n_steps):
    phi = arm_dynamics(phi, u, K, dt)

T = n_steps * dt
# Under constant torque from rest: theta_ddot = K @ u (constant)
# dtheta(T) = (K @ u) * T
# theta(T)  = 0.5 * (K @ u) * T^2
theta_ddot    = K @ u
dtheta_expect = theta_ddot * T
theta_expect  = 0.5 * theta_ddot * T**2

# Euler integration accumulates O(dt) error per step → O(dt * n_steps) = O(T*dt) total
tol = 10 * T * dt   # generous: 10x the leading-order Euler error

theta_err  = np.abs(phi[:3] - theta_expect).max()
dtheta_err = np.abs(phi[3:] - dtheta_expect).max()

print(f"  Euler error budget: {tol:.2e}")
print(f"  theta error:        {theta_err:.2e}  {PASS if theta_err < tol else FAIL}")
print(f"  dtheta error:       {dtheta_err:.2e}  {PASS if dtheta_err < tol else FAIL}")


# ── Test 3: End-effector velocity = J(phi) @ theta_dot ───────────────────────
section("Test 3: End-effector velocity = J @ theta_dot")

dt = 1e-5   # tiny for accurate finite difference
K  = np.diag([1.0, 2.0, 3.0])
L  = [1.0, 1.0, 1.0]

rng = np.random.default_rng(0)
max_err = 0.0

for _ in range(10):
    phi_flat = rng.uniform(-0.4, 0.4, size=6)
    phi_flat[3:] = rng.uniform(-0.3, 0.3, size=3)   # nonzero velocity
    u = rng.uniform(-1.0, 1.0, size=3)

    phi_col = phi_flat.reshape(6, 1)

    # Analytical velocity
    J    = jacobian(phi_col, L)
    v_an = J @ phi_col[3:6]

    # Numerical velocity via finite-difference of position
    phi_next = arm_dynamics(phi_flat, u, K, dt)
    p_now    = position_cup(phi_flat.reshape(6, 1), L).flatten()
    p_next   = position_cup(phi_next.reshape(6, 1), L).flatten()
    v_nu     = (p_next - p_now) / dt

    err = np.abs(v_an.flatten() - v_nu).max()
    max_err = max(max_err, err)

# O(dt) finite-diff error + O(dt) Euler error → O(dt) total
tol = 100 * dt
print(f"  Max velocity error over 10 random states: {max_err:.2e}  {PASS if max_err < tol else FAIL}")
print(f"  (tolerance = 100*dt = {tol:.2e})")


# ── Test 4: End-effector acceleration = J_dot @ theta_dot + J @ theta_ddot ───
section("Test 4: End-effector acceleration = J_dot @ θ_dot + J @ θ_ddot")

dt = 1e-5
K  = np.diag([1.0, 2.0, 3.0])
L  = [1.0, 1.0, 1.0]

rng = np.random.default_rng(1)
max_err = 0.0

for _ in range(10):
    phi_flat = rng.uniform(-0.4, 0.4, size=6)
    phi_flat[3:] = rng.uniform(-0.3, 0.3, size=3)
    u = rng.uniform(-1.0, 1.0, size=3)

    # Analytical acceleration
    a_an = get_cup_acceleration(phi_flat, u, K, L)

    # Numerical acceleration: differentiate velocity J(phi) @ theta_dot over one step
    phi_col = phi_flat.reshape(6, 1)
    J_now   = jacobian(phi_col, L)
    v_now   = (J_now @ phi_col[3:6]).flatten()

    phi_next    = arm_dynamics(phi_flat, u, K, dt)
    phi_next_col = phi_next.reshape(6, 1)
    J_next      = jacobian(phi_next_col, L)
    v_next      = (J_next @ phi_next_col[3:6]).flatten()
    a_nu        = (v_next - v_now) / dt

    err = np.abs(a_an - a_nu).max()
    max_err = max(max_err, err)

tol = 100 * dt
print(f"  Max acceleration error over 10 random states: {max_err:.2e}  {PASS if max_err < tol else FAIL}")
print(f"  (tolerance = 100*dt = {tol:.2e})")

print("\n" + "═"*55)
