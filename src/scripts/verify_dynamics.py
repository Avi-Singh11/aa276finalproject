"""Rigorous verification checks for the shared coffee-arm dynamics.
"""

from __future__ import annotations

import numpy as np

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.constants import (
    DEFAULT_K,
    DEFAULT_L,
    DEFAULT_L_EFF,
    DEFAULT_THETA_EPS,
    STATE_DIM,
)
from src.core.arm_dynamics import (
    arm_dynamics,
    position_cup,
    jacobian,
    jacobian_dot,
    get_cup_acceleration,
    A_matrix,
    B_matrix,
)
from src.core.slosh_dynamics import slosh_dynamics
from src.core.simulation import coupled_dynamics

PASS = "PASS"
FAIL = "FAIL"

def section(title: str) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


def max_abs_err(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def check_close(name: str, got, expected, tol: float) -> bool:
    err = max_abs_err(got, expected)
    ok = err <= tol
    print(f"  {name}: err={err:.3e}  tol={tol:.3e}  {PASS if ok else FAIL}")
    return ok


def main() -> None:
    all_ok = True

    # ------------------------------------------------------------------
    # Test 0: Structural consistency of the arm model
    # ------------------------------------------------------------------
    section("Test 0: Arm model structure — A and B matrices")

    A = A_matrix()
    B = B_matrix(DEFAULT_K)

    A_expected = np.zeros((6, 6), dtype=np.float64)
    A_expected[0:3, 3:6] = np.eye(3)

    B_expected = np.zeros((6, 3), dtype=np.float64)
    B_expected[3:6, :] = np.asarray(DEFAULT_K, dtype=np.float64)

    ok_A = check_close("A_matrix", A, A_expected, tol=1e-12)
    ok_B = check_close("B_matrix", B, B_expected, tol=1e-12)
    all_ok = all_ok and ok_A and ok_B

    # ------------------------------------------------------------------
    # Test 1: Zero torque -> constant velocity, linear positions
    # ------------------------------------------------------------------
    section("Test 1: Zero torque — constant velocity / linear position")

    dt = 1e-3
    n_steps = 500
    K = np.asarray(DEFAULT_K, dtype=np.float64)

    phi0 = np.array([0.1, -0.2, 0.3, 0.5, -0.4, 0.2], dtype=np.float64)
    u = np.zeros(3, dtype=np.float64)

    phi = phi0.copy()
    for _ in range(n_steps):
        phi = arm_dynamics(phi, u, K, dt)

    T = n_steps * dt
    theta_expected = phi0[:3] + phi0[3:] * T
    dtheta_expected = phi0[3:]

    ok_theta = check_close("theta", phi[:3], theta_expected, tol=1e-10)
    ok_dtheta = check_close("dtheta", phi[3:], dtheta_expected, tol=1e-10)
    all_ok = all_ok and ok_theta and ok_dtheta

    # ------------------------------------------------------------------
    # Test 2: Constant torque from rest -> closed-form Euler trajectory
    # ------------------------------------------------------------------
    section("Test 2: Constant torque from rest — discrete Euler trajectory")

    dt = 1e-3
    n_steps = 200
    K = np.asarray(DEFAULT_K, dtype=np.float64)

    phi0 = np.zeros(6, dtype=np.float64)
    u = np.array([1.0, 0.5, -0.5], dtype=np.float64)

    phi = phi0.copy()
    for _ in range(n_steps):
        phi = arm_dynamics(phi, u, K, dt)

    T = n_steps * dt
    theta_ddot = K @ u

    # Forward-Euler closed form for constant acceleration:
    # dq_n = n*dt*a
    # q_n  = 0.5 * a * dt^2 * n * (n-1)
    dtheta_expected = theta_ddot * T
    theta_expected = 0.5 * theta_ddot * (dt**2) * n_steps * (n_steps - 1)

    ok_theta = check_close("theta", phi[:3], theta_expected, tol=1e-10)
    ok_dtheta = check_close("dtheta", phi[3:], dtheta_expected, tol=1e-10)
    all_ok = all_ok and ok_theta and ok_dtheta

    # ------------------------------------------------------------------
    # Test 3: End-effector velocity = J(q) @ qdot
    # ------------------------------------------------------------------
    section("Test 3: End-effector velocity identity — v = J(q) qdot")

    rng = np.random.default_rng(0)
    dt_fd = 1e-6
    max_err_v = 0.0

    for _ in range(20):
        phi_flat = rng.uniform(-0.4, 0.4, size=6)
        phi_flat[3:] = rng.uniform(-0.3, 0.3, size=3)
        u = rng.uniform(-1.0, 1.0, size=3)

        phi_col = phi_flat.reshape(6, 1)
        J = jacobian(phi_col, DEFAULT_L)
        v_an = (J @ phi_col[3:6]).reshape(-1)

        phi_next = arm_dynamics(phi_flat, u, K, dt_fd)
        p_now = position_cup(phi_flat.reshape(6, 1), DEFAULT_L).reshape(-1)
        p_next = position_cup(phi_next.reshape(6, 1), DEFAULT_L).reshape(-1)
        v_fd = (p_next - p_now) / dt_fd

        err = max_abs_err(v_an, v_fd)
        max_err_v = max(max_err_v, err)

    tol_v = 5e-4
    ok_v = max_err_v <= tol_v
    print(f"  max velocity error over 20 random states: {max_err_v:.3e}")
    print(f"  tolerance: {tol_v:.3e}  {PASS if ok_v else FAIL}")
    all_ok = all_ok and ok_v

    # ------------------------------------------------------------------
    # Test 4: End-effector acceleration = Jdot qdot + J qddot
    # ------------------------------------------------------------------
    section("Test 4: End-effector acceleration identity")

    rng = np.random.default_rng(1)
    dt_fd = 1e-5
    max_err_a = 0.0

    for _ in range(20):
        phi_flat = rng.uniform(-0.4, 0.4, size=6)
        phi_flat[3:] = rng.uniform(-0.3, 0.3, size=3)
        u = rng.uniform(-1.0, 1.0, size=3)

        a_an = get_cup_acceleration(phi_flat, u, K, DEFAULT_L)

        phi_col = phi_flat.reshape(6, 1)
        J_now = jacobian(phi_col, DEFAULT_L)
        v_now = (J_now @ phi_col[3:6]).reshape(-1)

        phi_next = arm_dynamics(phi_flat, u, K, dt_fd)
        phi_next_col = phi_next.reshape(6, 1)
        J_next = jacobian(phi_next_col, DEFAULT_L)
        v_next = (J_next @ phi_next_col[3:6]).reshape(-1)

        a_fd = (v_next - v_now) / dt_fd

        err = max_abs_err(a_an, a_fd)
        max_err_a = max(max_err_a, err)

    tol_a = 5e-3
    ok_a = max_err_a <= tol_a
    print(f"  max acceleration error over 20 random states: {max_err_a:.3e}")
    print(f"  tolerance: {tol_a:.3e}  {PASS if ok_a else FAIL}")
    all_ok = all_ok and ok_a

    # ------------------------------------------------------------------
    # Test 5: Coupled 10D step sanity
    # ------------------------------------------------------------------
    section("Test 5: Coupled 10D step sanity")

    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[:6] = np.array([0.1, -0.2, 0.3, 0.05, 0.0, -0.05], dtype=np.float32)
    # UPDATED: Now using Cartesian slosh state [x, y, dx, dy]
    state[6:] = np.array([0.02, -0.01, 0.10, 0.0], dtype=np.float32)  
    u = np.array([0.2, -0.1, 0.05], dtype=np.float32)

    next_state = coupled_dynamics(
        state,
        u,
        DEFAULT_K,
        DEFAULT_L,
        dt=0.01,
        l_eff=DEFAULT_L_EFF,
        theta_eps=DEFAULT_THETA_EPS,
    )

    ok_shape = next_state.shape == (STATE_DIM,)
    ok_finite = np.all(np.isfinite(next_state))
    print(f"  next_state shape: {next_state.shape}  (expected ({STATE_DIM},))  {PASS if ok_shape else FAIL}")
    print(f"  all finite: {PASS if ok_finite else FAIL}")
    all_ok = all_ok and ok_shape and ok_finite

    # Check the arm part is consistent with direct arm stepping
    next_arm_direct = arm_dynamics(state[:6], u, DEFAULT_K, 0.01)
    arm_err = max_abs_err(next_state[:6], next_arm_direct)
    ok_arm = arm_err <= 1e-7
    print(f"  arm substep consistency: err={arm_err:.3e}  tol=1.000e-07  {PASS if ok_arm else FAIL}")
    all_ok = all_ok and ok_arm

    # ------------------------------------------------------------------
    # Test 6: Cartesian Origin Stability
    # ------------------------------------------------------------------
    section("Test 6: Cartesian Origin Stability")

    # UPDATED: Test that resting exactly at the origin (x=0, y=0) does not cause
    # division by zero in the new Cartesian constraints.
    slosh_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    arm_state = np.array([0.1, -0.2, 0.3, 0.05, 0.0, -0.05], dtype=np.float64)
    u = np.array([0.2, -0.1, 0.05], dtype=np.float64)

    next_slosh = slosh_dynamics(
        slosh_state=slosh_state,
        arm_state=arm_state,
        u=u,
        K=DEFAULT_K,
        L=DEFAULT_L,
        dt=0.01,
        l_eff=DEFAULT_L_EFF,
        theta_eps=DEFAULT_THETA_EPS,
    )
    ok_finite = np.all(np.isfinite(next_slosh))
    print(f"  next_slosh = {next_slosh}")
    print(f"  finite outputs at origin: {PASS if ok_finite else FAIL}")
    all_ok = all_ok and ok_finite

    # ------------------------------------------------------------------
    # Test 7: Kinematic Singularity (Fully Extended Arm)
    # ------------------------------------------------------------------
    section("Test 7: Kinematic Singularity (Fully Extended)")
    
    # Arm straight up/out: angles = 0, with some non-zero velocities
    phi_singularity = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    u_zero = np.zeros(3, dtype=np.float64)
    
    J_sing = jacobian(phi_singularity.reshape(6, 1), DEFAULT_L)
    a_sing = get_cup_acceleration(phi_singularity, u_zero, DEFAULT_K, DEFAULT_L)
    
    ok_J = np.all(np.isfinite(J_sing))
    ok_a = np.all(np.isfinite(a_sing))
    print(f"  Jacobian finite at singularity: {PASS if ok_J else FAIL}")
    print(f"  Acceleration finite at singularity: {PASS if ok_a else FAIL}")
    all_ok = all_ok and ok_J and ok_a

    # ------------------------------------------------------------------
    # Test 8: Smooth Cartesian Origin Crossing
    # ------------------------------------------------------------------
    section("Test 8: Smooth Cartesian Origin Crossing")
    
    # UPDATED: Ensure the fluid can swing directly through the bottom of the cup
    # without any coordinate flipping or velocity reversing glitches.
    # Start slightly positive x, moving rapidly in negative x direction.
    slosh_cross = np.array([1e-4, 0.0, -0.1, 0.0], dtype=np.float64)
    arm_static = np.zeros(6, dtype=np.float64)
    
    next_slosh = slosh_dynamics(
        slosh_state=slosh_cross,
        arm_state=arm_static,
        u=np.zeros(3),
        K=DEFAULT_K,
        L=DEFAULT_L,
        dt=0.05, 
        l_eff=DEFAULT_L_EFF
    )
    
    nx, ny, ndx, ndy = next_slosh
    
    ok_crossed = nx < 0.0
    ok_vel_maintained = ndx < 0.0
    print(f"  Smoothly passed origin (x < 0): {PASS if ok_crossed else FAIL}")
    print(f"  Velocity direction maintained: {PASS if ok_vel_maintained else FAIL}")
    
    all_ok = all_ok and ok_crossed and ok_vel_maintained

    # ------------------------------------------------------------------
    # Test 9: Unphysical Parameters (Zero length pendulum)
    # ------------------------------------------------------------------
    section("Test 9: Invalid physical parameters rejection")
    
    try:
        # UPDATED: Now targeting slosh_dynamics directly
        slosh_dynamics(
            slosh_state=np.zeros(4),
            arm_state=np.zeros(6),
            u=np.zeros(3),
            K=DEFAULT_K,
            L=DEFAULT_L,
            dt=0.01,
            l_eff=0.0  # Illegal value
        )
        print(f"  Rejected l_eff=0.0: {FAIL} (Did not raise Exception)")
        all_ok = False
    except (ValueError, ZeroDivisionError):
        # Depending on how strict your constants/math are, dividing by 0 
        # for omega_n should trigger an exception.
        print(f"  Rejected l_eff=0.0: {PASS}")

    # ------------------------------------------------------------------
    # Test 10: Extreme Control Inputs (Numerical Stability)
    # ------------------------------------------------------------------
    section("Test 10: Extreme Control Inputs")
    
    state_extreme = np.ones(STATE_DIM, dtype=np.float32)
    # Slam the system with an absurdly high torque vector
    u_extreme = np.array([1e5, -1e5, 1e5], dtype=np.float32) 
    
    next_state_extreme = coupled_dynamics(
        state_extreme,
        u_extreme,
        DEFAULT_K,
        DEFAULT_L,
        dt=1e-3
    )
    
    ok_extreme = np.all(np.isfinite(next_state_extreme))
    print(f"  System survived massive torque without NaN: {PASS if ok_extreme else FAIL}")
    all_ok = all_ok and ok_extreme

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "═" * 72)
    if all_ok:
        print("All verification checks PASSED.")
    else:
        print("Some verification checks FAILED.")
    print("═" * 72)


if __name__ == "__main__":
    main()