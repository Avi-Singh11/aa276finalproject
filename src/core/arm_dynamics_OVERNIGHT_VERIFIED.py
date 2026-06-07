"""Arm kinematics and linear joint dynamics for a fully free 3-DoF arm."""

from __future__ import annotations

import numpy as np


def _as_col(phi):
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim == 1:
        phi = phi.reshape(-1, 1)
    if phi.shape[0] != 6:
        raise ValueError(f"Expected 6D arm state, got shape {phi.shape}")
    return phi


def jacobian(phi, L):
    """3x3 translational end-effector Jacobian (unconstrained 3-DoF)."""
    phi = _as_col(phi)
    l1, l2, l3 = L[0], L[1], L[2]
    theta1, theta2, theta3 = phi[0, 0], phi[1, 0], phi[2, 0]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    s3, c3 = np.sin(theta3), np.cos(theta3)

    J = np.zeros((3, 3), dtype=np.float64)
    J[0, 0] = -l3 * s1 * c2 * c3 + l3 * s1 * s2 * s3 + -l2 * s1 * c2
    J[0, 1] = -l3 * c1 * s2 * c3 - l3 * c1 * c2 * s3 - l2 * c1 * s2
    J[0, 2] = -l3 * c1 * c2 * s3 - l3 * c1 * s2 * c3
    J[1, 0] = l3 * c2 * c3 * c1 - l3 * c1 * s2 * s3 + c2 * l2 * c1
    J[1, 1] = -l3 * s2 * c3 * s1 - l3 * s1 * c2 * s3 - s2 * l2 * s1
    J[1, 2] = -l3 * c2 * s3 * s1 - l3 * s1 * s2 * c3
    J[2, 0] = 0.0
    J[2, 1] = l2 * c2 - l3 * s2 * s3 + l3 * c3 * c2
    J[2, 2] = l3 * c2 * c3 - l3 * s3 * s2
    return J


def jacobian_dot(phi, L):
    """Time derivative of the translational Jacobian."""
    phi = _as_col(phi)
    l1, l2, l3 = L[0], L[1], L[2]
    theta1, theta2, theta3 = phi[0, 0], phi[1, 0], phi[2, 0]
    theta1d, theta2d, theta3d = phi[3, 0], phi[4, 0], phi[5, 0]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    s3, c3 = np.sin(theta3), np.cos(theta3)

    Jd = np.zeros((3, 3), dtype=np.float64)
    Jd[0, 0] = (
        (l3 * c1 * s2 * s3 - l3 * c1 * c2 * c3 - l2 * c1 * c2) * theta1d
        + (l3 * s1 * s2 * c3 + l3 * s1 * c2 * s3 + l2 * s1 * s2) * theta2d
        + (s1 * c2 * s3 + s1 * s2 * c3) * l3 * theta3d
    )
    Jd[0, 1] = (
        (l3 * s1 * s2 * c3 + l3 * s1 * c2 * s3 + l2 * s1 * s2) * theta1d
        + (l3 * c1 * s2 * s3 - l3 * c1 * c2 * c3 - l2 * c1 * c2) * theta2d
        + (c1 * s2 * s3 - c1 * c2 * c3) * l3 * theta3d
    )
    Jd[0, 2] = (
        (s1 * c2 * s3 + s1 * s2 * c3) * l3 * theta1d
        + (c1 * s2 * s3 - c1 * c2 * c3) * l3 * theta2d
        + (c1 * s2 * s3 - c1 * c2 * c3) * l3 * theta3d
    )
    Jd[1, 0] = (
        (l3 * s1 * s2 * s3 - l3 * c2 * c3 * s1 - l2 * c2 * s1) * theta1d
        - (l3 * s2 * c3 * c1 + l3 * c1 * c2 * s3 + l2 * c1 * s2) * theta2d
        - (c1 * s2 * c3 + c1 * c2 * s3) * l3 * theta3d
    )
    Jd[1, 1] = (
        -(l3 * s2 * c3 * c1 + l3 * c1 * c2 * s3 + l2 * c1 * s2) * theta1d
        + (l3 * s1 * s2 * s3 - l3 * c2 * c3 * s1 - l2 * s1 * c2) * theta2d
        + (s2 * s3 * s1 - s1 * c2 * c3) * l3 * theta3d
    )
    Jd[1, 2] = (
        -(c1 * c2 * s3 + c1 * s2 * c3) * l3 * theta1d
        + (s1 * s2 * s3 - s1 * c2 * c3) * l3 * theta2d
        + (s1 * s2 * s3 - s1 * c2 * c3) * l3 * theta3d
    )
    Jd[2, 0] = 0.0
    Jd[2, 1] = -(l2 * s2 + l3 * c2 * s3 + l3 * s2 * c3) * theta2d - (
        s2 * c3 + c2 * s3
    ) * l3 * theta3d
    Jd[2, 2] = -(s2 * c3 + c2 * s3) * l3 * theta2d - (c2 * s3 + s2 * c3) * l3 * theta3d
    return Jd


def A_matrix(damping=1.0):
    """Damped double integrator: dq_dot = -damping*dq + K*u."""
    A = np.zeros((6, 6), dtype=np.float64)
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 3:6] = -damping * np.eye(3)
    return A


def B_matrix(K):
    B = np.zeros((6, 3), dtype=np.float64)
    B[3:6, :] = np.asarray(K, dtype=np.float64)
    return B


def arm_dynamics(phi_flat, u_flat, K, dt, damping=1.0):
    """Forward Euler step for the damped 3-DoF arm model."""
    phi = np.asarray(phi_flat, dtype=np.float64).reshape(6, 1)
    u = np.asarray(u_flat, dtype=np.float64).reshape(3, 1)
    phi_dot = A_matrix(damping) @ phi + B_matrix(K) @ u

    return (phi.reshape(-1) + dt * phi_dot.reshape(-1)).astype(np.float64)


def position_cup(phi, L):
    """End-effector position in world frame."""
    phi = _as_col(phi)
    l1, l2, l3 = L[0], L[1], L[2]
    theta1, theta2, theta3 = phi[0, 0], phi[1, 0], phi[2, 0]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    s3, c3 = np.sin(theta3), np.cos(theta3)
    return np.array(
        [
            l3 * c1 * c2 * c3 - l3 * c1 * s2 * s3 + c1 * c2 * l2,
            l3 * c2 * c3 * s1 - l3 * s1 * s2 * s3 + c2 * l2 * s1,
            l1 + l2 * s2 + l3 * c2 * s3 + l3 * c3 * s2,
        ],
        dtype=np.float64,
    )


def get_link_positions(phi_flat, L):
    """Positions of base, joint 2, joint 3, and end-effector."""
    phi_flat = np.asarray(phi_flat, dtype=np.float64).reshape(-1)
    l1, l2 = L[0], L[1]
    theta1, theta2 = phi_flat[0], phi_flat[1]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)

    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    p1 = np.array([0.0, 0.0, l1], dtype=np.float64)
    p2 = np.array([l2 * c1 * c2, l2 * c2 * s1, l1 + l2 * s2], dtype=np.float64)
    p3 = position_cup(phi_flat, L)
    return [p0, p1, p2, p3]


def get_joint_jacobians(phi_flat, L):
    """Jacobians of each link endpoint wrt joint angles."""
    phi_flat = np.asarray(phi_flat, dtype=np.float64).reshape(-1)
    l2 = L[1]
    theta1, theta2 = phi_flat[0], phi_flat[1]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)

    J2 = np.zeros((3, 3), dtype=np.float64)
    J2[0, 0] = -l2 * s1 * c2
    J2[0, 1] = -l2 * c1 * s2
    J2[1, 0] = l2 * c1 * c2
    J2[1, 1] = -l2 * s1 * s2
    J2[2, 0] = 0.0
    J2[2, 1] = l2 * c2

    J3 = jacobian(phi_flat.reshape(6, 1), L)
    return [np.zeros((3, 3)), np.zeros((3, 3)), J2, J3]


def get_cup_acceleration(phi_flat, u_flat, K, L, damping=1.0):
    """Cartesian cup acceleration for the same damped arm used by arm_dynamics."""
    phi = np.asarray(phi_flat, dtype=np.float64).reshape(6, 1)
    u = np.asarray(u_flat, dtype=np.float64).reshape(3, 1)
    J = jacobian(phi, L)
    Jd = jacobian_dot(phi, L)
    theta_dot = phi[3:6]
    theta_ddot = -float(damping) * theta_dot + np.asarray(K, dtype=np.float64) @ u
    return (Jd @ theta_dot + J @ theta_ddot).reshape(-1)
