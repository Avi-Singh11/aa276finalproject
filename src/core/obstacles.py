"""Obstacle geometry and collision utilities."""

from __future__ import annotations

import numpy as np

from .arm_dynamics import get_link_positions, position_cup


def dist_point_to_segment(p, a, b):
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / ab_sq, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def check_arm_obstacle_collision(phi_flat, L, obstacles):
    """True if any arm link segment penetrates any spherical obstacle."""
    pts = get_link_positions(phi_flat, L)
    segments = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3])]
    for seg_a, seg_b in segments:
        for obs in obstacles:
            center = np.asarray(obs["center"], dtype=np.float64)
            radius = float(obs["radius"])
            if dist_point_to_segment(center, seg_a, seg_b) <= radius:
                return True
    return False


def check_ground_contact(phi_flat, L, z_min=0.0):
    cup_pos = position_cup(phi_flat, L)
    return float(cup_pos[2]) < float(z_min)


def check_joint_limits(q, joint_limits):
    """Check absolute joint-angle limits (symmetric)."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    joint_limits = np.asarray(joint_limits, dtype=np.float64).reshape(-1)
    if q.shape[0] != 3 or joint_limits.shape[0] != 3:
        raise ValueError("Expected 3 joint angles and 3 joint limits")
    return bool(np.any(np.abs(q) > joint_limits))