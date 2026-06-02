# src/core/__init__.py
"""Core mathematical models, kinematics, and physics simulation integrations."""

from src.core.arm_dynamics import arm_dynamics, position_cup, get_cup_acceleration
from src.core.slosh_dynamics import slosh_dynamics
from src.core.state import LAYOUT, split_state, join_state
from src.core.obstacles import (
    dist_point_to_segment,
    check_arm_obstacle_collision,
    check_ground_contact,
    check_joint_limits,
)

__all__ = [
    "arm_dynamics",
    "position_cup",
    "get_cup_acceleration",
    "slosh_dynamics",
    "LAYOUT",
    "split_state",
    "join_state",
    "dist_point_to_segment",
    "check_arm_obstacle_collision",
    "check_ground_contact",
    "check_joint_limits",
]