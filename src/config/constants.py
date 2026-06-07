"""Shared constants for the coffee arm project."""

from __future__ import annotations

import numpy as np

G = 9.81

# Core physical parameters
DEFAULT_L = np.array([0.30, 0.50, 0.30], dtype=np.float32)
DEFAULT_K = np.eye(3, dtype=np.float32)

# Slosh model parameters
DEFAULT_L_EFF = 0.025
DEFAULT_D_EFF = 0.2
SLOSH_DAMPING = DEFAULT_D_EFF

# Safety thresholds / limits
DEFAULT_U_MAX = 15.0
DEFAULT_THETA_EPS = 1e-5

# Cartesian slosh thresholds
DEFAULT_VARTHETA_MAX = 0.30
DEFAULT_SLOSH_RAD_MAX = float(DEFAULT_L_EFF * np.sin(DEFAULT_VARTHETA_MAX))

# Joint limits used by the failure set and safety checks
DEFAULT_JOINT_LIMITS = np.array([np.pi, np.pi, np.pi], dtype=np.float32)

# Nominal obstacle set used by CoffeeArmEnv
DEFAULT_OBSTACLES = [
    {"center": [0.24, -0.3, 0.08], "radius": 0.08},
    {"center": [0.39, 0, 0.30], "radius": 0.15},
    {"center": [0.21, 0.21, 0.08], "radius": 0.08},
]

# State layout for the 10D Cartesian model:
# x = [q1, q2, q3, dq1, dq2, dq3, x_slosh, y_slosh, vx_slosh, vy_slosh]
STATE_DIM = 10
ARM_STATE_DIM = 6
SLOSH_STATE_DIM = 4
OBS_DIM = 13

# Layout slices and indices
IDX_Q = slice(0, 3)
IDX_DQ = slice(3, 6)

IDX_X = 6
IDX_Y = 7
IDX_VX = 8
IDX_VY = 9
