"""Shared constants for the coffee arm project."""

from __future__ import annotations

import numpy as np

G = 9.81

# Core physical parameters
DEFAULT_L = np.array([0.30, 0.30, 0.25], dtype=np.float32)
DEFAULT_K = np.eye(3, dtype=np.float32)

# Slosh model parameters
DEFAULT_L_EFF = 0.025   # Length of the pendulum (m)
DEFAULT_D_EFF = 0.2     # Physical damping coefficient
SLOSH_DAMPING = DEFAULT_D_EFF

# Safety thresholds / limits
DEFAULT_U_MAX = 15.0
DEFAULT_THETA_EPS = 1e-5

# Cartesian slosh thresholds ---
DEFAULT_VARTHETA_MAX = 0.30   
DEFAULT_SLOSH_RAD_MAX = float(DEFAULT_L_EFF * np.sin(DEFAULT_VARTHETA_MAX))

# Joint limits used by the failure set and safety checks
DEFAULT_JOINT_LIMITS = np.array([np.pi, np.pi, np.pi], dtype=np.float32)

# Nominal obstacle set used by CoffeeArmEnv
DEFAULT_OBSTACLES = [
    {"center": [0.35, 0.15, 0.15], "radius": 0.08},  # espresso machine
    {"center": [0.55, -0.10, 0.20], "radius": 0.06},  # cup stack
]

# State layout for the 10D Cartesian model:
# x = [q1, q2, q3, dq1, dq2, dq3, x_slosh, y_slosh, vx_slosh, vy_slosh]
STATE_DIM = 10
ARM_STATE_DIM = 6
SLOSH_STATE_DIM = 4
OBS_DIM = 13  # 10 state + 3 goal-relative cup position

# Layout Slices and Indices
IDX_Q = slice(0, 3)
IDX_DQ = slice(3, 6)

IDX_X = 6
# Index 7 is y position because slosh_dynamics unpacks: x, y, dx, dy = slosh_state
IDX_Y = 7  
IDX_VX = 8
IDX_VY = 9



#DELETE
DEFAULT_A_MAX = 5.0
DEFAULT_ALPHA_MAX = 10.0
DEFAULT_B_DAMP = 5.0
DEFAULT_DAMPING = 3.0