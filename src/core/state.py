"""State layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.constants import (
    STATE_DIM,
    ARM_STATE_DIM,
    SLOSH_STATE_DIM,
    OBS_DIM,
    IDX_Q,
    IDX_DQ,
    IDX_X,
    IDX_Y,
    IDX_VX,
    IDX_VY,
)


@dataclass(frozen=True)
class StateLayout:
    state_dim: int = STATE_DIM
    arm_dim: int = ARM_STATE_DIM
    slosh_dim: int = SLOSH_STATE_DIM
    obs_dim: int = OBS_DIM

    q: slice = IDX_Q
    dq: slice = IDX_DQ

    # Cartesian names mapping cleanly to slosh_dynamics.py
    x: int = IDX_X
    y: int = IDX_Y
    vx: int = IDX_VX
    vy: int = IDX_VY


LAYOUT = StateLayout()


def split_state(x: np.ndarray):
    """
    Split augmented state into:
        arm   = [q1, q2, q3, dq1, dq2, dq3]
        slosh = [x_pos, y_pos, x_vel, y_vel]
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if x.shape[0] != STATE_DIM:
        raise ValueError(
            f"Expected {STATE_DIM}D state, got {x.shape[0]}D"
        )

    arm = x[:ARM_STATE_DIM]
    slosh = x[ARM_STATE_DIM:]

    return arm, slosh


def join_state(arm: np.ndarray, slosh: np.ndarray):
    """
    Construct augmented state:
        x = [arm, slosh]
    """
    arm = np.asarray(arm, dtype=np.float32).reshape(-1)
    slosh = np.asarray(slosh, dtype=np.float32).reshape(-1)

    if arm.shape[0] != ARM_STATE_DIM:
        raise ValueError(
            f"Expected {ARM_STATE_DIM}D arm state, got {arm.shape[0]}D"
        )

    if slosh.shape[0] != SLOSH_STATE_DIM:
        raise ValueError(
            f"Expected {SLOSH_STATE_DIM}D slosh state, got {slosh.shape[0]}D"
        )

    return np.concatenate((arm, slosh)).astype(np.float32)