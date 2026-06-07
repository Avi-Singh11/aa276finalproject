"""Coupled arm + slosh time stepping."""

from __future__ import annotations

import numpy as np

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .arm_dynamics import arm_dynamics, position_cup
from .arm_dynamics_OVERNIGHT_VERIFIED import get_cup_acceleration
from .slosh_dynamics_OVERNIGHT_VERIFIED import slosh_dynamics
from src.config.constants import DEFAULT_L_EFF, DEFAULT_THETA_EPS


def coupled_dynamics(state, u_flat, K, L,dt, l_eff=DEFAULT_L_EFF, theta_eps=DEFAULT_THETA_EPS):
    """Single forward-Euler step of the 10D coupled system."""
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape[0] != 10:
        raise ValueError(f"Expected 10D state, got {state.shape[0]}D")

    next_arm = arm_dynamics(state[:6], u_flat, K, dt)
    next_slosh = slosh_dynamics(state[6:],state[:6], u_flat, K, L, dt, l_eff=l_eff, theta_eps=theta_eps)

    # Combine in float64 before converting back to float32.
    combined = np.concatenate([next_arm, next_slosh])

    f32_info = np.finfo(np.float32)
    combined_clipped = np.clip(combined, f32_info.min, f32_info.max)

    return combined_clipped.astype(np.float32)


def step_info(state, action, K, L):
    """Diagnostics for environment logging."""
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    cup_pos = position_cup(state[:6], L)
    a_cup = get_cup_acceleration(state[:6], action, K, L)
    return {
        "cup_pos": cup_pos.astype(np.float32),
        "a_cup": a_cup.astype(np.float32),
        "a_norm": float(np.linalg.norm(a_cup)),
    }
