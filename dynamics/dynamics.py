"""Compatibility adapter for DeepReach-style reachability code.

This module intentionally keeps the import path used by the current scripts:
    from dynamics.dynamics import CoffeeArmDynamics

The implementation delegates to the shared core physics layer.
"""

from __future__ import annotations

import numpy as np
import torch

from src.config.constants import (
    DEFAULT_L,
    DEFAULT_K,
    DEFAULT_U_MAX,
    DEFAULT_SLOSH_RAD_MAX,  # Replaces DEFAULT_VARTHETA_MAX / THETA_MAX
    DEFAULT_JOINT_LIMITS,
    DEFAULT_OBSTACLES,
    DEFAULT_THETA_EPS,
    STATE_DIM,
)
from src.core.arm_dynamics import position_cup, get_link_positions
from src.core.obstacles import dist_point_to_segment
from src.core.simulation import coupled_dynamics


def _as_3vec(value, name):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr.item(), 3).astype(np.float32)
    if arr.size != 3:
        raise ValueError(f"{name} must be a scalar or length-3 vector")
    return arr


class CoffeeArmDynamics:
    """DeepReach adapter for the 10D coffee-arm system."""

    def __init__(
        self,
        L=None,
        K=None,
        u_max=DEFAULT_U_MAX,
        slosh_rad_max=DEFAULT_SLOSH_RAD_MAX,  # Now accepts radial limit in meters
        joint_limits=DEFAULT_JOINT_LIMITS,
        obstacles=DEFAULT_OBSTACLES,
        dt=0.01,
        l_eff=0.10,
    ):
        self.L = np.asarray(L if L is not None else DEFAULT_L, dtype=np.float32)
        self.K = np.asarray(K if K is not None else DEFAULT_K, dtype=np.float32)
        self.u_max = _as_3vec(u_max, "u_max")
        self.slosh_rad_max = float(slosh_rad_max)
        self.joint_limits = np.asarray(joint_limits, dtype=np.float32).reshape(3)
        self.obstacles = list(obstacles)
        self.dt = float(dt)
        self.l_eff = float(l_eff)
        self.theta_eps = float(DEFAULT_THETA_EPS)

        self.state_dim = STATE_DIM
        self.input_dim = STATE_DIM + 1  # time + state

    # --- coordinate / tensor helpers -------------------------------------------------

    def coord_to_input(self, coord):
        """Identity map: model input coordinates are [t, x]."""
        if isinstance(coord, torch.Tensor):
            return coord.to(dtype=torch.float32)
        return torch.as_tensor(coord, dtype=torch.float32)

    def input_to_coord(self, inp):
        """Identity map back to [t, x]."""
        if isinstance(inp, torch.Tensor):
            return inp.to(dtype=torch.float32)
        return torch.as_tensor(inp, dtype=torch.float32)

    def input_to_state(self, inp):
        inp = self.input_to_coord(inp)
        return inp[..., 1:]

    def coord_to_state(self, coord):
        coord = self.coord_to_input(coord)
        return coord[..., 1:]

    def io_to_value(self, model_in, model_out):
        """Interpret network output as the signed value function."""
        return model_out

    def io_to_dv(self, model_in, model_out):
        """Gradient of the network output wrt input coordinates."""
        if not isinstance(model_in, torch.Tensor):
            raise TypeError("model_in must be a torch.Tensor")
        if not model_in.requires_grad:
            model_in = model_in.detach().clone().requires_grad_(True)

        out = model_out
        if out.ndim > 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)

        grad = torch.autograd.grad(
            outputs=out.sum(),
            inputs=model_in,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        return grad

    # --- failure / boundary functions ------------------------------------------------

    def _state_to_numpy(self, state):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def failure_margin(self, state):
        """Positive = safe, negative = unsafe. Larger is farther from failure."""
        x = self._state_to_numpy(state)
        if x.shape[0] != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM}D state, got {x.shape[0]}D")

        q = x[:3]
        
        # New Cartesian Math: index 6 is x_slosh, index 7 is y_slosh
        x_slosh = float(x[6])
        y_slosh = float(x[7])
        slosh_disp = np.sqrt(x_slosh**2 + y_slosh**2)

        margins = []

        # Ground clearance
        cup_z = float(position_cup(x[:6], self.L)[2])
        margins.append(cup_z)

        # Spill threshold (safe if displacement is less than max radius)
        margins.append(self.slosh_rad_max - slosh_disp)

        # Joint limits
        margins.append(float(np.min(self.joint_limits - np.abs(q))))

        # Obstacle clearance (distance to obstacle surface)
        pts = get_link_positions(x[:6], self.L)
        segments = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3])]
        min_obs_margin = np.inf
        for seg_a, seg_b in segments:
            for obs in self.obstacles:
                c = np.asarray(obs["center"], dtype=np.float64)
                r = float(obs["radius"])
                min_obs_margin = min(min_obs_margin, dist_point_to_segment(c, seg_a, seg_b) - r)
        margins.append(float(min_obs_margin))

        return float(np.min(margins))

    def boundary_fn(self, coord):
        """DeepReach-compatible boundary function (state-only margin)."""
        state = self.coord_to_state(coord)
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        if state.ndim == 1:
            return self.failure_margin(state)
        return np.array([self.failure_margin(s) for s in state], dtype=np.float32)

    # Some DeepReach setups use these names.
    def target_fn(self, coord):
        return self.boundary_fn(coord)

    def value_fn(self, coord):
        return self.boundary_fn(coord)

    # --- dynamics helpers ------------------------------------------------------------

    def dynamics_step(self, state, u):
        return coupled_dynamics(
            state,
            u,
            self.K,
            self.L,
            self.dt,
            l_eff=self.l_eff,
            theta_eps=self.theta_eps,
        )

    def is_safe(self, state, action):
        """One-step successor-state check consistent with the report."""
        x = self._state_to_numpy(state)
        u = np.asarray(action, dtype=np.float32).reshape(-1)
        if u.shape[0] != 3:
            raise ValueError(f"Expected 3D action, got {u.shape[0]}D")

        if np.any(np.abs(u) > self.u_max + 1e-9):
            return False

        next_state = self.dynamics_step(x, u)
        return self.failure_margin(next_state) >= 0.0