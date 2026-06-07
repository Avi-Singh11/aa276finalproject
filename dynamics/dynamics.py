"""DeepReach dynamics for the 10D coffee arm."""

from __future__ import annotations

import numpy as np
import torch

from src.config.constants import (
    DEFAULT_L,
    DEFAULT_K,
    DEFAULT_U_MAX,
    DEFAULT_SLOSH_RAD_MAX,
    DEFAULT_JOINT_LIMITS,
    DEFAULT_OBSTACLES,
    DEFAULT_THETA_EPS,
    STATE_DIM,
    G,
    SLOSH_DAMPING,
)
from src.core.arm_dynamics import position_cup, get_link_positions
from src.core.obstacles import dist_point_to_segment
from src.core.simulation import coupled_dynamics

ARM_LINK_RADIUS = 0.04


def _as_3vec(value, name):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr.item(), 3).astype(np.float32)
    if arr.size != 3:
        raise ValueError(f"{name} must be a scalar or length-3 vector")
    return arr


class CoffeeArmDynamics:
    """DeepReach adapter for the 10D coffee-arm system."""

    loss_type = 'brt_hjivi'
    deepreach_model = 'sine'

    def __init__(
        self,
        L=None,
        K=None,
        u_max=DEFAULT_U_MAX,
        slosh_rad_max=DEFAULT_SLOSH_RAD_MAX,
        joint_limits=DEFAULT_JOINT_LIMITS,
        obstacles=DEFAULT_OBSTACLES,
        dt=0.01,
        l_eff=0.025,
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
        self.input_dim = STATE_DIM + 1

        # Keep sampled slosh states inside the pendulum sphere.
        _slosh_pos_scale = float(self.l_eff / np.sqrt(3))
        _slosh_vel_scale = 0.3

        self.state_scale = np.array([
            float(self.joint_limits[0]),
            float(self.joint_limits[1]),
            float(self.joint_limits[2]),
            5.0,
            5.0,
            5.0,
            _slosh_pos_scale,
            _slosh_pos_scale,
            _slosh_vel_scale,
            _slosh_vel_scale,
        ], dtype=np.float32)

    def coord_to_input(self, coord):
        """Convert physical coordinates to network coordinates."""
        coord = torch.as_tensor(coord, dtype=torch.float32)
        t = coord[..., :1]
        x = coord[..., 1:]
        scale = torch.as_tensor(self.state_scale, dtype=torch.float32, device=coord.device)
        return torch.cat([t, x / scale], dim=-1)

    def input_to_coord(self, inp):
        """Convert network coordinates to physical coordinates."""
        inp = torch.as_tensor(inp, dtype=torch.float32)
        t = inp[..., :1]
        z = inp[..., 1:]
        scale = torch.as_tensor(self.state_scale, dtype=torch.float32, device=inp.device)
        return torch.cat([t, z * scale], dim=-1)

    def input_to_state(self, inp):
        inp = self.input_to_coord(inp)
        return inp[..., 1:]

    def coord_to_state(self, coord):
        coord = self.coord_to_input(coord)
        return coord[..., 1:]

    def io_to_value(self, model_in, model_out):
        return model_out

    def io_to_dv(self, model_in, model_out):
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

    def _state_to_numpy(self, state):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def failure_margin(self, state):
        """Positive = safe, negative = unsafe."""
        x = self._state_to_numpy(state)
        if x.shape[0] != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM}D state, got {x.shape[0]}D")

        q = x[:3]
        slosh_disp = float(np.sqrt(x[6]**2 + x[7]**2))
        l_total = float(np.sum(self.L))

        pts = get_link_positions(x[:6], self.L)
        segments = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3])]

        # Ground clearance.
        cup_z = float(position_cup(x[:6], self.L)[2])
        m_ground = min(cup_z / l_total, 1.0)

        # Scale the slosh margin so it does not dominate the loss.
        m_slosh = (self.slosh_rad_max - slosh_disp) / (10.0 * self.slosh_rad_max)

        # Joint limits.
        m_joint = min(float(np.min(self.joint_limits - np.abs(q))) / float(self.joint_limits[0]), 1.0)

        # Obstacle clearance.
        m_obs = np.inf
        for seg_a, seg_b in segments:
            for obs in self.obstacles:
                c = np.asarray(obs["center"], dtype=np.float64)
                r = float(obs["radius"])
                raw = dist_point_to_segment(c, seg_a, seg_b) - r
                m_obs = min(m_obs, min(raw / l_total, 1.0))
        if np.isinf(m_obs):
            m_obs = 1.0

        # Self-collision clearance.
        p2 = np.asarray(pts[2], dtype=np.float64)
        p3 = np.asarray(pts[3], dtype=np.float64)
        p0z = float(pts[0][2])
        p1z = float(pts[1][2])
        sc_min_d = np.inf
        for t in np.linspace(0.0, 1.0, 20):
            pt = p2 + t * (p3 - p2)
            z_c = float(np.clip(pt[2], p0z, p1z))
            sc_min_d = min(sc_min_d, float(np.sqrt(pt[0]**2 + pt[1]**2 + (pt[2] - z_c)**2)))
        m_self = min((sc_min_d - ARM_LINK_RADIUS) / l_total, 1.0)

        return float(min(m_ground, m_slosh, m_joint, m_obs, m_self))

    def boundary_fn(self, coord):
        """Compute boundary values for a batch of states."""
        if isinstance(coord, torch.Tensor):
            coord = coord.detach().cpu().numpy()
        x = np.asarray(coord, dtype=np.float64)
        if x.ndim == 1:
            return self.failure_margin(x)

        # x has shape (N, 10).
        N = x.shape[0]
        q = x[:, :3]
        l1, l2, l3 = float(self.L[0]), float(self.L[1]), float(self.L[2])
        l_total = l1 + l2 + l3

        t1, t2, t3 = q[:, 0], q[:, 1], q[:, 2]
        s1, c1 = np.sin(t1), np.cos(t1)
        s2, c2 = np.sin(t2), np.cos(t2)
        s3, c3 = np.sin(t3), np.cos(t3)

        # Ground clearance.
        cup_z = l1 + l2*s2 + l3*(c2*s3 + c3*s2)

        # Slosh displacement.
        slosh_disp = np.sqrt(x[:, 6]**2 + x[:, 7]**2)

        # Joint limits.
        joint_margin = np.min(self.joint_limits - np.abs(q), axis=1)

        # Link endpoints for obstacle distance checks.
        p2x = l2*c1*c2
        p2y = l2*c2*s1
        p2z = l1 + l2*s2
        p3x = c1*(l2*c2 + l3*(c2*c3 - s2*s3))
        p3y = s1*(l2*c2 + l3*(c2*c3 - s2*s3))
        p3z = cup_z

        # Stack the three arm segments.
        zero = np.zeros(N)
        seg_A = np.stack([
            np.column_stack([zero, zero, zero]),
            np.column_stack([zero, zero, np.full(N, l1)]),
            np.column_stack([p2x, p2y, p2z]),
        ], axis=1)
        seg_B = np.stack([
            np.column_stack([zero, zero, np.full(N, l1)]),
            np.column_stack([p2x, p2y, p2z]),
            np.column_stack([p3x, p3y, p3z]),
        ], axis=1)

        # Obstacle clearance.
        obs_margin = np.full(N, np.inf)
        for obs in self.obstacles:
            c = np.asarray(obs["center"], dtype=np.float64)
            r = float(obs["radius"])
            ab = seg_B - seg_A
            ab_sq = np.sum(ab**2, axis=-1, keepdims=True)
            ab_sq = np.maximum(ab_sq, 1e-12)
            ac = c[None, None, :] - seg_A
            t = np.clip(np.sum(ac*ab, axis=-1, keepdims=True) / ab_sq, 0.0, 1.0)
            closest = seg_A + t * ab
            dist = np.sqrt(np.sum((c - closest)**2, axis=-1))
            raw_clr = np.min(dist, axis=1) - r
            obs_margin = np.minimum(obs_margin, np.minimum(raw_clr / l_total, 1.0))

        # Self-collision clearance.
        N_SC = 15
        t_sc = np.linspace(0.0, 1.0, N_SC)
        sc_px = p2x[:, None] + t_sc[None, :] * (p3x - p2x)[:, None]
        sc_py = p2y[:, None] + t_sc[None, :] * (p3y - p2y)[:, None]
        sc_pz = p2z[:, None] + t_sc[None, :] * (p3z - p2z)[:, None]
        sc_z_c = np.clip(sc_pz, 0.0, l1)
        sc_dist = np.sqrt(sc_px**2 + sc_py**2 + (sc_pz - sc_z_c)**2)
        link1_link3_d = np.min(sc_dist, axis=1)
        self_coll_margin = np.minimum((link1_link3_d - ARM_LINK_RADIUS) / l_total, 1.0)

        return np.min(np.stack([
            np.minimum(cup_z / l_total, 1.0),
            (self.slosh_rad_max - slosh_disp) / (10.0 * self.slosh_rad_max),
            np.minimum(joint_margin / float(self.joint_limits[0]), 1.0),
            obs_margin,
            self_coll_margin,
        ], axis=1), axis=1).astype(np.float32)

    def target_fn(self, coord):
        return self.boundary_fn(coord)

    def value_fn(self, coord):
        return self.boundary_fn(coord)

    def hamiltonian(self, state, dvds):
        """Compute H = max_u p dot f(x, u) for bounded controls."""
        shape = state.shape[:-1]
        # Convert normalized state and gradient to physical units.
        scale = torch.as_tensor(self.state_scale, dtype=torch.float32, device=state.device)
        s = (state.reshape(-1, 10).float()) * scale
        p = dvds.reshape(-1, 10).float() / scale
        N = s.shape[0]
        dv = s.device

        # Unpack the state.
        dq = s[:, 3:6]
        xs, ys = s[:, 6], s[:, 7]
        vxs, vys = s[:, 8], s[:, 9]

        L_t = torch.as_tensor(self.L, dtype=torch.float32, device=dv)
        K_t = torch.as_tensor(self.K, dtype=torch.float32, device=dv)
        l = self.l_eff

        # Clamp edge cases near the pendulum sphere.
        slosh_sq = xs**2 + ys**2
        feasible = slosh_sq < l**2 - 1e-8
        zs_sq = torch.clamp(l**2 - slosh_sq, min=1e-8)
        zs = -torch.sqrt(zs_sq)
        vzs = -(xs * vxs + ys * vys) / zs
        v_sq = vxs**2 + vys**2 + vzs**2

        # Batched arm Jacobian.
        l2, l3 = L_t[1], L_t[2]
        t1, t2, t3 = s[:, 0], s[:, 1], s[:, 2]
        s1, c1 = torch.sin(t1), torch.cos(t1)
        s2, c2 = torch.sin(t2), torch.cos(t2)
        s3, c3 = torch.sin(t3), torch.cos(t3)

        J = torch.zeros(N, 3, 3, device=dv)
        J[:, 0, 0] = -l3*s1*c2*c3 + l3*s1*s2*s3 - l2*s1*c2
        J[:, 0, 1] = -l3*c1*s2*c3 - l3*c1*c2*s3 - l2*c1*s2
        J[:, 0, 2] = -l3*c1*c2*s3 - l3*c1*s2*c3
        J[:, 1, 0] = l3*c2*c3*c1 - l3*c1*s2*s3 + l2*c2*c1
        J[:, 1, 1] = -l3*s2*c3*s1 - l3*s1*c2*s3 - l2*s2*s1
        J[:, 1, 2] = -l3*c2*s3*s1 - l3*s1*s2*c3
        J[:, 2, 0] = 0.0
        J[:, 2, 1] = l2*c2 - l3*s2*s3 + l3*c3*c2
        J[:, 2, 2] = l3*c2*c3 - l3*s3*s2

        # Batched Jacobian derivative.
        td1, td2, td3 = dq[:, 0], dq[:, 1], dq[:, 2]

        Jd = torch.zeros(N, 3, 3, device=dv)
        Jd[:, 0, 0] = (l3*c1*s2*s3 - l3*c1*c2*c3 - l2*c1*c2)*td1 + (l3*s1*s2*c3 + l3*s1*c2*s3 + l2*s1*s2)*td2 + (s1*c2*s3 + s1*s2*c3)*l3*td3
        Jd[:, 0, 1] = (l3*s1*s2*c3 + l3*s1*c2*s3 + l2*s1*s2)*td1 + (l3*c1*s2*s3 - l3*c1*c2*c3 - l2*c1*c2)*td2 + (c1*s2*s3 - c1*c2*c3)*l3*td3
        Jd[:, 0, 2] = (s1*c2*s3 + s1*s2*c3)*l3*td1 + (c1*s2*s3 - c1*c2*c3)*l3*td2 + (c1*s2*s3 - c1*c2*c3)*l3*td3
        Jd[:, 1, 0] = (l3*s1*s2*s3 - l3*c2*c3*s1 - l2*c2*s1)*td1 - (l3*s2*c3*c1 + l3*c1*c2*s3 + l2*c1*s2)*td2 - (c1*s2*c3 + c1*c2*s3)*l3*td3
        Jd[:, 1, 1] = -(l3*s2*c3*c1 + l3*c1*c2*s3 + l2*c1*s2)*td1 + (l3*s1*s2*s3 - l3*c2*c3*s1 - l2*s1*c2)*td2 + (s2*s3*s1 - s1*c2*c3)*l3*td3
        Jd[:, 1, 2] = -(c1*c2*s3 + c1*s2*c3)*l3*td1 + (s1*s2*s3 - s1*c2*c3)*l3*td2 + (s1*s2*s3 - s1*c2*c3)*l3*td3
        Jd[:, 2, 0] = 0.0
        Jd[:, 2, 1] = -(l2*s2 + l3*c2*s3 + l3*s2*c3)*td2 - (s2*c3 + c2*s3)*l3*td3
        Jd[:, 2, 2] = -(s2*c3 + c2*s3)*l3*td2 - (c2*s3 + s2*c3)*l3*td3

        # Cup acceleration with joint damping.
        dq_drift = -dq
        a_drift = ((Jd - J) @ dq.unsqueeze(-1)).squeeze(-1)
        ax_d, ay_d = a_drift[:, 0], a_drift[:, 1]

        # Pendulum constraint force with zero control.
        r_dot_a = xs*ax_d + ys*ay_d + zs*(G + a_drift[:, 2])
        lamb0 = (v_sq - r_dot_a) / l**2
        c_damp = 2.0 * SLOSH_DAMPING * (G / l) ** 0.5

        # Drift dynamics.
        f = torch.zeros(N, 10, device=dv)
        f[:, 0:3] = dq
        f[:, 3:6] = dq_drift
        f[:, 6] = vxs
        f[:, 7] = vys
        f[:, 8] = -ax_d - lamb0*xs - c_damp*vxs
        f[:, 9] = -ay_d - lamb0*ys - c_damp*vys

        # Control matrix for the arm and slosh acceleration.
        JK = J @ K_t.unsqueeze(0)
        r_sl = torch.stack([xs, ys, zs], dim=-1)
        r_JK = (r_sl.unsqueeze(1) @ JK).squeeze(1)

        g = torch.zeros(N, 10, 3, device=dv)
        g[:, 3:6, :] = K_t.unsqueeze(0).expand(N, -1, -1)
        g[:, 8, :] = -JK[:, 0, :] + xs.unsqueeze(-1) * r_JK / l**2
        g[:, 9, :] = -JK[:, 1, :] + ys.unsqueeze(-1) * r_JK / l**2

        # Maximize the affine Hamiltonian over box controls.
        p_g = (p.unsqueeze(1) @ g).squeeze(1)
        u_max_t = torch.as_tensor(self.u_max, dtype=torch.float32, device=dv)
        H = (p * f).sum(-1) + (p_g.abs() * u_max_t).sum(-1)

        # Ignore infeasible slosh samples.
        H = torch.where(feasible, H, torch.zeros_like(H))

        # Limit extreme residuals near the slosh boundary.
        H_MAX = 50.0
        H = H.clamp(min=-H_MAX, max=H_MAX)

        return H.reshape(shape)

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
        x = self._state_to_numpy(state)
        u = np.asarray(action, dtype=np.float32).reshape(-1)
        if u.shape[0] != 3:
            raise ValueError(f"Expected 3D action, got {u.shape[0]}D")
        if np.any(np.abs(u) > self.u_max + 1e-9):
            return False
        next_state = self.dynamics_step(x, u)
        return self.failure_margin(next_state) >= 0.0
