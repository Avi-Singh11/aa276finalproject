"""Stable-recovery dynamics for the separate BRT_STABLE_RECOVERY pipeline."""

from __future__ import annotations

import os
import sys
import numpy as np
import torch

VERIFIED_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(VERIFIED_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.constants import (
    DEFAULT_L,
    DEFAULT_K,
    DEFAULT_U_MAX,
    DEFAULT_SLOSH_RAD_MAX,  # Replaces DEFAULT_VARTHETA_MAX / THETA_MAX
    DEFAULT_JOINT_LIMITS,
    DEFAULT_OBSTACLES,
    DEFAULT_THETA_EPS,
    STATE_DIM,
    G,
    SLOSH_DAMPING,
)
from src.core.arm_dynamics import (
    position_cup,
    get_link_positions,
    jacobian,
    jacobian_dot,
)
from src.core.obstacles import dist_point_to_segment

# Physical arm-link radius used for self-collision clearance.
# Two links collide when their segment-to-segment distance < 2 × ARM_LINK_RADIUS.
# Here we express the margin as (dist - ARM_LINK_RADIUS) / ARM_LINK_RADIUS so it
# reaches 0 at the surface-to-surface contact boundary, matching the other margins.
ARM_LINK_RADIUS = 0.04   # metres


def _as_3vec(value, name):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr.item(), 3).astype(np.float32)
    if arr.size != 3:
        raise ValueError(f"{name} must be a scalar or length-3 vector")
    return arr


ARM_DAMPING = 1.0


class BRTStableRecoveryCoffeeArmDynamics:
    """DeepReach adapter for the 10D coffee-arm system."""

    # Required by DeepReach dataio and losses
    loss_type = 'brt_hjivi'
    deepreach_model = 'sine'
    time_scale = 10.0

    def __init__(
        self,
        L=None,
        K=None,
        u_max=DEFAULT_U_MAX,
        slosh_rad_max=DEFAULT_SLOSH_RAD_MAX,  # Now accepts radial limit in meters
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
        self.input_dim = STATE_DIM + 1  # time + state

        # Per-dimension scale factors mapping physical ranges to [-1, 1].
        # dataio samples the network input in [-1, 1], so these scales define
        # what physical region gets explored during BRT training.
        #
        # Slosh position scale: must satisfy 2*scale² < l_eff² so the training
        # square stays inside the feasibility disk xs²+ys² ≤ l_eff².
        # l_eff/sqrt(3) gives max xs²+ys² = 2/3·l_eff² — a 33% margin.
        # Using l_eff directly caused corner states with xs²+ys² > l_eff²,
        # making vzs→∞, lamb0→billions, H→-∞, diff_constraint_hom→billions.
        #
        # Slosh velocity scale: natural max speed ≈ ω·slosh_rad_max ≈ 0.15 m/s;
        # 0.3 m/s covers 2× that. The old 1.0 m/s sampled physically unreachable
        # states and amplified gradient errors through the 1/scale chain rule.
        _slosh_pos_scale = float(self.l_eff / np.sqrt(3))   # ≈ 0.01443 m
        _slosh_vel_scale = 0.3                               # m/s

        self.state_scale = np.array([
            float(self.joint_limits[0]),  # q1   [-π, π]
            float(self.joint_limits[1]),  # q2   [-π, π]
            float(self.joint_limits[2]),  # q3   [-π, π]
            15.0,                         # dq1: covers the damped |u| <= 15 equilibrium
            15.0,                         # dq2
            15.0,                         # dq3
            _slosh_pos_scale,             # x_slosh
            _slosh_pos_scale,             # y_slosh
            _slosh_vel_scale,             # vx_slosh
            _slosh_vel_scale,             # vy_slosh
        ], dtype=np.float32)

    # --- coordinate / tensor helpers -------------------------------------------------

    def coord_to_input(self, coord):
        """Physical [t, x] → normalised [t, z] fed to the network."""
        coord = torch.as_tensor(coord, dtype=torch.float32)
        t = coord[..., :1]
        x = coord[..., 1:]
        scale = torch.as_tensor(self.state_scale, dtype=torch.float32, device=coord.device)
        return torch.cat([t, x / scale], dim=-1)

    def input_to_coord(self, inp):
        """Normalised [t, z] → physical [t, x]."""
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
        """Enforce V(0, x) = l(x) exactly through the model representation."""
        raw = model_out.squeeze(-1) if model_out.ndim > 1 else model_out
        physical_state = self.input_to_coord(model_in)[..., 1:]
        boundary = self.boundary_fn(physical_state)
        return boundary + (model_in[..., 0] / self.time_scale) * raw

    def io_to_dv(self, model_in, model_out):
        """Gradient of the network output wrt input coordinates."""
        if not isinstance(model_in, torch.Tensor):
            raise TypeError("model_in must be a torch.Tensor")
        if not model_in.requires_grad:
            model_in = model_in.detach().clone().requires_grad_(True)

        out = self.io_to_value(model_in, model_out)

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
        """Positive = safe, negative = unsafe.

        All five constraints are normalised so that each margin reaches exactly 0
        at its boundary and is capped at +1.0 when well inside the safe set.
        This makes every constraint contribute comparably to min() and ensures
        the BRT training sees gradients from all failure modes equally.

        Constraints:
          cup_z    / l_total          ∈ (-∞, 1]  — ground clearance
          slosh    / slosh_rad_max    ∈ (-∞, 1]  — spill threshold
          joint    / joint_limits[0]  ∈ (-∞, 1]  — joint limits
          obs_clr  / obs_radius       ∈ (-∞, 1]  — obstacle clearance (per-obs)
          self_coll/ ARM_LINK_RADIUS  ∈ (-∞, 1]  — link self-collision
        """
        x = self._state_to_numpy(state)
        if x.shape[0] != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM}D state, got {x.shape[0]}D")

        q = x[:3]
        slosh_disp = float(np.sqrt(x[6]**2 + x[7]**2))
        l_total    = float(np.sum(self.L))

        pts = get_link_positions(x[:6], self.L)
        segments = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3])]

        # Ground clearance — cap at 1.0
        cup_z = float(position_cup(x[:6], self.L)[2])
        m_ground = min(cup_z / l_total, 1.0)

        # Slosh — cap at 1.0
        m_slosh = min((self.slosh_rad_max - slosh_disp) / self.slosh_rad_max, 1.0)

        # Joint limits — cap at 1.0
        m_joint = min(float(np.min(self.joint_limits - np.abs(q))) / float(self.joint_limits[0]), 1.0)

        # Obstacle clearance — normalise by each obstacle's own radius, cap at 1.0
        m_obs = np.inf
        for seg_a, seg_b in segments:
            for obs in self.obstacles:
                c = np.asarray(obs["center"], dtype=np.float64)
                r = float(obs["radius"])
                raw = dist_point_to_segment(c, seg_a, seg_b) - r
                m_obs = min(m_obs, min(raw / r, 1.0))
        if np.isinf(m_obs):
            m_obs = 1.0

        # Self-collision: Link 1 (base column p0→p1, along z-axis) vs Link 3 (p2→p3).
        # Sample 20 points along Link 3 and find the minimum distance to Link 1.
        p2 = np.asarray(pts[2], dtype=np.float64)
        p3 = np.asarray(pts[3], dtype=np.float64)
        p0z = float(pts[0][2])   # = 0.0
        p1z = float(pts[1][2])   # = l1
        sc_min_d = np.inf
        for t in np.linspace(0.0, 1.0, 20):
            pt = p2 + t * (p3 - p2)
            z_c = float(np.clip(pt[2], p0z, p1z))
            sc_min_d = min(sc_min_d, float(np.sqrt(pt[0]**2 + pt[1]**2 + (pt[2] - z_c)**2)))
        m_self = min((sc_min_d - ARM_LINK_RADIUS) / ARM_LINK_RADIUS, 1.0)

        return float(min(m_ground, m_slosh, m_joint, m_obs, m_self))

    def boundary_fn_numpy(self, coord):
        """NumPy reference boundary used only for equivalence testing."""
        x = np.asarray(coord, dtype=np.float64)
        if x.ndim == 1:
            return self.failure_margin(x)

        # x : (N, 10)
        N  = x.shape[0]
        q  = x[:, :3]                                           # (N, 3)
        l1, l2, l3 = float(self.L[0]), float(self.L[1]), float(self.L[2])

        t1, t2, t3 = q[:, 0], q[:, 1], q[:, 2]
        s1, c1 = np.sin(t1), np.cos(t1)
        s2, c2 = np.sin(t2), np.cos(t2)
        s3, c3 = np.sin(t3), np.cos(t3)

        # ── cup z (ground clearance) ──────────────────────────────────────
        cup_z = l1 + l2*s2 + l3*(c2*s3 + c3*s2)               # (N,)

        # ── slosh displacement ────────────────────────────────────────────
        slosh_disp = np.sqrt(x[:, 6]**2 + x[:, 7]**2)         # (N,)

        # ── joint limits ──────────────────────────────────────────────────
        joint_margin = np.min(self.joint_limits - np.abs(q), axis=1)  # (N,)

        # ── obstacle clearance (vectorised segment–sphere distance) ───────
        # link endpoints: p0=(0,0,0), p1=(0,0,l1), p2, p3=cup
        p2x = l2*c1*c2;  p2y = l2*c2*s1;  p2z = l1 + l2*s2
        p3x = c1*(l2*c2 + l3*(c2*c3 - s2*s3))
        p3y = s1*(l2*c2 + l3*(c2*c3 - s2*s3))
        p3z = cup_z

        # segment endpoints stacked: (N, num_segs, 3) for A and B
        zero = np.zeros(N)
        seg_A = np.stack([
            np.column_stack([zero, zero, zero]),       # p0
            np.column_stack([zero, zero, np.full(N, l1)]),  # p1
            np.column_stack([p2x,  p2y,  p2z]),        # p2
        ], axis=1)                                     # (N, 3, 3)
        seg_B = np.stack([
            np.column_stack([zero, zero, np.full(N, l1)]),  # p1
            np.column_stack([p2x,  p2y,  p2z]),        # p2
            np.column_stack([p3x,  p3y,  p3z]),        # p3
        ], axis=1)                                     # (N, 3, 3)

        # ── obstacle clearance ─────────────────────────────────────────────
        # Normalise by each obstacle's own radius and cap at 1.0 so that
        # every obstacle contributes comparably to the min() regardless of size.
        obs_margin = np.full(N, np.inf)
        for obs in self.obstacles:
            c  = np.asarray(obs["center"], dtype=np.float64)
            r  = float(obs["radius"])
            ab = seg_B - seg_A                                  # (N, 3, 3)
            ab_sq = np.sum(ab**2, axis=-1, keepdims=True)
            ab_sq = np.maximum(ab_sq, 1e-12)
            ac = c[None, None, :] - seg_A
            t  = np.clip(np.sum(ac*ab, axis=-1, keepdims=True) / ab_sq, 0.0, 1.0)
            closest = seg_A + t * ab
            dist = np.sqrt(np.sum((c - closest)**2, axis=-1))  # (N, 3)
            raw_clr = np.min(dist, axis=1) - r                 # (N,) raw clearance
            obs_margin = np.minimum(obs_margin, np.minimum(raw_clr / r, 1.0))

        # ── self-collision: Link 1 (z-axis) vs Link 3 (p2→p3) ─────────────
        # Sample N_SC points along Link 3 and measure min distance to Link 1.
        N_SC = 15
        t_sc = np.linspace(0.0, 1.0, N_SC)                     # (N_SC,)
        # Coordinates of sample points along Link 3: (N, N_SC)
        sc_px = p2x[:, None] + t_sc[None, :] * (p3x - p2x)[:, None]
        sc_py = p2y[:, None] + t_sc[None, :] * (p3y - p2y)[:, None]
        sc_pz = p2z[:, None] + t_sc[None, :] * (p3z - p2z)[:, None]
        # Closest point on Link 1 (base column: x=y=0, z∈[0, l1]) for each sample
        sc_z_c = np.clip(sc_pz, 0.0, l1)
        sc_dist = np.sqrt(sc_px**2 + sc_py**2 + (sc_pz - sc_z_c)**2)  # (N, N_SC)
        link1_link3_d = np.min(sc_dist, axis=1)                # (N,)
        # Normalise by ARM_LINK_RADIUS and cap at 1.0
        self_coll_margin = np.minimum(
            (link1_link3_d - ARM_LINK_RADIUS) / ARM_LINK_RADIUS, 1.0
        )

        l_total = l1 + l2 + l3

        return np.min(np.stack([
            np.minimum(cup_z / l_total,                                    1.0),
            np.minimum((self.slosh_rad_max - slosh_disp) / self.slosh_rad_max, 1.0),
            np.minimum(joint_margin / float(self.joint_limits[0]),         1.0),
            obs_margin,          # already capped at 1.0 per obstacle above
            self_coll_margin,    # already capped at 1.0
        ], axis=1), axis=1).astype(np.float32)

    def boundary_fn(self, coord):
        """Stable-recovery normalized safety margin, preserving device and dtype."""
        if not isinstance(coord, torch.Tensor):
            return self.boundary_fn_numpy(coord)

        x = coord
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)

        q = x[..., :3]
        dtype, device = x.dtype, x.device
        L = torch.as_tensor(self.L, dtype=dtype, device=device)
        limits = torch.as_tensor(self.joint_limits, dtype=dtype, device=device)
        l1, l2, l3 = L.unbind()

        t1, t2, t3 = q.unbind(dim=-1)
        s1, c1 = torch.sin(t1), torch.cos(t1)
        s2, c2 = torch.sin(t2), torch.cos(t2)
        s3, c3 = torch.sin(t3), torch.cos(t3)

        cup_z = l1 + l2*s2 + l3*(c2*s3 + c3*s2)
        slosh_disp = torch.linalg.vector_norm(x[..., 6:8], dim=-1)
        joint_margin = torch.amin(limits - torch.abs(q), dim=-1)

        p0 = torch.stack([
            torch.zeros_like(t1), torch.zeros_like(t1), torch.zeros_like(t1)
        ], dim=-1)
        p1 = torch.stack([
            torch.zeros_like(t1), torch.zeros_like(t1), torch.full_like(t1, l1)
        ], dim=-1)
        p2 = torch.stack([
            l2*c1*c2,
            l2*c2*s1,
            l1 + l2*s2,
        ], dim=-1)
        p3 = torch.stack([
            c1*(l2*c2 + l3*(c2*c3 - s2*s3)),
            s1*(l2*c2 + l3*(c2*c3 - s2*s3)),
            cup_z,
        ], dim=-1)

        seg_a = torch.stack([p0, p1, p2], dim=-2)
        seg_b = torch.stack([p1, p2, p3], dim=-2)
        ab = seg_b - seg_a
        ab_sq = torch.clamp(torch.sum(ab*ab, dim=-1, keepdim=True), min=1e-12)

        obs_margin = torch.full_like(cup_z, torch.inf)
        for obs in self.obstacles:
            center = torch.as_tensor(obs["center"], dtype=dtype, device=device)
            radius = torch.as_tensor(obs["radius"], dtype=dtype, device=device)
            ac = center - seg_a
            tau = torch.clamp(
                torch.sum(ac*ab, dim=-1, keepdim=True) / ab_sq, 0.0, 1.0
            )
            closest = seg_a + tau*ab
            dist = torch.linalg.vector_norm(center - closest, dim=-1)
            raw_clearance = torch.amin(dist, dim=-1) - radius
            obs_margin = torch.minimum(
                obs_margin, torch.clamp(raw_clearance / radius, max=1.0)
            )

        t_sc = torch.linspace(0.0, 1.0, 15, dtype=dtype, device=device)
        samples = p2.unsqueeze(-2) + t_sc.view(1, -1, 1) * (
            p3 - p2
        ).unsqueeze(-2)
        z_closest = torch.clamp(samples[..., 2], 0.0, float(self.L[0]))
        self_dist = torch.sqrt(
            samples[..., 0]**2
            + samples[..., 1]**2
            + (samples[..., 2] - z_closest)**2
        )
        self_margin = torch.clamp(
            (torch.amin(self_dist, dim=-1) - ARM_LINK_RADIUS) / ARM_LINK_RADIUS,
            max=1.0,
        )

        one = torch.ones_like(cup_z)
        margins = torch.stack([
            torch.minimum(cup_z / torch.sum(L), one),
            torch.minimum(
                (self.slosh_rad_max - slosh_disp) / self.slosh_rad_max, one
            ),
            torch.minimum(joint_margin / limits[0], one),
            obs_margin,
            self_margin,
        ], dim=-1)
        result = torch.amin(margins, dim=-1)
        return result[0] if squeeze else result

    # Some DeepReach setups use these names.
    def target_fn(self, coord):
        return self.boundary_fn(coord)

    def value_fn(self, coord):
        return self.boundary_fn(coord)

    # --- dynamics helpers ------------------------------------------------------------

    def hamiltonian(self, state, dvds):
        """H(x,p) = max_u [p · f(x,u)] for box-constrained u.

        For the affine-in-control system  ẋ = f_drift(x) + g(x)u:
            H = p · f_drift + Σ_i u_max_i · |(p^T g)_i|

        Fully vectorised in torch so it runs on GPU without CPU round-trips.
        Replicates the Jacobian / slosh math from arm_dynamics.py and
        slosh_dynamics.py, transposed to batched tensor operations.

        state : (..., 10) torch.Tensor
        dvds  : (..., 10) torch.Tensor  — ∂V/∂x
        returns: (...,)  torch.Tensor
        """
        shape = state.shape[:-1]
        # state arrives as normalised z ∈ [-1,1]; convert to physical x
        scale = torch.as_tensor(self.state_scale, dtype=torch.float32, device=state.device)
        s  = (state.reshape(-1, 10).float()) * scale      # physical state
        # dvds = ∂V/∂z; chain rule gives ∂V/∂x = ∂V/∂z / scale
        p  = dvds.reshape(-1, 10).float() / scale
        N  = s.shape[0]
        dv = s.device

        # ── unpack state ──────────────────────────────────────────────────
        dq  = s[:, 3:6]                          # joint velocities  (N,3)
        xs, ys  = s[:, 6], s[:, 7]              # slosh x, y
        vxs, vys = s[:, 8], s[:, 9]             # slosh vx, vy

        L_t = torch.as_tensor(self.L, dtype=torch.float32, device=dv)
        K_t = torch.as_tensor(self.K, dtype=torch.float32, device=dv)
        l   = self.l_eff

        # ── slosh geometry ────────────────────────────────────────────────
        # Guard: states outside the pendulum sphere are physically infeasible.
        # With the corrected state_scale this should not occur during training,
        # but clamp defensively to prevent vzs/lamb0 blowup at edge cases.
        slosh_sq = xs**2 + ys**2
        feasible = slosh_sq < l**2 - 1e-8                    # (N,) bool
        zs_sq    = torch.clamp(l**2 - slosh_sq, min=1e-8)
        zs       = -torch.sqrt(zs_sq)
        vzs      = -(xs * vxs + ys * vys) / zs
        v_sq     = vxs**2 + vys**2 + vzs**2

        # ── batched J(q) [N,3,3] — same formula as jacobian() ─────────────
        l2, l3 = L_t[1], L_t[2]
        t1, t2, t3 = s[:, 0], s[:, 1], s[:, 2]
        s1, c1 = torch.sin(t1), torch.cos(t1)
        s2, c2 = torch.sin(t2), torch.cos(t2)
        s3, c3 = torch.sin(t3), torch.cos(t3)

        J = torch.zeros(N, 3, 3, device=dv)
        J[:, 0, 0] = -l3*s1*c2*c3 + l3*s1*s2*s3 - l2*s1*c2
        J[:, 0, 1] = -l3*c1*s2*c3 - l3*c1*c2*s3 - l2*c1*s2
        J[:, 0, 2] = -l3*c1*c2*s3 - l3*c1*s2*c3
        J[:, 1, 0] =  l3*c2*c3*c1 - l3*c1*s2*s3 + l2*c2*c1
        J[:, 1, 1] = -l3*s2*c3*s1 - l3*s1*c2*s3 - l2*s2*s1
        J[:, 1, 2] = -l3*c2*s3*s1 - l3*s1*s2*c3
        J[:, 2, 0] = 0.0
        J[:, 2, 1] =  l2*c2 - l3*s2*s3 + l3*c3*c2
        J[:, 2, 2] =  l3*c2*c3 - l3*s3*s2

        # ── batched Jd(q, dq) [N,3,3] — same formula as jacobian_dot() ───
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

        # The simulator uses qddot = -ARM_DAMPING*dq + K*u. Cup acceleration
        # must include the damping term as Jd*dq + J*qddot.
        qddot_drift = -ARM_DAMPING * dq
        a_drift = (
            Jd @ dq.unsqueeze(-1) + J @ qddot_drift.unsqueeze(-1)
        ).squeeze(-1)
        ax_d, ay_d = a_drift[:, 0], a_drift[:, 1]

        # ── Lagrange multiplier (u=0) ─────────────────────────────────────
        r_dot_a = xs*ax_d + ys*ay_d + zs*(G + a_drift[:, 2])
        lamb0   = (v_sq - r_dot_a) / l**2
        c_damp  = 2.0 * SLOSH_DAMPING * (G / l) ** 0.5

        # ── drift vector f(x, 0)  [N,10] ─────────────────────────────────
        f = torch.zeros(N, 10, device=dv)
        f[:, 0:3] = dq                                  # q̇ = dq
        f[:, 3:6] = qddot_drift                         # dq̇ = -d*dq
        f[:, 6]   = vxs
        f[:, 7]   = vys
        f[:, 8]   = -ax_d - lamb0*xs - c_damp*vxs
        f[:, 9]   = -ay_d - lamb0*ys - c_damp*vys

        # ── control matrix g(x)  [N,10,3] ────────────────────────────────
        # Arm rows 3-5: dq̇ = K·u  →  g[3:6,:] = K
        # Slosh rows 8-9: see derivation in slosh_dynamics.py
        #   g_vx = -JK[0,:] + xs·(r_sl @ JK) / l²
        #   g_vy = -JK[1,:] + ys·(r_sl @ JK) / l²
        JK   = J @ K_t.unsqueeze(0)                    # (N,3,3)
        r_sl = torch.stack([xs, ys, zs], dim=-1)       # (N,3)
        r_JK = (r_sl.unsqueeze(1) @ JK).squeeze(1)     # (N,3)

        g = torch.zeros(N, 10, 3, device=dv)
        g[:, 3:6, :] = K_t.unsqueeze(0).expand(N, -1, -1)
        g[:, 8, :] = -JK[:, 0, :] + xs.unsqueeze(-1) * r_JK / l**2
        g[:, 9, :] = -JK[:, 1, :] + ys.unsqueeze(-1) * r_JK / l**2

        # ── H = p·f + Σ_i u_max_i·|(p^T g)_i| ───────────────────────────
        p_g     = (p.unsqueeze(1) @ g).squeeze(1)      # (N,3)
        u_max_t = torch.as_tensor(self.u_max, dtype=torch.float32, device=dv)
        H = (p * f).sum(-1) + (p_g.abs() * u_max_t).sum(-1)

        # Zero out H for any infeasible slosh states (xs²+ys² ≥ l²).
        H = torch.where(feasible, H, torch.zeros_like(H))

        return H.reshape(shape)

    def continuous_dynamics(self, state, u):
        """Exact continuous vector field used by the verified Hamiltonian."""
        x = np.asarray(state, dtype=np.float64).reshape(10)
        u = np.asarray(u, dtype=np.float64).reshape(3)
        q, dq = x[:3], x[3:6]
        xs, ys, vxs, vys = x[6:]

        phi = np.concatenate([q, dq])
        J = jacobian(phi, self.L)
        Jd = jacobian_dot(phi, self.L)
        qddot = -ARM_DAMPING * dq + np.asarray(self.K, dtype=np.float64) @ u
        a_cup = Jd @ dq + J @ qddot

        l = self.l_eff
        z = -np.sqrt(max(l*l - xs*xs - ys*ys, 1e-12))
        vz = -(xs*vxs + ys*vys) / z
        v_sq = vxs*vxs + vys*vys + vz*vz
        total_a = a_cup + np.array([0.0, 0.0, G])
        lamb = (v_sq - np.dot(np.array([xs, ys, z]), total_a)) / (l*l)
        c_damp = 2.0 * SLOSH_DAMPING * np.sqrt(G / l)

        return np.array([
            dq[0], dq[1], dq[2],
            qddot[0], qddot[1], qddot[2],
            vxs, vys,
            -a_cup[0] - lamb*xs - c_damp*vxs,
            -a_cup[1] - lamb*ys - c_damp*vys,
        ], dtype=np.float64)

    def dynamics_step(self, state, u):
        state = np.asarray(state, dtype=np.float64).reshape(10)
        return (state + self.dt * self.continuous_dynamics(state, u)).astype(np.float32)

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
