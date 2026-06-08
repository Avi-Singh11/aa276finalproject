"""Train the 10D coffee arm BRT with DeepReach."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEEPREACH_PATH = os.path.dirname(PROJECT_ROOT)
if DEEPREACH_PATH not in sys.path:
    sys.path.insert(0, DEEPREACH_PATH)

from dynamics.dynamics_OVERNIGHT_VERIFIED import ARM_LINK_RADIUS, CoffeeArmDynamics

try:
    from deepreach.utils import modules, losses
except Exception as exc:
    modules = losses = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "brt_model_final2", "model_final2.pth")

CFG = dict(
    epochs=200_000,
    lr=2e-5,
    pretrain_lr=2e-4,
    pretrain_iters=20_000,
    numpoints=32_000,
    num_src_samples=1500,
    tMin=0.0,
    tMax=10.0,
    counter_end=100_000,
    minWith="target",
    hidden_features=256,
    num_hidden_layers=3,
    save_dir="brt_model_OVERNIGHT_VERIFIED",
    checkpoint_every=5_000,
    n_targeted=512,
    targeted_freq=3,
    targeted_weight=3.0,
    n_pretrain_targeted=512,
    causal_eps=1.0,
    causal_n_bins=10,
    static_traj_n=512,
    static_traj_weight=5.0,
    pretrain_pool_size=500_000,
    pretrain_batch_size=32_000,
    static_pool_size=50_000,
    seed=42,
)

CFG_FAST = {
    **CFG,
    "epochs": 50_000,
    "pretrain_iters": 8_000,
    "numpoints": 16_000,
    "hidden_features": 128,
    "num_hidden_layers": 3,
    "n_targeted": 256,
    "static_traj_n": 256,
    "checkpoint_every": 10_000,
    "counter_end": 25_000,
    "save_dir": "brt_model_OVERNIGHT_VERIFIED_SMOKE",
}


def _boundary_fn_torch(dynamics, x: torch.Tensor) -> torch.Tensor:
    """Compute the boundary value for physical states."""
    q = x[..., :3]
    l1, l2, l3 = (
        torch.as_tensor(dynamics.L[i], dtype=x.dtype, device=x.device) for i in range(3)
    )
    l_total = l1 + l2 + l3

    t1, t2, t3 = q.unbind(dim=-1)
    s1, c1 = torch.sin(t1), torch.cos(t1)
    s2, c2 = torch.sin(t2), torch.cos(t2)
    s3, c3 = torch.sin(t3), torch.cos(t3)
    cup_z = l1 + l2 * s2 + l3 * (c2 * s3 + c3 * s2)
    slosh_disp = torch.linalg.vector_norm(x[..., 6:8], dim=-1)

    joint_limits = torch.as_tensor(
        dynamics.joint_limits, dtype=x.dtype, device=x.device
    )
    joint_margin = torch.amin(joint_limits - torch.abs(q), dim=-1)

    zero = torch.zeros_like(t1)
    p0 = torch.stack([zero, zero, zero], dim=-1)
    p1 = torch.stack([zero, zero, torch.ones_like(t1) * l1], dim=-1)
    p2 = torch.stack([l2 * c1 * c2, l2 * c2 * s1, l1 + l2 * s2], dim=-1)
    p3 = torch.stack(
        [
            c1 * (l2 * c2 + l3 * (c2 * c3 - s2 * s3)),
            s1 * (l2 * c2 + l3 * (c2 * c3 - s2 * s3)),
            cup_z,
        ],
        dim=-1,
    )
    seg_a = torch.stack([p0, p1, p2], dim=-2)
    seg_b = torch.stack([p1, p2, p3], dim=-2)
    ab = seg_b - seg_a
    ab_sq = torch.clamp(torch.sum(ab * ab, dim=-1, keepdim=True), min=1e-12)

    obs_margin = torch.full_like(t1, torch.inf)
    for obs in dynamics.obstacles:
        center = torch.as_tensor(obs["center"], dtype=x.dtype, device=x.device)
        radius = torch.as_tensor(obs["radius"], dtype=x.dtype, device=x.device)
        ac = center - seg_a
        projection = torch.clamp(
            torch.sum(ac * ab, dim=-1, keepdim=True) / ab_sq, 0.0, 1.0
        )
        closest = seg_a + projection * ab
        distance = torch.linalg.vector_norm(center - closest, dim=-1)
        clearance = torch.amin(distance, dim=-1) - radius
        obs_margin = torch.minimum(
            obs_margin, torch.clamp(clearance / l_total, max=1.0)
        )

    t_sc = torch.linspace(0.0, 1.0, 15, dtype=x.dtype, device=x.device)
    link3_points = p2.unsqueeze(-2) + t_sc.view(1, -1, 1) * (p3 - p2).unsqueeze(-2)
    sc_z_closest = torch.clamp(link3_points[..., 2], 0.0, float(dynamics.L[0]))
    sc_distance = torch.sqrt(
        link3_points[..., 0] ** 2
        + link3_points[..., 1] ** 2
        + (link3_points[..., 2] - sc_z_closest) ** 2
    )
    self_margin = torch.clamp(
        (torch.amin(sc_distance, dim=-1) - ARM_LINK_RADIUS) / l_total, max=1.0
    )

    margins = torch.stack(
        [
            torch.clamp(cup_z / l_total, max=1.0),
            (dynamics.slosh_rad_max - slosh_disp) / (10.0 * dynamics.slosh_rad_max),
            torch.clamp(joint_margin / joint_limits[0], max=1.0),
            obs_margin,
            self_margin,
        ],
        dim=-1,
    )
    return torch.amin(margins, dim=-1)


def _targeted_boundary_states(
    dynamics, n: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate physical states near each constraint boundary."""
    if rng is None:
        rng = np.random.default_rng()
    states: list[np.ndarray] = []
    L = dynamics.L
    l1 = float(L[0])
    l2 = float(L[1])

    n_per_obs = max(1, int(n * 0.35 / max(len(dynamics.obstacles), 1)))
    n_slosh = max(1, int(n * 0.15))
    n_ground = max(1, int(n * 0.10))
    n_selfc = max(1, int(n * 0.10))
    n_joint = max(
        1,
        n - len(dynamics.obstacles) * n_per_obs - n_slosh - n_ground - n_selfc,
    )

    for obs in dynamics.obstacles:
        cx, cy, cz = obs["center"]
        r = float(obs["radius"])
        q1_tgt = float(np.arctan2(cy, cx))
        elev = np.clip((cz - l1) / l2, -0.98, 0.98)
        q2_tgt = float(np.arcsin(elev))
        for _ in range(n_per_obs):
            q1 = q1_tgt + rng.uniform(-0.7, 0.7)
            q2 = q2_tgt + rng.uniform(-0.5, 0.5)
            q3 = rng.uniform(-0.6, 0.6)
            dq = rng.uniform(-4.0, 4.0, 3)
            sx = rng.uniform(
                -dynamics.slosh_rad_max * 1.3, dynamics.slosh_rad_max * 1.3
            )
            sy = rng.uniform(
                -dynamics.slosh_rad_max * 1.3, dynamics.slosh_rad_max * 1.3
            )
            vxy = rng.uniform(-0.25, 0.25, 2)
            states.append(np.array([q1, q2, q3, *dq, sx, sy, *vxy], dtype=np.float32))

    for _ in range(n_slosh):
        q = rng.uniform(-1.0, 1.0, 3)
        dq = rng.uniform(-3.0, 3.0, 3)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        rad = dynamics.slosh_rad_max * rng.uniform(0.65, 1.35)
        sx, sy = rad * np.cos(ang), rad * np.sin(ang)
        vxy = rng.uniform(-0.25, 0.25, 2)
        states.append(np.array([*q, *dq, sx, sy, *vxy], dtype=np.float32))

    for _ in range(n_ground):
        q1 = rng.uniform(-np.pi, np.pi)
        q2 = rng.uniform(-1.0, -0.3)
        q3 = rng.uniform(-0.5, 0.5)
        dq = rng.uniform(-3.0, 3.0, 3)
        sl = rng.uniform(-dynamics.slosh_rad_max * 0.5, dynamics.slosh_rad_max * 0.5, 4)
        states.append(np.array([q1, q2, q3, *dq, *sl], dtype=np.float32))

    for _ in range(n_selfc):
        sign = rng.choice([-1.0, 1.0])
        q1 = sign * (np.pi + rng.uniform(-0.5, 0.0))
        q2 = rng.uniform(-0.4, 0.6)
        q3 = rng.uniform(0.4, 1.6)
        dq = rng.uniform(-3.0, 3.0, 3)
        sl = rng.uniform(-dynamics.slosh_rad_max * 0.5, dynamics.slosh_rad_max * 0.5, 4)
        states.append(np.array([q1, q2, q3, *dq, *sl], dtype=np.float32))

    for _ in range(n_joint):
        k = int(rng.integers(3))
        q = rng.uniform(-0.8, 0.8, 3).astype(np.float32)
        q[k] = (
            float(rng.choice([-1.0, 1.0]))
            * float(dynamics.joint_limits[k])
            * rng.uniform(0.82, 1.05)
        )
        dq = rng.uniform(-3.0, 3.0, 3)
        sl = rng.uniform(-dynamics.slosh_rad_max * 0.5, dynamics.slosh_rad_max * 0.5, 4)
        states.append(np.array([*q, *dq, *sl], dtype=np.float32))

    return np.array(states, dtype=np.float32)


def build_model(dynamics, cfg=CFG):
    if modules is None:
        raise ImportError(f"DeepReach utils could not be imported: {_IMPORT_ERROR}")
    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type="sine",
        mode="mlp",
        final_layer_factor=1,
        hidden_features=cfg["hidden_features"],
        num_hidden_layers=cfg["num_hidden_layers"],
    )
    return model.to(DEVICE)


def train_brt(cfg=CFG, model=None, dynamics=None):
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "DeepReach utilities are unavailable. Install/point the deepreach package correctly "
            f"before training the BRT. Original import error: {_IMPORT_ERROR}"
        )

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    ckpt_dir = os.path.join(cfg["save_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(cfg["save_dir"], "training_config.json"), "w") as handle:
        json.dump(cfg, handle, indent=2, sort_keys=True)

    if dynamics is None:
        dynamics = CoffeeArmDynamics()
    if model is None:
        model = build_model(dynamics, cfg)

    loss_fn = losses.init_brt_hjivi_loss(
        dynamics,
        minWith=cfg["minWith"],
        dirichlet_loss_divisor=cfg["num_src_samples"],
        causal_eps=cfg.get("causal_eps", 0.0),
        n_causal_bins=cfg.get("causal_n_bins", 10),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    print(f"Training 10D BRT on {DEVICE}")
    print(f"  Epochs: {cfg['epochs']}, tMax={cfg['tMax']}s")
    print(f"  Network: {cfg['hidden_features']} x {cfg['num_hidden_layers']} SIREN\n")
    POOL_SIZE = cfg.get("pretrain_pool_size", 500_000)
    BATCH_SIZE = cfg.get("pretrain_batch_size", 32_000)
    rng_pool = np.random.default_rng(seed)

    print(f"Building pretrain pool ({POOL_SIZE:,} states, balanced safe/unsafe) ...")
    scale_np = dynamics.state_scale  # (10,)

    n_third = POOL_SIZE // 3
    n_targeted_pool = n_third
    n_uniform_pool = n_third
    n_safe_pool = POOL_SIZE - n_targeted_pool - n_uniform_pool

    # Add targeted boundary samples.
    tgt_chunks = []
    while sum(len(c) for c in tgt_chunks) < n_targeted_pool:
        tgt_chunks.append(_targeted_boundary_states(dynamics, 4096, rng_pool))
    tgt_pool_phys = np.concatenate(tgt_chunks, axis=0)[:n_targeted_pool]

    uni_norm = rng_pool.uniform(-1, 1, (n_uniform_pool, dynamics.state_dim)).astype(
        np.float32
    )
    uni_pool_phys = (uni_norm * scale_np).astype(np.float32)

    safe_chunks, collected = [], 0
    while collected < n_safe_pool:
        cands_norm = rng_pool.uniform(-1, 1, (8192, dynamics.state_dim)).astype(
            np.float32
        )
        cands_phys = (cands_norm * scale_np).astype(np.float32)
        keep = cands_phys[dynamics.boundary_fn(cands_phys) > 0]
        safe_chunks.append(keep)
        collected += len(keep)
    safe_pool_phys = np.concatenate(safe_chunks, axis=0)[:n_safe_pool]

    pool_phys = np.concatenate([tgt_pool_phys, uni_pool_phys, safe_pool_phys], axis=0)
    pool_lx = dynamics.boundary_fn(pool_phys).astype(np.float32)
    pool_norm = (pool_phys / scale_np).astype(np.float32)

    pool_norm_gpu = torch.tensor(pool_norm, dtype=torch.float32, device=DEVICE)
    pool_lx_gpu = torch.tensor(pool_lx, dtype=torch.float32, device=DEVICE)
    N_pool = len(pool_phys)

    safe_frac = float((pool_lx > 0).mean())

    print("Building static safe trajectory pool ...")
    static_safe_chunks, static_collected = [], 0
    N_STATIC_POOL = cfg.get("static_pool_size", 50_000)
    while static_collected < N_STATIC_POOL:
        q_rand = (
            rng_pool.uniform(-1, 1, (8192, 3)).astype(np.float32)
            * dynamics.joint_limits
        )
        zeros = np.zeros((8192, 7), dtype=np.float32)
        zero_slosh = np.concatenate([q_rand, zeros], axis=1)  # joints + zero slosh

        # Only zero slosh displacement and velocity is an equilibrium.
        cands = zero_slosh
        lx_cand = dynamics.boundary_fn(cands)
        keep = cands[lx_cand > 0.05]
        static_safe_chunks.append(keep)
        static_collected += len(keep)
    static_safe_phys = np.concatenate(static_safe_chunks)[:N_STATIC_POOL]
    static_safe_lx = dynamics.boundary_fn(static_safe_phys).astype(np.float32)
    static_safe_norm = (static_safe_phys / scale_np).astype(np.float32)
    static_safe_norm_gpu = torch.tensor(
        static_safe_norm, dtype=torch.float32, device=DEVICE
    )
    static_safe_lx_gpu = torch.tensor(
        static_safe_lx, dtype=torch.float32, device=DEVICE
    )
    N_static_safe = len(static_safe_phys)
    print(
        f"  Static safe pool: {N_static_safe:,} states, l(x) mean={static_safe_lx.mean():.4f}"
    )

    print("Building static unsafe trajectory pool ...")
    static_unsafe_chunks, static_unsafe_collected = [], 0
    while static_unsafe_collected < N_STATIC_POOL:
        q_rand = (
            rng_pool.uniform(-1, 1, (8192, 3)).astype(np.float32)
            * dynamics.joint_limits
        )
        zeros = np.zeros((8192, 7), dtype=np.float32)
        cands = np.concatenate([q_rand, zeros], axis=1)
        lx_cand = dynamics.boundary_fn(cands)
        keep = cands[lx_cand < -0.05]
        static_unsafe_chunks.append(keep)
        static_unsafe_collected += len(keep)
    static_unsafe_phys = np.concatenate(static_unsafe_chunks)[:N_STATIC_POOL]
    static_unsafe_lx = dynamics.boundary_fn(static_unsafe_phys).astype(np.float32)
    static_unsafe_norm = (static_unsafe_phys / scale_np).astype(np.float32)
    static_unsafe_norm_gpu = torch.tensor(
        static_unsafe_norm, dtype=torch.float32, device=DEVICE
    )
    static_unsafe_lx_gpu = torch.tensor(
        static_unsafe_lx, dtype=torch.float32, device=DEVICE
    )
    N_static_unsafe = len(static_unsafe_phys)
    print(
        f"  Static unsafe pool: {N_static_unsafe:,} states, l(x) mean={static_unsafe_lx.mean():.4f}\n"
    )
    print(
        f"  Pool ready: {N_pool:,} states, {safe_frac * 100:.1f}% safe, "
        f"l(x) std={pool_lx.std():.4f}"
    )

    model.train()
    pretrain_optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.get("pretrain_lr", 2e-4)
    )
    for i in tqdm(range(cfg["pretrain_iters"]), desc="Pretrain"):
        idx = torch.randint(0, N_pool, (BATCH_SIZE,), device=DEVICE)
        x_bat = pool_norm_gpu[idx]  # (B, 10)
        l_bat = pool_lx_gpu[idx]  # (B,)
        t_bat = torch.full((BATCH_SIZE, 1), cfg["tMin"], device=DEVICE)

        coords = torch.cat([t_bat, x_bat], dim=1)  # (B, 11)
        out = model({"coords": coords})
        V = dynamics.io_to_value(out["model_in"].detach(), out["model_out"].squeeze(-1))
        loss = torch.abs(V - l_bat).mean()
        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()

        if (i + 1) % 500 == 0:
            tqdm.write(
                f"  pretrain [{i + 1:>6}] loss={loss.item():.5f}  V_std={V.detach().std().item():.5f}"
            )
    print("Pretrain done.\n")

    # Generate training points directly on the GPU.
    n_src = cfg["num_src_samples"]
    n_pde = cfg["numpoints"] - n_src

    log_every = cfg.get("log_every", 500)
    model.train()
    for epoch in tqdm(range(cfg["epochs"]), desc="BRT training"):
        # Increase the time horizon during training.
        t_frac = min((epoch + 1) / cfg["counter_end"], 1.0)
        t_max_curr = cfg["tMin"] + (cfg["tMax"] - cfg["tMin"]) * t_frac
        t_max_curr = max(t_max_curr, cfg["tMin"] + 1e-4)

        # Sample interior PDE points.
        z_pde = torch.empty(n_pde, 10, device=DEVICE).uniform_(-1.0, 1.0)
        t_pde = torch.empty(n_pde, 1, device=DEVICE).uniform_(cfg["tMin"], t_max_curr)

        # Sample terminal boundary points.
        src_idx = torch.randint(0, N_pool, (n_src,), device=DEVICE)
        z_src = pool_norm_gpu[src_idx]  # (n_src, 10)
        t_src = torch.full((n_src, 1), cfg["tMin"], device=DEVICE)
        bv_src = pool_lx_gpu[src_idx]  # (n_src,)

        # Build [time, normalized state] coordinates.
        coords = torch.cat(
            [
                torch.cat([t_pde, z_pde], dim=1),
                torch.cat([t_src, z_src], dim=1),
            ],
            dim=0,
        ).unsqueeze(0)

        scale = torch.as_tensor(dynamics.state_scale, dtype=z_pde.dtype, device=DEVICE)
        bv_pde = _boundary_fn_torch(dynamics, z_pde * scale)
        bv = torch.cat([bv_pde, bv_src], dim=0).unsqueeze(0)  # (1, numpoints)

        dirichlet_mask = torch.zeros(cfg["numpoints"], dtype=torch.bool, device=DEVICE)
        dirichlet_mask[n_pde:] = True
        dirichlet_mask = dirichlet_mask.unsqueeze(0)  # (1, numpoints)

        model_input = {"model_coords": coords}
        gt = {"boundary_values": bv, "dirichlet_masks": dirichlet_mask}

        results = model({"coords": model_input["model_coords"]})
        states = results["model_in"].detach()[..., 1:]
        values = dynamics.io_to_value(
            results["model_in"].detach(), results["model_out"].squeeze(-1)
        )
        dvs = dynamics.io_to_dv(results["model_in"], results["model_out"].squeeze(-1))

        step_losses = loss_fn(
            states,
            values,
            dvs[..., 0],
            dvs[..., 1:],
            gt["boundary_values"],
            gt["dirichlet_masks"],
            results["model_out"],
            times=model_input["model_coords"][..., 0],
        )

        total_loss = (
            step_losses["dirichlet"] + step_losses["diff_constraint_hom"] / n_pde
        )

        if (epoch + 1) % cfg.get("targeted_freq", 3) == 0:
            tgt_idx = torch.randint(0, n_third, (cfg["n_targeted"],), device=DEVICE)
            tgt_norm = pool_norm_gpu[tgt_idx]
            l_tgt = pool_lx_gpu[tgt_idx]
            t_zero = torch.zeros(cfg["n_targeted"], 1, device=DEVICE)
            tgt_coords = torch.cat([t_zero, tgt_norm], dim=1)
            tgt_out = model({"coords": tgt_coords})
            V_tgt = dynamics.io_to_value(
                tgt_out["model_in"].detach(), tgt_out["model_out"].squeeze(-1)
            )
            targeted_loss = torch.abs(V_tgt - l_tgt).mean()
            total_loss = total_loss + cfg.get("targeted_weight", 3.0) * targeted_loss

        n_st = cfg.get("static_traj_n", 512)
        n_st_half = n_st // 2
        t_range = (cfg["tMin"], max(t_max_curr, cfg["tMin"] + 0.01))

        st_idx = torch.randint(0, N_static_safe, (n_st_half,), device=DEVICE)
        st_norm = static_safe_norm_gpu[st_idx]
        st_lx = static_safe_lx_gpu[st_idx]
        st_times = torch.empty(n_st_half, 1, device=DEVICE).uniform_(*t_range)
        st_coords = torch.cat([st_times, st_norm], dim=1)
        st_out = model({"coords": st_coords})
        V_st = dynamics.io_to_value(
            st_out["model_in"].detach(), st_out["model_out"].squeeze(-1)
        )
        static_traj_loss = torch.abs(V_st - st_lx).mean()

        ust_idx = torch.randint(0, N_static_unsafe, (n_st_half,), device=DEVICE)
        ust_norm = static_unsafe_norm_gpu[ust_idx]
        ust_lx = static_unsafe_lx_gpu[ust_idx]
        ust_times = torch.empty(n_st_half, 1, device=DEVICE).uniform_(*t_range)
        ust_coords = torch.cat([ust_times, ust_norm], dim=1)
        ust_out = model({"coords": ust_coords})
        V_ust = dynamics.io_to_value(
            ust_out["model_in"].detach(), ust_out["model_out"].squeeze(-1)
        )
        static_traj_loss = static_traj_loss + torch.abs(V_ust - ust_lx).mean()

        total_loss = total_loss + cfg.get("static_traj_weight", 5.0) * static_traj_loss

        if not torch.isfinite(total_loss):
            failure_path = os.path.join(
                ckpt_dir, f"NONFINITE_epoch_{epoch + 1:06d}.pth"
            )
            torch.save(
                {"epoch": epoch + 1, "model": model.state_dict(), "cfg": cfg},
                failure_path,
            )
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch + 1}; saved {failure_path}"
            )

        optimizer.zero_grad()
        total_loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                failure_path = os.path.join(
                    ckpt_dir, f"NONFINITE_GRAD_epoch_{epoch + 1:06d}.pth"
                )
                torch.save(
                    {"epoch": epoch + 1, "model": model.state_dict(), "cfg": cfg},
                    failure_path,
                )
                raise FloatingPointError(
                    f"Non-finite gradient in {name} at epoch {epoch + 1}; "
                    f"saved {failure_path}"
                )
        optimizer.step()

        if (epoch + 1) % log_every == 0:
            loss_parts = "  ".join(
                f"{k}={v.mean().item():.5f}" for k, v in step_losses.items()
            )
            v_std = values.detach().std().item()
            tqdm.write(
                f"  [{epoch + 1:>6}] {loss_parts}  traj={static_traj_loss.item():.5f}  V_std={v_std:.5f}"
            )

        if (epoch + 1) % cfg["checkpoint_every"] == 0:
            path = os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:06d}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                },
                path,
            )
            t_trained = cfg["tMin"] + (cfg["tMax"] - cfg["tMin"]) * min(
                (epoch + 1) / cfg["counter_end"], 1.0
            )
            model.eval()
            with torch.no_grad():

                def _probe(state_np, t):
                    coord = torch.cat(
                        [torch.tensor([t]), torch.tensor(state_np, dtype=torch.float32)]
                    ).unsqueeze(0)
                    inp = dynamics.coord_to_input(coord).to(DEVICE)
                    out = model({"coords": inp})
                    return float(
                        dynamics.io_to_value(
                            out["model_in"].detach(),
                            out["model_out"].squeeze(-1).detach(),
                        ).item()
                    )

                safe_equilibrium = static_safe_phys[0].copy()
                slosh_unsafe = safe_equilibrium.copy()
                slosh_unsafe[6] = dynamics.slosh_rad_max * 1.2
                states_to_probe = {
                    "safe_eq": safe_equilibrium,
                    "unsafe_eq": static_unsafe_phys[0],
                    "slosh": slosh_unsafe,
                    "gnd": np.array(
                        [0, -1.2, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
                    ),
                }
                probes_trained = {
                    k: _probe(v, t_trained) for k, v in states_to_probe.items()
                }
                probes_full = {
                    k: _probe(v, cfg["tMax"]) for k, v in states_to_probe.items()
                }
                probe_targets = {
                    k: float(dynamics.boundary_fn(v))
                    for k, v in states_to_probe.items()
                }
            h_diag = dynamics.hamiltonian(states, dvs[..., 1:]).detach()
            h_abs_max = float(h_diag.abs().max().item())
            h_clipped = int((h_diag.abs() >= 49.999).sum().item())
            model.train()
            trained_str = "  ".join(
                f"V({k})={v:+.3f}" for k, v in probes_trained.items()
            )
            full_str = "  ".join(f"V({k})={v:+.3f}" for k, v in probes_full.items())
            target_str = "  ".join(f"l({k})={v:+.3f}" for k, v in probe_targets.items())
            tqdm.write(
                f"  [{epoch + 1}] loss={total_loss.item():.5f}  t_trained={t_trained:.2f}s\n"
                f"target:     {target_str}\n"
                f"@t_trained: {trained_str}\n"
                f"@t_full:    {full_str}\n"
                f"H_abs_max={h_abs_max:.3f}  H_clipped={h_clipped}/{h_diag.numel()}\n"
                f"saved {path}"
            )

    final = os.path.join(cfg["save_dir"], "model_final.pth")
    torch.save(model.state_dict(), final)
    print(f"\nSaved: {final}")
    return model, dynamics


def load_model(model_path=FINAL_MODEL_PATH):
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "DeepReach utilities are unavailable; cannot load a trained BRT model through this script."
        )
    dynamics = CoffeeArmDynamics()
    model = build_model(dynamics)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, dynamics


def query_value(model, dynamics, state_10d, t=None):
    """Query V(x, t) for a single 10D state."""
    if t is None:
        t = CFG["tMax"]

    if not isinstance(state_10d, torch.Tensor):
        state_10d = torch.tensor(state_10d, dtype=torch.float32)

    coord = torch.cat(
        [torch.tensor([t], dtype=torch.float32), state_10d.cpu()]
    ).unsqueeze(0)
    inp = dynamics.coord_to_input(coord).to(DEVICE)

    with torch.no_grad():
        out = model({"coords": inp})
        V = dynamics.io_to_value(
            out["model_in"].detach(), out["model_out"].squeeze(-1).detach()
        )
    return float(V.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Load saved model and run query checks instead of training",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick ~15-min run with CFG_FAST for validating fixes",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run a tiny end-to-end training check before the overnight job",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth to fine-tune from",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=20_000,
        help="Number of additional epochs when using --resume (default: 20000)",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=5e-6,
        help="Learning rate for fine-tuning (default: 5e-6, lower than training to avoid forgetting)",
    )
    parser.add_argument(
        "--finetune-dir",
        type=str,
        default="brt_model_finetuned",
        help="Output directory for fine-tuned model (default: brt_model_finetuned)",
    )
    args = parser.parse_args()

    if args.test:
        model, dyn = load_model()
        print("Model loaded. Running query checks...\n")

        checks = [
            (np.zeros(10), "origin (all zero)", "> 0 (safe)"),
            (
                np.array([0] * 6 + [dyn.slosh_rad_max * 0.5, 0, 0, 0]),
                "x_slosh = 0.5*max",
                "> 0 (safe)",
            ),
            (
                np.array([0] * 6 + [dyn.slosh_rad_max * 0.95, 0, 0, 0]),
                "x_slosh = 0.95*max",
                "near 0",
            ),
            (
                np.array([0] * 6 + [dyn.slosh_rad_max * 1.1, 0, 0, 0]),
                "x_slosh = 1.1*max",
                "< 0 (unsafe)",
            ),
            (
                np.array([0, -1.2, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                "q2=-1.2 (below ground)",
                "< 0 (unsafe)",
            ),
        ]

        for s, desc, expect in checks:
            v = query_value(model, dyn, s)
            print(f"  V({desc:40s}) = {v:+.4f}  (expect {expect})")

    elif args.resume is not None:
        # Fine-tune a checkpoint with the corrected static pool.
        cfg = CFG_FAST if args.fast else CFG
        ft_cfg = {
            **cfg,
            "epochs": args.finetune_epochs,
            "lr": args.finetune_lr,
            "pretrain_iters": 0,  # skip pretrain — model already initialised
            "save_dir": args.finetune_dir,
            "checkpoint_every": max(1000, args.finetune_epochs // 10),
        }
        print(f"── FINE-TUNE mode: loading {args.resume} ──")
        print(
            f"   epochs={ft_cfg['epochs']}  lr={ft_cfg['lr']}  → {ft_cfg['save_dir']}"
        )
        dyn = CoffeeArmDynamics()
        ft_model = build_model(dyn, ft_cfg)
        ft_model.load_state_dict(torch.load(args.resume, map_location=DEVICE))
        train_brt(cfg=ft_cfg, model=ft_model, dynamics=dyn)

    elif args.preflight:
        preflight_cfg = {
            **CFG_FAST,
            "epochs": 20,
            "pretrain_iters": 20,
            "numpoints": 2048,
            "num_src_samples": 256,
            "hidden_features": 64,
            "num_hidden_layers": 2,
            "n_targeted": 64,
            "static_traj_n": 64,
            "pretrain_pool_size": 20_000,
            "pretrain_batch_size": 2048,
            "static_pool_size": 2_000,
            "checkpoint_every": 10,
            "counter_end": 20,
            "log_every": 5,
            "save_dir": "brt_model_OVERNIGHT_VERIFIED_PREFLIGHT",
        }
        train_brt(cfg=preflight_cfg)

    else:
        cfg = CFG_FAST if args.fast else CFG
        if args.fast:
            print("── FAST mode: using CFG_FAST (~15 min) ──")
        train_brt(cfg=cfg)
