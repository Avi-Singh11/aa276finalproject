"""Train a BRT for the 10D coffee arm model using DeepReach.

This script keeps the current user-facing interface but uses the shared core
physics via dynamics.dynamics.CoffeeArmDynamics.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. Add your project root so Python can find 'dynamics.dynamics'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. Add the parent of the deepreach package so 'from deepreach.utils import ...' resolves
DEEPREACH_PATH = os.path.dirname(PROJECT_ROOT)
if DEEPREACH_PATH not in sys.path:
    sys.path.insert(0, DEEPREACH_PATH)

# Now these imports will resolve flawlessly relative to your true folder structure!
from dynamics.dynamics import CoffeeArmDynamics

try:
    from deepreach.utils import modules, dataio, losses
except Exception as exc:
    modules = dataio = losses = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = dict(
    epochs=200_000,
    lr=2e-5,
    pretrain_lr=2e-4,
    pretrain_iters=20_000,       # double: need longer to cover all 5 constraints
    numpoints=32_000,
    num_src_samples=1500,        # 3× more boundary anchoring
    tMin=0.0,
    tMax=3.0,
    counter_end=100_000,
    minWith='target',
    hidden_features=256,
    num_hidden_layers=3,
    save_dir='brt_model',
    checkpoint_every=5_000,
    # ── targeted boundary augmentation ───────────────────────────────────────
    # Every `targeted_freq` epochs we inject `n_targeted` states sampled near
    # every constraint boundary (obstacles, slosh, ground, self-collision, joints)
    # and enforce V(x,0)=l(x) on them with a weighted extra loss term.
    # This fixes the core problem: uniform sampling leaves obstacle/self-collision
    # boundaries severely under-represented in 10D space.
    n_targeted=512,
    targeted_freq=3,             # apply every 3 epochs
    targeted_weight=3.0,         # extra weight on targeted boundary loss
    n_pretrain_targeted=512,     # targeted states mixed into each pretrain step
    # ── causal PDE weighting (Wang et al. 2022) ───────────────────────────────
    # Weights PDE residuals by exp(-eps * cumulative_earlier_residual) so the
    # network must satisfy t≈0 before fitting t>0, preventing the degenerate
    # V=const solution. eps=1.0 is a good default; set to 0 to disable.
    causal_eps=1.0,
    causal_n_bins=10,
    # ── Static trajectory loss (safe AND unsafe) ─────────────────────────────
    # For arm configs with zero velocity/slosh, u=0 is optimal (arm stays
    # still), so V(x,t) = l(x) for ALL t — regardless of sign.
    # safe pool:   l(x) > 0  → prevents degenerate V=const<0 solution
    # unsafe pool: l(x) < 0  → prevents network drifting positive in ground-
    #              collision / other failure regions (the joint-space bug)
    static_traj_n=512,        # states per step (split evenly safe/unsafe)
    static_traj_weight=5.0,   # loss weight; needs to dominate PDE loss (~0.02)
)

CFG_FAST = {
    **CFG,
    'epochs':           50_000,
    'pretrain_iters':   8_000,
    'numpoints':        16_000,
    'hidden_features':  128,
    'num_hidden_layers': 3,
    'n_targeted':       256,
    'static_traj_n':    256,
    'checkpoint_every': 10_000,
    'counter_end':      25_000,   # curriculum completes by epoch 25K; rest refines at tMax
    'save_dir':         'brt_model_fast',
}

def _targeted_boundary_states(dynamics, n: int) -> np.ndarray:
    """Generate physical states analytically near every constraint boundary.

    Distribution (approximate):
      ~35% near each obstacle  (analytically targeted by azimuth/elevation)
      ~15% near slosh limit    (random arm, slosh near threshold)
      ~10% near ground         (arm tilted down, cup near z=0)
      ~10% near self-collision (arm wrapping back toward base column)
      ~10% near joint limits   (one joint at ±pi)
    All states are returned in *physical* units (not normalised).
    """
    rng = np.random.default_rng()
    states: list[np.ndarray] = []
    L  = dynamics.L
    l1 = float(L[0])
    l2 = float(L[1])

    n_per_obs  = max(1, int(n * 0.35 / max(len(dynamics.obstacles), 1)))
    n_slosh    = max(1, int(n * 0.15))
    n_ground   = max(1, int(n * 0.10))
    n_selfc    = max(1, int(n * 0.10))
    n_joint    = max(1, n - 2*n_per_obs - n_slosh - n_ground - n_selfc)

    # ── Obstacle-targeted ────────────────────────────────────────────────────
    for obs in dynamics.obstacles:
        cx, cy, cz = obs['center']
        r = float(obs['radius'])
        q1_tgt = float(np.arctan2(cy, cx))
        elev   = np.clip((cz - l1) / l2, -0.98, 0.98)
        q2_tgt = float(np.arcsin(elev))
        for _ in range(n_per_obs):
            # Spread uniformly around the analytically targeted config
            q1  = q1_tgt + rng.uniform(-0.7, 0.7)
            q2  = q2_tgt + rng.uniform(-0.5, 0.5)
            q3  = rng.uniform(-0.6, 0.6)
            dq  = rng.uniform(-4.0, 4.0, 3)
            # Slosh: full range including near-boundary
            sx  = rng.uniform(-dynamics.slosh_rad_max * 1.3, dynamics.slosh_rad_max * 1.3)
            sy  = rng.uniform(-dynamics.slosh_rad_max * 1.3, dynamics.slosh_rad_max * 1.3)
            vxy = rng.uniform(-0.25, 0.25, 2)
            states.append(np.array([q1, q2, q3, *dq, sx, sy, *vxy], dtype=np.float32))

    # ── Near slosh limit ─────────────────────────────────────────────────────
    for _ in range(n_slosh):
        q   = rng.uniform(-1.0, 1.0, 3)
        dq  = rng.uniform(-3.0, 3.0, 3)
        ang = rng.uniform(0.0, 2.0*np.pi)
        rad = dynamics.slosh_rad_max * rng.uniform(0.65, 1.35)
        sx, sy = rad*np.cos(ang), rad*np.sin(ang)
        vxy = rng.uniform(-0.25, 0.25, 2)
        states.append(np.array([*q, *dq, sx, sy, *vxy], dtype=np.float32))

    # ── Near ground (cup z ≈ 0) ──────────────────────────────────────────────
    # Straddle the cup_z=0 boundary (at q2≈-0.57 rad for q3=0) so the network
    # learns where l(x)=0 is, not just that it's negative deep inside.
    for _ in range(n_ground):
        q1  = rng.uniform(-np.pi, np.pi)
        q2  = rng.uniform(-1.0, -0.3)       # straddles cup_z=0 boundary at q2≈-0.57
        q3  = rng.uniform(-0.5, 0.5)
        dq  = rng.uniform(-3.0, 3.0, 3)
        sl  = rng.uniform(-dynamics.slosh_rad_max*0.5, dynamics.slosh_rad_max*0.5, 4)
        states.append(np.array([q1, q2, q3, *dq, *sl], dtype=np.float32))

    # ── Near self-collision (Link 3 toward base column) ──────────────────────
    for _ in range(n_selfc):
        sign = rng.choice([-1.0, 1.0])
        q1   = sign * (np.pi + rng.uniform(-0.5, 0.0))
        q2   = rng.uniform(-0.4, 0.6)
        q3   = rng.uniform(0.4, 1.6)
        dq   = rng.uniform(-3.0, 3.0, 3)
        sl   = rng.uniform(-dynamics.slosh_rad_max*0.5, dynamics.slosh_rad_max*0.5, 4)
        states.append(np.array([q1, q2, q3, *dq, *sl], dtype=np.float32))

    # ── Near joint limits ────────────────────────────────────────────────────
    for _ in range(n_joint):
        k   = int(rng.integers(3))
        q   = rng.uniform(-0.8, 0.8, 3).astype(np.float32)
        q[k] = float(rng.choice([-1.0, 1.0])) * float(dynamics.joint_limits[k]) * rng.uniform(0.82, 1.05)
        dq  = rng.uniform(-3.0, 3.0, 3)
        sl  = rng.uniform(-dynamics.slosh_rad_max*0.5, dynamics.slosh_rad_max*0.5, 4)
        states.append(np.array([*q, *dq, *sl], dtype=np.float32))

    return np.array(states, dtype=np.float32)


def build_model(dynamics, cfg=CFG):
    if modules is None:
        raise ImportError(f"DeepReach utils could not be imported: {_IMPORT_ERROR}")
    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type='sine',
        mode='mlp',
        final_layer_factor=1,
        hidden_features=cfg['hidden_features'],
        num_hidden_layers=cfg['num_hidden_layers'],
    )
    return model.to(DEVICE)

def train_brt(cfg=CFG):
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "DeepReach utilities are unavailable. Install/point the deepreach package correctly "
            f"before training the BRT. Original import error: {_IMPORT_ERROR}"
        )

    os.makedirs(cfg['save_dir'], exist_ok=True)
    ckpt_dir = os.path.join(cfg['save_dir'], 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    dynamics = CoffeeArmDynamics()
    model = build_model(dynamics, cfg)

    loss_fn = losses.init_brt_hjivi_loss(
        dynamics,
        minWith=cfg['minWith'],
        dirichlet_loss_divisor=cfg['num_src_samples'],
        causal_eps=cfg.get('causal_eps', 0.0),
        n_causal_bins=cfg.get('causal_n_bins', 10),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    print(f"Training 10D BRT on {DEVICE}")
    print(f"  Epochs: {cfg['epochs']}, tMax={cfg['tMax']}s")
    print(f"  Network: {cfg['hidden_features']} x {cfg['num_hidden_layers']} SIREN\n")

    # Pretrain V(x,t) = l(x) for all t in [tMin, tMax].
    # The original DeepReach pretrain only uses t=0, leaving V at t>0 completely
    # untrained. When the PDE loss kicks in, those random values cause a large
    # initial residual that overwrites the spatial structure. Pretraining across
    # all t gives a globally consistent warm start where V_t=0 and the PDE
    # residual max(-H, 0)=0 for safe states (H>0), so training begins smoothly.
    # ── Build a balanced pretrain pool once on CPU, then sample GPU mini-batches ──
    # 85% of uniform states have l(x) < 0, so per-step uniform sampling causes the
    # L1 loss to collapse to a constant negative output (V_std ≈ 0, no spatial structure).
    # Fix: pre-generate a large pool that is ~50% safe / 50% unsafe, compute all
    # boundary_fn values upfront, then mini-batch sample from GPU tensors every step.
    # This also eliminates per-step CPU boundary_fn calls, saturating the GPU.
    POOL_SIZE   = cfg.get('pretrain_pool_size', 500_000)
    BATCH_SIZE  = cfg.get('pretrain_batch_size', 32_000)
    rng_pool    = np.random.default_rng(42)

    print(f"Building pretrain pool ({POOL_SIZE:,} states, balanced safe/unsafe) ...")
    scale_np = dynamics.state_scale  # (10,)

    # Pool composition (three equal thirds):
    #   1/3 targeted   — analytically near every constraint boundary (~50/50 safe/unsafe)
    #   1/3 uniform    — natural distribution, preserves deeply unsafe states for correct
    #                    gradient structure away from the boundary (needed so the PDE
    #                    propagates the zero level set correctly)
    #   1/3 safe-only  — rejection-sampled safe states to rebalance away from the 85%
    #                    unsafe skew of pure uniform sampling
    n_third          = POOL_SIZE // 3
    n_targeted_pool  = n_third
    n_uniform_pool   = n_third
    n_safe_pool      = POOL_SIZE - n_targeted_pool - n_uniform_pool

    # Targeted third
    tgt_chunks = []
    while sum(len(c) for c in tgt_chunks) < n_targeted_pool:
        tgt_chunks.append(_targeted_boundary_states(dynamics, 4096))
    tgt_pool_phys = np.concatenate(tgt_chunks, axis=0)[:n_targeted_pool]

    # Uniform third — no filtering, keeps deep unsafe states for correct V shape
    uni_norm      = rng_pool.uniform(-1, 1, (n_uniform_pool, dynamics.state_dim)).astype(np.float32)
    uni_pool_phys = (uni_norm * scale_np).astype(np.float32)

    # Safe third — rebalances the overall pool away from the 85%-unsafe skew
    safe_chunks, collected = [], 0
    while collected < n_safe_pool:
        cands_norm = rng_pool.uniform(-1, 1, (8192, dynamics.state_dim)).astype(np.float32)
        cands_phys = (cands_norm * scale_np).astype(np.float32)
        keep       = cands_phys[dynamics.boundary_fn(cands_phys) > 0]
        safe_chunks.append(keep)
        collected += len(keep)
    safe_pool_phys = np.concatenate(safe_chunks, axis=0)[:n_safe_pool]

    pool_phys = np.concatenate([tgt_pool_phys, uni_pool_phys, safe_pool_phys], axis=0)
    pool_lx   = dynamics.boundary_fn(pool_phys).astype(np.float32)
    pool_norm = (pool_phys / scale_np).astype(np.float32)

    # Move pool to GPU tensors once
    pool_norm_gpu = torch.tensor(pool_norm, dtype=torch.float32, device=DEVICE)
    pool_lx_gpu   = torch.tensor(pool_lx,   dtype=torch.float32, device=DEVICE)
    N_pool        = len(pool_phys)

    safe_frac = float((pool_lx > 0).mean())

    # ── Static safe trajectory pool ──────────────────────────────────────────
    # States with zero velocity and zero slosh where l(x)>0.
    # For these states optimal control is u=0 (stay still), so V(x,t)=l(x)
    # for all t.  Used to enforce the trajectory loss that breaks V=const<0.
    print("Building static safe trajectory pool ...")
    static_safe_chunks, static_collected = [], 0
    N_STATIC_POOL = 50_000
    while static_collected < N_STATIC_POOL:
        q_rand  = rng_pool.uniform(-1, 1, (8192, 3)).astype(np.float32) * dynamics.joint_limits
        zeros   = np.zeros((8192, 7), dtype=np.float32)
        cands   = np.concatenate([q_rand, zeros], axis=1)  # (8192, 10) physical
        lx_cand = dynamics.boundary_fn(cands)
        keep    = cands[lx_cand > 0.05]
        static_safe_chunks.append(keep)
        static_collected += len(keep)
    static_safe_phys = np.concatenate(static_safe_chunks)[:N_STATIC_POOL]
    static_safe_lx   = dynamics.boundary_fn(static_safe_phys).astype(np.float32)
    static_safe_norm = (static_safe_phys / scale_np).astype(np.float32)
    static_safe_norm_gpu = torch.tensor(static_safe_norm, dtype=torch.float32, device=DEVICE)
    static_safe_lx_gpu   = torch.tensor(static_safe_lx,   dtype=torch.float32, device=DEVICE)
    N_static_safe        = len(static_safe_phys)
    print(f"  Static safe pool: {N_static_safe:,} states, l(x) mean={static_safe_lx.mean():.4f}")

    # ── Static unsafe trajectory pool ────────────────────────────────────────
    # Symmetric counterpart to the safe pool. For static states with l(x)<0
    # (e.g. arm below ground), u=0 keeps the arm there, so V(x,t)=l(x)<0 for
    # all t. Without this, the safe-only static loss creates an imbalance that
    # lets the network drift positive at unsafe static configurations.
    print("Building static unsafe trajectory pool ...")
    static_unsafe_chunks, static_unsafe_collected = [], 0
    while static_unsafe_collected < N_STATIC_POOL:
        q_rand   = rng_pool.uniform(-1, 1, (8192, 3)).astype(np.float32) * dynamics.joint_limits
        zeros    = np.zeros((8192, 7), dtype=np.float32)
        cands    = np.concatenate([q_rand, zeros], axis=1)
        lx_cand  = dynamics.boundary_fn(cands)
        keep     = cands[lx_cand < -0.05]
        static_unsafe_chunks.append(keep)
        static_unsafe_collected += len(keep)
    static_unsafe_phys = np.concatenate(static_unsafe_chunks)[:N_STATIC_POOL]
    static_unsafe_lx   = dynamics.boundary_fn(static_unsafe_phys).astype(np.float32)
    static_unsafe_norm = (static_unsafe_phys / scale_np).astype(np.float32)
    static_unsafe_norm_gpu = torch.tensor(static_unsafe_norm, dtype=torch.float32, device=DEVICE)
    static_unsafe_lx_gpu   = torch.tensor(static_unsafe_lx,   dtype=torch.float32, device=DEVICE)
    N_static_unsafe        = len(static_unsafe_phys)
    print(f"  Static unsafe pool: {N_static_unsafe:,} states, l(x) mean={static_unsafe_lx.mean():.4f}\n")
    print(f"  Pool ready: {N_pool:,} states, {safe_frac*100:.1f}% safe, "
          f"l(x) std={pool_lx.std():.4f}")

    model.train()
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('pretrain_lr', 2e-4))
    for i in tqdm(range(cfg['pretrain_iters']), desc='Pretrain'):
        # Random mini-batch from the pre-built pool (pure GPU, no CPU work per step)
        idx   = torch.randint(0, N_pool, (BATCH_SIZE,), device=DEVICE)
        x_bat = pool_norm_gpu[idx]                                       # (B, 10)
        l_bat = pool_lx_gpu[idx]                                         # (B,)
        t_bat = torch.empty(BATCH_SIZE, 1, device=DEVICE).uniform_(cfg['tMin'], cfg['tMax'])

        coords = torch.cat([t_bat, x_bat], dim=1)                        # (B, 11)
        out    = model({'coords': coords})
        V      = dynamics.io_to_value(out['model_in'].detach(), out['model_out'].squeeze(-1))

        loss = torch.abs(V - l_bat).mean()
        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()

        if (i + 1) % 500 == 0:
            tqdm.write(f"  pretrain [{i+1:>6}] loss={loss.item():.5f}  V_std={V.detach().std().item():.5f}")
    print("Pretrain done.\n")

    # Use the time curriculum (counter_start=0): training begins near t=0 where
    # V ≈ l(x) is well-defined, then gradually expands toward tMax. This forces
    # the PDE to propagate spatial structure (safe vs unsafe) forward through time.
    # The old counter_start=counter_end bypassed the curriculum, letting the network
    # find a degenerate solution V(z,t>0) ≈ f(t) with no spatial variation in slosh.
    dataset = dataio.ReachabilityDataset(
        dynamics=dynamics,
        numpoints=cfg['numpoints'],
        pretrain=False,
        pretrain_iters=0,
        tMin=cfg['tMin'],
        tMax=cfg['tMax'],
        counter_start=0,
        counter_end=cfg['counter_end'],
        num_src_samples=cfg['num_src_samples'],
        num_target_samples=0,
    )
    loader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)

    log_every = cfg.get('log_every', 500)
    model.train()
    for epoch in tqdm(range(cfg['epochs']), desc='BRT training'):
        for model_input, gt in loader:
            model_input = {k: v.to(DEVICE) for k, v in model_input.items()}
            gt = {k: v.to(DEVICE) for k, v in gt.items()}

            results = model({'coords': model_input['model_coords']})
            states = results['model_in'].detach()[..., 1:]
            values = dynamics.io_to_value(results['model_in'].detach(), results['model_out'].squeeze(-1))
            dvs    = dynamics.io_to_dv(results['model_in'], results['model_out'].squeeze(-1))

            step_losses = loss_fn(
                states, values,
                dvs[..., 0], dvs[..., 1:],
                gt['boundary_values'], gt['dirichlet_masks'],
                results['model_out'],
                times=model_input['model_coords'][..., 0],
            )

            n_pde = cfg['numpoints'] - cfg['num_src_samples']
            total_loss = step_losses['dirichlet'] + step_losses['diff_constraint_hom'] / n_pde

            # ── Targeted boundary augmentation ──────────────────────────────
            # Sample from the targeted portion of the pre-built GPU pool
            # (first n_third entries, generated by _targeted_boundary_states).
            # Pure GPU index ops — no CPU work, no effect on results vs.
            # fresh CPU generation (same distribution, pre-computed values).
            if (epoch + 1) % cfg.get('targeted_freq', 3) == 0:
                tgt_idx    = torch.randint(0, n_third, (cfg['n_targeted'],), device=DEVICE)
                tgt_norm   = pool_norm_gpu[tgt_idx]          # (N, 10) already normalised (z = x/scale)
                l_tgt      = pool_lx_gpu[tgt_idx]            # (N,) boundary values for those states
                t_zero     = torch.zeros(cfg['n_targeted'], 1, device=DEVICE)
                tgt_coords = torch.cat([t_zero, tgt_norm], dim=1)   # (N, 11) = [t, z]
                # Feed normalised coords directly — do NOT call coord_to_input here,
                # which would divide by scale again giving [t, z/scale] = [t, x/scale²].
                tgt_out    = model({'coords': tgt_coords})
                V_tgt      = dynamics.io_to_value(
                    tgt_out['model_in'].detach(), tgt_out['model_out'].squeeze(-1)
                )
                targeted_loss = torch.abs(V_tgt - l_tgt).mean()
                total_loss = total_loss + cfg.get('targeted_weight', 3.0) * targeted_loss

            # ── Static trajectory loss (safe + unsafe) ───────────────────────
            # V(x_static, t) = l(x) for all t: arm at rest stays wherever it is.
            n_st = cfg.get('static_traj_n', 512)
            n_st_half = n_st // 2
            t_curr = cfg['tMin'] + (cfg['tMax'] - cfg['tMin']) * min(
                dataset.counter / cfg['counter_end'], 1.0)
            t_range = (cfg['tMin'], max(t_curr, cfg['tMin'] + 0.01))

            # safe half
            st_idx    = torch.randint(0, N_static_safe,   (n_st_half,), device=DEVICE)
            st_norm   = static_safe_norm_gpu[st_idx]
            st_lx     = static_safe_lx_gpu[st_idx]
            st_times  = torch.empty(n_st_half, 1, device=DEVICE).uniform_(*t_range)
            st_coords = torch.cat([st_times, st_norm], dim=1)
            st_out    = model({'coords': st_coords})
            V_st      = dynamics.io_to_value(st_out['model_in'].detach(), st_out['model_out'].squeeze(-1))
            static_traj_loss = torch.abs(V_st - st_lx).mean()

            # unsafe half — symmetric: prevents network drifting positive at
            # ground-collision and other static failure configurations
            ust_idx    = torch.randint(0, N_static_unsafe, (n_st_half,), device=DEVICE)
            ust_norm   = static_unsafe_norm_gpu[ust_idx]
            ust_lx     = static_unsafe_lx_gpu[ust_idx]
            ust_times  = torch.empty(n_st_half, 1, device=DEVICE).uniform_(*t_range)
            ust_coords = torch.cat([ust_times, ust_norm], dim=1)
            ust_out    = model({'coords': ust_coords})
            V_ust      = dynamics.io_to_value(ust_out['model_in'].detach(), ust_out['model_out'].squeeze(-1))
            static_traj_loss = static_traj_loss + torch.abs(V_ust - ust_lx).mean()

            total_loss = total_loss + cfg.get('static_traj_weight', 5.0) * static_traj_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if (epoch + 1) % log_every == 0:
            loss_parts = "  ".join(f"{k}={v.mean().item():.5f}" for k, v in step_losses.items())
            v_std = values.detach().std().item()
            tqdm.write(f"  [{epoch+1:>6}] {loss_parts}  traj={static_traj_loss.item():.5f}  V_std={v_std:.5f}")

        if (epoch + 1) % cfg['checkpoint_every'] == 0:
            path = os.path.join(ckpt_dir, f'model_epoch_{epoch+1:06d}.pth')
            torch.save({'epoch': epoch + 1, 'model': model.state_dict()}, path)
            # Diagnostic: query safe/unsafe states at two times:
            #   t_trained = current training horizon (within the curriculum window)
            #   t_full    = tMax (only meaningful once curriculum covers the full range)
            t_trained = cfg['tMin'] + (cfg['tMax'] - cfg['tMin']) * min(
                dataset.counter / cfg['counter_end'], 1.0)
            model.eval()
            with torch.no_grad():
                def _probe(state_np, t):
                    coord = torch.cat([torch.tensor([t]), torch.tensor(state_np, dtype=torch.float32)]).unsqueeze(0)
                    inp = dynamics.coord_to_input(coord).to(DEVICE)
                    out = model({'coords': inp})
                    return float(dynamics.io_to_value(out['model_in'].detach(), out['model_out'].squeeze(-1).detach()).item())
                states_to_probe = {
                    'safe':  np.zeros(10),
                    'slosh': np.array([0]*6 + [dynamics.slosh_rad_max*1.2, 0, 0, 0]),
                    'obs1':  np.array([-0.21,-0.36, 0.25, 0,0,0, 0,0,0,0], dtype=np.float32),
                    'obs2':  np.array([-0.21,-0.23, 0.00, 0,0,0, 0,0,0,0], dtype=np.float32),
                    'gnd':   np.array([0,-1.2,0.5, 0,0,0, 0,0,0,0], dtype=np.float32),
                }
                probes_trained = {k: _probe(v, t_trained) for k, v in states_to_probe.items()}
                probes_full    = {k: _probe(v, cfg['tMax']) for k, v in states_to_probe.items()}
            model.train()
            trained_str = '  '.join(f"V({k})={v:+.3f}" for k, v in probes_trained.items())
            full_str    = '  '.join(f"V({k})={v:+.3f}" for k, v in probes_full.items())
            tqdm.write(
                f"  [{epoch+1}] loss={total_loss.item():.5f}  t_trained={t_trained:.2f}s\n"
                f"    @t_trained: {trained_str}\n"
                f"    @t_full:    {full_str}\n"
                f"    saved {path}"
            )

    final = os.path.join(cfg['save_dir'], 'model_final.pth')
    torch.save(model.state_dict(), final)
    print(f"\nSaved: {final}")
    return model, dynamics

def load_model(save_dir=CFG['save_dir']):
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "DeepReach utilities are unavailable; cannot load a trained BRT model through this script."
        )
    dynamics = CoffeeArmDynamics()
    model = build_model(dynamics)
    path = os.path.join(save_dir, 'model_final.pth')
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model, dynamics

def query_value(model, dynamics, state_10d, t=None):
    """Query V(x, t) for a single 10D state."""
    if t is None:
        t = CFG['tMax']

    if not isinstance(state_10d, torch.Tensor):
        state_10d = torch.tensor(state_10d, dtype=torch.float32)

    coord = torch.cat([torch.tensor([t], dtype=torch.float32), state_10d.cpu()]).unsqueeze(0)
    inp = dynamics.coord_to_input(coord).to(DEVICE)

    with torch.no_grad():
        out = model({'coords': inp})
        V = dynamics.io_to_value(out['model_in'].detach(), out['model_out'].squeeze(-1).detach())
    return float(V.item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Load saved model and run query checks instead of training')
    parser.add_argument('--fast', action='store_true', help='Quick ~15-min run with CFG_FAST for validating fixes')
    args = parser.parse_args()

    if args.test:
        model, dyn = load_model()
        print("Model loaded. Running query checks...\n")

        checks = [
            (np.zeros(10), "origin (all zero)", "> 0 (safe)"),
            (np.array([0]*6 + [dyn.slosh_rad_max*0.5, 0, 0, 0]), "x_slosh = 0.5*max", "> 0 (safe)"),
            (np.array([0]*6 + [dyn.slosh_rad_max*0.95, 0, 0, 0]), "x_slosh = 0.95*max", "near 0"),
            (np.array([0]*6 + [dyn.slosh_rad_max*1.1, 0, 0, 0]), "x_slosh = 1.1*max", "< 0 (unsafe)"),
            (np.array([0, -1.2, 0.5, 0,0,0, 0,0,0,0], dtype=np.float32), "q2=-1.2 (below ground)", "< 0 (unsafe)"),
        ]

        for s, desc, expect in checks:
            v = query_value(model, dyn, s)
            print(f"  V({desc:40s}) = {v:+.4f}  (expect {expect})")
    else:
        cfg = CFG_FAST if args.fast else CFG
        if args.fast:
            print("── FAST mode: using CFG_FAST (~15 min) ──")
        train_brt(cfg=cfg)