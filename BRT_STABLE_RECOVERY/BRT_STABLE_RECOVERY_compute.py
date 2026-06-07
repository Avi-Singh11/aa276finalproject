"""Stable-recovery BRT training entry point for the 10D coffee arm model."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

VERIFIED_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(VERIFIED_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEEPREACH_PATH = os.path.dirname(PROJECT_ROOT)
if DEEPREACH_PATH not in sys.path:
    sys.path.insert(0, DEEPREACH_PATH)

from BRT_STABLE_RECOVERY_dynamics import BRTStableRecoveryCoffeeArmDynamics
from BRT_STABLE_RECOVERY_preflight import run_preflight

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
    pretrain_iters=0,
    numpoints=32_000,
    num_src_samples=1500,        
    tMin=0.0,
    tMax=10.0,
    counter_end=100_000,
    minWith='target',
    hidden_features=256,
    num_hidden_layers=3,
    save_dir='BRT_STABLE_RECOVERY_model',
    checkpoint_every=5_000,
    n_targeted=512,
    targeted_freq=3,             
    targeted_weight=3.0,        
    n_pretrain_targeted=512,    
    causal_eps=1.0,
    causal_n_bins=10,
    equilibrium_n=512,
    equilibrium_weight=5.0,
    grad_clip_norm=1.0,
)

CFG_FAST = {
    **CFG,
    'epochs':           50_000,
    'pretrain_iters':   0,
    'numpoints':        16_000,
    'hidden_features':  128,
    'num_hidden_layers': 3,
    'n_targeted':       256,
    'checkpoint_every': 10_000,
    'counter_end':      25_000,   
    'save_dir':         'BRT_STABLE_RECOVERY_model_fast',
}

CFG_BENCHMARK = {
    **CFG,
    'epochs':             20,
    'pretrain_iters':     0,
    'pretrain_pool_size': 50_000,
    'checkpoint_every':   1_000_000,
    'counter_end':        20,
    'save_dir':           'BRT_STABLE_RECOVERY_benchmark',
    'log_every':          5,
}

CFG_PILOT = {
    **CFG,
    'epochs':             10_000,
    'pretrain_iters':     0,
    'pretrain_pool_size': 200_000,
    'counter_end':        10_000,
    'checkpoint_every':   2_000,
    'save_dir':           'BRT_STABLE_RECOVERY_model_pilot',
    'log_every':          250,
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
    L = dynamics.L
    l1 = float(L[0])
    l2 = float(L[1])

    n_per_obs = max(1, int(n * 0.35 / max(len(dynamics.obstacles), 1)))
    n_slosh = max(1, int(n * 0.15))
    n_ground = max(1, int(n * 0.10))
    n_selfc = max(1, int(n * 0.10))
    n_obstacle = n_per_obs * len(dynamics.obstacles)
    n_joint = max(1, n - n_obstacle - n_slosh - n_ground - n_selfc)

    for obs in dynamics.obstacles:
        cx, cy, cz = obs['center']
        r = float(obs['radius'])
        q1_tgt = float(np.arctan2(cy, cx))
        elev = np.clip((cz - l1) / l2, -0.98, 0.98)
        q2_tgt = float(np.arcsin(elev))
        for _ in range(n_per_obs):
            q1 = q1_tgt + rng.uniform(-0.7, 0.7)
            q2 = q2_tgt + rng.uniform(-0.5, 0.5)
            q3 = rng.uniform(-0.6, 0.6)
            dq = rng.uniform(-4.0, 4.0, 3)
            sx = rng.uniform(-dynamics.slosh_rad_max * 1.3, dynamics.slosh_rad_max * 1.3)
            sy = rng.uniform(-dynamics.slosh_rad_max * 1.3, dynamics.slosh_rad_max * 1.3)
            vxy = rng.uniform(-0.25, 0.25, 2)
            states.append(np.array([q1, q2, q3, *dq, sx, sy, *vxy], dtype=np.float32))

    for _ in range(n_slosh):
        q = rng.uniform(-1.0, 1.0, 3)
        dq = rng.uniform(-3.0, 3.0, 3)
        ang = rng.uniform(0.0, 2.0*np.pi)
        rad = dynamics.slosh_rad_max * rng.uniform(0.65, 1.35)
        sx, sy = rad*np.cos(ang), rad*np.sin(ang)
        vxy = rng.uniform(-0.25, 0.25, 2)
        states.append(np.array([*q, *dq, sx, sy, *vxy], dtype=np.float32))

    for _ in range(n_ground):
        q1 = rng.uniform(-np.pi, np.pi)
        q2 = rng.uniform(-1.0, -0.3)      
        q3 = rng.uniform(-0.5, 0.5)
        dq = rng.uniform(-3.0, 3.0, 3)
        sl = rng.uniform(-dynamics.slosh_rad_max*0.5, dynamics.slosh_rad_max*0.5, 4)
        states.append(np.array([q1, q2, q3, *dq, *sl], dtype=np.float32))

    for _ in range(n_selfc):
        sign = rng.choice([-1.0, 1.0])
        q1 = sign * (np.pi + rng.uniform(-0.5, 0.0))
        q2 = rng.uniform(-0.4, 0.6)
        q3 = rng.uniform(0.4, 1.6)
        dq = rng.uniform(-3.0, 3.0, 3)
        sl = rng.uniform(-dynamics.slosh_rad_max*0.5, dynamics.slosh_rad_max*0.5, 4)
        states.append(np.array([q1, q2, q3, *dq, *sl], dtype=np.float32))

    for _ in range(n_joint):
        k = int(rng.integers(3))
        q = rng.uniform(-0.8, 0.8, 3).astype(np.float32)
        q[k] = float(rng.choice([-1.0, 1.0])) * float(dynamics.joint_limits[k]) * rng.uniform(0.82, 1.05)
        dq = rng.uniform(-3.0, 3.0, 3)
        sl = rng.uniform(-dynamics.slosh_rad_max*0.5, dynamics.slosh_rad_max*0.5, 4)
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
    model = model.to(DEVICE)
    # Start extremely close to V=l while preserving gradient flow into all layers.
    final_linear = model.net.net[-1][0]
    torch.nn.init.normal_(final_linear.weight, mean=0.0, std=1e-4)
    torch.nn.init.zeros_(final_linear.bias)
    return model


class TorchReachabilitySampler:
    """Generate curriculum coordinates and labels directly on the GPU."""

    def __init__(self, dynamics, cfg, device):
        self.dynamics = dynamics
        self.numpoints = cfg['numpoints']
        self.num_src_samples = cfg['num_src_samples']
        self.t_min = cfg['tMin']
        self.t_max = cfg['tMax']
        self.counter = 0
        self.counter_end = cfg['counter_end']
        self.device = device

    def sample(self):
        states = torch.empty(
            self.numpoints, self.dynamics.state_dim, device=self.device
        ).uniform_(-1.0, 1.0)
        max_time = (self.t_max - self.t_min) * min(
            self.counter / self.counter_end, 1.0
        )
        times = self.t_min + torch.empty(
            self.numpoints, 1, device=self.device
        ).uniform_(0.0, max_time)
        times[-self.num_src_samples:, 0] = self.t_min
        coords = torch.cat([times, states], dim=-1)
        physical_states = self.dynamics.input_to_coord(coords)[..., 1:]
        boundary_values = self.dynamics.boundary_fn(physical_states)
        dirichlet_masks = coords[:, 0] == self.t_min
        if self.counter < self.counter_end:
            self.counter += 1
        return coords, boundary_values, dirichlet_masks

def train_brt(cfg=CFG, model=None, dynamics=None):
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "DeepReach utilities are unavailable. Install/point the deepreach package correctly "
            f"before training the BRT. Original import error: {_IMPORT_ERROR}"
        )

    if dynamics is None:
        dynamics = BRTStableRecoveryCoffeeArmDynamics()
    run_preflight(dynamics, verbose=True)

    os.makedirs(cfg['save_dir'], exist_ok=True)
    ckpt_dir = os.path.join(cfg['save_dir'], 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    if model is None:
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
    POOL_SIZE = cfg.get('pretrain_pool_size', 500_000)
    BATCH_SIZE = cfg.get('pretrain_batch_size', 32_000)
    rng_pool = np.random.default_rng(42)

    print(f"Building pretrain pool ({POOL_SIZE:,} states, balanced safe/unsafe) ...")
    scale_gpu = torch.as_tensor(
        dynamics.state_scale, dtype=torch.float32, device=DEVICE
    )

    n_third = POOL_SIZE // 3
    n_targeted_pool = n_third
    n_uniform_pool = n_third
    n_safe_pool = POOL_SIZE - n_targeted_pool - n_uniform_pool

    # Targeted third
    tgt_chunks = []
    while sum(len(c) for c in tgt_chunks) < n_targeted_pool:
        tgt_chunks.append(_targeted_boundary_states(dynamics, 4096))
    tgt_pool_phys = torch.as_tensor(
        np.concatenate(tgt_chunks, axis=0)[:n_targeted_pool],
        dtype=torch.float32,
        device=DEVICE,
    )

    uni_pool_phys = torch.empty(
        n_uniform_pool, dynamics.state_dim, device=DEVICE
    ).uniform_(-1.0, 1.0) * scale_gpu

    safe_chunks, collected = [], 0
    while collected < n_safe_pool:
        cands_phys = torch.empty(
            8192, dynamics.state_dim, device=DEVICE
        ).uniform_(-1.0, 1.0) * scale_gpu
        keep = cands_phys[dynamics.boundary_fn(cands_phys) > 0]
        safe_chunks.append(keep)
        collected += len(keep)
    safe_pool_phys = torch.cat(safe_chunks, dim=0)[:n_safe_pool]

    pool_phys = torch.cat([tgt_pool_phys, uni_pool_phys, safe_pool_phys], dim=0)
    pool_lx_gpu = dynamics.boundary_fn(pool_phys)
    pool_norm_gpu = pool_phys / scale_gpu
    N_pool = pool_phys.shape[0]

    safe_frac = float((pool_lx_gpu > 0).float().mean().item())

    print(f"  Pool ready: {N_pool:,} states, {safe_frac*100:.1f}% safe, "
          f"l(x) std={pool_lx_gpu.std().item():.4f}")

    equilibrium_q = torch.empty(50_000, 3, device=DEVICE).uniform_(-1.0, 1.0)
    equilibrium_q = equilibrium_q * torch.as_tensor(
        dynamics.joint_limits, dtype=torch.float32, device=DEVICE
    )
    equilibrium_phys = torch.zeros(50_000, dynamics.state_dim, device=DEVICE)
    equilibrium_phys[:, :3] = equilibrium_q
    equilibrium_lx = dynamics.boundary_fn(equilibrium_phys)
    equilibrium_norm = equilibrium_phys / scale_gpu

    model.train()
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('pretrain_lr', 2e-4))
    for i in tqdm(range(cfg['pretrain_iters']), desc='Pretrain'):
        idx = torch.randint(0, N_pool, (BATCH_SIZE,), device=DEVICE)
        x_bat = pool_norm_gpu[idx]                                       # (B, 10)
        l_bat = pool_lx_gpu[idx]                                         # (B,)
        t_bat = torch.full((BATCH_SIZE, 1), cfg['tMin'], device=DEVICE)

        coords = torch.cat([t_bat, x_bat], dim=1)                        # (B, 11)
        out = model({'coords': coords})
        V = dynamics.io_to_value(out['model_in'].detach(), out['model_out'].squeeze(-1)) 
        loss = torch.abs(V - l_bat).mean()
        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()

        if (i + 1) % 500 == 0:
            tqdm.write(f"  pretrain [{i+1:>6}] loss={loss.item():.5f}  V_std={V.detach().std().item():.5f}")
    print("Pretrain done.\n")

    sampler = TorchReachabilitySampler(dynamics, cfg, DEVICE)

    log_every = cfg.get('log_every', 500)
    model.train()
    for epoch in tqdm(range(cfg['epochs']), desc='BRT training'):
        model_coords, boundary_values, dirichlet_masks = sampler.sample()
        results = model({'coords': model_coords})
        states = results['model_in'].detach()[..., 1:]
        values = dynamics.io_to_value(
            results['model_in'].detach(), results['model_out'].squeeze(-1)
        )
        dvs = dynamics.io_to_dv(
            results['model_in'], results['model_out'].squeeze(-1)
        )

        step_losses = loss_fn(
            states, values,
            dvs[..., 0], dvs[..., 1:],
            boundary_values, dirichlet_masks,
            results['model_out'],
            times=model_coords[..., 0],
        )

        n_pde = cfg['numpoints'] - cfg['num_src_samples']
        total_loss = (
            step_losses['dirichlet']
            + step_losses['diff_constraint_hom'] / n_pde
        )

        if (epoch + 1) % cfg.get('targeted_freq', 3) == 0:
            tgt_idx = torch.randint(
                0, n_third, (cfg['n_targeted'],), device=DEVICE
            )
            tgt_norm = pool_norm_gpu[tgt_idx]
            l_tgt = pool_lx_gpu[tgt_idx]
            t_zero = torch.zeros(cfg['n_targeted'], 1, device=DEVICE)
            tgt_coords = torch.cat([t_zero, tgt_norm], dim=1)
            tgt_out = model({'coords': tgt_coords})
            V_tgt = dynamics.io_to_value(
                tgt_out['model_in'].detach(),
                tgt_out['model_out'].squeeze(-1),
            )
            total_loss = total_loss + cfg.get('targeted_weight', 3.0) * torch.abs(
                V_tgt - l_tgt
            ).mean()

        n_eq = cfg.get('equilibrium_n', 512)
        eq_idx = torch.randint(
            0, equilibrium_norm.shape[0], (n_eq,), device=DEVICE
        )
        eq_times = torch.empty(n_eq, 1, device=DEVICE).uniform_(
            cfg['tMin'], cfg['tMax']
        )
        eq_coords = torch.cat([eq_times, equilibrium_norm[eq_idx]], dim=-1)
        eq_out = model({'coords': eq_coords})
        eq_values = dynamics.io_to_value(
            eq_out['model_in'], eq_out['model_out']
        )
        equilibrium_loss = torch.abs(
            eq_values - equilibrium_lx[eq_idx]
        ).mean()
        total_loss = total_loss + cfg.get(
            'equilibrium_weight', 5.0
        ) * equilibrium_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch + 1}; last checkpoint is intact"
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.get('grad_clip_norm', 1.0),
            error_if_nonfinite=False,
        )
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite gradient at epoch {epoch + 1}; update was skipped"
            )
        optimizer.step()

        for parameter in model.parameters():
            if not torch.all(torch.isfinite(parameter)):
                raise FloatingPointError(
                    f"Non-finite parameter after epoch {epoch + 1}"
                )

        if (epoch + 1) % log_every == 0:
            loss_parts = "  ".join(f"{k}={v.mean().item():.5f}" for k, v in step_losses.items())
            v_std = values.detach().std().item()
            tqdm.write(
                f"  [{epoch+1:>6}] {loss_parts}  "
                f"equilibrium={equilibrium_loss.item():.5f}  "
                f"grad={float(grad_norm):.4f}  V_std={v_std:.5f}"
            )

        if (epoch + 1) % cfg['checkpoint_every'] == 0:
            path = os.path.join(ckpt_dir, f'model_epoch_{epoch+1:06d}.pth')
            torch.save({'epoch': epoch + 1, 'model': model.state_dict()}, path)
            t_trained = cfg['tMin'] + (cfg['tMax'] - cfg['tMin']) * min(
                sampler.counter / cfg['counter_end'], 1.0)
            model.eval()
            with torch.no_grad():
                def _probe(state_np, t):
                    coord = torch.cat([torch.tensor([t]), torch.tensor(state_np, dtype=torch.float32)]).unsqueeze(0)
                    inp = dynamics.coord_to_input(coord).to(DEVICE)
                    out = model({'coords': inp})
                    return float(dynamics.io_to_value(out['model_in'].detach(), out['model_out'].squeeze(-1).detach()).item())
                states_to_probe = {
                    'safe':  np.array(
                        [-0.54, 0.65, 0.66, 0,0,0, 0,0,0,0],
                        dtype=np.float32,
                    ),
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
    dynamics = BRTStableRecoveryCoffeeArmDynamics()
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
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run 20 full-size pretrain and PDE steps to estimate runtime',
    )
    parser.add_argument(
        '--pilot',
        action='store_true',
        help='Run a full-architecture 10k-epoch pilot before the full solve',
    )
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pth to fine-tune from (e.g. brt_model/model_final.pth)')
    parser.add_argument('--finetune-epochs', type=int, default=20_000,
                        help='Number of additional epochs when using --resume (default: 20000)')
    parser.add_argument('--finetune-lr', type=float, default=5e-6,
                        help='Learning rate for fine-tuning (default: 5e-6, lower than training to avoid forgetting)')
    parser.add_argument(
        '--recover',
        type=str,
        default=None,
        help='Resume safely from a finite checkpoint using gradient clipping',
    )
    parser.add_argument('--recover-epochs', type=int, default=4_000)
    parser.add_argument('--recover-lr', type=float, default=2e-6)
    parser.add_argument('--finetune-dir', type=str, default='brt_model_finetuned',
                        help='Output directory for fine-tuned model (default: brt_model_finetuned)')
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

    elif args.recover is not None:
        recovery_cfg = {
            **CFG_PILOT,
            'epochs': args.recover_epochs,
            'counter_end': args.recover_epochs,
            'lr': args.recover_lr,
            'pretrain_iters': 0,
            'checkpoint_every': 500,
            'log_every': 100,
            'save_dir': 'BRT_STABLE_RECOVERY_model',
            'grad_clip_norm': 1.0,
        }
        print(f"── RECOVERY mode: loading {args.recover} ──")
        dyn = BRTStableRecoveryCoffeeArmDynamics()
        recovery_model = build_model(dyn, recovery_cfg)
        checkpoint = torch.load(args.recover, map_location=DEVICE)
        recovery_model.load_state_dict(checkpoint.get('model', checkpoint))
        train_brt(
            cfg=recovery_cfg,
            model=recovery_model,
            dynamics=dyn,
        )

    elif args.resume is not None:
        # Fine-tune an existing checkpoint with the corrected static pool.
        # Uses a low learning rate to fix the slosh-static V error without
        # destabilising the already-correct obstacle and joint-limit regions.
        cfg = CFG_FAST if args.fast else CFG
        ft_cfg = {
            **cfg,
            'epochs': args.finetune_epochs,
            'lr': args.finetune_lr,
            'pretrain_iters': 0,      # skip pretrain — model already initialised
            'save_dir': args.finetune_dir,
            'checkpoint_every': max(1000, args.finetune_epochs // 10),
        }
        print(f"── FINE-TUNE mode: loading {args.resume} ──")
        print(f"   epochs={ft_cfg['epochs']}  lr={ft_cfg['lr']}  → {ft_cfg['save_dir']}")
        dyn = BRTStableRecoveryCoffeeArmDynamics()
        ft_model = build_model(dyn, ft_cfg)
        ft_model.load_state_dict(torch.load(args.resume, map_location=DEVICE))
        train_brt(cfg=ft_cfg, model=ft_model, dynamics=dyn)

    else:
        if args.benchmark:
            cfg = CFG_BENCHMARK
        elif args.pilot:
            cfg = CFG_PILOT
        elif args.fast:
            cfg = CFG_FAST
        else:
            cfg = CFG
        if args.benchmark:
            print("── BENCHMARK mode: 20 full-size pretrain + PDE steps ──")
        if args.pilot:
            print("── PILOT mode: full architecture, 10k HJ epochs ──")
        if args.fast:
            print("── FAST mode: using CFG_FAST (~15 min) ──")
        train_brt(cfg=cfg)
