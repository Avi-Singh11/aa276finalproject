#!/usr/bin/env python3
"""Quick diagnostic: load latest checkpoint and probe at the current curriculum horizon.

Run from aa276finalproject directory:
  python probe_checkpoint.py
"""
import sys, os, glob
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEEPREACH_PARENT = os.path.dirname(PROJECT_ROOT)
for p in [PROJECT_ROOT, DEEPREACH_PARENT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from deepreach.utils import modules
from dynamics.dynamics import CoffeeArmDynamics
from src.reachability.compute_brt import CFG

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_latest(save_dir):
    ckpts = sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'model_epoch_*.pth')))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {save_dir}/checkpoints/")
    path = ckpts[-1]
    data = torch.load(path, map_location=DEVICE)
    epoch = data['epoch']
    return data['model'], epoch, path

def probe(model, dynamics, state_np, t):
    coord = torch.cat([
        torch.tensor([t], dtype=torch.float32),
        torch.tensor(state_np, dtype=torch.float32),
    ]).unsqueeze(0)
    inp = dynamics.coord_to_input(coord).to(DEVICE)
    with torch.no_grad():
        out = model({'coords': inp})
        return float(dynamics.io_to_value(
            out['model_in'].detach(), out['model_out'].squeeze(-1).detach()).item())

if __name__ == '__main__':
    dynamics = CoffeeArmDynamics()
    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim, out_features=1,
        type='sine', mode='mlp', final_layer_factor=1,
        hidden_features=CFG['hidden_features'],
        num_hidden_layers=CFG['num_hidden_layers'],
    ).to(DEVICE)

    state_dict, epoch, path = load_latest(CFG['save_dir'])
    model.load_state_dict(state_dict)
    model.eval()

    t_trained = CFG['tMin'] + (CFG['tMax'] - CFG['tMin']) * min(epoch / CFG['counter_end'], 1.0)

    print(f"\nCheckpoint: {path}  (epoch {epoch})")
    print(f"Training window: t ∈ [0, {t_trained:.3f}s]  (tMax={CFG['tMax']}s)\n")

    states = {
        'safe (origin)':      np.zeros(10),
        'slosh over limit':   np.array([0]*6 + [dynamics.slosh_rad_max*1.2, 0, 0, 0]),
        'obstacle 1':         np.array([-0.21,-0.36, 0.25, 0,0,0, 0,0,0,0], dtype=np.float32),
        'obstacle 2':         np.array([-0.21,-0.23, 0.00, 0,0,0, 0,0,0,0], dtype=np.float32),
        'cup below ground':   np.array([0,-1.2,0.5, 0,0,0, 0,0,0,0], dtype=np.float32),
        'slosh safe (0.3x)':  np.array([0]*6 + [dynamics.slosh_rad_max*0.3, 0, 0, 0]),
    }

    l_values = {k: float(np.asarray(dynamics.boundary_fn(v.reshape(1,-1))).flat[0]) for k, v in states.items()}

    print(f"{'State':<25}  {'l(x)':>8}  {'V(t=0)':>8}  {'V(t_trained)':>12}  {'V(tMax)':>8}  status")
    print("-" * 80)
    for name, s in states.items():
        lx   = l_values[name]
        v0   = probe(model, dynamics, s, CFG['tMin'])
        vt   = probe(model, dynamics, s, t_trained)
        vmax = probe(model, dynamics, s, CFG['tMax'])
        # At t=0 the model should output ≈ l(x) (dirichlet enforced)
        t0_ok = abs(v0 - lx) < 0.1
        # At t_trained: safe states (l>0) should have V>0, unsafe (l<0) should have V<0
        if lx > 0:
            struct_ok = vt > 0
            flag = "GOOD" if struct_ok else "WRONG-SIGN"
        else:
            struct_ok = vt < 0
            flag = "GOOD" if struct_ok else "WRONG-SIGN"
        t0_flag = "ok" if t0_ok else f"BAD(err={abs(v0-lx):.3f})"
        print(f"  {name:<23}  {lx:>+8.4f}  {v0:>+8.4f}  {vt:>+12.4f}  {vmax:>+8.4f}  [{flag}] t0:{t0_flag}")

    print()
    safe_vt  = probe(model, dynamics, states['safe (origin)'], t_trained)
    slosh_vt = probe(model, dynamics, states['slosh over limit'], t_trained)
    sep = safe_vt - slosh_vt
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    print(f"  V(safe) - V(slosh) at t_trained = {sep:+.4f}  (want > 0)  [{PASS if sep > 0 else FAIL}]")
    print(f"  All l(x) values: {[f'{v:+.3f}' for v in l_values.values()]}")
