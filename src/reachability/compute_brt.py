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

# 2. Add the deepreach directory specifically so 'utils' can be discovered 
DEEPREACH_PATH = os.path.join(PROJECT_ROOT, 'deepreach')
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
    epochs=100_000,
    lr=2e-5,
    numpoints=65_000,
    num_src_samples=1_000,
    pretrain_iters=2_000,
    tMin=0.0,
    tMax=1.0,
    counter_end=100_000,
    minWith='zero',
    hidden_features=512,
    num_hidden_layers=3,
    save_dir='brt_model',
    checkpoint_every=10_000,
)

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

    dataset = dataio.ReachabilityDataset(
        dynamics=dynamics,
        numpoints=cfg['numpoints'],
        pretrain=True,
        pretrain_iters=cfg['pretrain_iters'],
        tMin=cfg['tMin'],
        tMax=cfg['tMax'],
        counter_start=0,
        counter_end=cfg['counter_end'],
        num_src_samples=cfg['num_src_samples'],
        num_target_samples=0,
    )

    loss_fn = losses.init_brt_hjivi_loss(
        dynamics,
        minWith=cfg['minWith'],
        dirichlet_loss_divisor=cfg['num_src_samples'],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)

    print(f"Training 10D BRT on {DEVICE}")
    print(f"  Epochs: {cfg['epochs']}, tMax={cfg['tMax']}s")
    print(f"  Network: {cfg['hidden_features']} x {cfg['num_hidden_layers']} SIREN\n")

    model.train()
    for epoch in tqdm(range(cfg['epochs']), desc='BRT training'):
        for model_input, gt in loader:
            model_input = {k: v.to(DEVICE) for k, v in model_input.items()}
            gt = {k: v.to(DEVICE) for k, v in gt.items()}

            results = model({'coords': model_input['model_coords']})
            states = dynamics.input_to_coord(results['model_in'].detach())[..., 1:]
            values = dynamics.io_to_value(results['model_in'].detach(), results['model_out'].squeeze(-1))
            dvs = dynamics.io_to_dv(results['model_in'], results['model_out'].squeeze(-1))

            step_losses = loss_fn(
                states, values,
                dvs[..., 0], dvs[..., 1:],
                gt['boundary_values'], gt['dirichlet_masks'],
                results['model_out'],
            )

            total_loss = sum(v.mean() for v in step_losses.values())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if (epoch + 1) % cfg['checkpoint_every'] == 0:
            path = os.path.join(ckpt_dir, f'model_epoch_{epoch+1:06d}.pth')
            torch.save({'epoch': epoch + 1, 'model': model.state_dict()}, path)
            tqdm.write(f"  [{epoch+1}] loss={total_loss.item():.5f}  saved {path}")

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
    args = parser.parse_args()

    if args.test:
        model, dyn = load_model()
        print("Model loaded. Running query checks...\n")

        # FIXED: Updated to use Cartesian slosh_rad_max check parameters
        checks = [
            (np.zeros(10), "origin (all zero)", "> 0 (safe)"),
            (np.array([0]*6 + [dyn.slosh_rad_max*0.5, 0, 0, 0]), "x_slosh = 0.5*max", "> 0 (safe)"),
            (np.array([0]*6 + [dyn.slosh_rad_max*0.95, 0, 0, 0]), "x_slosh = 0.95*max", "near 0"),
            (np.array([0]*6 + [dyn.slosh_rad_max*1.1, 0, 0, 0]), "x_slosh = 1.1*max", "< 0 (unsafe)"),
        ]

        for s, desc, expect in checks:
            v = query_value(model, dyn, s)
            print(f"  V({desc:30s}) = {v:+.4f}  (expect {expect})")
    else:
        train_brt()