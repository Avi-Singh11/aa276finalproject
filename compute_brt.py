# Train BRT for the 8D coffee arm using DeepReach
# Solves HJ PDE: dV/dt + min(0, H(x, dV/dx)) = 0
# failure set: |alpha| > alpha_max, obstacle collision, cup below ground
#
# Usage: python compute_brt.py        (train)
#        python compute_brt.py --test (query test states from saved model)

import sys
import os
import argparse

# Add deepreach to path 
DEEPREACH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'deepreach')
)
sys.path.insert(0, DEEPREACH_PATH)

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dynamics.dynamics import CoffeeArmDynamics
from utils import modules, dataio, losses

# Training config 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = dict(
    # Training
    epochs = 100_000,
    lr = 2e-5,
    # Data
    numpoints = 65_000,   # points sampled per training step
    num_src_samples = 1_000,    # boundary condition samples included per step
    pretrain_iters = 2_000,    # steps of BC-only pretraining before PDE loss kicks in
    # Time horizon
    tMin = 0.0,
    tMax = 1.0, # seconds of backward reachability
    counter_end = 100_000, # curriculum: linearly ramp up tMax over this many steps
    # BRT formulation
    minWith = 'zero', # 'zero' = BRT tube, 'target' = BRS endpoint
    # Network architecture (SIREN)
    hidden_features = 512,
    num_hidden_layers = 3,
    # Checkpointing
    save_dir = 'brt_model',
    checkpoint_every = 10_000,
)


# Model construction 
def build_model(dynamics, cfg=CFG):
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


# Training
def train_brt(cfg=CFG):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    ckpt_dir = os.path.join(cfg['save_dir'], 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    dynamics = CoffeeArmDynamics()
    model = build_model(dynamics, cfg)

    dataset = dataio.ReachabilityDataset(
        dynamics = dynamics,
        numpoints = cfg['numpoints'],
        pretrain = True,
        pretrain_iters = cfg['pretrain_iters'],
        tMin = cfg['tMin'],
        tMax = cfg['tMax'],
        counter_start = 0,
        counter_end = cfg['counter_end'],
        num_src_samples = cfg['num_src_samples'],
        num_target_samples = 0,
    )

    loss_fn = losses.init_brt_hjivi_loss(dynamics, minWith=cfg['minWith'], dirichlet_loss_divisor=cfg['num_src_samples'])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)

    print(f"Training 8D BRT on {DEVICE}")
    print(f"  Epochs: {cfg['epochs']}, tMax={cfg['tMax']}s")
    print(f"  Network: {cfg['hidden_features']} x {cfg['num_hidden_layers']} SIREN")
    print()

    model.train()
    for epoch in tqdm(range(cfg['epochs']), desc='BRT training'):
        for model_input, gt in loader:
            model_input = {k: v.to(DEVICE) for k, v in model_input.items()}
            gt = {k: v.to(DEVICE) for k, v in gt.items()}

            results = model({'coords': model_input['model_coords']})
            states = dynamics.input_to_coord(results['model_in'].detach())[..., 1:]
            values = dynamics.io_to_value(
                results['model_in'].detach(),
                results['model_out'].squeeze(-1),
            )
            dvs = dynamics.io_to_dv(
                results['model_in'],
                results['model_out'].squeeze(-1),
            )

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


# Inference 
def load_model(save_dir=CFG['save_dir']):
    dynamics = CoffeeArmDynamics()
    model = build_model(dynamics)
    path = os.path.join(save_dir, 'model_final.pth')
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model, dynamics


def query_value(model, dynamics, state_8d, t=None):
    """Query V(x, t) for a single 8D numpy/tensor state.

    Args:
        model: trained SIREN model (from train_brt or load_model)
        dynamics: CoffeeArmDynamics instance
        state_8d: (8,) array [theta1..3, dtheta1..3, alpha, dalpha]
        t: query time; defaults to tMax (most conservative BRT slice)

    Returns:
        scalar float V. V < 0 means the state is inside the BRT (unsafe).
    """
    if t is None:
        t = CFG['tMax']

    if not isinstance(state_8d, torch.Tensor):
        state_8d = torch.tensor(state_8d, dtype=torch.float32)

    coord = torch.cat([torch.tensor([t], dtype=torch.float32), state_8d.cpu()]).unsqueeze(0)
    inp = dynamics.coord_to_input(coord).to(DEVICE)

    with torch.no_grad():
        out = model({'coords': inp})
        V = dynamics.io_to_value(
            out['model_in'].detach(),
            out['model_out'].squeeze(-1).detach(),
        )
    return float(V.item())

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='Load saved model and run query checks instead of training')
    args = parser.parse_args()

    if args.test:
        model, dyn = load_model()
        print("Model loaded. Running query checks...\n")

        checks = [
            (np.zeros(8), "origin (all zero)", "> 0 (safe)"),
            (np.array([0]*6 + [dyn.ALPHA_MAX*0.5, 0]), "alpha = 0.5*alpha_max", "> 0 (safe)"),
            (np.array([0]*6 + [dyn.ALPHA_MAX*0.95, 0]), "alpha = 0.95*alpha_max",  "near 0"),
            (np.array([0]*6 + [dyn.ALPHA_MAX*1.1, 0]), "alpha = 1.1*alpha_max", "< 0 (unsafe)"),
            (np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "near obstacle center", "< 0 if inside"),
        ]

        for s, desc, expect in checks:
            v = query_value(model, dyn, s)
            print(f"  V({desc:30s}) = {v:+.4f}  (expect {expect})")
    else:
        train_brt()
