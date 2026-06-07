"""Visualize the trained BRT value function and CBF safe set.

Produces 2D slices of V(x, t*) across the most safety-relevant state subspaces.
The zero level set  V = 0  is the CBF boundary — states with V >= 0 are safe.

Run from aa276finalproject/:
    python src/scripts/visualize_brt.py
    python src/scripts/visualize_brt.py --ckpt brt_model/checkpoints/model_epoch_030000.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from dynamics.dynamics import CoffeeArmDynamics
from src.reachability.compute_brt import build_model, CFG, DEVICE

RESOLUTION = 80   # grid points per axis


def load(ckpt_path: str):
    dyn = CoffeeArmDynamics()
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # Infer hidden_features and num_hidden_layers from checkpoint weights
    # so the visualizer works with both CFG and CFG_FAST models.
    first_w = state["net.net.0.0.weight"]
    hidden_features = first_w.shape[0]
    num_hidden_layers = sum(1 for k in state if k.startswith("net.net.") and k.endswith(".weight")) - 2
    cfg = {**CFG, "hidden_features": hidden_features, "num_hidden_layers": num_hidden_layers}
    model = build_model(dyn, cfg)
    model.load_state_dict(state)
    model.eval()
    return model, dyn


@torch.no_grad()
def query_grid(model, dyn, states_np: np.ndarray, t: float) -> np.ndarray:
    """Query V(x, t) for a batch of states. Returns numpy array."""
    N = states_np.shape[0]
    t_col = torch.full((N, 1), t, dtype=torch.float32)
    x_t   = torch.tensor(states_np, dtype=torch.float32)
    coords = torch.cat([t_col, x_t], dim=1).to(DEVICE)
    inp    = dyn.coord_to_input(coords)
    out    = model({"coords": inp})
    V      = dyn.io_to_value(out["model_in"].detach(), out["model_out"].squeeze(-1).detach())
    return V.cpu().numpy()


def make_base_state():
    """Return a nominal 10D state (arm at home, slosh at rest).
    q1=1.2 (az~69deg), q2=0.3 (el~17deg) — mid-range of the safe reset distribution,
    clear of all obstacles.
    """
    s = np.zeros(10, dtype=np.float32)
    s[0] = 1.2   # azimuth ~69 deg (reset range is 60-90 deg)
    s[1] = 0.3   # elevation ~17 deg
    return s


def plot_slice(ax, V_grid, x_vals, y_vals, xlabel, ylabel, title, dyn):
    V = V_grid.reshape(RESOLUTION, RESOLUTION)
    vmax = np.percentile(np.abs(V), 95)
    vmax = max(vmax, 0.01)

    im = ax.contourf(x_vals, y_vals, V, levels=40,
                     cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.contour(x_vals, y_vals, V, levels=[0.0],
               colors="black", linewidths=2.0)
    plt.colorbar(im, ax=ax, label="V(x, t*)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal" if xlabel == ylabel else "auto")


def main(ckpt_path: str, t: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading: {ckpt_path}")
    model, dyn = load(ckpt_path)
    print(f"Querying on {DEVICE}  (resolution {RESOLUTION}x{RESOLUTION} per plot)")

    base = make_base_state()

    # ── 1. Slosh position plane (most directly safety-relevant) ───────────
    # Use the training range (state_scale) not l_eff — corners of the l_eff
    # square are outside the feasibility disk xs²+ys² ≤ l_eff² and extrapolate.
    slosh_scale = float(dyn.state_scale[6])   # l_eff / sqrt(3)
    xs_vals  = np.linspace(-slosh_scale, slosh_scale, RESOLUTION)
    ys_vals  = np.linspace(-slosh_scale, slosh_scale, RESOLUTION)
    XX, YY   = np.meshgrid(xs_vals, ys_vals)
    states   = np.tile(base, (RESOLUTION * RESOLUTION, 1))
    states[:, 6] = XX.ravel()
    states[:, 7] = YY.ravel()
    # zero slosh velocity for this slice
    states[:, 8] = 0.0
    states[:, 9] = 0.0
    V_slosh = query_grid(model, dyn, states, t)

    # ── 2. Slosh velocity plane (fixed at slosh centre) ───────────────────
    v_max    = float(dyn.state_scale[8])   # training range for slosh velocity
    vx_vals  = np.linspace(-v_max, v_max, RESOLUTION)
    vy_vals  = np.linspace(-v_max, v_max, RESOLUTION)
    VX, VY   = np.meshgrid(vx_vals, vy_vals)
    states2  = np.tile(base, (RESOLUTION * RESOLUTION, 1))
    states2[:, 6] = 0.0
    states2[:, 7] = 0.0
    states2[:, 8] = VX.ravel()
    states2[:, 9] = VY.ravel()
    V_vel = query_grid(model, dyn, states2, t)

    # ── 3. Slosh x-position vs x-velocity (phase portrait) ───────────────
    states3  = np.tile(base, (RESOLUTION * RESOLUTION, 1))
    XS, VXS  = np.meshgrid(xs_vals, vx_vals)
    states3[:, 6] = XS.ravel()
    states3[:, 8] = VXS.ravel()
    states3[:, 7] = 0.0
    states3[:, 9] = 0.0
    V_phase = query_grid(model, dyn, states3, t)

    # ── 4. Joint angles q1 vs q2 (arm reachability) ──────────────────────
    q_lim   = float(dyn.joint_limits[0])
    q1_vals = np.linspace(-q_lim, q_lim, RESOLUTION)
    q2_vals = np.linspace(-q_lim, q_lim, RESOLUTION)
    Q1, Q2  = np.meshgrid(q1_vals, q2_vals)
    states4 = np.tile(base, (RESOLUTION * RESOLUTION, 1))
    states4[:, 0] = Q1.ravel()
    states4[:, 1] = Q2.ravel()
    V_joints = query_grid(model, dyn, states4, t)

    # ── Plot all four ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"BRT Value Function V(x, t*)   —   {os.path.basename(ckpt_path)}\n"
        f"Black contour = CBF boundary (V=0).  Green = safe, Red = unsafe.",
        fontsize=12,
    )

    plot_slice(axes[0, 0], V_slosh,  xs_vals, ys_vals,
               "x_slosh (m)", "y_slosh (m)",
               "Slosh position  (vx=vy=0, arm at home)", dyn)
    # Draw slosh radius limit
    theta_c = np.linspace(0, 2*np.pi, 200)
    axes[0, 0].plot(dyn.slosh_rad_max * np.cos(theta_c),
                    dyn.slosh_rad_max * np.sin(theta_c),
                    "b--", lw=1.5, label=f"slosh_rad_max={dyn.slosh_rad_max:.4f}m")
    axes[0, 0].legend(fontsize=7)

    plot_slice(axes[0, 1], V_vel,  vx_vals, vy_vals,
               "vx_slosh (m/s)", "vy_slosh (m/s)",
               "Slosh velocity  (x=y=0, arm at home)", dyn)

    plot_slice(axes[1, 0], V_phase, xs_vals, vx_vals,
               "x_slosh (m)", "vx_slosh (m/s)",
               "Slosh phase portrait  (y=vy=0, arm at home)", dyn)

    plot_slice(axes[1, 1], V_joints, q1_vals, q2_vals,
               "q1 (rad)", "q2 (rad)",
               "Joint angle plane  (slosh at rest)", dyn)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "brt_value_slices.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # ── CBF stats ─────────────────────────────────────────────────────────
    print("\n── CBF summary (slosh position slice) ──")
    print(f"  V range : [{V_slosh.min():.3f}, {V_slosh.max():.3f}]")
    safe_frac = (V_slosh >= 0).mean()
    print(f"  Safe fraction of slosh grid: {safe_frac*100:.1f}%")
    print(f"  (slosh_rad_max={dyn.slosh_rad_max:.4f}m  training range ±{slosh_scale:.4f}m)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="brt_model/model_final.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--t", type=float, default=CFG["tMax"],
                        help="Time horizon to query (default: tMax)")
    parser.add_argument("--out", default="brt_visualizations",
                        help="Output directory for plots")
    args = parser.parse_args()
    main(args.ckpt, args.t, args.out)
