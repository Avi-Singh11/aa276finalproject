"""Plot velocity-dependent slices of the completed verified BRT."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

from dynamics.dynamics_OVERNIGHT_VERIFIED import CoffeeArmDynamics
from src.reachability.compute_brt_OVERNIGHT_VERIFIED import CFG, DEVICE, build_model


def main():
    checkpoint = "brt_model_final2/model_final2.pth"
    output = "brt_visualizations/final2_velocity_near_boundary.png"
    dynamics = CoffeeArmDynamics()
    state_dict = torch.load(checkpoint, map_location=DEVICE)
    model = build_model(dynamics, CFG)
    model.load_state_dict(state_dict)
    model.eval()

    resolution = 120
    velocity = np.linspace(-0.3, 0.3, resolution)
    vx, vy = np.meshgrid(velocity, velocity)
    base = np.zeros(10, dtype=np.float32)
    base[0], base[1] = 1.2, 0.3
    fractions = [0.0, 0.5, 0.9]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, fraction in zip(axes, fractions):
        states = np.tile(base, (resolution * resolution, 1))
        states[:, 6] = fraction * dynamics.slosh_rad_max
        states[:, 8] = vx.ravel()
        states[:, 9] = vy.ravel()
        t = torch.full((len(states), 1), CFG["tMax"], dtype=torch.float32)
        coords = torch.cat([t, torch.tensor(states)], dim=1).to(DEVICE)
        with torch.no_grad():
            inputs = dynamics.coord_to_input(coords)
            values = model({"coords": inputs})["model_out"].squeeze().cpu().numpy()
        values = values.reshape(resolution, resolution)
        limit = max(float(np.percentile(np.abs(values), 98)), 0.01)
        image = axis.contourf(
            velocity,
            velocity,
            values,
            levels=40,
            cmap="RdYlGn",
            vmin=-limit,
            vmax=limit,
        )
        if values.min() <= 0.0 <= values.max():
            axis.contour(
                velocity, velocity, values, levels=[0.0], colors="black", linewidths=2
            )
        axis.set_title(f"x_slosh = {fraction:.1f} x spill limit")
        axis.set_xlabel("vx_slosh (m/s)")
        axis.set_ylabel("vy_slosh (m/s)")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, label="V(x, 10 s)")

    fig.suptitle(
        "Verified BRT: velocity-dependent safe set near the spill boundary\n"
        "Black contour is V=0; axes remain inside the trained [-0.3, 0.3] m/s domain"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == "__main__":
    main()
