"""Create matched controller animations, metrics, and intervention action logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEEPREACH_PARENT = os.path.dirname(PROJECT_ROOT)
for path in (PROJECT_ROOT, DEEPREACH_PARENT):
    if path not in sys.path:
        sys.path.insert(0, path)

from deepreach.utils import modules
from dynamics.dynamics_OVERNIGHT_VERIFIED import CoffeeArmDynamics
from src.config.constants import DEFAULT_SLOSH_RAD_MAX
from src.core.arm_dynamics import get_link_positions
from src.envs.base_env import CoffeeArmEnv, inverse_kinematics
from src.reachability.brt_safety_filter import (
    safety_filter as brt_safety_filter,
)

PPO_MODEL = os.path.join(PROJECT_ROOT, "ppo_baseline_final.zip")
PPO_VECNORM = os.path.join(PROJECT_ROOT, "ppo_baseline_vecnormalize.pkl")
BRT_MODEL = os.path.join(PROJECT_ROOT, "brt_model_final2", "model_final2.pth")
ENV_KWARGS = dict(u_max=15.0, T=10.0, dt=0.01)
DT = ENV_KWARGS["dt"]


class PDController:
    def __init__(self, env, kp=5.0, kd=3.0):
        self.q_goal = inverse_kinematics(
            *[float(v) for v in env.goal_pos], env.L, elbow="up"
        )
        self.kp = float(kp)
        self.kd = float(kd)
        self.u_max = float(env.u_max)

    def __call__(self, state, observation):
        del observation
        action = self.kp * (self.q_goal - state[:3]) - self.kd * state[3:6]
        return np.clip(action, -self.u_max, self.u_max).astype(np.float32)


def load_ppo():
    dummy = DummyVecEnv([lambda: CoffeeArmEnv(**ENV_KWARGS)])
    vecnorm = VecNormalize.load(PPO_VECNORM, dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = PPO.load(PPO_MODEL, device="cpu")

    def controller(state, observation):
        del state
        normalized = vecnorm.normalize_obs(observation.reshape(1, -1))
        action, _ = model.predict(normalized, deterministic=True)
        return np.clip(action[0], -15.0, 15.0).astype(np.float32)

    return controller


def load_brt():
    dynamics = CoffeeArmDynamics()
    state_dict = torch.load(BRT_MODEL, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    hidden_features = state_dict["net.net.0.0.weight"].shape[0]
    num_hidden_layers = (
        sum(
            key.startswith("net.net.") and key.endswith(".weight") for key in state_dict
        )
        - 2
    )
    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type="sine",
        mode="mlp",
        final_layer_factor=1,
        hidden_features=hidden_features,
        num_hidden_layers=num_hidden_layers,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, dynamics


def terminal_cause(info):
    if info.get("spill_slosh", False):
        return "spill"
    if info.get("obstacle_hit", False):
        return "obstacle"
    if info.get("joint_violation", False):
        return "joint"
    if info.get("below_ground", False):
        return "ground"
    if info.get("dist_to_goal", float("inf")) < 0.1:
        return "goal"
    return "timeout"


def record_episode(seed, controller, use_filter, brt_model, brt_dynamics, margin):
    env = CoffeeArmEnv(**ENV_KWARGS)
    observation, _ = env.reset(seed=seed)
    records = []
    initial_points = np.asarray(get_link_positions(env.state[:6], env.L))
    initial_state = env.state.copy()
    last_info = {}

    while True:
        state_before = env.state.copy()
        nominal = controller(state_before, observation.copy())
        if use_filter:
            executed, intervened = brt_safety_filter(
                brt_model,
                brt_dynamics,
                state_before,
                nominal,
                t=10.0,
                margin=margin,
            )
        else:
            executed = nominal.copy()
            intervened = False

        observation, reward, terminated, truncated, info = env.step(executed)
        state_after = env.state.copy()
        records.append(
            {
                "step": len(records),
                "time": (len(records) + 1) * DT,
                "state": state_after,
                "link_points": np.asarray(get_link_positions(state_after[:6], env.L)),
                "slosh_radius": float(info["slosh_rad"]),
                "slosh_speed": float(np.linalg.norm(state_after[8:10])),
                "distance_to_goal": float(info["dist_to_goal"]),
                "nominal": np.asarray(nominal, dtype=np.float32),
                "executed": np.asarray(executed, dtype=np.float32),
                "intervened": bool(intervened),
                "reward": float(reward),
            }
        )
        last_info = info
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "initial_state": initial_state,
        "initial_points": initial_points,
        "records": records,
        "cause": terminal_cause(last_info),
        "goal": env.goal_pos.copy(),
        "obstacles": env.obstacles,
        "slosh_limit": env.slosh_rad_max,
    }


def array(traj, key):
    if key in ("nominal", "executed", "link_points", "state"):
        return np.asarray([row[key] for row in traj["records"]])
    return np.asarray([row[key] for row in traj["records"]])


def sphere_mesh(center, radius, n=12):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    return (
        center[0] + radius * np.outer(np.cos(u), np.sin(v)),
        center[1] + radius * np.outer(np.sin(u), np.sin(v)),
        center[2] + radius * np.outer(np.ones_like(u), np.cos(v)),
    )


def setup_scene(ax, title, traj):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(-0.65, 0.75)
    ax.set_ylim(-0.65, 0.75)
    ax.set_zlim(0.0, 1.05)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=25, azim=-55)
    xx, yy = np.meshgrid([-0.65, 0.75], [-0.65, 0.75])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="gray", alpha=0.08)
    for obstacle in traj["obstacles"]:
        xyz = sphere_mesh(np.asarray(obstacle["center"]), obstacle["radius"])
        ax.plot_surface(*xyz, color="#8c564b", alpha=0.42, linewidth=0)
    goal = traj["goal"]
    ax.scatter(*goal, marker="*", s=180, color="gold", edgecolor="black")


def create_pair_gif(label, baseline, filtered, output, stride=4, fps=18):
    fig = plt.figure(figsize=(12, 5.5))
    ax_left = fig.add_subplot(121, projection="3d")
    ax_right = fig.add_subplot(122, projection="3d")
    setup_scene(ax_left, f"{label} baseline", baseline)
    setup_scene(ax_right, f"{label} + BRT (margin=0.005)", filtered)

    artists = []
    for ax, color in ((ax_left, "#d62728"), (ax_right, "#2ca02c")):
        (arm,) = ax.plot([], [], [], "-o", lw=4, ms=6, color=color)
        (trail,) = ax.plot([], [], [], lw=1.5, alpha=0.55, color=color)
        status = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, va="top")
        artists.append((arm, trail, status))

    max_steps = max(len(baseline["records"]), len(filtered["records"]))
    frame_steps = list(range(0, max_steps, stride))
    if frame_steps[-1] != max_steps - 1:
        frame_steps.append(max_steps - 1)

    def update(frame_index):
        step = frame_steps[frame_index]
        changed = []
        for traj, (arm, trail, status) in zip((baseline, filtered), artists):
            idx = min(step, len(traj["records"]) - 1)
            points = array(traj, "link_points")[idx]
            arm.set_data(points[:, 0], points[:, 1])
            arm.set_3d_properties(points[:, 2])
            history = array(traj, "link_points")[: idx + 1, -1]
            trail.set_data(history[:, 0], history[:, 1])
            trail.set_3d_properties(history[:, 2])
            row = traj["records"][idx]
            frozen = step >= len(traj["records"])
            state = traj["cause"] if frozen else "running"
            int_text = " | FILTER ACTIVE" if row["intervened"] else ""
            status.set_text(
                f"t={row['time']:.2f}s | {state}{int_text}\n"
                f"slosh={1000 * row['slosh_radius']:.2f} mm | "
                f"goal distance={row['distance_to_goal']:.3f} m"
            )
            status.set_color("darkorange" if row["intervened"] else "black")
            changed.extend((arm, trail, status))
        fig.suptitle(
            f"Matched seed {baseline['seed']}: {label} navigation",
            fontsize=14,
            fontweight="bold",
        )
        return changed

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_steps), interval=1000 / fps, blit=False
    )
    ani.save(output, writer=animation.PillowWriter(fps=fps), dpi=95)
    plt.close(fig)


def shade_spill_region(ax, limit, ymax):
    ax.axhspan(limit, ymax, color="red", alpha=0.13, label="spill region")
    ax.axhline(
        limit,
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=f"spill threshold ({1000 * limit:.2f} mm)",
    )


def create_metrics_plot(trajectories, output):
    labels = list(trajectories)
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex="col")
    colors = ["#d62728", "#2ca02c", "#9467bd"]
    for col, label in enumerate(labels):
        traj = trajectories[label]
        time = array(traj, "time")
        slosh = array(traj, "slosh_radius")
        slosh_speed = array(traj, "slosh_speed")
        executed = array(traj, "executed")
        distance = array(traj, "distance_to_goal")
        interventions = array(traj, "intervened").astype(bool)

        ymax = max(float(slosh.max()) * 1.15, traj["slosh_limit"] * 1.4)
        axes[0, col].plot(time, slosh * 1000, color="#1f77b4", lw=1.8)
        shade_spill_region(axes[0, col], traj["slosh_limit"] * 1000, ymax * 1000)
        axes[0, col].set_ylim(0, ymax * 1000)
        axes[0, col].set_title(f"{label}\nresult: {traj['cause']}")

        for joint in range(3):
            axes[1, col].plot(
                time,
                executed[:, joint],
                color=colors[joint],
                lw=1.1,
                label=f"u{joint + 1}",
            )
        axes[1, col].axhline(15, color="black", ls=":", lw=0.8)
        axes[1, col].axhline(-15, color="black", ls=":", lw=0.8)

        axes[2, col].plot(time, distance, color="#222222", lw=1.8)
        axes[2, col].axhspan(0, 0.1, color="green", alpha=0.14)
        axes[2, col].axhline(0.1, color="green", ls="--", lw=1.2)

        if interventions.any():
            for row in range(3):
                axes[row, col].fill_between(
                    time,
                    0,
                    1,
                    where=interventions,
                    transform=axes[row, col].get_xaxis_transform(),
                    color="orange",
                    alpha=0.12,
                    linewidth=0,
                )
        for row in range(3):
            axes[row, col].grid(alpha=0.25)

    axes[0, 0].set_ylabel("Slosh displacement (mm)")
    axes[1, 0].set_ylabel("Executed control")
    axes[2, 0].set_ylabel("Distance to goal (m)")
    for ax in axes[2]:
        ax.set_xlabel("Time (s)")
    axes[1, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Matched rollout metrics (orange shading = safety-filter intervention)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def contiguous_intervention_windows(mask):
    windows = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            windows.append((start, index - 1))
            start = None
    if start is not None:
        windows.append((start, len(mask) - 1))
    return windows


def create_action_plot(label, traj, output):
    time = array(traj, "time")
    nominal = array(traj, "nominal")
    executed = array(traj, "executed")
    intervened = array(traj, "intervened").astype(bool)
    difference = np.linalg.norm(executed - nominal, axis=1)
    colors = ["#d62728", "#2ca02c", "#9467bd"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for joint in range(3):
        axes[joint].plot(
            time,
            nominal[:, joint],
            color=colors[joint],
            ls="--",
            alpha=0.65,
            label=f"nominal u{joint + 1}",
        )
        axes[joint].plot(
            time,
            executed[:, joint],
            color=colors[joint],
            lw=1.5,
            label=f"safe/executed u{joint + 1}",
        )
        axes[joint].set_ylabel(f"u{joint + 1}")
        axes[joint].legend(loc="upper right", fontsize=8)
    axes[3].plot(time, difference, color="black", lw=1.5)
    axes[3].scatter(
        time[intervened],
        difference[intervened],
        color="darkorange",
        s=9,
        label="intervention",
    )
    axes[3].set_ylabel("||safe - nominal||")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right", fontsize=8)

    for ax in axes:
        for start, end in contiguous_intervention_windows(intervened):
            ax.axvspan(time[start], time[end] + DT, color="orange", alpha=0.13)
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"{label}: nominal actions and safety-filter replacements\n"
        f"{intervened.sum()} interventions / {len(intervened)} steps",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_action_csv(label, traj, output):
    fieldnames = [
        "controller",
        "step",
        "time_s",
        "intervened",
        "nominal_u1",
        "nominal_u2",
        "nominal_u3",
        "executed_u1",
        "executed_u2",
        "executed_u3",
        "action_change_norm",
        "slosh_mm",
        "distance_to_goal_m",
    ]
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in traj["records"]:
            nominal = row["nominal"]
            executed = row["executed"]
            writer.writerow(
                {
                    "controller": label,
                    "step": row["step"],
                    "time_s": f"{row['time']:.4f}",
                    "intervened": int(row["intervened"]),
                    "nominal_u1": f"{nominal[0]:.7f}",
                    "nominal_u2": f"{nominal[1]:.7f}",
                    "nominal_u3": f"{nominal[2]:.7f}",
                    "executed_u1": f"{executed[0]:.7f}",
                    "executed_u2": f"{executed[1]:.7f}",
                    "executed_u3": f"{executed[2]:.7f}",
                    "action_change_norm": (f"{np.linalg.norm(executed - nominal):.7f}"),
                    "slosh_mm": f"{1000 * row['slosh_radius']:.5f}",
                    "distance_to_goal_m": f"{row['distance_to_goal']:.6f}",
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "controller_visualizations"),
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ppo = load_ppo()
    brt_model, brt_dynamics = load_brt()
    pd = PDController(CoffeeArmEnv(**ENV_KWARGS))

    configurations = [
        ("PD", pd, False),
        ("PD + BRT", pd, True),
        ("PPO", ppo, False),
        ("PPO + BRT", ppo, True),
    ]
    trajectories = {}
    for label, controller, use_filter in configurations:
        print(f"Recording {label}...", flush=True)
        trajectories[label] = record_episode(
            args.seed,
            controller,
            use_filter,
            brt_model,
            brt_dynamics,
            args.margin,
        )

    create_pair_gif(
        "PD",
        trajectories["PD"],
        trajectories["PD + BRT"],
        os.path.join(args.output_dir, f"pd_side_by_side_seed{args.seed}.gif"),
    )
    create_pair_gif(
        "PPO",
        trajectories["PPO"],
        trajectories["PPO + BRT"],
        os.path.join(args.output_dir, f"ppo_side_by_side_seed{args.seed}.gif"),
    )
    create_metrics_plot(
        trajectories,
        os.path.join(args.output_dir, f"all_metrics_seed{args.seed}.png"),
    )

    for label in ("PD + BRT", "PPO + BRT"):
        stem = label.lower().replace(" ", "_").replace("+", "plus")
        create_action_plot(
            label,
            trajectories[label],
            os.path.join(args.output_dir, f"{stem}_actions_seed{args.seed}.png"),
        )
        write_action_csv(
            label,
            trajectories[label],
            os.path.join(args.output_dir, f"{stem}_actions_seed{args.seed}.csv"),
        )

    summary = {
        label: {
            "steps": len(traj["records"]),
            "duration_s": len(traj["records"]) * DT,
            "cause": traj["cause"],
            "peak_slosh_mm": float(1000 * array(traj, "slosh_radius").max()),
            "minimum_goal_distance_m": float(array(traj, "distance_to_goal").min()),
            "interventions": int(array(traj, "intervened").sum()),
        }
        for label, traj in trajectories.items()
    }
    with open(
        os.path.join(args.output_dir, f"summary_seed{args.seed}.json"), "w"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
