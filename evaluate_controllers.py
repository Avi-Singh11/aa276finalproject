"""Matched evaluation of a fixed-horizon BRT with a value safety margin."""

from __future__ import annotations

import argparse
import json
import os
import sys

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
from src.envs.base_env import CoffeeArmEnv, inverse_kinematics
from src.reachability.brt_safety_filter import (
    safety_filter as brt_safety_filter,
)


PPO_MODEL = os.path.join(PROJECT_ROOT, "ppo_baseline_final.zip")
PPO_VECNORM = os.path.join(PROJECT_ROOT, "ppo_baseline_vecnormalize.pkl")
BRT_MODEL = os.path.join(PROJECT_ROOT, "brt_model_final2", "model_final2.pth")
ENV_KWARGS = dict(u_max=15.0, T=10.0, dt=0.01)


class PDController:
    def __init__(self, env: CoffeeArmEnv, kp=5.0, kd=3.0):
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
    model = PPO.load(PPO_MODEL)

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
    first_weight = state_dict["net.net.0.0.weight"]
    hidden_features = first_weight.shape[0]
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


def run_episode(seed, controller, use_filter, brt_model, brt_dynamics, filter_margin):
    env = CoffeeArmEnv(**ENV_KWARGS)
    observation, _ = env.reset(seed=seed)
    reward_total = 0.0
    interventions = 0
    peak_slosh = 0.0
    minimum_cup_z = float("inf")
    minimum_goal_distance = float("inf")
    last_info = {}

    while True:
        nominal = controller(env.state.copy(), observation.copy())
        if use_filter:
            action, intervened = brt_safety_filter(
                brt_model,
                brt_dynamics,
                env.state.copy(),
                nominal,
                t=10.0,
                margin=filter_margin,
            )
            interventions += int(intervened)
        else:
            action = nominal

        observation, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        peak_slosh = max(peak_slosh, float(info["slosh_rad"]))
        minimum_cup_z = min(minimum_cup_z, float(info["cup_pos"][2]))
        minimum_goal_distance = min(minimum_goal_distance, float(info["dist_to_goal"]))
        last_info = info
        if terminated or truncated:
            break

    failed = bool(
        last_info.get("spill_slosh", False)
        or last_info.get("obstacle_hit", False)
        or last_info.get("joint_violation", False)
        or last_info.get("below_ground", False)
    )
    completed = bool(not failed and last_info.get("dist_to_goal", float("inf")) < 0.1)
    cause = (
        "spill"
        if last_info.get("spill_slosh", False)
        else "obstacle"
        if last_info.get("obstacle_hit", False)
        else "joint"
        if last_info.get("joint_violation", False)
        else "ground"
        if last_info.get("below_ground", False)
        else "goal"
        if completed
        else "timeout"
    )
    steps = int(last_info.get("step_count_ep", env.step_count))
    return {
        "seed": int(seed),
        "reward": reward_total,
        "failed": failed,
        "completed": completed,
        "cause": cause,
        "steps": steps,
        "peak_slosh": peak_slosh,
        "minimum_cup_z": minimum_cup_z,
        "minimum_goal_distance": minimum_goal_distance,
        "interventions": interventions,
        "intervention_rate": interventions / max(steps, 1),
    }


def summarize(episodes):
    return {
        "episodes": len(episodes),
        "failure_rate": float(np.mean([ep["failed"] for ep in episodes])),
        "completion_rate": float(np.mean([ep["completed"] for ep in episodes])),
        "timeout_rate": float(np.mean([ep["cause"] == "timeout" for ep in episodes])),
        "mean_reward": float(np.mean([ep["reward"] for ep in episodes])),
        "mean_steps": float(np.mean([ep["steps"] for ep in episodes])),
        "mean_peak_slosh": float(np.mean([ep["peak_slosh"] for ep in episodes])),
        "mean_minimum_goal_distance": float(
            np.mean([ep["minimum_goal_distance"] for ep in episodes])
        ),
        "mean_intervention_rate": float(
            np.mean([ep["intervention_rate"] for ep in episodes])
        ),
        "causes": {
            cause: sum(ep["cause"] == cause for ep in episodes)
            for cause in ["goal", "timeout", "spill", "obstacle", "joint", "ground"]
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--margin", type=float, required=True)
    args = parser.parse_args()
    if args.margin < 0.0:
        parser.error("--margin must be nonnegative")

    margin_tag = f"{args.margin:.4f}".replace(".", "p")
    output_path = os.path.join(
        PROJECT_ROOT,
        f"controller_results_margin_{margin_tag}.json",
    )

    ppo_controller = load_ppo()
    brt_model, brt_dynamics = load_brt()
    template_env = CoffeeArmEnv(**ENV_KWARGS)
    pd_controller = PDController(template_env)
    seeds = range(args.seed_start, args.seed_start + args.episodes)

    conditions = [
        ("PD", pd_controller, False),
        ("PD+BRT", pd_controller, True),
        ("PPO", ppo_controller, False),
        ("PPO+BRT", ppo_controller, True),
    ]
    output = {}
    for label, controller, use_filter in conditions:
        print(f"\n=== {label} ===", flush=True)
        episodes = []
        for seed in seeds:
            result = run_episode(
                seed,
                controller,
                use_filter,
                brt_model,
                brt_dynamics,
                args.margin,
            )
            episodes.append(result)
            print(
                f"seed={seed:2d} cause={result['cause']:8s} "
                f"steps={result['steps']:4d} reward={result['reward']:8.2f} "
                f"peak_slosh={result['peak_slosh']:.5f} "
                f"intervene={100 * result['intervention_rate']:.1f}%",
                flush=True,
            )
        summary = summarize(episodes)
        output[label] = {"summary": summary, "episodes": episodes}
        print(json.dumps(summary, indent=2), flush=True)

    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
