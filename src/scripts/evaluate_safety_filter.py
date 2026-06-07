"""Evaluate PPO baseline with and without the BRT safety filter.

Runs N_EPISODES rollouts under each condition and reports:
  - Spill / unsafe termination rate
  - Task completion rate (dist_to_goal < 0.1 at truncation)
  - Filter intervention rate (how often the QP overrides the policy)
  - Mean episode reward
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEEPREACH_PATH = os.path.dirname(PROJECT_ROOT)
if DEEPREACH_PATH not in sys.path:
    sys.path.insert(0, DEEPREACH_PATH)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.base_env import CoffeeArmEnv
from src.reachability.safety_filter import safety_filter as brt_safety_filter
from src.reachability.compute_brt import CFG as BRT_CFG

N_EPISODES = 50
MODEL_PATH = os.path.join(PROJECT_ROOT, 'ppo_baseline_final.zip')
VECNORM_PATH = os.path.join(PROJECT_ROOT, 'ppo_baseline_vecnormalize.pkl')
BRT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'brt_model', 'model_final.pth')

ENV_KWARGS = dict(u_max=15.0, T=10.0, dt=0.01)


def make_env(seed=0):
    def _init():
        return CoffeeArmEnv(**ENV_KWARGS)
    return _init


def load_ppo(model_path, vecnorm_path):
    vec_env = DummyVecEnv([make_env(seed=42)])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env)
    return model, vec_env


def load_brt():
    from dynamics.dynamics import CoffeeArmDynamics
    try:
        from deepreach.utils import modules
    except Exception as e:
        raise ImportError(f"Could not import DeepReach: {e}")

    dynamics = CoffeeArmDynamics()
    brt_model = modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type='sine',
        mode='mlp',
        final_layer_factor=1,
        hidden_features=BRT_CFG['hidden_features'],
        num_hidden_layers=BRT_CFG['num_hidden_layers'],
    )
    state_dict = torch.load(BRT_MODEL_PATH, map_location='cpu')
    brt_model.load_state_dict(state_dict)
    brt_model.eval()
    return brt_model, dynamics


def run_episodes(model, vec_env, n_episodes, use_filter=False, brt_model=None, brt_dynamics=None):
    rewards, spills, completions, interventions = [], [], [], []

    raw_env = vec_env.venv.envs[0]  # unwrapped CoffeeArmEnv

    for ep in range(n_episodes):
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_interventions = 0
        ep_steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            u_nom = action[0].copy()  # (3,) nominal action from PPO

            if use_filter:
                state_10d = raw_env.state.copy()
                u_safe, intervened = brt_safety_filter(
                    brt_model, brt_dynamics, state_10d, u_nom,
                    t=BRT_CFG['tMax'],
                )
                if intervened:
                    ep_interventions += 1
                action[0] = u_safe

            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])
            ep_reward += float(reward[0])
            ep_steps += 1

        info = info_arr[0]
        unsafe = bool(
            info.get('spill_slosh', False)
            or info.get('obstacle_hit', False)
            or info.get('joint_violation', False)
            or info.get('below_ground', False)
        )
        completed = bool(info.get('dist_to_goal', float('inf')) < 0.1)

        rewards.append(ep_reward)
        spills.append(float(unsafe))
        completions.append(float(completed))
        interventions.append(ep_interventions / max(ep_steps, 1))

        tag = 'F' if use_filter else '-'
        mark = 'UNSAFE' if unsafe else ('DONE' if completed else 'trunc')
        print(f"  [{tag}] ep {ep+1:3d}/{n_episodes}  reward={ep_reward:7.1f}  {mark}"
              + (f"  intervened={ep_interventions}/{ep_steps}" if use_filter else ""))

    return dict(
        mean_reward=float(np.mean(rewards)),
        spill_rate=float(np.mean(spills)),
        completion_rate=float(np.mean(completions)),
        intervention_rate=float(np.mean(interventions)) if use_filter else None,
    )


def print_results(label, stats):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Mean reward:       {stats['mean_reward']:8.2f}")
    print(f"  Spill/unsafe rate: {stats['spill_rate']*100:6.1f}%")
    print(f"  Completion rate:   {stats['completion_rate']*100:6.1f}%")
    if stats['intervention_rate'] is not None:
        print(f"  Intervention rate: {stats['intervention_rate']*100:6.1f}%")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=N_EPISODES)
    parser.add_argument('--no-filter', action='store_true', help='Skip the safety-filter run')
    parser.add_argument('--filter-only', action='store_true', help='Skip the baseline run')
    args = parser.parse_args()

    print(f"Loading PPO from {MODEL_PATH}")
    model, vec_env = load_ppo(MODEL_PATH, VECNORM_PATH)
    print("PPO loaded.\n")

    brt_model = brt_dynamics = None
    if not args.no_filter:
        print(f"Loading BRT model from {BRT_MODEL_PATH}")
        brt_model, brt_dynamics = load_brt()
        print("BRT model loaded.\n")

    if not args.filter_only:
        print(f"--- Running {args.episodes} episodes: PPO baseline (no filter) ---")
        baseline_stats = run_episodes(model, vec_env, args.episodes, use_filter=False)
        print_results("PPO Baseline (no safety filter)", baseline_stats)

    if not args.no_filter:
        print(f"--- Running {args.episodes} episodes: PPO + BRT safety filter ---")
        filter_stats = run_episodes(
            model, vec_env, args.episodes,
            use_filter=True, brt_model=brt_model, brt_dynamics=brt_dynamics,
        )
        print_results("PPO + BRT Safety Filter", filter_stats)

    if not args.filter_only and not args.no_filter:
        print("--- Delta (filter vs baseline) ---")
        print(f"  Reward:      {filter_stats['mean_reward'] - baseline_stats['mean_reward']:+.2f}")
        print(f"  Spill rate:  {(filter_stats['spill_rate'] - baseline_stats['spill_rate'])*100:+.1f}pp")
        print(f"  Completion:  {(filter_stats['completion_rate'] - baseline_stats['completion_rate'])*100:+.1f}pp")
