"""PPO training for the coffee arm.

Training rollouts are intentionally unfiltered. Safety is handled only at
deployment time through the reachability-based filter described in the report.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import gymnasium as gym
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, HumanOutputFormat, KVWriter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.envs.base_env import CoffeeArmEnv

N_ENVS = 8
TOTAL_TIMESTEPS = 5_000_000
# LOG_INTERVAL must be a multiple of (N_ENVS * n_steps) = 8192
LOG_INTERVAL = 8_192
SAVE_INTERVAL = 100_000
RUN_NAME = "same_sector_v1"
CKPT_DIR = f"checkpoints/{RUN_NAME}"

# Environment parameters (Cleaned up old a_max and alpha_max variables)
ENV_KWARGS = dict(
    u_max=15.0,
    T=10.0,
    dt=0.01,
)


class WandbOutputFormat(KVWriter):
    """Pipes every SB3 logger.dump() call directly to wandb.log()."""
    def write(self, key_values, key_excluded, step=0):
        # SB3 uses numpy scalars, not Python int/float, so use try/float() not isinstance
        payload = {}
        for k, v in key_values.items():
            try:
                payload[k] = float(v)
            except (TypeError, ValueError):
                pass
        if payload:
            wandb.log(payload, step=step)

    def close(self):
        pass


class SafetyMetricsCallback(BaseCallback):
    def __init__(self, log_interval, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._reset_buffers()

    def _reset_buffers(self):
        self.ep_spills  = []
        self.ep_ground  = []
        self.ep_obs     = []
        self.ep_completions = []
        self.ep_lengths = []

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                spill   = float(info.get("spill_slosh", False))
                ground  = float(info.get("below_ground", False))
                obs_hit = float(
                    info.get("obstacle_hit", False)
                    or info.get("joint_violation", False)
                )
                dist    = info.get("dist_to_goal", float("inf"))
                n_steps = info.get("step_count_ep", info.get("episode", {}).get("l", 1))

                self.ep_spills.append(spill)
                self.ep_ground.append(ground)
                self.ep_obs.append(obs_hit)
                self.ep_completions.append(float(dist < 0.1))
                self.ep_lengths.append(n_steps)

        if self.num_timesteps % self.log_interval == 0 and self.ep_spills:
            wandb.log({
                "safety/spill_rate":      np.mean(self.ep_spills),
                "safety/ground_rate":     np.mean(self.ep_ground),
                "safety/obstacle_rate":   np.mean(self.ep_obs),
                "safety/completion_rate": np.mean(self.ep_completions),
                "safety/steps_per_episode": np.mean(self.ep_lengths),
            }, step=self.num_timesteps)
            self._reset_buffers()

        return True


def make_env(seed=0):
    def _init():
        env = CoffeeArmEnv(**ENV_KWARGS)
        env = Monitor(
            env,
            info_keywords=(
                "spill_slosh",
                "below_ground",
                "obstacle_hit",
                "joint_violation",
                "dist_to_goal",
            ),
        )
        return env
    return _init


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)

    run = wandb.init(
        project="aa276-coffee-pouring",
        name=RUN_NAME,
        config={
            "algorithm": "PPO",
            "env": "CoffeeArmEnv",
            "use_safety_filter": False,
            "norm_reward": False,
            "total_timesteps": TOTAL_TIMESTEPS,
            **ENV_KWARGS,
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "n_envs": N_ENVS,
            "batch_size": 2048,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.05,
            "goal": "[0.55, 0.10, 0.10]",
            "start_az_deg": "[70, 90]",
            "at_goal_bonus": 100.0,
        },
        sync_tensorboard=False,
        save_code=True,
    )

    print(f"Run: {RUN_NAME}  ({run.url})")
    print("Filter: False")
    print(f"Steps: {TOTAL_TIMESTEPS:,}\n")

    vec_env = DummyVecEnv([make_env(seed=i) for i in range(N_ENVS)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        device="cuda",
        verbose=1,
        tensorboard_log=None,
    )

    model.set_logger(Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), WandbOutputFormat()],
    ))

    callbacks = CallbackList(
        [
            SafetyMetricsCallback(log_interval=LOG_INTERVAL),
            CheckpointCallback(
                save_freq=SAVE_INTERVAL,
                save_path=CKPT_DIR,
                name_prefix=f"ppo_{RUN_NAME}",
                save_vecnormalize=True,
            ),
        ]
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    model.save(f"ppo_{RUN_NAME}_final")
    vec_env.save(f"ppo_{RUN_NAME}_vecnormalize.pkl")
    wandb.save(f"ppo_{RUN_NAME}_final.zip")
    wandb.save(f"ppo_{RUN_NAME}_vecnormalize.pkl")
    run.finish()
    print(f"\nSaved ppo_{RUN_NAME}_final.zip and normalization stats.")