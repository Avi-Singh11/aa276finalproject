# PPO training for the coffee pouring arm
# USE_SAFETY_FILTER = False: baseline run
# USE_SAFETY_FILTER = True: filtered training (needs BRT from compute_brt.py)

import numpy as np
import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import os

from coffee_pouring_env import CoffeePouringEnv, safety_filter

# Flags
USE_SAFETY_FILTER = False   # set True once BRT is computed and sloshing model ready

TOTAL_TIMESTEPS= 2_000_000
LOG_INTERVAL = 5_000 # steps between metric logs
SAVE_INTERVAL= 50_000 # steps between model checkpoints
RUN_NAME = "baseline" if not USE_SAFETY_FILTER else "filtered"
CKPT_DIR = f"checkpoints/{RUN_NAME}"

# Environment parameters
# a_max relaxed during early training — tighten once policy is learning
ENV_KWARGS = dict(a_max=20.0, alpha_max=0.3, u_max=2.0, T=10.0, dt=0.01,)

class FilteredEnv(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.intervention_count = 0
        self.step_count_ep = 0

    def reset(self, **kwargs):
        self.intervention_count = 0
        self.step_count_ep = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        u_safe, intervened = safety_filter(self.env, self.env.state, action)
        if intervened:
            self.intervention_count += 1
        self.step_count_ep += 1

        obs, reward, terminated, truncated, info = self.env.step(u_safe)
        info["intervened"] = intervened
        info["intervention_count"] = self.intervention_count
        info["step_count_ep"] = self.step_count_ep
        return obs, reward, terminated, truncated, info


class SafetyMetricsCallback(BaseCallback):

    def __init__(self, log_interval, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._reset_buffers()

    def _reset_buffers(self):
        self.ep_spills = []
        self.ep_completions = []
        self.ep_lengths = []
        self.ep_interventions = []

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                spill = float(info.get("spill_accel", False) or info.get("spill_slosh", False))
                dist = info.get("dist_to_goal", float("inf"))
                n_steps = info.get("step_count_ep", info.get("episode", {}).get("l", 1))
                n_ints = info.get("intervention_count", 0)

                self.ep_spills.append(spill)
                self.ep_completions.append(float(dist < 0.1))
                self.ep_lengths.append(n_steps)
                self.ep_interventions.append(n_ints / max(n_steps, 1))

        if self.num_timesteps % self.log_interval == 0 and self.ep_spills:
            metrics = {
                "safety/spill_rate": np.mean(self.ep_spills),
                "safety/completion_rate": np.mean(self.ep_completions),
                "safety/intervention_rate": np.mean(self.ep_interventions),
                "safety/steps_per_episode": np.mean(self.ep_lengths),
            }
            for k, v in metrics.items():
                self.logger.record(k, v)
            self.logger.dump(self.num_timesteps)
            self._reset_buffers()

        return True


def make_env(seed=0):
    def _init():
        env = CoffeePouringEnv(**ENV_KWARGS)
        if USE_SAFETY_FILTER:
            env = FilteredEnv(env)
        env = Monitor(env, info_keywords=("spill_accel", "spill_slosh", "dist_to_goal"))
        return env
    return _init

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)

    run = wandb.init(
        project="aa276-coffee-pouring",
        name=RUN_NAME,
        config={
            "algorithm":        "PPO",
            "use_safety_filter": USE_SAFETY_FILTER,
            "total_timesteps":  TOTAL_TIMESTEPS,
            **ENV_KWARGS,
            "learning_rate":    3e-4,
            "n_steps":          2048,
            "batch_size":       64,
            "n_epochs":         10,
            "gamma":            0.99,
            "gae_lambda":       0.95,
            "clip_range":       0.2,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    print(f"Run: {RUN_NAME}  ({run.url})")
    print(f"Filter {USE_SAFETY_FILTER}")
    print(f"Steps: {TOTAL_TIMESTEPS:,}")
    print()

    vec_env = DummyVecEnv([make_env(seed=0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu",
        verbose=1,
        tensorboard_log=f"logs/{RUN_NAME}",
    )
    callbacks = CallbackList([
        SafetyMetricsCallback(log_interval=LOG_INTERVAL),
        CheckpointCallback(
            save_freq=SAVE_INTERVAL,
            save_path=CKPT_DIR,
            name_prefix=f"ppo_{RUN_NAME}",
            save_vecnormalize=True,
        ),
    ])
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    model.save(f"ppo_{RUN_NAME}_final")
    vec_env.save(f"ppo_{RUN_NAME}_vecnormalize.pkl")
    wandb.save(f"ppo_{RUN_NAME}_final.zip")
    wandb.save(f"ppo_{RUN_NAME}_vecnormalize.pkl")
    run.finish()
    print(f"\nSaved ppo_{RUN_NAME}_final.zip and normalization stats.")
