"""Gym environment for safe RL coffee pouring.

This is the simpler non-obstacle variant. It uses the shared physics core and
keeps the 10D state ordering:

x = [q1, q2, q3, dq1, dq2, dq3, x_slosh, y_slosh, vx_slosh, vy_slosh]
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.constants import (
    DEFAULT_L, DEFAULT_K, DEFAULT_U_MAX, 
    DEFAULT_SLOSH_RAD_MAX, DEFAULT_L_EFF, 
    STATE_DIM, OBS_DIM
)
from src.core.arm_dynamics import position_cup
from src.core.simulation import coupled_dynamics
from src.core.obstacles import check_ground_contact

def safety_filter(env, state, u_policy, n_samples=100):
    """Backward-compatible random-sampling safety filter."""
    if env.is_safe(state, u_policy):
        return np.asarray(u_policy, dtype=np.float32), False

    for _ in range(n_samples):
        candidate = env.action_space.sample()
        if env.is_safe(state, candidate):
            return np.asarray(candidate, dtype=np.float32), True

    return np.zeros(3, dtype=np.float32), True

class CoffeePouringEnv(gym.Env):
    """Observation: (13,) [state(10), goal-relative cup vector(3)]"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        L=None,
        K=None,
        dt=0.01,
        T=10.0,
        u_max=DEFAULT_U_MAX,
        slosh_rad_max=DEFAULT_SLOSH_RAD_MAX,
        goal_pos=None,
        randomize_goal=False,
        l_eff=DEFAULT_L_EFF,
    ):
        super().__init__()
        self.L = np.asarray(L if L is not None else DEFAULT_L, dtype=np.float32)
        self.K = np.asarray(K if K is not None else DEFAULT_K, dtype=np.float32)
        self.dt = float(dt)
        self.T = float(T)
        self.max_steps = int(T / dt)
        self.u_max = float(u_max)
        self.slosh_rad_max = float(slosh_rad_max)
        self.l_eff = float(l_eff)
        self.goal_pos = np.asarray(goal_pos if goal_pos is not None else [0.5, 0.5, 1.5], dtype=np.float32)
        self.randomize_goal = bool(randomize_goal)

        # Updated bounds for Cartesian slosh (bounded by l_eff)
        obs_high = np.array(
            [np.pi]*3 + [10.0]*3 + [self.l_eff, self.l_eff, 10.0, 10.0] + [6.0]*3,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        
        # Bounded joint actuation commands |u_i| <= u_max
        self.action_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(3,), dtype=np.float32)

        self.state = None
        self.step_count = 0

    def _goal_vec(self):
        cup_pos = position_cup(self.state[:6], self.L)
        return (self.goal_pos - cup_pos).astype(np.float32)

    def _make_obs(self):
        return np.concatenate([self.state, self._goal_vec()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        theta_init = self.np_random.uniform(-0.3, 0.3, size=3)
        self.state = np.concatenate([theta_init, np.zeros(7)]).astype(np.float32)
        self.step_count = 0
        if self.randomize_goal:
            self.goal_pos = self.np_random.uniform(low=[-1.5, -1.5, 0.5], high=[1.5, 1.5, 2.5]).astype(np.float32)
        return self._make_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -self.u_max, self.u_max)

        next_state = coupled_dynamics(
            self.state, action, self.K, self.L, self.dt, l_eff=self.l_eff
        ).astype(np.float32)

        # Cartesian Math for slosh
        x_slosh, y_slosh = float(next_state[6]), float(next_state[7])
        slosh_rad = np.sqrt(x_slosh**2 + y_slosh**2)
        
        cup_pos = position_cup(next_state[:6], self.L)
        dist_to_goal = float(np.linalg.norm(cup_pos - self.goal_pos))

        # Failure set conditions
        spill_slosh = bool(slosh_rad > self.slosh_rad_max)
        below_ground = bool(cup_pos[2] < 0.0)
        terminated = spill_slosh or below_ground

        self.step_count += 1
        truncated = bool(self.step_count >= self.max_steps)

        reward = -0.1 * dist_to_goal - 0.01 * float(np.linalg.norm(action) ** 2)
        if dist_to_goal < 0.1:
            reward += 10.0
        if spill_slosh:
            reward -= 50.0

        self.state = next_state

        info = {
            "cup_pos": cup_pos.astype(np.float32),
            "slosh_rad": slosh_rad,
            "dist_to_goal": dist_to_goal,
            "spill_slosh": spill_slosh,
            "below_ground": below_ground,
            "step_count_ep": self.step_count,
        }
        return self._make_obs(), reward, terminated, truncated, info

    def is_safe(self, state, action):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        
        next_state = coupled_dynamics(
            state, action, self.K, self.L, self.dt, l_eff=self.l_eff
        )
        
        slosh_rad = np.sqrt(next_state[6]**2 + next_state[7]**2)
        
        return (
            slosh_rad <= self.slosh_rad_max
            and not check_ground_contact(next_state[:6], self.L)
        )

    def render(self):
        pass

if __name__ == "__main__":
    print("=== CoffeePouringEnv sanity check ===")
    env = CoffeePouringEnv()
    obs, _ = env.reset(seed=0)
    print(f"Obs shape: {obs.shape}  (expected {OBS_DIM})")
    
    total_reward = 0.0
    for i in range(env.max_steps + 5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended at step {i+1} | terminated={terminated} truncated={truncated}")
            print(f"cup_pos={info['cup_pos'].round(3)}, slosh_rad={info['slosh_rad']:.3f}")
            print(f"Total reward: {total_reward:.2f}")
            break