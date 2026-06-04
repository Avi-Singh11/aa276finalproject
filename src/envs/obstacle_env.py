"""CoffeeArmEnv extends the base pouring environment with obstacle collisions."""

from __future__ import annotations

import numpy as np
import gymnasium as gym

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.constants import DEFAULT_OBSTACLES, DEFAULT_JOINT_LIMITS
from src.core.arm_dynamics import position_cup
from src.core.obstacles import (
    check_joint_limits, 
    check_arm_obstacle_collision,
    check_ground_contact
)
from src.core.simulation import coupled_dynamics
from src.envs.base_env import CoffeePouringEnv

class CoffeeArmEnv(CoffeePouringEnv):
    """Obstacle-aware version of the task environment."""

    DEFAULT_OBSTACLES = DEFAULT_OBSTACLES

    def __init__(self, *args, obstacles=None, joint_limits=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacles = list(obstacles) if obstacles is not None else list(self.DEFAULT_OBSTACLES)
        self.joint_limits = np.asarray(joint_limits if joint_limits is not None else DEFAULT_JOINT_LIMITS, dtype=np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -self.u_max, self.u_max)

        next_state = coupled_dynamics(
            self.state, action, self.K, self.L, self.dt, l_eff=self.l_eff
        ).astype(np.float32)

        cup_pos = position_cup(next_state[:6], self.L)
        dist_to_goal = float(np.linalg.norm(cup_pos - self.goal_pos))
        
        # Cartesian Math for slosh
        x_slosh, y_slosh = float(next_state[6]), float(next_state[7])
        slosh_rad = np.sqrt(x_slosh**2 + y_slosh**2)

        # Full failure set F
        spill_slosh = bool(slosh_rad > self.slosh_rad_max)
        below_ground = bool(cup_pos[2] < 0.0)
        obstacle_hit = bool(check_arm_obstacle_collision(next_state[:6], self.L, self.obstacles))
        joint_violation = bool(check_joint_limits(next_state[:3], self.joint_limits))

        terminated = spill_slosh or below_ground or obstacle_hit or joint_violation

        self.step_count += 1
        truncated = bool(self.step_count >= self.max_steps)

        reward = -0.1 * dist_to_goal - 0.01 * float(np.linalg.norm(action) ** 2)
        if dist_to_goal < 0.1:
            reward += 10.0
        if spill_slosh:
            reward -= 50.0
        if below_ground:
            reward -= 50.0
        if obstacle_hit or joint_violation:
            reward -= 50.0

        self.state = next_state

        info = {
            "cup_pos": cup_pos.astype(np.float32),
            "slosh_rad": slosh_rad,
            "dist_to_goal": dist_to_goal,
            "spill_slosh": spill_slosh,
            "below_ground": below_ground,
            "obstacle_hit": obstacle_hit,
            "joint_violation": joint_violation,
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
            and not check_arm_obstacle_collision(next_state[:6], self.L, self.obstacles)
            and not check_joint_limits(next_state[:3], self.joint_limits)
        )

if __name__ == "__main__":
    print("=== CoffeeArmEnv sanity check ===")
    env = CoffeeArmEnv()
    obs, _ = env.reset(seed=0)
    
    total_reward = 0.0
    for i in range(env.max_steps + 5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended at step {i+1} | terminated={terminated} truncated={truncated}")
            print(f"cup_pos={info['cup_pos'].round(3)}, slosh_rad={info['slosh_rad']:.3f}")
            print(f"spill_slosh={info['spill_slosh']}, obstacle_hit={info['obstacle_hit']}, joint_violation={info['joint_violation']}")
            break