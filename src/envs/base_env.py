"""Coffee-arm gym environment.

10D state: [q1, q2, q3, dq1, dq2, dq3, x_slosh, y_slosh, vx_slosh, vy_slosh]

Failure set:
  - cup below ground  (cup_z < 0)
  - slosh radius > slosh_rad_max  (spill)
  - any arm link hits an obstacle
  - joint angle exceeds joint_limits

CoffeeArmEnv is an alias for CoffeePouringEnv (same class).
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
    DEFAULT_OBSTACLES, DEFAULT_JOINT_LIMITS,
    STATE_DIM, OBS_DIM,
)
from src.core.arm_dynamics import position_cup
from src.core.simulation import coupled_dynamics
from src.core.obstacles import (
    check_ground_contact,
    check_arm_obstacle_collision,
    check_joint_limits,
)


def safety_filter(env, state, u_policy, n_samples=100):
    """Random-sampling safety filter (backward-compatible helper)."""
    if env.is_safe(state, u_policy):
        return np.asarray(u_policy, dtype=np.float32), False
    for _ in range(n_samples):
        candidate = env.action_space.sample()
        if env.is_safe(state, candidate):
            return np.asarray(candidate, dtype=np.float32), True
    return np.zeros(3, dtype=np.float32), True


def inverse_kinematics(x, y, z, L, elbow="up"):
    l1, l2, l3 = float(L[0]), float(L[1]), float(L[2])
    theta1 = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    z_rel = z - l1
    D = (rho**2 + z_rel**2 - l2**2 - l3**2) / (2.0 * l2 * l3)
    if D < -1.000001 or D > 1.000001:
        raise ValueError("Target is unreachable")
    D = np.clip(D, -1.0, 1.0)
    if elbow == "up":
        theta3 = np.arctan2(np.sqrt(1.0 - D**2), D)
    else:
        theta3 = np.arctan2(-np.sqrt(1.0 - D**2), D)
    theta2 = np.arctan2(z_rel, rho) - np.arctan2(
        l3 * np.sin(theta3), l2 + l3 * np.cos(theta3)
    )
    return np.array([theta1, theta2, theta3], dtype=np.float32)


class CoffeePouringEnv(gym.Env):
    """Coffee-arm environment.

    Observation: (13,) = [state(10), goal-relative cup vector(3)]
    Action:      (3,)  = joint acceleration commands, clipped to ±u_max
    """

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
        obstacles=None,
        joint_limits=None,
    ):
        super().__init__()
        self.L             = np.asarray(L if L is not None else DEFAULT_L, dtype=np.float32)
        self.K             = np.asarray(K if K is not None else DEFAULT_K, dtype=np.float32)
        self.dt            = float(dt)
        self.T             = float(T)
        self.max_steps     = int(T / dt)
        self.u_max         = float(u_max)
        self.slosh_rad_max = float(slosh_rad_max)
        self.l_eff         = float(l_eff)
        self.goal_pos      = np.asarray(
            goal_pos if goal_pos is not None else [0.55, 0.10, 0.10], dtype=np.float32
        )
        self.randomize_goal = bool(randomize_goal)
        self.obstacles      = list(obstacles if obstacles is not None else DEFAULT_OBSTACLES)
        self.joint_limits   = np.asarray(
            joint_limits if joint_limits is not None else DEFAULT_JOINT_LIMITS, dtype=np.float32
        )

        obs_high = np.array(
            [np.pi]*3 + [10.0]*3 + [self.l_eff, self.l_eff, 10.0, 10.0] + [6.0]*3,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(
            low=-self.u_max, high=self.u_max, shape=(3,), dtype=np.float32
        )

        self.state      = None
        self.step_count = 0
        self._prev_dist = None

    def _goal_vec(self):
        cup_pos = position_cup(self.state[:6], self.L)
        return (self.goal_pos - cup_pos).astype(np.float32)

    def _make_obs(self):
        return np.concatenate([self.state, self._goal_vec()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        while True:
            r  = self.np_random.uniform(0.5, 0.65)
            az = np.deg2rad(self.np_random.uniform(70, 90))
            el = np.deg2rad(self.np_random.uniform(0, 30))
            x  = r * np.cos(el) * np.cos(az)
            y  = r * np.cos(el) * np.sin(az)
            z  = r * np.sin(el)
            try:
                theta_init = inverse_kinematics(x, y, z, self.L, elbow="up")
                break
            except ValueError:
                pass

        self.state = np.concatenate([theta_init, np.zeros(7)]).astype(np.float32)
        self.step_count = 0
        if self.randomize_goal:
            self.goal_pos = self.np_random.uniform(
                low=[-1.5, -1.5, 0.5], high=[1.5, 1.5, 2.5]
            ).astype(np.float32)
        self._prev_dist = float(
            np.linalg.norm(position_cup(self.state[:6], self.L) - self.goal_pos)
        )
        return self._make_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -self.u_max, self.u_max)

        next_state = coupled_dynamics(
            self.state, action, self.K, self.L, self.dt, l_eff=self.l_eff
        ).astype(np.float32)

        x_slosh, y_slosh = float(next_state[6]), float(next_state[7])
        slosh_rad    = np.sqrt(x_slosh**2 + y_slosh**2)
        cup_pos      = position_cup(next_state[:6], self.L)
        dist_to_goal = float(np.linalg.norm(cup_pos - self.goal_pos))

        spill_slosh     = bool(slosh_rad > self.slosh_rad_max)
        below_ground    = bool(cup_pos[2] < 0.0)
        obstacle_hit    = bool(check_arm_obstacle_collision(next_state[:6], self.L, self.obstacles))
        joint_violation = bool(check_joint_limits(next_state[:3], self.joint_limits))
        at_goal         = bool(dist_to_goal < 0.1)

        terminated = spill_slosh or below_ground or obstacle_hit or joint_violation or at_goal
        self.step_count += 1
        truncated = bool(self.step_count >= self.max_steps)

        progress = self._prev_dist - dist_to_goal
        reward   = 10.0 * progress - 0.01 * float(np.linalg.norm(action) ** 2)
        if spill_slosh:
            reward -= 50.0
        if below_ground:
            reward -= 50.0
        if obstacle_hit or joint_violation:
            reward -= 50.0
        if at_goal:
            reward += 100.0

        self._prev_dist = dist_to_goal
        self.state      = next_state

        info = {
            "cup_pos":         cup_pos.astype(np.float32),
            "slosh_rad":       slosh_rad,
            "dist_to_goal":    dist_to_goal,
            "spill_slosh":     spill_slosh,
            "below_ground":    below_ground,
            "obstacle_hit":    obstacle_hit,
            "joint_violation": joint_violation,
            "step_count_ep":   self.step_count,
        }
        return self._make_obs(), reward, terminated, truncated, info

    def is_safe(self, state, action):
        state  = np.asarray(state,  dtype=np.float32).reshape(-1)
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

    def render(self):
        pass


# Alias — CoffeeArmEnv and CoffeePouringEnv are the same class.
CoffeeArmEnv = CoffeePouringEnv


if __name__ == "__main__":
    print("=== CoffeePouringEnv / CoffeeArmEnv sanity check ===")
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
            print(f"cup_pos={info['cup_pos'].round(3)}, slosh_rad={info['slosh_rad']:.4f}")
            print(f"obstacle_hit={info['obstacle_hit']}, joint_violation={info['joint_violation']}")
            print(f"Total reward: {total_reward:.2f}")
            break
