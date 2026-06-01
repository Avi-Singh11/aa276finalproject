# 3-DOF arm env for navigating with a cup of liquid
# State (8D): [theta1..3, dtheta1..3, alpha, dalpha]
# Obs (11D): state + goal-relative cup position
# arm kinematics from groupmate's verified model, sloshing stubbed for now

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Physical constants (match coffee_pouring_env.py)
G      = 9.81   # gravity (m/s^2)
L_EFF  = 0.1    # effective pendulum length for sloshing (m) — stub parameter
B_DAMP = 0.1    # sloshing damping coefficient — stub parameter


def jacobian(phi, L):
    """3x3 end-effector Jacobian. phi: (6,1) column."""
    l1, l2, l3 = L[0], L[1], L[2]
    theta1, theta2, theta3 = phi[0, 0], phi[1, 0], phi[2, 0]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    s3, c3 = np.sin(theta3), np.cos(theta3)
    J = np.zeros((3, 3))
    J[0, 0] = -l3*s1*c2*c3 + l3*s1*s2*s3 - l2*s1*c2
    J[0, 1] = -l3*c1*s2*c3 - l3*c1*c2*s3 - l2*c1*s2
    J[0, 2] = -l3*c1*c2*s3 - l3*c1*s2*c3
    J[1, 0] = l3*c2*c3*c1 - l3*c1*s2*s3 + c2*l2*c1
    J[1, 1] = -l3*s2*c3*s1 - l3*s1*c2*s3 - s2*l2*s1
    J[1, 2] = -l3*c2*s3*s1 - l3*s1*s2*c3
    J[2, 0] = 0.
    J[2, 1] = l2*c2 - l3*s2*s3 + l3*c3*c2   # was +l3*s2*s3
    J[2, 2] = l3*c2*c3 - l3*s3*s2
    return J


def jacobian_dot(phi, L):
    """Time derivative of end-effector Jacobian. phi: (6,1) column."""
    l1, l2, l3 = L[0], L[1], L[2]
    theta1, theta2, theta3 = phi[0, 0], phi[1, 0], phi[2, 0]
    theta1d, theta2d, theta3d = phi[3, 0], phi[4, 0], phi[5, 0]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    s3, c3 = np.sin(theta3), np.cos(theta3)
    J = np.zeros((3, 3))
    J[0, 0] = (l3*c1*s2*s3 - l3*c1*c2*c3 - l2*c1*c2)*theta1d + (l3*s1*s2*c3 + l3*s1*c2*s3 + l2*s1*s2)*theta2d + (s1*c2*s3 + s1*s2*c3)*l3*theta3d  # s2*s3 -> s2*c3
    J[0, 1] = (l3*s1*s2*c3 + l3*s1*c2*s3 + l2*s1*s2)*theta1d + (l3*c1*s2*s3 - l3*c1*c2*c3 - l2*c1*c2)*theta2d + (c1*s2*s3 - c1*c2*c3)*l3*theta3d
    J[0, 2] = (s1*c2*s3 + s1*s2*c3)*l3*theta1d + (c1*s2*s3 - c1*c2*c3)*l3*theta2d + (c1*s2*s3 - c1*c2*c3)*l3*theta3d
    J[1, 0] = (l3*s1*s2*s3 - l3*c2*c3*s1 - l2*c2*s1)*theta1d - (l3*s2*c3*c1 + l3*c1*c2*s3 + l2*c1*s2)*theta2d - (c1*s2*c3 + c1*c2*s3)*l3*theta3d  # c2*c3+s2*c3 -> s2*c3+c2*s3
    J[1, 1] = -(l3*s2*c3*c1 + l3*c1*c2*s3 + l2*c1*s2)*theta1d + (l3*s1*s2*s3 - l3*c2*c3*s1 - l2*s1*c2)*theta2d + (s2*s3*s1 - s1*c2*c3)*l3*theta3d
    J[1, 2] = -(c1*c2*s3 + c1*s2*c3)*l3*theta1d + (s1*s2*s3 - s1*c2*c3)*l3*theta2d + (s1*s2*s3 - s1*c2*c3)*l3*theta3d  # was copy of J[0,2]
    J[2, 0] = 0.
    J[2, 1] = -(l2*s2 + l3*c2*s3 + l3*s2*c3)*theta2d - (s2*c3 + c2*s3)*l3*theta3d
    J[2, 2] = -(s2*c3 + c2*s3)*l3*theta2d - (c2*s3 + s2*c3)*l3*theta3d
    return J


def A_matrix(damping=0.5):
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 3:6] = -damping * np.eye(3)
    return A


def B_matrix(K):
    B = np.zeros((6, 3))
    B[3:6, :] = K
    return B


def arm_dynamics(phi_flat, u_flat, K, dt):
    phi = phi_flat.reshape(6, 1)
    u   = u_flat.reshape(3, 1)
    A = A_matrix()
    B = B_matrix(K)
    phi_dot = A @ phi + B @ u
    return (phi_flat + phi_dot.flatten() * dt)


def position_cup(phi, L):
    """End-effector (cup) position. phi: (6,1) or (6,) array. Returns (3,)."""
    if phi.ndim == 1:
        phi = phi.reshape(6, 1)
    l1, l2, l3 = L[0], L[1], L[2]
    theta1, theta2, theta3 = phi[0, 0], phi[1, 0], phi[2, 0]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    s3, c3 = np.sin(theta3), np.cos(theta3)
    return np.array([
        l3*c1*c2*c3 - l3*c1*s2*s3 + c1*c2*l2,
        l3*c2*c3*s1 - l3*s1*s2*s3 + c2*l2*s1,
        l1 + l2*s2 + l3*c2*s3 + l3*c3*s2,
    ])


def get_link_positions(phi_flat, L):
    """Positions of all link endpoints for collision checking.

    Returns [p0, p1, p2, p3]:
      p0 = base (fixed at origin)
      p1 = joint 2 (top of vertical link 1)
      p2 = joint 3 (end of link 2)
      p3 = end-effector / cup (end of link 3)
    """
    if phi_flat.ndim > 1:
        phi_flat = phi_flat.flatten()
    l1, l2, l3 = L[0], L[1], L[2]
    theta1, theta2 = phi_flat[0], phi_flat[1]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, l1])
    p2 = np.array([l2*c1*c2, l2*c2*s1, l1 + l2*s2])
    p3 = position_cup(phi_flat, L)
    return [p0, p1, p2, p3]


def get_joint_jacobians(phi_flat, L):
    """Jacobians (dp_i/dtheta) for each link endpoint.

    Returns [J0, J1, J2, J3] each (3,3).
    J0, J1 are zero (fixed or independent-of-control).
    J2: Jacobian of joint-3 position w.r.t. joint angles.
    J3: full end-effector Jacobian.
    """
    if phi_flat.ndim > 1:
        phi_flat = phi_flat.flatten()
    l2 = L[1]
    theta1, theta2 = phi_flat[0], phi_flat[1]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)

    # p2 = [l2*c1*c2, l2*c2*s1, l1 + l2*s2]
    J2 = np.zeros((3, 3))
    J2[0, 0] = -l2*s1*c2;  J2[0, 1] = -l2*c1*s2
    J2[1, 0] =  l2*c1*c2;  J2[1, 1] = -l2*s1*s2
    J2[2, 0] = 0.;          J2[2, 1] =  l2*c2

    J3 = jacobian(phi_flat.reshape(6, 1), L)
    return [np.zeros((3, 3)), np.zeros((3, 3)), J2, J3]



def get_cup_acceleration(phi_flat, u_flat, K, L):
    """Cup (end-effector) acceleration given current state and control.

    Matches groupmate's verified implementation in coffee_pouring_env.py:
      a_cup = J_dot @ theta_dot + J @ (K @ u)
    """
    phi = phi_flat.reshape(6, 1)
    u = u_flat.reshape(3, 1)
    J = jacobian(phi, L)
    J_dot = jacobian_dot(phi, L)
    theta_dot = phi[3:6]
    theta_ddot = K @ u
    return (J_dot @ theta_dot + J @ theta_ddot).flatten()


# TODO: replace with real damped pendulum model
# slosh_state = [alpha, dalpha], arm_state = (6,), u = (3,)
def slosh_dynamics(slosh_state, arm_state, u, dt):
    return slosh_state.copy()


def coupled_dynamics(state, u_flat, K, L, dt):
    next_arm   = arm_dynamics(state[:6], u_flat, K, dt)
    next_slosh = slosh_dynamics(state[6:], state[:6], u_flat, dt)
    return np.concatenate([next_arm, next_slosh])


def _dist_point_to_segment(p, a, b):
    ab = b - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-12:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / ab_sq, 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


def check_arm_obstacle_collision(phi_flat, L, obstacles):
    """True if any arm link segment penetrates any obstacle sphere."""
    pts = get_link_positions(phi_flat, L)
    segments = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3])]
    for seg_a, seg_b in segments:
        for obs in obstacles:
            c = np.asarray(obs["center"])
            if _dist_point_to_segment(c, seg_a, seg_b) < obs["radius"]:
                return True
    return False


class CoffeeArmEnv(gym.Env):
    """Obs (11D): [theta1..3, dtheta1..3, alpha, dalpha, goal-cup vec], Action (3D): torques"""

    metadata = {"render_modes": []}

    DEFAULT_OBSTACLES = [
        {"center": [0.8, 0.8, 1.5],  "radius": 0.25},
        {"center": [-0.8, 0.6, 2.0], "radius": 0.25},
    ]

    def __init__(
        self,
        L=None, K=None, dt=0.01, T=10.0,
        u_max=2.0, a_max=5.0, alpha_max=0.3,
        goal_pos=None, obstacles=None,
        randomize_goal=False,
    ):
        super().__init__()
        self.L  = L if L is not None else [1.0, 1.0, 1.0]
        self.K  = K if K is not None else np.diag([1.0, 2.0, 3.0])
        self.dt = dt
        self.T  = T
        self.max_steps  = int(T / dt)
        self.u_max      = u_max
        self.a_max      = a_max
        self.alpha_max  = alpha_max
        self.goal_pos   = np.asarray(goal_pos if goal_pos is not None else [1.5, 0.0, 2.0], dtype=np.float64)
        self.obstacles  = obstacles if obstacles is not None else self.DEFAULT_OBSTACLES
        self.randomize_goal = randomize_goal

        # Observation: 8D state + 3D relative goal vector
        obs_high = np.array([np.pi]*3 + [10.0]*3 + [np.pi, 10.0] + [4.0]*3, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-u_max, high=u_max, shape=(3,), dtype=np.float32)
        self.state      = None
        self.step_count = 0

    def _goal_vec(self):
        cup_pos = position_cup(self.state[:6], self.L)
        return (self.goal_pos - cup_pos).astype(np.float32)

    def _make_obs(self):
        return np.concatenate([self.state, self._goal_vec()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        theta_init  = self.np_random.uniform(-0.3, 0.3, size=3)
        self.state  = np.concatenate([theta_init, np.zeros(5)]).astype(np.float32)
        self.step_count = 0
        if self.randomize_goal:
            self.goal_pos = self.np_random.uniform(
                low=[-1.5, -1.5, 0.5], high=[1.5, 1.5, 2.5]
            )
        return self._make_obs(), {}

    def step(self, action):
        action = np.clip(action, -self.u_max, self.u_max)

        a_cup  = get_cup_acceleration(self.state[:6], action, self.K, self.L)
        a_norm = np.linalg.norm(a_cup)

        next_state    = coupled_dynamics(self.state, action, self.K, self.L, self.dt).astype(np.float32)
        alpha         = next_state[6]
        cup_pos       = position_cup(next_state[:6], self.L)
        dist_to_goal  = np.linalg.norm(cup_pos - self.goal_pos)

        spill_slosh   = bool(abs(alpha) > self.alpha_max)
        below_ground  = bool(cup_pos[2] < 0.0)
        obstacle_hit  = check_arm_obstacle_collision(next_state[:6], self.L, self.obstacles)
        terminated    = spill_slosh or below_ground or obstacle_hit

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        reward  = -0.1 * dist_to_goal
        reward -= 0.01 * float(np.linalg.norm(action) ** 2)
        if dist_to_goal < 0.1:
            reward += 10.0
        if spill_slosh:
            reward -= 50.0
        if obstacle_hit:
            reward -= 50.0

        self.state = next_state

        info = {
            "cup_pos":      cup_pos,
            "a_norm":       a_norm,
            "alpha":        float(alpha),
            "dist_to_goal": dist_to_goal,
            "spill_slosh":  spill_slosh,
            "obstacle_hit": obstacle_hit,
        }
        return self._make_obs(), reward, terminated, truncated, info

    def is_safe(self, state, action):
        a_cup = get_cup_acceleration(state[:6], action, self.K, self.L)
        next_state = coupled_dynamics(state, action, self.K, self.L, self.dt)
        return (
            np.linalg.norm(a_cup) <= self.a_max
            and abs(next_state[6]) <= self.alpha_max
            and not check_arm_obstacle_collision(next_state[:6], self.L, self.obstacles)
        )

    def render(self):
        pass


if __name__ == "__main__":
    print("=== CoffeeArmEnv sanity check ===")
    env = CoffeeArmEnv(a_max=20.0)
    obs, _ = env.reset(seed=0)
    print(f"Obs shape: {obs.shape}  (expected 11)")
    print(f"Initial state: {obs[:8].round(3)}")
    print(f"Initial goal vec: {obs[8:].round(3)}")

    total_reward = 0
    for i in range(env.max_steps + 5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"\nEpisode ended at step {i+1} | terminated={terminated} truncated={truncated}")
            print(f"  cup_pos={info['cup_pos'].round(3)}, alpha={info['alpha']:.3f}")
            print(f"  spill_slosh={info['spill_slosh']}, obstacle_hit={info['obstacle_hit']}")
            print(f"  Total reward: {total_reward:.2f}")
            break
