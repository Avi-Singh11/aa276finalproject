"""
CoffeePouringEnv: Gymnasium environment for safe RL coffee pouring.

State space (8D):
    phi    [0:6] = [theta1, theta2, theta3, dtheta1, dtheta2, dtheta3]
    slosh  [6:8] = [alpha, dalpha]   <-- stubbed

Action space (3D):
    u = [u1, u2, u3]  joint torque commands, bounded by u_max

Plug-in interface:
    Replace slosh_dynamics() with the real damped pendulum model when ready.
    Everything else (Gym env, PPO training, safety filter) stays unchanged.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def jacobian(φ, L):
    l1, l2, l3 = L[0], L[1], L[2]
    θ1, θ2, θ3 = φ[0, 0], φ[1, 0], φ[2, 0]
    s1, c1 = np.sin(θ1), np.cos(θ1)
    s2, c2 = np.sin(θ2), np.cos(θ2)
    s3, c3 = np.sin(θ3), np.cos(θ3)
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


def jacobian_dot(φ, L):
    l1, l2, l3 = L[0], L[1], L[2]
    θ1, θ2, θ3 = φ[0, 0], φ[1, 0], φ[2, 0]
    θ1d, θ2d, θ3d = φ[3, 0], φ[4, 0], φ[5, 0]
    s1, c1 = np.sin(θ1), np.cos(θ1)
    s2, c2 = np.sin(θ2), np.cos(θ2)
    s3, c3 = np.sin(θ3), np.cos(θ3)
    J = np.zeros((3, 3))
    J[0, 0] = (l3*c1*s2*s3 - l3*c1*c2*c3 - l2*c1*c2)*θ1d + (l3*s1*s2*c3 + l3*s1*c2*s3 + l2*s1*s2)*θ2d + (s1*c2*s3 + s1*s2*c3)*l3*θ3d  # s2*s3 -> s2*c3
    J[0, 1] = (l3*s1*s2*c3 + l3*s1*c2*s3 + l2*s1*s2)*θ1d + (l3*c1*s2*s3 - l3*c1*c2*c3 - l2*c1*c2)*θ2d + (c1*s2*s3 - c1*c2*c3)*l3*θ3d
    J[0, 2] = (s1*c2*s3 + s1*s2*c3)*l3*θ1d + (c1*s2*s3 - c1*c2*c3)*l3*θ2d + (c1*s2*s3 - c1*c2*c3)*l3*θ3d
    J[1, 0] = (l3*s1*s2*s3 - l3*c2*c3*s1 - l2*c2*s1)*θ1d - (l3*s2*c3*c1 + l3*c1*c2*s3 + l2*c1*s2)*θ2d - (c1*s2*c3 + c1*c2*s3)*l3*θ3d  # c2*c3+s2*c3 -> s2*c3+c2*s3
    J[1, 1] = -(l3*s2*c3*c1 + l3*c1*c2*s3 + l2*c1*s2)*θ1d + (l3*s1*s2*s3 - l3*c2*c3*s1 - l2*s1*c2)*θ2d + (s2*s3*s1 - s1*c2*c3)*l3*θ3d
    J[1, 2] = -(c1*c2*s3 + c1*s2*c3)*l3*θ1d + (s1*s2*s3 - s1*c2*c3)*l3*θ2d + (s1*s2*s3 - s1*c2*c3)*l3*θ3d  # wsa copy of J[0,2]
    J[2, 0] = 0.
    J[2, 1] = -(l2*s2 + l3*c2*s3 + l3*s2*c3)*θ2d - (s2*c3 + c2*s3)*l3*θ3d
    J[2, 2] = -(s2*c3 + c2*s3)*l3*θ2d - (c2*s3 + s2*c3)*l3*θ3d
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

def position_cup(φ, L):
    l1, l2, l3 = L[0], L[1], L[2]
    θ1, θ2, θ3 = φ[0, 0], φ[1, 0], φ[2, 0]
    s1, c1 = np.sin(θ1), np.cos(θ1)
    s2, c2 = np.sin(θ2), np.cos(θ2)
    s3, c3 = np.sin(θ3), np.cos(θ3)
    P = np.zeros((3, 1))
    P[0, 0] = l3*c1*c2*c3 - l3*c1*s2*s3 + c1*c2*l2
    P[1, 0] = l3*c2*c3*s1 - l3*s1*s2*s3 + c2*l2*s1
    P[2, 0] = l1 + l2*s2 + l3*c2*s3 + l3*c3*s2
    return P


def get_cup_acceleration(phi_flat, u_flat, K, L):
    phi = phi_flat.reshape(6, 1)
    u = u_flat.reshape(3, 1)
    J = jacobian(phi, L)
    J_dot = jacobian_dot(phi, L)
    theta_dot = phi[3:6]
    theta_ddot = K @ u
    a = J_dot @ theta_dot + J @ theta_ddot
    return a.flatten()


# ─── Sloshing dynamics stub ───────────────────────────────────────────────────
# TODO: Replace this function with the real damped pendulum model.
#
# Expected signature:
#   slosh_state_next = slosh_dynamics(slosh_state, arm_state, u, dt)
#
# Where:
#   slosh_state : (2,) array  [alpha, dalpha]
#                 alpha = liquid CoM angle from vertical (rad)
#                 dalpha = angular velocity of liquid CoM (rad/s)
#   arm_state : (6,) array current arm state (for coupling term)
#   u : (3,) array current control input
#   dt : float timestep
#

def slosh_dynamics(slosh_state, arm_state, u, dt):
    """STUB: returns unchanged sloshing state.
    """
    return slosh_state.copy()


# Coupled dynamics
def coupled_dynamics(state, u_flat, K, L, dt):
    arm_state   = state[:6]
    slosh_state = state[6:]

    next_arm   = arm_dynamics(arm_state, u_flat, K, dt)
    next_slosh = slosh_dynamics(slosh_state, arm_state, u_flat, dt)

    return np.concatenate([next_arm, next_slosh])


# Gymnasium Environment
class CoffeePouringEnv(gym.Env):
    """
    Observation: (8,)  [theta1..3, dtheta1..3, alpha, dalpha]
    Action:      (3,)  joint torques in [-u_max, u_max]
    """

    metadata = {"render_modes": []}

    def __init__(self, L=None, K=None, dt=0.01, T=10.0, u_max=2.0, a_max=5.0, alpha_max=0.3, goal_pos=None):
        super().__init__()
        self.L = L if L is not None else [1.0, 1.0, 1.0]
        self.K = K if K is not None else np.diag([1.0, 2.0, 3.0])
        self.dt = dt
        self.T  = T
        self.max_steps = int(T / dt)
        self.u_max     = u_max
        self.a_max     = a_max
        self.alpha_max = alpha_max
        self.goal_pos = goal_pos if goal_pos is not None else np.array([0.5, 0.5, 1.5])

        obs_high = np.array([np.pi]*3 + [10.0]*3 + [np.pi, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-u_max, high=u_max, shape=(3,), dtype=np.float32)
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        theta_init = self.np_random.uniform(-0.3, 0.3, size=3)
        dtheta_init = np.zeros(3)
        slosh_init = np.zeros(2)
        self.state = np.concatenate([theta_init, dtheta_init, slosh_init]).astype(np.float32)
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        action = np.clip(action, -self.u_max, self.u_max)

        # Evaluate acceleration at current state before propagating
        a_vec = get_cup_acceleration(self.state[:6], action, self.K, self.L)  # was next_state[:6]
        a_norm = np.linalg.norm(a_vec)

        next_state = coupled_dynamics(self.state, action, self.K, self.L, self.dt).astype(np.float32)
        alpha = next_state[6]
        phi_col = next_state[:6].reshape(6, 1)
        cup_pos = position_cup(phi_col, self.L).flatten()
        dist_to_goal = np.linalg.norm(cup_pos - self.goal_pos)

        spill_accel = a_norm > self.a_max
        spill_slosh = abs(alpha) > self.alpha_max
        below_ground = cup_pos[2] < 0.0
        terminated = bool(spill_accel or spill_slosh or below_ground)

        self.step_count += 1
        truncated = (self.step_count >= self.max_steps)

        reward = -dist_to_goal * 0.1
        reward -= 0.01 * float(np.linalg.norm(action) ** 2)
        if dist_to_goal < 0.1:
            reward += 10.0
        if terminated:
            reward -= 50.0

        self.state = next_state

        info = {
            "cup_pos":      cup_pos,
            "a_norm":       a_norm,
            "alpha":        alpha,
            "dist_to_goal": dist_to_goal,
            "spill_accel":  spill_accel,
            "spill_slosh":  spill_slosh,
        }

        return self.state.copy(), reward, terminated, truncated, info

    def is_safe(self, state, action):
        a_vec = get_cup_acceleration(state[:6], action, self.K, self.L)  # fixed: was next_state[:6]
        next_state = coupled_dynamics(state, action, self.K, self.L, self.dt)
        alpha = next_state[6]
        return (np.linalg.norm(a_vec) <= self.a_max) and (abs(alpha) <= self.alpha_max)

    def render(self):
        pass


# Safety filter (placeholder for BRT)
def safety_filter(env, state, u_policy, n_samples=100):
    """Least-restrictive safety filter using random sampling fallback.
    Replace with QP + BRT value function when hj_reachability is ready.
    """
    if env.is_safe(state, u_policy):
        return u_policy, False

    for _ in range(n_samples):
        u_candidate = env.action_space.sample()
        if env.is_safe(state, u_candidate):
            return u_candidate, True

    return np.zeros(3, dtype=np.float32), True


# Verification 
def _check_jacobian(phi_flat, L, eps=1e-6):
    """Numerical vs analytical Jacobian check"""
    phi_col = phi_flat.reshape(6, 1)
    J_analytical = jacobian(phi_col, L)
    J_numerical = np.zeros((3, 3))
    for i in range(3):
        phi_plus = phi_flat.copy(); phi_plus[i]  += eps
        phi_minus = phi_flat.copy(); phi_minus[i] -= eps
        p_plus = position_cup(phi_plus.reshape(6, 1),  L).flatten()
        p_minus = position_cup(phi_minus.reshape(6, 1), L).flatten()
        J_numerical[:, i] = (p_plus - p_minus) / (2 * eps)
    err = np.abs(J_analytical - J_numerical).max()
    return err, J_analytical, J_numerical


def _check_jacobian_dot(phi_flat, L, eps=1e-6):
    """Numerical vs analytical J_dot check via finite difference of J."""
    phi_col = phi_flat.reshape(6, 1)
    J_dot_analytical = jacobian_dot(phi_col, L)
    # dJ/dt = sum_i (dJ/dtheta_i) * theta_i_dot
    J_dot_numerical = np.zeros((3, 3))
    for i in range(3):
        phi_plus  = phi_flat.copy(); phi_plus[i]  += eps
        phi_minus = phi_flat.copy(); phi_minus[i] -= eps
        dJ_dtheta_i = (jacobian(phi_plus.reshape(6,1), L) - jacobian(phi_minus.reshape(6,1), L)) / (2 * eps)
        J_dot_numerical += dJ_dtheta_i * phi_flat[3 + i]
    err = np.abs(J_dot_analytical - J_dot_numerical).max()
    return err, J_dot_analytical, J_dot_numerical


if __name__ == "__main__":
    print("Jacobian checks")
    rng = np.random.default_rng(42)
    all_ok = True
    for trial in range(5):
        phi_test = rng.uniform(-0.5, 0.5, size=6)
        L_test = [1.0, 1.0, 1.0]
        err_J, J_an, J_nu = _check_jacobian(phi_test, L_test)
        err_Jd, Jd_an, Jd_nu = _check_jacobian_dot(phi_test, L_test)
        ok_J  = err_J  < 1e-6
        ok_Jd = err_Jd < 1e-6
        all_ok = all_ok and ok_J and ok_Jd
        print(f"  Trial {trial+1}: J_err={err_J:.2e} {'OK' if ok_J else 'FAIL'}  |  J_dot_err={err_Jd:.2e} {'OK' if ok_Jd else 'FAIL'}")

    if all_ok:
        print("\nAll Jacobian checks PASSED.")
    else:
        print("\nSome checks FAILED — inspect J_an vs J_nu above.")

    print("\n=== Episode sanity check ===")
    env = CoffeePouringEnv(a_max=20.0)
    obs, _ = env.reset(seed=0)
    print(f"Initial state: {obs}")

    total_reward = 0
    for i in range(env.max_steps + 5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended at step {i+1} (max_steps={env.max_steps}). "
                  f"terminated={terminated}, truncated={truncated}")
            print(f"  cup_pos={info['cup_pos'].round(3)}, a_norm={info['a_norm']:.3f}, alpha={info['alpha']:.3f}")
            print(f"  Total reward: {total_reward:.2f}")
            break
    else:
        print(f"WARNING: episode did not terminate after {env.max_steps + 5} steps")
