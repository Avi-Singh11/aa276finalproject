"""
BRT computation for the coffee pouring system via hj_reachability.

STATE_DIM flag controls dimensionality:
  6  →  [theta2, theta3, dtheta2, dtheta3, alpha, dalpha]
         theta1 dropped (exact rotational symmetry)
         dtheta1 treated as bounded adversarial disturbance
  8  →  [theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, alpha, dalpha]
         full state, no reduction

Swapping STATE_DIM changes the grid, dynamics, and query function.
The QP safety filter (safety_filter.py) calls query_value() which handles
the mapping from full 8D state automatically.

Plug-in points (marked TODO):
  - slosh_coupling_open_loop(): replace stub zeros with Shai's damped pendulum
  - slosh_coupling_control_jacobian(): how arm torques excite liquid
  - slosh_coupling_disturbance_jacobian(): how dtheta1 excites liquid (6D only)
"""

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from functools import partial

STATE_DIM = 6   # change to 8 to use full state (coarser grid, no reductions)

# ─── Physical parameters (must match coffee_pouring_env.py)

L  = [1.0, 1.0, 1.0]
K  = np.diag([1.0, 2.0, 3.0])
G  = 9.81
L_EFF = 0.1 # effective pendulum length 
B_DAMP = 0.1 # sloshing damping coefficient
U_MAX = 2.0 # joint torque bound
DTHETA_MAX = 2.0 # joint velocity bound (used for disturbance bound)
ALPHA_MAX = 0.3 # sloshing angle spill threshold (rad)

# Grid bounds

BOUNDS_6D = {
    "lo": np.array([-np.pi/2, -np.pi/2, -DTHETA_MAX, -DTHETA_MAX, -ALPHA_MAX*1.5, -3.0]),
    "hi": np.array([ np.pi/2,  np.pi/2,  DTHETA_MAX,  DTHETA_MAX,  ALPHA_MAX*1.5,  3.0]),
}

BOUNDS_8D = {
    "lo": np.array([-np.pi, -np.pi/2, -np.pi/2, -DTHETA_MAX, -DTHETA_MAX, -DTHETA_MAX, -ALPHA_MAX*1.5, -3.0]),
    "hi": np.array([ np.pi,  np.pi/2,  np.pi/2,  DTHETA_MAX,  DTHETA_MAX,  DTHETA_MAX,  ALPHA_MAX*1.5,  3.0]),
}

# Points per dimension — tune to fit GPU memory.
# 6D: 22^6 ≈ 113M cells (~0.4 GB float32)   good resolution
# 8D: 10^8 ≈ 100M cells (~0.4 GB float32)
RESOLUTION_6D = 30   # 30^6 = 729M cells (~2.9 GB float32)
RESOLUTION_8D = 12   # 12^8 = 429M cells (~1.7 GB float32)


# State index mpas
# Full 8D state convention: [theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, alpha, dalpha]
IDX_8D = dict(theta1=0, theta2=1, theta3=2, dtheta1=3, dtheta2=4, dtheta3=5, alpha=6, dalpha=7)

IDX_6D = dict(theta2=0, theta3=1, dtheta2=2, dtheta3=3, alpha=4, dalpha=5)

STATE_INDICES = IDX_6D if STATE_DIM == 6 else IDX_8D


# Sloshing coupling stubs
# Signature is fixed — do not change.

def slosh_coupling_open_loop(state):
    """Open-loop (drift) contribution to dalpha_dot from current state.
    Returns scalar. With real model: -(G/L_EFF)*jnp.sin(alpha) - B_DAMP*dalpha
    (the gravity + damping terms — the part that doesn't depend on control or disturbance).
    """
    # TODO: replace with real model
    if STATE_DIM == 6:
        alpha  = state[IDX_6D["alpha"]]
        dalpha = state[IDX_6D["dalpha"]]
    else:
        alpha  = state[IDX_8D["alpha"]]
        dalpha = state[IDX_8D["dalpha"]]
    return -(G / L_EFF) * jnp.sin(alpha) - B_DAMP * dalpha


def slosh_coupling_control_jacobian(state):
    """How control u=[u1,u2,u3] drives dalpha_dot via arm acceleration.
    Returns (3,) array — one coefficient per control input.
    With real model: coupling = (1/L_EFF) * horizontal_acceleration_from_u
    """
    # TODO: replace with real model
    return jnp.zeros(3)


def slosh_coupling_disturbance_jacobian(state):
    """(6D only) How dtheta1 disturbance drives dalpha_dot via Coriolis acceleration.
    Returns (1,) array.
    """
    # TODO: replace with real model
    return jnp.zeros(1)


# Dynamics: 6D

class ArmSlosh6D(hj.ControlAndDisturbanceAffineDynamics):
    """
    State: [theta2, theta3, dtheta2, dtheta3, alpha, dalpha]
    Control: [u1, u2, u3] (u1 enters through sloshing coupling only)
    Disturbance:[dtheta1] (adversarial worst-case for dtheta1 Coriolis effect)

    Controller maximizes V (tries to stay safe).
    Disturbance minimizes V (adversarial dtheta1).
    """

    def __init__(self):
        control_space     = hj.sets.Box(lo=jnp.full(3, -U_MAX),    hi=jnp.full(3, U_MAX))
        disturbance_space = hj.sets.Box(lo=jnp.array([-DTHETA_MAX]), hi=jnp.array([DTHETA_MAX]))
        super().__init__(
            control_mode="max", # controller maximizes V to stay safe
            disturbance_mode="min", # adversarial disturbance minimizes V
            control_space=control_space,
            disturbance_space=disturbance_space,
        )

    def open_loop_dynamics(self, state, time):
        i = IDX_6D
        theta2_dot  = state[i["dtheta2"]]
        theta3_dot  = state[i["dtheta3"]]
        dtheta2_dot = 0.0
        dtheta3_dot = 0.0
        alpha_dot   = state[i["dalpha"]]
        dalpha_dot  = slosh_coupling_open_loop(state)
        return jnp.array([theta2_dot, theta3_dot, dtheta2_dot, dtheta3_dot, alpha_dot, dalpha_dot])

    def control_jacobian(self, state, time):
        # u = [u1, u2, u3]
        # dtheta2_dot += K[1,1]*u2,  dtheta3_dot += K[2,2]*u3
        # dalpha_dot  += slosh_coupling_control_jacobian(state)
        slosh_ctrl = slosh_coupling_control_jacobian(state)   # (3,)
        G_u = jnp.array([
            [0., 0., 0.], # theta2_dot
            [0., 0., 0.], # theta3_dot
            [0., K[1, 1], 0.], # dtheta2_dot
            [0., 0., K[2, 2]], # dtheta3_dot
            [0., 0., 0.], # alpha_dot
            [slosh_ctrl[0], slosh_ctrl[1], slosh_ctrl[2]], # dalpha_dot
        ])
        return G_u

    def disturbance_jacobian(self, state, time):
        slosh_dist = slosh_coupling_disturbance_jacobian(state)   # (1,)
        G_d = jnp.array([
            [0.], # theta2_dot
            [0.], # theta3_dot
            [0.], # dtheta2_dot
            [0.], # dtheta3_dot
            [0.], # alpha_dot
            [slosh_dist[0]], # dalpha_dot
        ])
        return G_d


# Dynamics: 8D
class ArmSlosh8D(hj.ControlAndDisturbanceAffineDynamics):
    """
    State: [theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, alpha, dalpha]
    Control: [u1, u2, u3]
    Disturbance: none (use a zero-dimensional box)
    """

    def __init__(self):
        control_space = hj.sets.Box(lo=jnp.full(3, -U_MAX), hi=jnp.full(3, U_MAX))
        disturbance_space = hj.sets.Box(lo=jnp.zeros(0), hi=jnp.zeros(0))
        super().__init__(
            control_mode="max",
            disturbance_mode="min",
            control_space=control_space,
            disturbance_space=disturbance_space,
        )

    def open_loop_dynamics(self, state, time):
        i = IDX_8D
        theta1_dot = state[i["dtheta1"]]
        theta2_dot = state[i["dtheta2"]]
        theta3_dot = state[i["dtheta3"]]
        dtheta1_dot = 0.0
        dtheta2_dot = 0.0
        dtheta3_dot = 0.0
        alpha_dot = state[i["dalpha"]]
        dalpha_dot = slosh_coupling_open_loop(state)
        return jnp.array([theta1_dot, theta2_dot, theta3_dot, dtheta1_dot, dtheta2_dot, dtheta3_dot, alpha_dot, dalpha_dot])

    def control_jacobian(self, state, time):
        slosh_ctrl = slosh_coupling_control_jacobian(state) # (3,)
        G_u = jnp.array([
            [0., 0., 0.], # theta1_dot
            [0., 0., 0.], # theta2_dot
            [0., 0., 0. ], # theta3_dot
            [K[0, 0], 0., 0.], # dtheta1_dot
            [0., K[1, 1], 0.], # dtheta2_dot
            [0., 0., K[2, 2]], # dtheta3_dot
            [0., 0., 0.], # alpha_dot
            [slosh_ctrl[0], slosh_ctrl[1], slosh_ctrl[2]],   # dalpha_dot
        ])
        return G_u

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((8, 0))


# Failure set (l(x) < 0 means unsafe)

def failure_values(grid):
    """
    l(x) = min(alpha_max - |alpha|, ...)
    Negative where unsafe (inside failure set), positive where safe.

    Sloshing: |alpha| > alpha_max
    Add more constraints here when ready (z_cup, joint velocity limits).
    """
    if STATE_DIM == 6:
        alpha = grid.states[..., IDX_6D["alpha"]]
    else:
        alpha = grid.states[..., IDX_8D["alpha"]]

    l_slosh = ALPHA_MAX - jnp.abs(alpha) # negative when |alpha| > alpha_max
    return l_slosh


# Build grid and dynamics
def build_grid():
    if STATE_DIM == 6:
        bounds = BOUNDS_6D
        shape  = tuple([RESOLUTION_6D] * 6)
    else:
        bounds = BOUNDS_8D
        shape  = tuple([RESOLUTION_8D] * 8)

    domain = hj.sets.Box(lo=bounds["lo"], hi=bounds["hi"])
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(domain, shape)


def build_dynamics():
    return ArmSlosh6D() if STATE_DIM == 6 else ArmSlosh8D()

# Value function query (used by safety filter at runtime)
def query_value(values, grid, full_state_8d):
    """
    Query V at a full 8D state, handling the dimension reduction automatically.

    full_state_8d: (8,) array [theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, alpha, dalpha]
    values: value function on the BRT grid (output of hj.solve)
    grid: the hj_reachability Grid object

    Returns: scalar V(x). V < 0 means state is inside the BRT (unsafe).
    """
    if STATE_DIM == 6:
        brt_state = jnp.array([
            full_state_8d[IDX_8D["theta2"]],
            full_state_8d[IDX_8D["theta3"]],
            full_state_8d[IDX_8D["dtheta2"]],
            full_state_8d[IDX_8D["dtheta3"]],
            full_state_8d[IDX_8D["alpha"]],
            full_state_8d[IDX_8D["dalpha"]],
        ])
    else:
        brt_state = jnp.array(full_state_8d)

    return grid.interpolate(values, brt_state)


# Main: run BRT computation
if __name__ == "__main__":
    import time as time_module

    print(f"Computing {STATE_DIM}D BRT")
    print(f"  Resolution: {RESOLUTION_6D if STATE_DIM == 6 else RESOLUTION_8D} pts/dim")
    if STATE_DIM == 6:
        total_cells = RESOLUTION_6D ** 6
    else:
        total_cells = RESOLUTION_8D ** 8
    print(f"  Grid cells: {total_cells:,}  (~{total_cells * 4 / 1e9:.2f} GB float32)")
    print()

    grid = build_grid()
    dynamics = build_dynamics()
    T_horizon = 2.0
    times = np.linspace(0, -T_horizon, 50)   # negative = backward in time

    init_values = failure_values(grid)

    solver_settings = hj.SolverSettings.with_accuracy(
        "medium",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
    )

    print("Starting BRT solve...")
    t0 = time_module.time()
    all_values = hj.solve(solver_settings, dynamics, grid, times, init_values, progress_bar=True)
    elapsed = time_module.time() - t0
    print(f"Solve complete in {elapsed:.1f}s")

    # Final value function = last time slice (most backward)
    brt_values = all_values[-1]

    unsafe_frac = float(jnp.mean(brt_values < 0))
    print(f"BRT unsafe fraction: {unsafe_frac:.3f} ({unsafe_frac*100:.1f}% of grid)")

    # Save
    np.save(f"brt_values_{STATE_DIM}d.npy", np.array(brt_values))
    print(f"Saved brt_values_{STATE_DIM}d.npy")

    # Quick query test
    test_state = np.zeros(8)
    test_state[IDX_8D["alpha"]] = ALPHA_MAX * 1.1 # slightly over threshold → should be unsafe
    v_unsafe = query_value(brt_values, grid, test_state)

    test_state[IDX_8D["alpha"]] = 0.0 # at origin → should be safe
    v_safe = query_value(brt_values, grid, test_state)

    print(f"\nQuery check:")
    print(f"  V(alpha=1.1*alpha_max) = {float(v_unsafe):.4f}  (expect < 0)")
    print(f"  V(alpha=0) = {float(v_safe):.4f}   (expect > 0)")
