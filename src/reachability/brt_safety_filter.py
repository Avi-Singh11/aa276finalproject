"""Fixed-horizon BRT safety filter with a value margin."""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from scipy.optimize import minimize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEEPREACH_PATH = os.path.dirname(PROJECT_ROOT)
if DEEPREACH_PATH not in sys.path:
    sys.path.insert(0, DEEPREACH_PATH)

from src.config.constants import DEFAULT_U_MAX
from src.core.simulation import coupled_dynamics
from dynamics.dynamics import CoffeeArmDynamics

GAMMA = 1.0


def _brt_value_and_gradient(model, dynamics, state_10d_np, t):
    """Query value and spatial gradient from the DeepReach model."""
    coord = torch.cat(
        [
            torch.tensor([t], dtype=torch.float32),
            torch.tensor(state_10d_np, dtype=torch.float32),
        ]
    ).unsqueeze(0)

    inp = dynamics.coord_to_input(coord)
    result = model({"coords": inp})
    dv = dynamics.io_to_dv(result["model_in"], result["model_out"].squeeze(-1))
    V = dynamics.io_to_value(
        result["model_in"].detach(), result["model_out"].squeeze(-1).detach()
    )

    dvdt = float(dv[0, 0].item())
    dvds = dv[0, 1:].detach().cpu().numpy().reshape(-1) / dynamics.state_scale
    V_val = float(V.item())
    return V_val, dvdt, dvds


def _build_cbf_numerical(model, dynamics, state_10d, u_nom, t, gamma=GAMMA, margin=0.0):
    """Build the fixed-horizon linear CBF constraint."""
    state_10d = np.asarray(state_10d, dtype=np.float32).flatten()
    V, _dvdt, dvds = _brt_value_and_gradient(model, dynamics, state_10d, t)

    dvds = dvds.flatten()
    dt = 1e-4
    u_dim = 3
    u_zero = np.zeros(u_dim, dtype=np.float32)
    x_next_drift = coupled_dynamics(
        state_10d, u_zero, dynamics.K, dynamics.L, dt, l_eff=dynamics.l_eff
    )
    x_next_drift = np.asarray(x_next_drift, dtype=np.float32).flatten()
    f_x = (x_next_drift - state_10d) / dt
    g_x = np.zeros((len(state_10d), u_dim), dtype=np.float32)
    eps = 1.0

    for i in range(u_dim):
        u_perturb = np.zeros(u_dim, dtype=np.float32)
        u_perturb[i] = eps
        x_next_perturbed = coupled_dynamics(
            state_10d, u_perturb, dynamics.K, dynamics.L, dt, l_eff=dynamics.l_eff
        )
        x_next_perturbed = np.asarray(x_next_perturbed, dtype=np.float32).flatten()
        g_x[:, i] = ((x_next_perturbed - state_10d) / dt - f_x) / eps

    a_cbf = dvds @ g_x
    drift_term = np.dot(dvds, f_x).item()
    b_cbf = -gamma * (V - margin) - drift_term
    return a_cbf.astype(np.float32), float(b_cbf)


def _solve_qp(u_nom, a_cbf, b_cbf, u_max=DEFAULT_U_MAX):
    u_nom = np.asarray(u_nom, dtype=np.float32).reshape(-1)

    if isinstance(u_max, (int, float)):
        u_bounds_max = np.array([u_max, u_max, u_max], dtype=np.float32)
    else:
        u_bounds_max = np.asarray(u_max, dtype=np.float32).flatten()
        if u_bounds_max.size == 1:
            val = float(u_bounds_max[0])
            u_bounds_max = np.array([val, val, val], dtype=np.float32)

    if np.dot(a_cbf, u_nom) >= b_cbf - 1e-6:
        return u_nom.copy(), False

    bounds = [(-u_bounds_max[i], u_bounds_max[i]) for i in range(3)]

    res = minimize(
        fun=lambda u: 0.5 * np.dot(u - u_nom, u - u_nom),
        x0=u_nom.copy(),
        jac=lambda u: u - u_nom,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda u: np.dot(a_cbf, u) - b_cbf,
                "jac": lambda u: a_cbf,
            }
        ],
        options={"ftol": 1e-9, "maxiter": 200},
    )

    if res.success:
        u_safe = np.clip(res.x, -u_bounds_max, u_bounds_max).astype(np.float32)
        intervened = float(np.linalg.norm(u_safe - u_nom)) > 1e-4
        return u_safe, intervened

    # Directional fallback if QP fails.
    u_fb = np.array(
        [
            u_bounds_max[0] * np.sign(a_cbf[0]) if abs(a_cbf[0]) > 1e-5 else 0.0,
            u_bounds_max[1] * np.sign(a_cbf[1]) if abs(a_cbf[1]) > 1e-5 else 0.0,
            u_bounds_max[2] * np.sign(a_cbf[2]) if abs(a_cbf[2]) > 1e-5 else 0.0,
        ],
        dtype=np.float32,
    )
    return u_fb, True


def safety_filter(model, dynamics, state_10d, u_nom, t=None, gamma=GAMMA, margin=0.0):
    """Apply the BRT as a fixed-horizon CBF with an explicit value margin."""
    if t is None:
        from src.reachability.compute_brt import CFG
        t = CFG["tMax"]

    u_nom = np.asarray(u_nom, dtype=np.float32)
    a_cbf, b_cbf = _build_cbf_numerical(
        model=model,
        dynamics=dynamics,
        state_10d=state_10d,
        u_nom=u_nom,
        t=t,
        gamma=gamma,
        margin=margin,
    )
    return _solve_qp(u_nom, a_cbf, b_cbf, u_max=dynamics.u_max)


if __name__ == "__main__":
    print("=== Safety filter smoke check ===")

    from dynamics.dynamics import CoffeeArmDynamics
    dynamics = CoffeeArmDynamics()

    try:
        from deepreach.utils import modules
        model = modules.SingleBVPNet(
            in_features=dynamics.input_dim,
            out_features=1,
            type="sine",
            mode="mlp",
            final_layer_factor=1,
            hidden_features=64,
            num_hidden_layers=2,
        )
        model.eval()

        state = np.zeros(10, dtype=np.float32)
        state[6] = 0.02

        u_nom = np.array([2.0, 1.0, -1.0], dtype=np.float32)
        u_safe, intervened = safety_filter(model, dynamics, state, u_nom)
        print(f"u_nom = {u_nom}")
        print(f"u_safe = {u_safe}")
        print(f"intervened = {intervened}")
    except Exception as exc:
        import traceback
        print(f"Smoke test skipped due to error:\n{traceback.format_exc()}")
