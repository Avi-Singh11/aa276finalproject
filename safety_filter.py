# CBF-based safety filter using the DeepReach BRT value function
# CBF condition: dV/dt + dV/dx * f(x,u) >= -gamma*V
# dynamics are control-affine so this is a linear constraint in u -> QP

import sys
import os
import numpy as np
import torch
from scipy.optimize import minimize

DEEPREACH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'deepreach')
)
if DEEPREACH_PATH not in sys.path:
    sys.path.insert(0, DEEPREACH_PATH)

from dynamics.dynamics import CoffeeArmDynamics

# must match coffee_arm_env.py
K_DIAG = [1.0, 2.0, 3.0]   # joint gain matrix diagonal
DAMPING = 0.5                 # joint velocity damping
U_MAX = 2.0                 # torque limit

# CBF decay rate — larger γ = more aggressive intervention
GAMMA = 1.0


def _brt_value_and_gradient(model, dynamics, state_8d_np, t):
    """Return (V, dvdt, dvds) at the given state and time."""
    coord = torch.cat([
        torch.tensor([t], dtype=torch.float32),
        torch.tensor(state_8d_np, dtype=torch.float32),
    ]).unsqueeze(0)                             # (1, 9) in real units

    inp = dynamics.coord_to_input(coord)        # model-unit input

    result = model({'coords': inp})
    # model_in is the tensor that passed through the graph — required for autograd
    dv = dynamics.io_to_dv(result['model_in'], result['model_out'].squeeze(-1))
    V = dynamics.io_to_value(result['model_in'].detach(), result['model_out'].squeeze(-1).detach())

    dvdt = float(dv[0, 0].item())
    dvds = dv[0, 1:].detach().cpu().numpy()    # (8,)
    V_val = float(V.item())
    return V_val, dvdt, dvds


def _build_cbf_constraint(state_8d, dvdt, dvds, V, gamma=GAMMA):
    # builds the linear CBF constraint a_cbf @ u >= b_cbf
    dtheta = state_8d[3:6]    # [dtheta1, dtheta2, dtheta3]
    dalpha = state_8d[7]

    # drift = dV/dx * f_drift(x)
    # dalpha_dot term is 0 for now (sloshing stub)
    drift = (
        dvds[0]*dtheta[0] + dvds[1]*dtheta[1] + dvds[2]*dtheta[2]
        - DAMPING*(dvds[3]*dtheta[0] + dvds[4]*dtheta[1] + dvds[5]*dtheta[2])
        + dvds[6]*dalpha
        # dvds[7] * dalpha_dot_stub(0) = 0
    )

    # control coefficient: dV/dx * G(x), G(x)@u = [0,0,0, K0*u1, K1*u2, K2*u3, 0, 0]
    a_arm = np.array([
        dvds[3] * K_DIAG[0],
        dvds[4] * K_DIAG[1],
        dvds[5] * K_DIAG[2],
    ])

    # TODO: add sloshing coupling when model is ready
    # a_slosh = dvds[7] * (J[0,:] @ K_matrix) / L_EFF
    # also add drift term: dvds[7] * (-(G/L_eff)*sin(alpha) - B_damp*dalpha + J_dot[0,:]@dtheta/L_eff)
    a_cbf = a_arm

    b_cbf = -gamma * V - dvdt - drift

    return a_cbf, b_cbf


def _solve_qp(u_nom, a_cbf, b_cbf, u_max=U_MAX):
    """min ||u - u_nom||^2  s.t.  a_cbf @ u >= b_cbf, |u| <= u_max"""
    # Fast check: is u_nom already feasible?
    if np.dot(a_cbf, u_nom) >= b_cbf - 1e-6:
        return u_nom.copy(), False

    result = minimize(
        fun=lambda u: 0.5 * np.dot(u - u_nom, u - u_nom),
        x0=u_nom.copy(),
        jac=lambda u: u - u_nom,
        method='SLSQP',
        bounds=[(-u_max, u_max)] * 3,
        constraints=[{
            'type': 'ineq',
            'fun':  lambda u: np.dot(a_cbf, u) - b_cbf,
            'jac':  lambda u: a_cbf,
        }],
        options={'ftol': 1e-9, 'maxiter': 200},
    )

    if result.success:
        u_safe = np.clip(result.x, -u_max, u_max).astype(np.float32)
        intervened = np.linalg.norm(u_safe - u_nom) > 1e-4
        return u_safe, intervened

    # fallback: bang-bang on the BRT gradient
    u_fallback = np.array([
        u_max * np.sign(a_cbf[0]),
        u_max * np.sign(a_cbf[1]),
        u_max * np.sign(a_cbf[2]),
    ], dtype=np.float32)
    return u_fallback, True


def safety_filter(model, dynamics, state_8d, u_nom, t=None, gamma=GAMMA):
    """Returns (u_safe, intervened). Queries BRT, builds CBF constraint, solves QP."""
    if t is None:
        from compute_brt import CFG
        t = CFG['tMax']

    u_nom = np.asarray(u_nom, dtype=np.float32)

    V, dvdt, dvds = _brt_value_and_gradient(model, dynamics, state_8d, t)
    a_cbf, b_cbf  = _build_cbf_constraint(state_8d, dvdt, dvds, V, gamma)

    return _solve_qp(u_nom, a_cbf, b_cbf)


if __name__ == '__main__':
    print("Safety filter smoke test (random BRT model -- values are not meaningful)")

    dynamics = CoffeeArmDynamics()

    # Build an untrained model just to test the pipeline
    import sys
    sys.path.insert(0, DEEPREACH_PATH)
    from utils import modules
    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim, out_features=1,
        type='sine', mode='mlp', final_layer_factor=1,
        hidden_features=64, num_hidden_layers=2,
    )
    model.eval()

    state = np.zeros(8)
    state[6] = 0.1   # alpha slightly above zero

    u_nom = np.array([1.0, 0.5, -0.5], dtype=np.float32)
    u_safe, intervened = safety_filter(model, dynamics, state, u_nom)

    print(f" u_nom  = {u_nom}")
    print(f" u_safe = {u_safe}")
    print(f" intervened = {intervened}")
    print("Pipeline OK")
