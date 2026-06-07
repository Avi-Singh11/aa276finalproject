"""Correctness and equivalence gate for BRT_STABLE_RECOVERY."""

from __future__ import annotations

import numpy as np
import torch


def run_preflight(dynamics, n_tests=64, seed=7, verbose=False):
    rng = np.random.default_rng(seed)
    scale = np.asarray(dynamics.state_scale, dtype=np.float64)
    u_max = np.asarray(dynamics.u_max, dtype=np.float64)
    worst_h_error = 0.0

    for _ in range(n_tests):
        z = rng.uniform(-0.7, 0.7, 10)
        x = z * scale

        # Keep the Cartesian pendulum state comfortably feasible.
        max_r = 0.75 * dynamics.l_eff
        r = np.hypot(x[6], x[7])
        if r > max_r:
            x[6:8] *= max_r / r
            z = x / scale

        p_phys = rng.normal(size=10)
        p_normalized = p_phys * scale

        h_model = float(dynamics.hamiltonian(
            torch.tensor(z[None, :], dtype=torch.float32),
            torch.tensor(p_normalized[None, :], dtype=torch.float32),
        ).item())

        f0 = dynamics.continuous_dynamics(x, np.zeros(3))
        g = np.column_stack([
            dynamics.continuous_dynamics(x, np.eye(3)[i]) - f0
            for i in range(3)
        ])
        h_expected = float(p_phys @ f0 + np.sum(u_max * np.abs(p_phys @ g)))
        error = abs(h_model - h_expected)
        worst_h_error = max(worst_h_error, error)

    probe = rng.uniform(-0.5, 0.5, (4096, 10)) * scale
    probe_t = torch.tensor(probe, dtype=torch.float32)
    vectorized = dynamics.boundary_fn(probe_t).detach().cpu().numpy()
    numpy_reference = dynamics.boundary_fn_numpy(probe)
    scalar = np.array([dynamics.failure_margin(x) for x in probe])
    boundary_error = float(np.max(np.abs(vectorized - scalar)))
    torch_numpy_error = float(np.max(np.abs(vectorized - numpy_reference)))
    sign_agreement = float(np.mean(
        np.signbit(vectorized) == np.signbit(numpy_reference)
    ))

    # A displaced pendulum at zero velocity is not a static trajectory.
    displaced = np.zeros(10)
    displaced[6] = 0.5 * dynamics.slosh_rad_max
    displaced_accel = abs(float(dynamics.continuous_dynamics(displaced, np.zeros(3))[8]))

    exact_in = torch.randn(512, 11, dtype=torch.float32)
    exact_in[:, 0] = 0.0
    exact_in.requires_grad_(True)
    raw_out = torch.randn(512, dtype=torch.float32, requires_grad=True)
    exact_value = dynamics.io_to_value(exact_in, raw_out)
    exact_boundary = dynamics.boundary_fn(
        dynamics.input_to_coord(exact_in)[..., 1:]
    )
    exact_boundary_error = float(
        torch.max(torch.abs(exact_value - exact_boundary)).detach().item()
    )

    if worst_h_error > 2e-3:
        raise RuntimeError(
            f"Hamiltonian/vector-field mismatch: max error {worst_h_error:.3e}"
        )
    if boundary_error > 2e-6:
        raise RuntimeError(
            f"Scalar/vectorized boundary mismatch: max error {boundary_error:.3e}"
        )
    if torch_numpy_error > 2e-6 or sign_agreement < 1.0:
        raise RuntimeError(
            "Torch/NumPy boundary mismatch: "
            f"error={torch_numpy_error:.3e}, sign agreement={sign_agreement:.6f}"
        )
    if displaced_accel < 1e-6:
        raise RuntimeError("Slosh model incorrectly treats displaced coffee as static")
    if exact_boundary_error > 1e-7:
        raise RuntimeError(
            f"Exact boundary representation failed: {exact_boundary_error:.3e}"
        )

    if verbose:
        print("BRT_STABLE_RECOVERY preflight passed")
        print(f"  Hamiltonian max error : {worst_h_error:.3e}")
        print(f"  Boundary max error    : {boundary_error:.3e}")
        print(f"  Torch/NumPy max error : {torch_numpy_error:.3e}")
        print(f"  Boundary sign match   : {sign_agreement*100:.2f}%")
        print(f"  Displaced slosh accel : {displaced_accel:.3e} m/s^2")
        print(f"  Exact t=0 error       : {exact_boundary_error:.3e}")

    return {
        "hamiltonian_max_error": worst_h_error,
        "boundary_max_error": boundary_error,
        "torch_numpy_max_error": torch_numpy_error,
        "boundary_sign_agreement": sign_agreement,
        "displaced_slosh_accel": displaced_accel,
        "exact_boundary_error": exact_boundary_error,
    }


if __name__ == "__main__":
    from BRT_STABLE_RECOVERY_dynamics import BRTStableRecoveryCoffeeArmDynamics

    run_preflight(BRTStableRecoveryCoffeeArmDynamics(), verbose=True)
