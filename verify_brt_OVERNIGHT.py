"""Numerical preflight checks for the overnight BRT implementation."""

from __future__ import annotations

import itertools

import numpy as np
import torch

from dynamics.dynamics_OVERNIGHT_VERIFIED import CoffeeArmDynamics
from src.core.simulation_OVERNIGHT_VERIFIED import coupled_dynamics
from src.reachability.compute_brt_OVERNIGHT_VERIFIED import _boundary_fn_torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    rng = np.random.default_rng(276)
    dynamics = CoffeeArmDynamics()
    scale = dynamics.state_scale.astype(np.float64)

    states = rng.uniform(-1.0, 1.0, (10_000, 10)) * scale
    boundary_numpy = dynamics.boundary_fn(states)
    boundary_torch = (
        _boundary_fn_torch(dynamics, torch.tensor(states, dtype=torch.float32))
        .detach()
        .cpu()
        .numpy()
    )
    boundary_error = float(np.max(np.abs(boundary_numpy - boundary_torch)))
    sign_mismatches = int(
        np.count_nonzero((boundary_numpy <= 0.0) != (boundary_torch <= 0.0))
    )
    require(boundary_error < 2e-6, f"Boundary mismatch: {boundary_error}")
    require(sign_mismatches == 0, f"Boundary sign mismatches: {sign_mismatches}")

    max_h_error = 0.0
    max_dynamics_error = 0.0
    dt = 1e-4
    corners = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
    for _ in range(80):
        state = rng.uniform(-0.65, 0.65, 10) * scale
        state[6:8] *= 0.5
        # Keep this arbitrary-gradient check below H_MAX so it tests the
        # analytic dynamics rather than the deliberate stabilization clamp.
        physical_gradient = 0.1 * rng.normal(size=10)
        normalized_gradient = physical_gradient * scale

        z = torch.tensor(state / scale, dtype=torch.float32).unsqueeze(0)
        pz = torch.tensor(normalized_gradient, dtype=torch.float32).unsqueeze(0)
        analytic_h = float(dynamics.hamiltonian(z, pz).item())

        directional_values = []
        for signs in corners:
            control = signs * dynamics.u_max
            successor = coupled_dynamics(
                state,
                control,
                dynamics.K,
                dynamics.L,
                dt,
                l_eff=dynamics.l_eff,
                theta_eps=dynamics.theta_eps,
            ).astype(np.float64)
            derivative = (successor - state) / dt
            directional_values.append(float(physical_gradient @ derivative))

        finite_difference_h = max(directional_values)
        max_h_error = max(max_h_error, abs(analytic_h - finite_difference_h))

        zero_successor = coupled_dynamics(
            state,
            np.zeros(3),
            dynamics.K,
            dynamics.L,
            dt,
            l_eff=dynamics.l_eff,
            theta_eps=dynamics.theta_eps,
        ).astype(np.float64)
        fd_arm = (zero_successor[:6] - state[:6]) / dt
        expected_arm = np.concatenate([state[3:6], -state[3:6]])
        max_dynamics_error = max(
            max_dynamics_error, float(np.max(np.abs(fd_arm - expected_arm)))
        )

    require(max_dynamics_error < 0.02, f"Arm dynamics mismatch: {max_dynamics_error}")
    require(max_h_error < 0.01, f"Hamiltonian mismatch: {max_h_error}")

    z = torch.empty(100_000, 10).uniform_(-1.0, 1.0).requires_grad_(True)
    physical = z * torch.tensor(dynamics.state_scale)
    boundary = _boundary_fn_torch(dynamics, physical)
    gradient = torch.autograd.grad(boundary.sum(), z)[0]
    hamiltonian = dynamics.hamiltonian(z.detach(), gradient.detach())
    require(torch.isfinite(hamiltonian).all().item(), "Non-finite Hamiltonian")
    require(torch.isfinite(gradient).all().item(), "Non-finite boundary gradient")
    clamp_count = int((torch.abs(hamiltonian) >= 49.999).sum().item())
    require(clamp_count == 0, f"Hamiltonian clamp activated {clamp_count} times")

    print(f"boundary_max_abs_error={boundary_error:.3e}")
    print(f"boundary_sign_mismatches={sign_mismatches}")
    print(f"arm_dynamics_max_abs_error={max_dynamics_error:.3e}")
    print(f"hamiltonian_max_abs_error={max_h_error:.3e}")
    print(f"hamiltonian_clamp_count={clamp_count}/100000")
    print("ALL OVERNIGHT BRT PREFLIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
