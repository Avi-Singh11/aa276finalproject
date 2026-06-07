"""Cartesian pendulum model for coffee slosh."""

from __future__ import annotations
import numpy as np
from src.config.constants import G, DEFAULT_L_EFF, SLOSH_DAMPING
from .arm_dynamics import get_cup_acceleration


def slosh_dynamics(slosh_state, arm_state, u, K, L, dt, l_eff=DEFAULT_L_EFF, theta_eps=1e-9):
    """Integrate the Cartesian slosh state [x, y, dx, dy]."""
    x, y, dx, dy = slosh_state
    ax, ay, az = get_cup_acceleration(arm_state, u, K, L)

    # Recover z and dz from the pendulum constraint.
    z_sq = l_eff**2 - x**2 - y**2
    z = -np.sqrt(max(z_sq, 1e-12))
    dz = -(x * dx + y * dy) / z

    cart_state = np.array([x, y, z, dx, dy, dz], dtype=np.float64)

    # Convert the damping ratio to a damping coefficient.
    omega_n = np.sqrt(G / l_eff)
    c_damping = 2.0 * SLOSH_DAMPING * omega_n

    def cart_derivatives(state):
        cx, cy, cz, cdx, cdy, cdz = state

        total_ax = ax
        total_ay = ay
        total_az = G + az

        # Compute the pendulum constraint force.
        v_sq = cdx**2 + cdy**2 + cdz**2
        r_dot_a = cx * total_ax + cy * total_ay + cz * total_az
        lamb = (v_sq - r_dot_a) / (l_eff**2)

        # Acceleration includes cup motion, tension, and damping.
        ddx = -total_ax - lamb * cx - c_damping * cdx
        ddy = -total_ay - lamb * cy - c_damping * cdy
        ddz = -total_az - lamb * cz - c_damping * cdz

        return np.array([cdx, cdy, cdz, ddx, ddy, ddz], dtype=np.float64)

    # RK4 integration.
    k1 = cart_derivatives(cart_state)
    k2 = cart_derivatives(cart_state + 0.5 * dt * k1)
    k3 = cart_derivatives(cart_state + 0.5 * dt * k2)
    k4 = cart_derivatives(cart_state + dt * k3)

    next_cart = cart_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    nx, ny, nz, ndx, ndy, ndz = next_cart

    # Project back onto the pendulum sphere.
    pos_len = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = (nx / pos_len) * l_eff
    ny = (ny / pos_len) * l_eff

    # Keep velocity tangent to the sphere.
    nz_updated = -np.sqrt(max(l_eff**2 - nx**2 - ny**2, 1e-12))
    v_drift = (nx * ndx + ny * ndy + nz_updated * ndz) / (l_eff**2)
    ndx -= v_drift * nx
    ndy -= v_drift * ny

    return np.array([nx, ny, ndx, ndy], dtype=np.float64)
