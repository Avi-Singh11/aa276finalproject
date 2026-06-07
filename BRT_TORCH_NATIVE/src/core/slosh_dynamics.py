"""Pure Cartesian pendulum slosh dynamics to completely eliminate coordinate poles."""

from __future__ import annotations
import numpy as np
from src.config.constants import G, DEFAULT_L_EFF, SLOSH_DAMPING
from .arm_dynamics import get_cup_acceleration

def slosh_dynamics(slosh_state, arm_state, u, K, L, dt, l_eff=DEFAULT_L_EFF, theta_eps=1e-9):
    """Integrates slosh dynamics using a pure Cartesian state vector [x, y, dx, dy]
    to completely eliminate trigonometric coordinate pole vulnerabilities.
    """
    x, y, dx, dy = slosh_state
    ax, ay, az = get_cup_acceleration(arm_state, u, K, L)
    
    # 1. Reconstruct dependent Z elements from the spherical shell constraint
    z_sq = l_eff**2 - x**2 - y**2
    z = -np.sqrt(max(z_sq, 1e-12))
    dz = -(x * dx + y * dy) / z
    
    cart_state = np.array([x, y, z, dx, dy, dz], dtype=np.float64)
    
    # Compute natural frequency and scale damping as a true dimensionless ratio (zeta)
    omega_n = np.sqrt(G / l_eff)
    c_damping = 2.0 * SLOSH_DAMPING * omega_n
    
    # 2. Define the pure Cartesian derivative function
    def cart_derivatives(state):
        cx, cy, cz, cdx, cdy, cdz = state
        
        total_ax = ax
        total_ay = ay
        total_az = G + az
        
        # Calculate the string tension multiplier (lambda) via kinematic constraint
        v_sq = cdx**2 + cdy**2 + cdz**2
        r_dot_a = cx * total_ax + cy * total_ay + cz * total_az
        lamb = (v_sq - r_dot_a) / (l_eff**2)
        
        # Equations of motion: Accel = -Cup_Accel - Tension - Damping
        ddx = -total_ax - lamb * cx - c_damping * cdx
        ddy = -total_ay - lamb * cy - c_damping * cdy
        ddz = -total_az - lamb * cz - c_damping * cdz
        
        return np.array([cdx, cdy, cdz, ddx, ddy, ddz], dtype=np.float64)
    
    # 3. 4th-Order Runge-Kutta Step
    k1 = cart_derivatives(cart_state)
    k2 = cart_derivatives(cart_state + 0.5 * dt * k1)
    k3 = cart_derivatives(cart_state + 0.5 * dt * k2)
    k4 = cart_derivatives(cart_state + dt * k3)
    
    next_cart = cart_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    nx, ny, nz, ndx, ndy, ndz = next_cart
    
    # 4. Rigid algebraic projection to prevent numerical string stretching
    pos_len = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = (nx / pos_len) * l_eff
    ny = (ny / pos_len) * l_eff
    
    # Enforce velocity constraint (must remain strictly tangent to sphere surface)
    nz_updated = -np.sqrt(max(l_eff**2 - nx**2 - ny**2, 1e-12))
    v_drift = (nx * ndx + ny * ndy + nz_updated * ndz) / (l_eff**2)
    ndx -= v_drift * nx
    ndy -= v_drift * ny
        
    return np.array([nx, ny, ndx, ndy], dtype=np.float64)