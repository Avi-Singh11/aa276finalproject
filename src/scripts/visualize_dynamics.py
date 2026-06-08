"""Unified Robot Arm Kinematics and Slosh Dynamics Real-Time Visualization."""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

# Ensure project root is in the system path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.slosh_dynamics import slosh_dynamics
from src.core.arm_dynamics import position_cup, jacobian, arm_dynamics, get_cup_acceleration
from src.config.constants import (
    DEFAULT_L,
    DEFAULT_K,
    DEFAULT_L_EFF,
    DEFAULT_THETA_EPS,
    DEFAULT_VARTHETA_MAX
)

print("Running unified 3D arm and slosh visualization with unconstrained joints...")

# Simulation configurations
micro_dt = 0.0001
macro_dt = 0.01 
timesteps = 500
l_eff = DEFAULT_L_EFF
substeps = int(macro_dt / micro_dt)
playback_interval_ms = int(macro_dt * 1000)  

# Initialize states using your project constants
slosh_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
arm_state = np.zeros(6, dtype=np.float64)
arm_state[1] = np.pi / 4.0   
arm_state[2] = -np.pi / 4.0  # Just an initial state; joint 3 is free to move.

K = DEFAULT_K.astype(np.float64)
L = DEFAULT_L.astype(np.float64)  

u_profile = np.zeros((timesteps, 3), dtype=np.float64)
for i in range(timesteps):
    t = i * macro_dt
    if t < 1.0:
        u_profile[i] = [3.0, 0.0, 0.0]     
    elif t < 2.0:
        u_profile[i] = [0.0, 4.0, 0.0]     
    elif t < 3.0:
        u_profile[i] = [-2.0, -2.0, 0.0]   
    else:
        u_profile[i] = [0.0, 0.0, 0.0]

arm_history = []
slosh_history = []
a_cup_history = []
ee_trail = []
jac_vel_history = []

def get_joint_positions_clean(phi_flat, L):
    """Computes joint positions for the fully free 3-DoF arm."""
    l1, l2, l3 = L
    theta1 = phi_flat[0]
    theta2 = phi_flat[1]
    
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    
    base = np.array([0.0, 0.0, 0.0])
    j2 = np.array([0.0, 0.0, l1])
    j3 = np.array([
        c1 * l2 * c2,
        s1 * l2 * c2,
        l1 + l2 * s2
    ])
    ee = position_cup(phi_flat.reshape(6, 1), L).flatten()
    return [base, j2, j3, ee]

print("Pre-computing dynamic trajectories...")
for i in range(timesteps):
    arm_history.append(arm_state.copy())
    ee_pos = position_cup(arm_state.reshape(6, 1), L).flatten()
    ee_trail.append(ee_pos)
    
    v_jac = (jacobian(arm_state.reshape(6, 1), L) @ arm_state[3:6].reshape(3, 1)).flatten()
    jac_vel_history.append(v_jac)
    
    u_current = u_profile[i]
    a_cup = get_cup_acceleration(arm_state, u_current, K, L)
    a_cup_history.append(a_cup)
    
    for _ in range(substeps):
        slosh_state = slosh_dynamics(
            slosh_state=slosh_state,
            arm_state=arm_state,
            u=u_current,
            K=K,
            L=L,
            dt=micro_dt,
            l_eff=l_eff,
            theta_eps=DEFAULT_THETA_EPS,
        )

    z_current = -np.sqrt(max(l_eff**2 - slosh_state[0]**2 - slosh_state[1]**2, 0.0))
    slosh_history.append([slosh_state[0], slosh_state[1], z_current])
    
    arm_state = arm_dynamics(arm_state, u_current, K, macro_dt)

arm_history = np.array(arm_history)
slosh_history = np.array(slosh_history)
a_cup_history = np.array(a_cup_history)
ee_trail = np.array(ee_trail)
jac_vel_history = np.array(jac_vel_history)

x_pend = slosh_history[:, 0]
y_pend = slosh_history[:, 1]
z_pend = slosh_history[:, 2]

xy_dist = np.sqrt(x_pend**2 + y_pend**2)
z_depth = np.abs(z_pend)
angles_rad = np.full_like(xy_dist, np.pi / 2.0)  # Default to 90 degrees
mask = z_depth > 1e-6
angles_rad[mask] = np.arctan(xy_dist[mask] / z_depth[mask])
time_array = np.arange(timesteps) * macro_dt

fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1], width_ratios=[1.0, 1.0])

ax_arm = fig.add_subplot(gs[0, 0], projection='3d')
lim_arm = 0.85 
ax_arm.set_xlim([-lim_arm, lim_arm])
ax_arm.set_ylim([-lim_arm, lim_arm])
ax_arm.set_zlim([0, lim_arm])
ax_arm.set_xlabel("X (m)")
ax_arm.set_ylabel("Y (m)")
ax_arm.set_zlabel("Z (m)")
xx, yy = np.meshgrid([-lim_arm, lim_arm], [-lim_arm, lim_arm])
ax_arm.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.04, color="gray")

arm_line, = ax_arm.plot([], [], [], "-o", color="steelblue", linewidth=3.0, markersize=6, label="Robot Arm")
ee_dot = ax_arm.scatter([], [], [], color="crimson", s=50, zorder=5)
trail_line, = ax_arm.plot([], [], [], "-", color="lightblue", linewidth=1.5, alpha=0.7, label="EE Trail")
arrow_arm = [None]
ax_arm.view_init(elev=20, azim=45)
ax_arm.legend(loc='upper left', fontsize=8)

ax3d = fig.add_subplot(gs[0, 1], projection='3d')
max_range = l_eff * 1.5
ax3d.set_xlim([-max_range, max_range])
ax3d.set_ylim([-max_range, max_range])
ax3d.set_zlim([-l_eff * 1.2, max_range * 0.5])
ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Z (m)")

ax3d.plot(x_pend, y_pend, z_pend, 'k-', alpha=0.15, lw=1.5, label='Fluid Path')
line3d, = ax3d.plot([], [], [], 'o-', lw=3, color='dodgerblue', markersize=8, label='Pendulum')
quiver_handle = [None]

orange_arrow = mlines.Line2D([], [], color='darkorange', marker='>', markersize=8, label='Excitation Accel')
handles, labels = ax3d.get_legend_handles_labels()
handles.append(orange_arrow)
ax3d.legend(handles=handles, loc='upper left', fontsize=8)
ax3d.view_init(elev=20, azim=45)

ax2d = fig.add_subplot(gs[1, :])
ax2d.set_xlim(0, timesteps * macro_dt)
ax2d.set_ylim(0, max(DEFAULT_VARTHETA_MAX * 1.5, np.max(angles_rad) * 1.1))
ax2d.set_xlabel("Time (s)")
ax2d.set_ylabel("Slosh Angle (rad)")
ax2d.grid(True, linestyle=':', alpha=0.7)

ax2d.axhline(DEFAULT_VARTHETA_MAX, color='red', linestyle='--', lw=2, label='Spill Threshold (VARTHETA_MAX)')
line2d, = ax2d.plot([], [], '-', lw=2.5, color='dodgerblue', label='Current Angle')
ax2d.legend(loc='upper right')

def init():
    arm_line.set_data([], [])
    arm_line.set_3d_properties([])
    trail_line.set_data([], [])
    trail_line.set_3d_properties([])
    line3d.set_data_3d([], [], [])
    line2d.set_data([], [])
    return arm_line, trail_line, line3d, line2d

def animate(i):
    t = time_array[i]
    
    pts = get_joint_positions_clean(arm_history[i], L)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    arm_line.set_data(xs, ys)
    arm_line.set_3d_properties(zs)
    
    ee = pts[-1]
    ee_dot._offsets3d = ([ee[0]], [ee[1]], [ee[2]])
    trail_line.set_data(ee_trail[:i+1, 0], ee_trail[:i+1, 1])
    trail_line.set_3d_properties(ee_trail[:i+1, 2])
    
    # Render Robot acceleration vector arrow
    if arrow_arm[0] is not None:
        arrow_arm[0].remove()
    a_macro = a_cup_history[i]  # Now using acceleration!
    if np.linalg.norm(a_macro) > 0.01:
        # Fixed scale so it visually matches the right-side plot's behavior
        scale_a_macro = 0.05 
        arrow_arm[0] = ax_arm.quiver(
            ee[0], ee[1], ee[2],
            a_macro[0]*scale_a_macro, a_macro[1]*scale_a_macro, a_macro[2]*scale_a_macro,
            color="darkorange", linewidth=2.5, arrow_length_ratio=0.4
        )
    else:
        arrow_arm[0] = None
    
    line3d.set_data_3d([0, x_pend[i]], [0, y_pend[i]], [0, z_pend[i]])
    line2d.set_data(time_array[:i+1], angles_rad[:i+1])
    
    if angles_rad[i] > DEFAULT_VARTHETA_MAX:
        line2d.set_color('red')
        line3d.set_color('red')
    else:
        line2d.set_color('dodgerblue')
        line3d.set_color('dodgerblue')
    
    if quiver_handle[0] is not None:
        quiver_handle[0].remove()
        
    a_x, a_y, a_z = a_cup_history[i]
    accel_mag = np.linalg.norm([a_x, a_y, a_z])
    
    if accel_mag > 0.01:
        scale_a = 0.05 
        quiver_handle[0] = ax3d.quiver(
            0, 0, 0, 
            a_x * scale_a, a_y * scale_a, a_z * scale_a,
            color='darkorange', lw=2.5, arrow_length_ratio=0.4
        )
    else:
        quiver_handle[0] = None

    phase = "Y-Axis Push" if t < 1.0 else "X/Z-Axis Push" if t < 2.0 else "Mixed Push" if t < 3.0 else "Free Swing"
    ax_arm.set_title(f"Macro Arm View (Free 3-DoF) | Phase: {phase}", fontsize=9)
    ax3d.set_title(f"Gimbaled Cup Frame | Time: {t:.2f}s", fontsize=9)
    
    return arm_line, trail_line, line3d, line2d

ani = animation.FuncAnimation(
    fig, 
    animate, 
    init_func=init, 
    frames=timesteps, 
    interval=playback_interval_ms,
    blit=False
)

plt.tight_layout()
plt.show()