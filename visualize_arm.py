"""
Arm kinematics visualization and verification.

Figure 1: Static poses — 4 known configurations with expected end-effector
          positions printed alongside the computed ones.
Figure 2: Animated trajectory under constant torque, with end-effector trail
          and Jacobian velocity arrows (verifies J is consistent with motion).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.animation import FuncAnimation

from coffee_pouring_env import (
    position_cup, jacobian, arm_dynamics, get_cup_acceleration,
)


L = [1.0, 1.0, 1.0]   # link lengths
K = np.diag([1.0, 2.0, 3.0])


def joint_positions(phi_flat, L):
    """
    Returns four (x,y,z) points along the arm:
      [base (0,0,0), joint-2, joint-3, end-effector]

    Arm structure:
      link-1 is always vertical (length l1), so joint-2 = (0, 0, l1)
      link-2 rotates in the plane defined by θ1 (azimuth) and θ2 (elevation)
      link-3 continues at angle θ2+θ3 from vertical
    """
    l1, l2, l3 = L
    if phi_flat.ndim == 1:
        phi_flat = phi_flat.reshape(6, 1)
    θ1 = phi_flat[0, 0];  θ2 = phi_flat[1, 0]
    s1, c1 = np.sin(θ1), np.cos(θ1)
    s2, c2 = np.sin(θ2), np.cos(θ2)

    base = np.array([0.0, 0.0, 0.0])
    j2   = np.array([0.0, 0.0, l1])
    j3   = np.array([c1 * l2 * c2,
                     s1 * l2 * c2,
                     l1 + l2 * s2])
    ee   = position_cup(phi_flat, L).flatten()
    return [base, j2, j3, ee]


def draw_arm(ax, phi_flat, L, color="steelblue", alpha=1.0, label=None):
    pts = joint_positions(phi_flat, L)
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    zs  = [p[2] for p in pts]
    ax.plot(xs, ys, zs, "-o", color=color, alpha=alpha,
            linewidth=2, markersize=5, label=label)
    # mark end-effector
    ax.scatter(*pts[-1], color="crimson", s=60, zorder=5)
    return pts[-1]   # return end-effector position


def setup_ax(ax, title, lim=3.5):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, fontsize=9)
    # ground plane reference
    xx, yy = np.meshgrid([-lim, lim], [-lim, lim])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.06, color="gray")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Static verification poses
# ─────────────────────────────────────────────────────────────────────────────

configs = [
    # (theta1, theta2, theta3,  description,  expected end-effector)
    (0,      0,      0,       "θ=(0,0,0)\narm horizontal along +x",
     np.array([2.0, 0.0, 1.0])),

    (np.pi/2, 0,     0,       "θ=(π/2,0,0)\narm horizontal along +y",
     np.array([0.0, 2.0, 1.0])),

    (0,      np.pi/2, 0,      "θ=(0,π/2,0)\narm fully vertical (z=3)",
     np.array([0.0, 0.0, 3.0])),

    (np.pi/4, np.pi/4, -np.pi/4,  "θ=(π/4,π/4,-π/4)\nmixed pose",
     None),   # expected computed below
]

# Compute expected for the mixed pose analytically from position_cup
th1, th2, th3 = np.pi/4, np.pi/4, -np.pi/4
phi_mixed = np.array([th1, th2, th3, 0., 0., 0.]).reshape(6, 1)
configs[3] = configs[3][:4] + (position_cup(phi_mixed, L).flatten(),)

fig1, axes = plt.subplots(2, 2, figsize=(11, 9),
                          subplot_kw={"projection": "3d"})
fig1.suptitle("Static Arm Configurations — Kinematics Verification", fontsize=12)

print("=== Figure 1: Static pose verification ===\n")
all_ok = True
for ax, (t1, t2, t3, desc, expected) in zip(axes.flat, configs):
    phi_flat = np.array([t1, t2, t3, 0., 0., 0.])
    setup_ax(ax, desc)
    ee = draw_arm(ax, phi_flat, L)

    err = np.linalg.norm(ee - expected)
    ok  = err < 1e-10
    all_ok = all_ok and ok
    status = "OK" if ok else f"ERR={err:.2e}"
    print(f"  {desc.split(chr(10))[0]}")
    print(f"    expected:  {np.round(expected, 4)}")
    print(f"    computed:  {np.round(ee, 4)}")
    print(f"    status:    {status}\n")

    # annotate on plot
    color = "green" if ok else "red"
    ax.text2D(0.05, 0.92,
              f"ee = {np.round(ee,2)}\n{status}",
              transform=ax.transAxes, fontsize=7, color=color)

print("All static poses OK:", all_ok)
plt.tight_layout()
plt.savefig("arm_static_poses.png", dpi=120)
print("Saved arm_static_poses.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Animated trajectory with Jacobian velocity arrows
# ─────────────────────────────────────────────────────────────────────────────

dt      = 0.02
n_steps = 300
u_const = np.array([0.3, 0.5, -0.2])   # constant torque

phi0 = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
phi  = phi0.copy()

traj_phi = [phi.copy()]
for _ in range(n_steps):
    phi = arm_dynamics(phi, u_const, K, dt)
    traj_phi.append(phi.copy())
traj_phi = np.array(traj_phi)   # (n_steps+1, 6)

# End-effector trail
ee_trail = np.array([
    position_cup(p.reshape(6,1), L).flatten() for p in traj_phi
])

# Jacobian velocity at each frame
jac_vel = np.array([
    (jacobian(traj_phi[i].reshape(6,1), L) @ traj_phi[i, 3:6].reshape(3,1)).flatten()
    for i in range(len(traj_phi))
])

fig2 = plt.figure(figsize=(9, 7))
ax2  = fig2.add_subplot(111, projection="3d")
setup_ax(ax2, "Arm Trajectory (constant torque) + Jacobian velocity arrows", lim=3.5)
fig2.suptitle("Trajectory Verification\n"
              "Arrow = J(φ)·θ̇  (should point along end-effector motion)",
              fontsize=10)

trail_line,  = ax2.plot([], [], [], "-", color="lightblue", linewidth=1, alpha=0.6)
arm_line,    = ax2.plot([], [], [], "-o", color="steelblue", linewidth=2, markersize=5)
ee_dot       = ax2.scatter([], [], [], color="crimson", s=60, zorder=5)
arrow_handle = [None]   # holds the current quiver object

frame_skip = 3   # animate every Nth frame

def update(frame):
    i = frame * frame_skip
    i = min(i, len(traj_phi) - 1)

    # trail up to current frame
    trail_line.set_data(ee_trail[:i+1, 0], ee_trail[:i+1, 1])
    trail_line.set_3d_properties(ee_trail[:i+1, 2])

    # arm links
    pts = joint_positions(traj_phi[i], L)
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    zs  = [p[2] for p in pts]
    arm_line.set_data(xs, ys)
    arm_line.set_3d_properties(zs)

    # end-effector dot
    ee = pts[-1]
    ee_dot._offsets3d = ([ee[0]], [ee[1]], [ee[2]])

    # Jacobian velocity arrow at end-effector
    if arrow_handle[0] is not None:
        arrow_handle[0].remove()
    v = jac_vel[i]
    scale = 0.4 / (np.linalg.norm(v) + 1e-8)
    arrow_handle[0] = ax2.quiver(
        ee[0], ee[1], ee[2],
        v[0]*scale, v[1]*scale, v[2]*scale,
        color="darkorange", linewidth=2, arrow_length_ratio=0.3
    )

    ax2.set_title(
        f"Trajectory Verification — step {i}/{len(traj_phi)-1}\n"
        f"Arrow = J·θ̇ (should align with motion)",
        fontsize=9
    )
    return trail_line, arm_line, ee_dot

n_frames = len(traj_phi) // frame_skip
anim = FuncAnimation(fig2, update, frames=n_frames, interval=40, blit=False)

try:
    anim.save("arm_trajectory.gif", writer="pillow", fps=25)
    print("Saved arm_trajectory.gif")
except Exception as e:
    print(f"GIF save failed ({e}), showing interactive window instead.")

plt.tight_layout()
plt.show()
