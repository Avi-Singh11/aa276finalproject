import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Obstacles
# ---------------------------------
from constants import DEFAULT_OBSTACLES

# ---------------------------------
# Figure
# ---------------------------------
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# ---------------------------------
# Draw obstacles (spheres)
# ---------------------------------
u = np.linspace(0, 2 * np.pi, 80)
v = np.linspace(0, np.pi, 80)

for i, obs in enumerate(DEFAULT_OBSTACLES):
    cx, cy, cz = obs["center"]
    r = obs["radius"]

    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(
        x, y, z,
        alpha=0.65,
        linewidth=0,
        shade=True
    )

    ax.scatter(cx, cy, cz, s=40, color="black")
    ax.text(cx, cy, cz, f" {i+1}")

# ---------------------------------
# Workspace hemisphere (radius 0.8 m)
# Only z >= 0
# ---------------------------------
R = 0.8

phi = np.linspace(0, 2 * np.pi, 120)
theta = np.linspace(0, np.pi / 2, 120)

PHI, THETA = np.meshgrid(phi, theta)

X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
Z = R * np.cos(THETA)

ax.plot_surface(
    X,
    Y,
    Z,
    color="lightgray",
    alpha=0.22,
    linewidth=0,
    shade=True
)

# ---------------------------------
# XY plane (z = 0)
# ---------------------------------
rho = np.linspace(0, R, 100)
phi_disk = np.linspace(0, 2 * np.pi, 200)

RHO, PHI = np.meshgrid(rho, phi_disk)

Xdisk = RHO * np.cos(PHI)
Ydisk = RHO * np.sin(PHI)
Zdisk = np.zeros_like(Xdisk)

ax.plot_surface(
    Xdisk,
    Ydisk,
    Zdisk,
    color="lightgray",
    alpha=0.10,
    linewidth=0,
    shade=False
)

# Boundary circle of workspace
t = np.linspace(0, 2 * np.pi, 500)

ax.plot(
    R * np.cos(t),
    R * np.sin(t),
    np.zeros_like(t),
    color="gray",
    linewidth=1.5
)

# ---------------------------------
# Coordinate axes
# ---------------------------------

# X-axis
ax.plot(
    [-R, R],
    [0, 0],
    [0, 0],
    linewidth=2
)

# Y-axis
ax.plot(
    [0, 0],
    [-R, R],
    [0, 0],
    linewidth=2
)

# Z-axis
ax.plot(
    [0, 0],
    [0, 0],
    [-R, R],
    linewidth=3
)

# Origin
ax.scatter(
    [0],
    [0],
    [0],
    s=60,
    color="black"
)

# Axis labels
ax.text(R, 0, 0, "x", fontsize=12)
ax.text(0, R, 0, "y", fontsize=12)
ax.text(0, 0, R, "z", fontsize=12)

# ---------------------------------
# End marker
# ---------------------------------

# End point
end_x, end_y, end_z = 0.4, -0.5, 0.15

ax.scatter(
    [end_x], [end_y], [end_z],
    marker='x',
    s=200,
    linewidths=3,
    color='black'
)

ax.text(
    end_x,
    end_y,
    end_z + 0.03,
    "end",
    fontsize=12
)

# ---------------------------------
# Initialization region
# ---------------------------------

r_vals = np.linspace(0.5, 0.7, 10)
az_vals = np.deg2rad(np.linspace(60, 90, 20))
el_vals = np.deg2rad(np.linspace(0, 30, 20))

Rr, AZ, EL = np.meshgrid(
    r_vals,
    az_vals,
    el_vals,
    indexing="ij"
)

Xinit = Rr * np.cos(EL) * np.cos(AZ)
Yinit = Rr * np.cos(EL) * np.sin(AZ)
Zinit = Rr * np.sin(EL)

ax.scatter(
    Xinit.flatten(),
    Yinit.flatten(),
    Zinit.flatten(),
    s=3,
    alpha=0.15,
    color="green"
)

ax.text(
    0.05,
    0.55,
    0.25,
    "start region",
    color="green",
    fontsize=12
)

# ---------------------------------
# Equal scaling
# ---------------------------------
ax.set_xlim(-R, R)
ax.set_ylim(-R, R)
ax.set_zlim(-R, R)

# True 1:1:1 aspect ratio
ax.set_box_aspect([1, 1, 1])

# ---------------------------------
# Labels and view
# ---------------------------------
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

ax.set_title("CoffeeArm Workspace and Obstacles")

# Nice robotics-style viewing angle
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()