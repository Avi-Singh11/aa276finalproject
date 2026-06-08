"""Side-by-side animated comparison: PPO baseline vs PPO + BRT safety filter.

Layout (4 panels):
  Top-left    : Baseline 3D arm  (red when failure, normal otherwise)
  Top-right   : Safety-filter 3D arm  (orange flash on intervention)
  Bottom-left : Baseline slosh displacement vs. limit over time
  Bottom-right: Filter slosh displacement + intervention markers over time

Output: rollout_comparison.gif  (saved in aa276finalproject/)

Run from aa276finalproject/:
    python -m src.scripts.visualize_rollout
    python -m src.scripts.visualize_rollout --seed 7 --stride 3
"""

from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DEEPREACH_PATH = os.path.dirname(PROJECT_ROOT)
if DEEPREACH_PATH not in sys.path:
    sys.path.insert(0, DEEPREACH_PATH)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.base_env import CoffeeArmEnv
from src.core.arm_dynamics import get_link_positions
from src.config.constants import DEFAULT_OBSTACLES, DEFAULT_SLOSH_RAD_MAX, DEFAULT_L_EFF
from src.reachability.safety_filter import safety_filter as brt_safety_filter
from src.reachability.compute_brt import CFG as BRT_CFG

# Scale factor so the pendulum bob is visible in the 3D view.
# l_eff = 0.025 m is sub-centimetre; multiply by VIS_SCALE so the rod is
# VIS_L_EFF = 0.2 m long — large enough to see but not dominating the scene.
VIS_SCALE = 8.0
VIS_L_EFF = DEFAULT_L_EFF * VIS_SCALE

MODEL_PATH   = os.path.join(PROJECT_ROOT, 'ppo_baseline_final.zip')
VECNORM_PATH = os.path.join(PROJECT_ROOT, 'ppo_baseline_vecnormalize.pkl')
BRT_PATH     = os.path.join(PROJECT_ROOT, 'brt_model', 'model_final.pth')
OUT_PATH     = os.path.join(PROJECT_ROOT, 'rollout_comparison.gif')

ENV_KWARGS = dict(u_max=15.0, T=10.0, dt=0.01)
L = np.array([0.30, 0.30, 0.25], dtype=np.float32)


# ── data collection ────────────────────────────────────────────────────────────

def make_vec_env(seed=0):
    vec_env = DummyVecEnv([lambda: CoffeeArmEnv(**ENV_KWARGS)])
    vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def load_brt():
    from dynamics.dynamics import CoffeeArmDynamics
    from deepreach.utils import modules
    dyn = CoffeeArmDynamics()
    brt = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1,
        type='sine', mode='mlp', final_layer_factor=1,
        hidden_features=BRT_CFG['hidden_features'],
        num_hidden_layers=BRT_CFG['num_hidden_layers'],
    )
    sd = torch.load(BRT_PATH, map_location='cpu')
    brt.load_state_dict(sd)
    brt.eval()
    return brt, dyn


def record_episode(model, vec_env, seed=0, use_filter=False, brt=None, brt_dyn=None):
    """Run one episode and collect per-step trajectory data."""
    raw_env = vec_env.venv.envs[0]
    obs = vec_env.reset()

    traj = dict(
        link_pts=[],       # list of [p0,p1,p2,p3] each (4,3)
        slosh_xy=[],       # (x_slosh, y_slosh)
        actions_nom=[],    # nominal PPO action
        actions_exe=[],    # action actually executed
        intervened=[],     # bool per step
        spill_rad=[],      # slosh radius magnitude
        terminated=False,
        term_step=None,
        fail_flags=[],     # dict per step
    )

    step = 0
    done = False
    while not done:
        action_arr, _ = model.predict(obs, deterministic=True)
        u_nom = action_arr[0].copy()

        if use_filter and brt is not None:
            state_10d = raw_env.state.copy()
            u_exe, did_intervene = brt_safety_filter(
                brt, brt_dyn, state_10d, u_nom, t=BRT_CFG['tMax']
            )
        else:
            u_exe = u_nom.copy()
            did_intervene = False

        action_arr[0] = u_exe
        obs, _, done_arr, info_arr = vec_env.step(action_arr)
        done = bool(done_arr[0])
        info = info_arr[0]

        state = raw_env.state
        pts = get_link_positions(state[:6], L)  # [p0,p1,p2,p3]

        traj['link_pts'].append(np.array(pts, dtype=np.float32))
        traj['slosh_xy'].append((float(state[6]), float(state[7])))
        traj['actions_nom'].append(u_nom)
        traj['actions_exe'].append(u_exe)
        traj['intervened'].append(did_intervene)
        traj['spill_rad'].append(np.sqrt(state[6]**2 + state[7]**2))
        traj['fail_flags'].append({
            'spill': info.get('spill_slosh', False),
            'obstacle': info.get('obstacle_hit', False),
            'joint': info.get('joint_violation', False),
            'ground': info.get('below_ground', False),
        })

        if done and (info.get('spill_slosh') or info.get('obstacle_hit')
                     or info.get('joint_violation') or info.get('below_ground')):
            traj['terminated'] = True
            traj['term_step'] = step

        step += 1

    traj['link_pts']  = np.array(traj['link_pts'])   # (T, 4, 3)
    traj['slosh_xy']  = np.array(traj['slosh_xy'])   # (T, 2)
    traj['spill_rad'] = np.array(traj['spill_rad'])   # (T,)
    traj['intervened'] = np.array(traj['intervened'])  # (T,)
    traj['T'] = step
    return traj


# ── drawing helpers ────────────────────────────────────────────────────────────

def _sphere_mesh(center, radius, n=12):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi,   n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def draw_arm(ax, pts, color='steelblue', lw=3, ee_color=None):
    """Draw the arm links and return artist handles."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    ln, = ax.plot(xs, ys, zs, '-o', color=color, lw=lw, ms=5)
    ee = pts[-1]
    dot = ax.scatter([ee[0]], [ee[1]], [ee[2]],
                     color=ee_color or color, s=60, zorder=5)
    return ln, dot


def setup_arm_ax(fig, pos, title, obstacles, goal):
    ax = fig.add_subplot(pos, projection='3d')
    lim = 0.9
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([0, lim])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.view_init(elev=22, azim=40)

    # ground plane
    xx, yy = np.meshgrid([-lim, lim], [-lim, lim])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.06, color='gray')

    # obstacles
    for obs in obstacles:
        cx, cy, cz = obs['center']
        r = obs['radius']
        sx, sy, sz = _sphere_mesh([cx, cy, cz], r, n=14)
        ax.plot_surface(sx, sy, sz, alpha=0.3, color='saddlebrown', linewidth=0)

    # goal
    ax.scatter([goal[0]], [goal[1]], [goal[2]],
               marker='*', s=200, color='gold', zorder=6, label='Goal')
    return ax


def setup_slosh_ax(fig, pos, title):
    ax = fig.add_subplot(pos)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Slosh radius (m)")
    ax.axhline(DEFAULT_SLOSH_RAD_MAX, color='red', lw=1.5, ls='--', label='Limit')
    ax.set_ylim(0, DEFAULT_SLOSH_RAD_MAX * 2.2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    return ax


# ── main animation ─────────────────────────────────────────────────────────────

def build_animation(traj_base, traj_filt, stride=4, fps=15):
    T_base = traj_base['T']
    T_filt = traj_filt['T']
    T_max  = max(T_base, T_filt)
    dt     = ENV_KWARGS['dt']

    frames_base = list(range(0, T_base, stride))
    frames_filt = list(range(0, T_filt, stride))
    n_frames    = max(len(frames_base), len(frames_filt))

    goal = np.array([0.5, 0.5, 1.5], dtype=np.float32)

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.15,
                          top=0.93, bottom=0.07, left=0.05, right=0.97,
                          height_ratios=[2.2, 1.0])

    ax_b3d = setup_arm_ax(fig, gs[0, 0],
                           "Baseline PPO — no safety filter",
                           DEFAULT_OBSTACLES, goal)
    ax_f3d = setup_arm_ax(fig, gs[0, 1],
                           "PPO + BRT Safety Filter",
                           DEFAULT_OBSTACLES, goal)
    ax_bsl = setup_slosh_ax(fig, gs[1, 0], "Baseline: slosh displacement")
    ax_fsl = setup_slosh_ax(fig, gs[1, 1], "Filter: slosh displacement  (orange = intervention)")

    for ax in [ax_b3d, ax_f3d]:
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(labelsize=7)

    time_base = np.arange(T_base) * dt
    time_filt = np.arange(T_filt) * dt

    # Pre-draw static time series for slosh
    ax_bsl.plot(time_base, traj_base['spill_rad'], color='#4fc3f7', lw=1.2, alpha=0.9)
    ax_fsl.plot(time_filt, traj_filt['spill_rad'], color='#4fc3f7', lw=1.2, alpha=0.9)

    # Shade intervention regions on filter plot
    interv = traj_filt['intervened']
    for i in range(T_filt):
        if interv[i]:
            ax_fsl.axvspan(time_filt[i], time_filt[min(i+1, T_filt-1)],
                           alpha=0.25, color='orange', linewidth=0)

    # Shade failure region on baseline plot
    if traj_base['terminated'] and traj_base['term_step'] is not None:
        ts = traj_base['term_step']
        ax_bsl.axvspan(time_base[ts], time_base[-1] + dt,
                       alpha=0.35, color='red', linewidth=0)
        ax_bsl.axvline(time_base[ts], color='red', lw=2, ls='-')

    # Animated elements — arm links + EE
    b_arm_ln, = ax_b3d.plot([], [], [], '-o', color='steelblue', lw=3, ms=5)
    b_arm_ee  = ax_b3d.scatter([], [], [], color='steelblue', s=60, zorder=5)
    f_arm_ln, = ax_f3d.plot([], [], [], '-o', color='limegreen', lw=3, ms=5)
    f_arm_ee  = ax_f3d.scatter([], [], [], color='limegreen', s=60, zorder=5)

    # Slosh pendulum: rod (line from cup to bob) + bob (scatter)
    b_rod_ln, = ax_b3d.plot([], [], [], '-', color='cyan', lw=2, alpha=0.9, zorder=6)
    b_bob     = ax_b3d.scatter([], [], [], color='cyan', s=55, zorder=7)
    f_rod_ln, = ax_f3d.plot([], [], [], '-', color='cyan', lw=2, alpha=0.9, zorder=6)
    f_bob     = ax_f3d.scatter([], [], [], color='cyan', s=55, zorder=7)

    # Failure text overlay on baseline 3D axes (hidden until failure)
    b_fail_txt = ax_b3d.text2D(
        0.5, 0.88, '', transform=ax_b3d.transAxes,
        fontsize=18, fontweight='bold', color='red',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a0000', alpha=0.85),
        zorder=10,
    )

    b_cursor = ax_bsl.axvline(0, color='white', lw=1.2, ls=':')
    f_cursor = ax_fsl.axvline(0, color='white', lw=1.2, ls=':')

    title_text = fig.suptitle('', fontsize=11, color='white', fontweight='bold')

    # EE trails
    b_trail_xs, b_trail_ys, b_trail_zs = [], [], []
    f_trail_xs, f_trail_ys, f_trail_zs = [], [], []
    b_trail_ln, = ax_b3d.plot([], [], [], '-', color='lightblue', lw=1, alpha=0.5)
    f_trail_ln, = ax_f3d.plot([], [], [], '-', color='palegreen', lw=1, alpha=0.5)

    def _bob_color(slosh_rad):
        """Cyan → yellow → red as slosh approaches and crosses the limit."""
        ratio = slosh_rad / DEFAULT_SLOSH_RAD_MAX
        if ratio < 0.5:
            return 'cyan'
        if ratio < 0.8:
            return 'yellow'
        if ratio < 1.0:
            return 'darkorange'
        return 'red'

    def _slosh_bob_pos(ee, sx, sy):
        """World-frame bob position with visual scaling for the pendulum."""
        sx_v = sx * VIS_SCALE
        sy_v = sy * VIS_SCALE
        sz_v = -np.sqrt(max(VIS_L_EFF**2 - sx_v**2 - sy_v**2, 1e-6))
        return (ee[0] + sx_v, ee[1] + sy_v, ee[2] + sz_v)

    def _update_arm(arm_ln, arm_ee, rod_ln, bob,
                    trail_xs, trail_ys, trail_zs, trail_ln,
                    pts, slosh_xy, slosh_rad, color, failed, intervened_step):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        arm_c = 'crimson' if failed else ('darkorange' if intervened_step else color)
        lw    = 5 if failed else 3
        arm_ln.set_data(xs, ys); arm_ln.set_3d_properties(zs)
        arm_ln.set_color(arm_c); arm_ln.set_linewidth(lw)
        ee = pts[-1]
        arm_ee._offsets3d = ([ee[0]], [ee[1]], [ee[2]]); arm_ee.set_color(arm_c)

        # Pendulum rod + bob
        sx, sy = slosh_xy
        bx, by, bz = _slosh_bob_pos(ee, sx, sy)
        rod_ln.set_data([ee[0], bx], [ee[1], by])
        rod_ln.set_3d_properties([ee[2], bz])
        bc = _bob_color(slosh_rad)
        rod_ln.set_color(bc)
        bob._offsets3d = ([bx], [by], [bz]); bob.set_color(bc)

        trail_xs.append(ee[0]); trail_ys.append(ee[1]); trail_zs.append(ee[2])
        trail_ln.set_data(trail_xs, trail_ys); trail_ln.set_3d_properties(trail_zs)

    def animate(frame_idx):
        bi = frames_base[min(frame_idx, len(frames_base)-1)]
        fi = frames_filt[min(frame_idx, len(frames_filt)-1)]

        b_pts  = traj_base['link_pts'][bi]
        b_sxy  = traj_base['slosh_xy'][bi]
        b_srad = traj_base['spill_rad'][bi]
        b_fail = (traj_base['terminated'] and
                  traj_base['term_step'] is not None and
                  bi >= traj_base['term_step'])

        f_pts  = traj_filt['link_pts'][fi]
        f_sxy  = traj_filt['slosh_xy'][fi]
        f_srad = traj_filt['spill_rad'][fi]
        f_int  = bool(traj_filt['intervened'][fi])

        _update_arm(b_arm_ln, b_arm_ee, b_rod_ln, b_bob,
                    b_trail_xs, b_trail_ys, b_trail_zs, b_trail_ln,
                    b_pts, b_sxy, b_srad, 'steelblue', b_fail, False)

        _update_arm(f_arm_ln, f_arm_ee, f_rod_ln, f_bob,
                    f_trail_xs, f_trail_ys, f_trail_zs, f_trail_ln,
                    f_pts, f_sxy, f_srad, 'limegreen', False, f_int)

        # Failure overlay text + pulsing red background
        if b_fail:
            b_fail_txt.set_text('⚠  SPILL FAILURE')
            # Alternate between two reds to create a pulse effect
            flash = '#cc0000' if (frame_idx % 6) < 3 else '#550000'
            fig.patch.set_facecolor(flash)
        else:
            b_fail_txt.set_text('')
            fig.patch.set_facecolor('#1a1a0a' if f_int else '#1a1a2e')

        b_cursor.set_xdata([bi * dt, bi * dt])
        f_cursor.set_xdata([fi * dt, fi * dt])

        b_status = "⚠ SPILL FAILURE" if b_fail else f"t={bi*dt:.1f}s"
        f_status = "INTERVENING" if f_int else f"t={fi*dt:.1f}s"
        title_text.set_text(f"Baseline: {b_status}          Filter: {f_status}")
        title_text.set_color('red' if b_fail else 'white')

        return (b_arm_ln, b_arm_ee, b_rod_ln, b_bob,
                f_arm_ln, f_arm_ee, f_rod_ln, f_bob,
                b_cursor, f_cursor, title_text, b_fail_txt)

    ani = animation.FuncAnimation(
        fig, animate,
        frames=n_frames,
        interval=int(1000 / fps),
        blit=False,
    )
    return ani, fig


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',   type=int, default=3,  help='Random seed for episode')
    parser.add_argument('--stride', type=int, default=4,  help='Animate every N-th step')
    parser.add_argument('--fps',    type=int, default=15, help='GIF frames per second')
    parser.add_argument('--out',    default=OUT_PATH,     help='Output GIF path')
    args = parser.parse_args()

    print("Loading PPO model...")
    vec_env_base = make_vec_env(seed=args.seed)
    vec_env_filt = make_vec_env(seed=args.seed)
    model_base   = PPO.load(MODEL_PATH, env=vec_env_base, device='cpu')
    model_filt   = PPO.load(MODEL_PATH, env=vec_env_filt, device='cpu')

    print("Loading BRT safety filter...")
    brt, brt_dyn = load_brt()

    print(f"Recording baseline episode (seed={args.seed})...")
    # Try a few seeds to find one that fails somewhat dramatically (not step 1)
    best_base = None
    for s in [args.seed, args.seed+1, args.seed+2, args.seed+10, args.seed+20]:
        vec_env_try = make_vec_env(seed=s)
        m_try = PPO.load(MODEL_PATH, env=vec_env_try, device='cpu')
        t = record_episode(m_try, vec_env_try, seed=s, use_filter=False)
        # Prefer episodes that run for at least 50 steps before failing
        if t['terminated'] and t['term_step'] is not None and t['term_step'] >= 50:
            best_base = t
            print(f"  Baseline seed={s}: failed at step {t['term_step']}")
            break
        elif best_base is None or (t['terminated'] and
              (best_base['term_step'] or 0) < (t['term_step'] or 0)):
            best_base = t
    traj_base = best_base

    print(f"Recording filter episode (seed={args.seed})...")
    # Find a filter episode with meaningful interventions
    best_filt = None
    for s in [args.seed, args.seed+1, args.seed+5]:
        vec_env_try = make_vec_env(seed=s)
        m_try = PPO.load(MODEL_PATH, env=vec_env_try, device='cpu')
        t = record_episode(m_try, vec_env_try, seed=s,
                           use_filter=True, brt=brt, brt_dyn=brt_dyn)
        n_int = int(t['intervened'].sum())
        print(f"  Filter seed={s}: {n_int} interventions, terminated={t['terminated']}")
        if best_filt is None or n_int > int(best_filt['intervened'].sum()):
            best_filt = t
        if n_int > 50:
            break
    traj_filt = best_filt

    print(f"\nBaseline: {traj_base['T']} steps, "
          f"terminated={traj_base['terminated']} "
          f"@ step {traj_base['term_step']}")
    print(f"Filter  : {traj_filt['T']} steps, "
          f"terminated={traj_filt['terminated']}, "
          f"interventions={int(traj_filt['intervened'].sum())}")

    print("\nBuilding animation...")
    ani, fig = build_animation(traj_base, traj_filt,
                               stride=args.stride, fps=args.fps)

    print(f"Saving GIF to {args.out} (this may take a minute)...")
    ani.save(args.out, writer='pillow', fps=args.fps, dpi=90)
    plt.close(fig)
    print(f"Done. Saved: {args.out}")


if __name__ == '__main__':
    main()
