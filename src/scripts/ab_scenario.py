"""A→B cup transport scenario: scripted PD trajectory + BRT safety filter.

The arm starts with the cup at point A and delivers it to point B while keeping
the liquid inside the safe slosh set.  The scripted controller tracks a
two-phase joint-space trajectory that is deliberately fast enough to cause
spills without the filter.

Output: ab_scenario.gif  (side-by-side baseline vs. safety filter)

Run from aa276finalproject/:
    python -m src.scripts.ab_scenario
"""

from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DEEPREACH_PATH = os.path.dirname(PROJECT_ROOT)
if DEEPREACH_PATH not in sys.path:
    sys.path.insert(0, DEEPREACH_PATH)

import torch
from src.envs.obstacle_env import CoffeeArmEnv
from src.core.arm_dynamics import get_link_positions, position_cup
from src.config.constants import (
    DEFAULT_OBSTACLES, DEFAULT_SLOSH_RAD_MAX, DEFAULT_L_EFF, DEFAULT_L,
)
from src.reachability.safety_filter import safety_filter as brt_filter
from src.reachability.compute_brt import CFG as BRT_CFG

PPO_MODEL_PATH   = os.path.join(PROJECT_ROOT, 'ppo_baseline_final.zip')
PPO_VECNORM_PATH = os.path.join(PROJECT_ROOT, 'checkpoints', 'baseline',
                                'ppo_baseline_vecnormalize_2000000_steps.pkl')

OUT_PATH = os.path.join(PROJECT_ROOT, 'ab_scenario.gif')
L = DEFAULT_L.astype(np.float64)

# ── Scenario definition ────────────────────────────────────────────────────────
# IK solutions for A = (0.35, -0.25, 0.45 m) and B ≈ (0.35, 0.40, 0.45 m)
Q_A    = np.array([-0.620, -0.200,  1.195], dtype=np.float32)
Q_B    = np.array([ 0.852,  0.275,  0.000], dtype=np.float32)
# Via-point: swing slightly up to clear the obstacle zone
Q_MID  = (Q_A + Q_B) / 2 + np.array([0.0, 0.0, 0.30], dtype=np.float32)
GOAL_B = position_cup(np.concatenate([Q_B, np.zeros(3)]), L).astype(np.float32)
POS_A  = position_cup(np.concatenate([Q_A, np.zeros(3)]), L).astype(np.float32)

# Trajectory timing / controller gains
T1, T2 = 1.0, 1.0          # seconds for each phase (fast enough to cause slosh)
KP, KD = 40.0, 5.0
ENV_KWARGS = dict(u_max=15.0, T=10.0, dt=0.01)
DT = ENV_KWARGS['dt']

# Visual scale for slosh pendulum bob (l_eff is 25 mm; scale ×8 to be visible)
VIS_SCALE = 8.0
VIS_L_EFF = DEFAULT_L_EFF * VIS_SCALE


# ── PPO controller ─────────────────────────────────────────────────────────────

def load_ppo():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    print(f"Loading PPO from {PPO_MODEL_PATH}")
    dummy = DummyVecEnv([lambda: CoffeeArmEnv(**ENV_KWARGS)])
    venv = VecNormalize.load(PPO_VECNORM_PATH, dummy)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(PPO_MODEL_PATH)
    return model, venv

def make_ppo_controller(model, venv, goal_pos):
    def ppo_u(state: np.ndarray, t: float) -> np.ndarray:
        cup_pos = position_cup(state[:6], L)
        raw_obs = np.concatenate([state, goal_pos - cup_pos]).reshape(1, -1).astype(np.float32)
        norm_obs = venv.normalize_obs(raw_obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        return np.clip(action.flatten().astype(np.float32), -15.0, 15.0)
    return ppo_u


# ── Scripted PD controller ─────────────────────────────────────────────────────

def scripted_u(state: np.ndarray, t: float) -> np.ndarray:
    """Two-phase joint-space PD: A → via-point → B."""
    if t < T1:
        alpha  = t / T1
        q_ref  = Q_A + alpha * (Q_MID - Q_A)
        dq_ref = (Q_MID - Q_A) / T1
    elif t < T1 + T2:
        alpha  = (t - T1) / T2
        q_ref  = Q_MID + alpha * (Q_B - Q_MID)
        dq_ref = (Q_B - Q_MID) / T2
    else:
        q_ref  = Q_B
        dq_ref = np.zeros(3, dtype=np.float32)

    q = state[:3].astype(np.float32)
    dq = state[3:6].astype(np.float32)
    u = KP * (q_ref - q) + KD * (dq_ref - dq)
    return np.clip(u, -15.0, 15.0).astype(np.float32)


# ── Episode recording ──────────────────────────────────────────────────────────

def record_episode(use_filter: bool, brt=None, brt_dyn=None, controller=None):
    env = CoffeeArmEnv(**ENV_KWARGS)
    env.reset(seed=0)
    env.state = np.concatenate([Q_A, np.zeros(7)]).astype(np.float32)
    env.goal_pos = GOAL_B.copy()

    ctrl = controller if controller is not None else scripted_u

    traj = dict(
        link_pts=[], slosh_xy=[], spill_rad=[],
        intervened=[], actions_nom=[], actions_exe=[],
        terminated=False, term_step=None, fail_cause=None, T=0,
    )

    done = False
    step = 0
    while not done:
        t = step * DT
        u_nom = ctrl(env.state, t)

        if use_filter and brt is not None:
            u_exe, did_intervene = brt_filter(
                brt, brt_dyn, env.state.copy(), u_nom, t=BRT_CFG['tMax']
            )
        else:
            u_exe = u_nom.copy()
            did_intervene = False

        _, _, term, trunc, info = env.step(u_exe)
        done = bool(term or trunc)

        pts = get_link_positions(env.state[:6], L)
        traj['link_pts'].append(np.array(pts, dtype=np.float32))
        traj['slosh_xy'].append((float(env.state[6]), float(env.state[7])))
        traj['spill_rad'].append(np.sqrt(env.state[6]**2 + env.state[7]**2))
        traj['intervened'].append(did_intervene)
        traj['actions_nom'].append(u_nom)
        traj['actions_exe'].append(u_exe)

        if term and (info.get('spill_slosh') or info.get('obstacle_hit') or
                     info.get('joint_violation') or info.get('below_ground')):
            traj['terminated'] = True
            traj['term_step']  = step
            traj['fail_cause'] = (
                f"spill={info.get('spill_slosh',False)}  "
                f"ground={info.get('below_ground',False)}  "
                f"obstacle={info.get('obstacle_hit',False)}  "
                f"joint={info.get('joint_violation',False)}"
            )

        step += 1

    traj['link_pts']   = np.array(traj['link_pts'])
    traj['slosh_xy']   = np.array(traj['slosh_xy'])
    traj['spill_rad']  = np.array(traj['spill_rad'])
    traj['intervened'] = np.array(traj['intervened'])
    traj['T']          = step
    return traj


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _sphere_mesh(center, radius, n=14):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _bob_color(slosh_rad):
    r = slosh_rad / DEFAULT_SLOSH_RAD_MAX
    if r < 0.5:  return 'cyan'
    if r < 0.8:  return 'yellow'
    if r < 1.0:  return 'darkorange'
    return 'red'


def _slosh_bob_pos(ee, sx, sy):
    sx_v = sx * VIS_SCALE
    sy_v = sy * VIS_SCALE
    sz_v = -np.sqrt(max(VIS_L_EFF**2 - sx_v**2 - sy_v**2, 1e-6))
    return (ee[0] + sx_v, ee[1] + sy_v, ee[2] + sz_v)


def _ref_cup_path():
    """Precompute the reference cup-tip path for the annotated dashed line."""
    pts = []
    for step in range(int(ENV_KWARGS['T'] / DT)):
        t = step * DT
        if t < T1:
            alpha = t / T1
            q = Q_A + alpha * (Q_MID - Q_A)
        elif t < T1 + T2:
            alpha = (t - T1) / T2
            q = Q_MID + alpha * (Q_B - Q_MID)
        else:
            q = Q_B
        state = np.concatenate([q, np.zeros(3)])
        pts.append(position_cup(state, L))
    return np.array(pts)


def setup_ax(fig, pos, title):
    ax = fig.add_subplot(pos, projection='3d')
    lim = 0.9
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([0, lim])
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=9, fontweight='bold', color='white')
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(labelsize=7, colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.view_init(elev=22, azim=40)

    # Ground plane
    xx, yy = np.meshgrid([-lim, lim], [-lim, lim])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.06, color='gray')

    # Obstacles
    for obs in DEFAULT_OBSTACLES:
        sx, sy, sz = _sphere_mesh(obs['center'], obs['radius'])
        ax.plot_surface(sx, sy, sz, alpha=0.35, color='saddlebrown', linewidth=0)

    # Reference cup path (dashed white line)
    ref = _ref_cup_path()
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], '--', color='white',
            lw=1.0, alpha=0.4, label='Reference path')

    # Point A marker
    ax.scatter([POS_A[0]], [POS_A[1]], [POS_A[2]],
               marker='o', s=120, color='deepskyblue', zorder=7, label='A (start)')
    ax.text(POS_A[0]+0.03, POS_A[1]+0.03, POS_A[2]+0.03,
            'A', color='deepskyblue', fontsize=9, fontweight='bold')

    # Point B marker
    ax.scatter([GOAL_B[0]], [GOAL_B[1]], [GOAL_B[2]],
               marker='*', s=200, color='gold', zorder=7, label='B (goal)')
    ax.text(GOAL_B[0]+0.03, GOAL_B[1]+0.03, GOAL_B[2]+0.03,
            'B', color='gold', fontsize=9, fontweight='bold')

    ax.legend(fontsize=7, loc='upper left', facecolor='#1a1a2e',
              labelcolor='white', framealpha=0.7)
    return ax


def setup_slosh_ax(fig, pos, title, traj, color_shade):
    ax = fig.add_subplot(pos)
    ax.set_facecolor('#0d0d1a')
    ax.set_title(title, fontsize=9, fontweight='bold', color='white')
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Slosh radius (m)", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    ax.axhline(DEFAULT_SLOSH_RAD_MAX, color='red', lw=1.5, ls='--', label='Spill limit')
    ax.set_ylim(0, DEFAULT_SLOSH_RAD_MAX * 2.4)
    ax.grid(True, alpha=0.2, color='white')

    time_ax = np.arange(traj['T']) * DT
    ax.plot(time_ax, traj['spill_rad'], color=color_shade, lw=1.2, alpha=0.9)

    # Phase markers
    ax.axvline(T1,      color='white', lw=0.8, ls=':', alpha=0.6)
    ax.axvline(T1 + T2, color='white', lw=0.8, ls=':', alpha=0.6)
    ax.text(T1/2,       DEFAULT_SLOSH_RAD_MAX*2.1, 'A→mid', color='white', fontsize=6, ha='center')
    ax.text(T1+T2/2,    DEFAULT_SLOSH_RAD_MAX*2.1, 'mid→B', color='white', fontsize=6, ha='center')
    ax.text(T1+T2+0.3,  DEFAULT_SLOSH_RAD_MAX*2.1, 'hold',  color='white', fontsize=6, ha='left')

    if traj['terminated'] and traj['term_step'] is not None:
        ts = traj['term_step']
        ax.axvspan(ts*DT, traj['T']*DT, alpha=0.3, color='red', linewidth=0)
        ax.axvline(ts*DT, color='red', lw=2.0)

    interv = traj['intervened']
    time_ax_full = np.arange(traj['T']) * DT
    for i in range(traj['T']):
        if interv[i]:
            ax.axvspan(time_ax_full[i], time_ax_full[min(i+1, traj['T']-1)],
                       alpha=0.3, color='orange', linewidth=0)

    ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.7)
    return ax


# ── Main animation ─────────────────────────────────────────────────────────────

def build_animation(traj_base, traj_filt, stride=4, fps=15, ctrl_label="Scripted PD"):
    frames_base = list(range(0, traj_base['T'], stride))
    frames_filt = list(range(0, traj_filt['T'], stride))
    n_frames    = max(len(frames_base), len(frames_filt))

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.12,
                          top=0.93, bottom=0.07, left=0.05, right=0.97,
                          height_ratios=[2.2, 1.0])

    ax_b = setup_ax(fig, gs[0, 0], f"Baseline ({ctrl_label}) — A→B (no safety filter)")
    ax_f = setup_ax(fig, gs[0, 1], f"BRT Safety Filter ({ctrl_label}) — A→B")
    ax_bsl = setup_slosh_ax(fig, gs[1, 0], "Baseline: slosh displacement", traj_base, '#4fc3f7')
    ax_fsl = setup_slosh_ax(fig, gs[1, 1], "Filter: slosh  (orange = intervention)", traj_filt, '#81c784')

    # ── Animated arm artists ──
    b_arm_ln, = ax_b.plot([], [], [], '-o', color='steelblue', lw=3, ms=5)
    b_arm_ee  = ax_b.scatter([], [], [], color='steelblue', s=60, zorder=5)
    b_rod_ln, = ax_b.plot([], [], [], '-',  color='cyan', lw=2, alpha=0.9, zorder=6)
    b_bob     = ax_b.scatter([], [], [], color='cyan', s=55, zorder=7)

    f_arm_ln, = ax_f.plot([], [], [], '-o', color='limegreen', lw=3, ms=5)
    f_arm_ee  = ax_f.scatter([], [], [], color='limegreen', s=60, zorder=5)
    f_rod_ln, = ax_f.plot([], [], [], '-',  color='cyan', lw=2, alpha=0.9, zorder=6)
    f_bob     = ax_f.scatter([], [], [], color='cyan', s=55, zorder=7)

    # Failure text overlay on baseline panel
    b_fail_txt = ax_b.text2D(
        0.5, 0.88, '', transform=ax_b.transAxes,
        fontsize=17, fontweight='bold', color='red', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a0000', alpha=0.85),
        zorder=10,
    )

    # EE trail lines
    b_tr_xs, b_tr_ys, b_tr_zs = [], [], []
    f_tr_xs, f_tr_ys, f_tr_zs = [], [], []
    b_trail_ln, = ax_b.plot([], [], [], '-', color='lightblue', lw=1.0, alpha=0.5)
    f_trail_ln, = ax_f.plot([], [], [], '-', color='palegreen', lw=1.0, alpha=0.5)

    b_cursor = ax_bsl.axvline(0, color='white', lw=1.2, ls=':')
    f_cursor = ax_fsl.axvline(0, color='white', lw=1.2, ls=':')

    title_txt = fig.suptitle('', fontsize=11, fontweight='bold', color='white')

    def _update(arm_ln, arm_ee, rod_ln, bob,
                tr_xs, tr_ys, tr_zs, trail_ln,
                pts, slosh_xy, slosh_rad, color, failed, intervened):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        c  = 'crimson' if failed else ('darkorange' if intervened else color)
        lw = 5 if failed else 3
        arm_ln.set_data(xs, ys); arm_ln.set_3d_properties(zs)
        arm_ln.set_color(c); arm_ln.set_linewidth(lw)
        ee = pts[-1]
        arm_ee._offsets3d = ([ee[0]], [ee[1]], [ee[2]]); arm_ee.set_color(c)

        sx, sy = slosh_xy
        bx, by, bz = _slosh_bob_pos(ee, sx, sy)
        bc = _bob_color(slosh_rad)
        rod_ln.set_data([ee[0], bx], [ee[1], by]); rod_ln.set_3d_properties([ee[2], bz])
        rod_ln.set_color(bc)
        bob._offsets3d = ([bx], [by], [bz]); bob.set_color(bc)

        tr_xs.append(ee[0]); tr_ys.append(ee[1]); tr_zs.append(ee[2])
        trail_ln.set_data(tr_xs, tr_ys); trail_ln.set_3d_properties(tr_zs)

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

        _update(b_arm_ln, b_arm_ee, b_rod_ln, b_bob,
                b_tr_xs, b_tr_ys, b_tr_zs, b_trail_ln,
                b_pts, b_sxy, b_srad, 'steelblue', b_fail, False)

        _update(f_arm_ln, f_arm_ee, f_rod_ln, f_bob,
                f_tr_xs, f_tr_ys, f_tr_zs, f_trail_ln,
                f_pts, f_sxy, f_srad, 'limegreen', False, f_int)

        # Failure flash
        if b_fail:
            b_fail_txt.set_text('⚠  SPILL FAILURE')
            fig.patch.set_facecolor('#cc0000' if (frame_idx % 6) < 3 else '#550000')
        else:
            b_fail_txt.set_text('')
            fig.patch.set_facecolor('#1a1a0a' if f_int else '#1a1a2e')

        b_cursor.set_xdata([bi * DT, bi * DT])
        f_cursor.set_xdata([fi * DT, fi * DT])

        b_status = "⚠ SPILL FAILURE" if b_fail else f"t={bi*DT:.1f}s"
        f_status = "INTERVENING" if f_int else f"t={fi*DT:.1f}s"
        title_txt.set_text(
            f"Cup transport A→B  |  Baseline: {b_status}    Filter: {f_status}"
        )
        title_txt.set_color('red' if b_fail else 'white')

        return (b_arm_ln, b_arm_ee, b_rod_ln, b_bob,
                f_arm_ln, f_arm_ee, f_rod_ln, f_bob,
                b_cursor, f_cursor, title_txt, b_fail_txt)

    ani = animation.FuncAnimation(fig, animate, frames=n_frames,
                                  interval=int(1000 / fps), blit=False)
    return ani, fig


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride',  type=int,            default=3,       help='Animate every N-th step')
    parser.add_argument('--fps',     type=int,            default=20,      help='GIF frames per second')
    parser.add_argument('--out',     default=OUT_PATH)
    parser.add_argument('--use-ppo', action='store_true', default=False,   help='Use PPO as nominal controller instead of scripted PD')
    args = parser.parse_args()

    # Load BRT
    print("Loading BRT safety filter...")
    from dynamics.dynamics import CoffeeArmDynamics
    from deepreach.utils import modules
    brt_dyn = CoffeeArmDynamics()
    brt = modules.SingleBVPNet(
        in_features=brt_dyn.input_dim, out_features=1,
        type='sine', mode='mlp', final_layer_factor=1,
        hidden_features=BRT_CFG['hidden_features'],
        num_hidden_layers=BRT_CFG['num_hidden_layers'],
    )
    brt.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, 'brt_model', 'model_final.pth'), map_location='cpu'))
    brt.eval()

    # Optionally load PPO
    controller = None
    ctrl_label = "Scripted PD"
    if args.use_ppo:
        ppo_model, ppo_venv = load_ppo()
        controller = make_ppo_controller(ppo_model, ppo_venv, GOAL_B)
        ctrl_label = "PPO"

    print(f"Controller: {ctrl_label}")
    print(f"Point A: {POS_A.round(3)}")
    print(f"Point B: {GOAL_B.round(3)}\n")

    print("Recording baseline episode (no filter)...")
    traj_base = record_episode(use_filter=False, controller=controller)
    print(f"  Steps={traj_base['T']}  terminated={traj_base['terminated']}"
          f"  @step={traj_base['term_step']}")
    if traj_base.get('fail_cause'):
        print(f"  Cause: {traj_base['fail_cause']}")

    print("Recording filtered episode...")
    traj_filt = record_episode(use_filter=True, brt=brt, brt_dyn=brt_dyn, controller=controller)
    n_int = int(traj_filt['intervened'].sum())
    print(f"  Steps={traj_filt['T']}  terminated={traj_filt['terminated']}"
          f"  interventions={n_int} ({100*n_int/traj_filt['T']:.1f}% of steps)")
    if traj_filt.get('fail_cause'):
        print(f"  Cause: {traj_filt['fail_cause']}")

    out_path = args.out
    if args.use_ppo and out_path == OUT_PATH:
        out_path = os.path.join(PROJECT_ROOT, 'ab_scenario_ppo.gif')

    print("\nBuilding animation...")
    ani, fig = build_animation(traj_base, traj_filt,
                               stride=args.stride, fps=args.fps, ctrl_label=ctrl_label)

    print(f"Saving GIF to {out_path}...")
    ani.save(out_path, writer='pillow', fps=args.fps, dpi=90)
    plt.close(fig)
    print(f"Done. Saved: {out_path}")


if __name__ == '__main__':
    main()
