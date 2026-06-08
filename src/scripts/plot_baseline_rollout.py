"""
Static paper figure: PPO baseline rollout with failure-set annotations.

Shows a representative episode from the 100k-step checkpoint (which has real
ground-contact failures) alongside several background episodes for context.

Panels:
  A  3D cup trajectory — trail colored blue → red as arm approaches ground
  B  Cup height (z) vs time — red shaded region = below-ground failure set
  C  Slosh radius vs time — red dashed line = spill threshold
  D  Distance to goal vs time — green dashed line = completion threshold

Failure-set entries (below_ground=True) are marked with a red vertical band
on every time-series panel and a red marker on the 3D trajectory.

Usage (from aa276finalproject/):
    python -m src.scripts.plot_baseline_rollout
    python -m src.scripts.plot_baseline_rollout --ckpt final  # use final model
    python -m src.scripts.plot_baseline_rollout --seed 5 --out figures/baseline.pdf
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D                   # noqa: F401

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.base_env import CoffeePouringEnv
from src.core.arm_dynamics import get_link_positions, position_cup
from src.config.constants import DEFAULT_SLOSH_RAD_MAX, DEFAULT_L

ENV_KWARGS  = dict(u_max=15.0, T=10.0, dt=0.01)
CKPT_BASE   = os.path.join(PROJECT_ROOT, 'checkpoints', 'baseline')
FINAL_MODEL = os.path.join(PROJECT_ROOT, 'ppo_baseline_final.zip')
FINAL_VN    = os.path.join(PROJECT_ROOT, 'ppo_baseline_vecnormalize.pkl')

COMPLETION_DIST = 0.1   # metres — task done
GROUND_Z        = 0.0   # metres — failure set boundary


def make_env():
    return CoffeePouringEnv(**ENV_KWARGS)


def load_model(ckpt: str):
    if ckpt == 'final':
        model    = PPO.load(FINAL_MODEL)
        vec_env  = VecNormalize.load(FINAL_VN, DummyVecEnv([make_env]))
    else:
        steps    = int(ckpt)
        model    = PPO.load(f'{CKPT_BASE}/ppo_baseline_{steps}_steps.zip')
        vec_env  = VecNormalize.load(
            f'{CKPT_BASE}/ppo_baseline_vecnormalize_{steps}_steps.pkl',
            DummyVecEnv([make_env])
        )
    vec_env.training  = False
    vec_env.norm_reward = False
    return model, vec_env


def record_episode(model, vec_env, seed: int, deterministic: bool = True):
    """Run one episode; return per-step data dict."""
    raw_env = vec_env.venv.envs[0]
    # Seed before VecNormalize calls env.reset() internally.
    raw_env._np_random = np.random.default_rng(seed)
    obs     = vec_env.reset()

    cup_pos, slosh_rad, dist_goal = [], [], []
    actions, below_ground_flag, spill_flag = [], [], []
    link_pts_all = []
    t = 0

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done_arr, info_arr = vec_env.step(action)
        done = bool(done_arr[0])
        info = info_arr[0]

        state = raw_env.state
        cup   = position_cup(state[:6], DEFAULT_L)
        lpts  = get_link_positions(state[:6], DEFAULT_L)

        cup_pos.append(cup.copy())
        slosh_rad.append(info.get('slosh_rad', 0.0))
        dist_goal.append(info.get('dist_to_goal', float('nan')))
        actions.append(action[0].copy())
        below_ground_flag.append(bool(info.get('below_ground', False)))
        spill_flag.append(bool(info.get('spill_slosh', False)))
        link_pts_all.append(np.array(lpts, dtype=np.float32))
        t += 1

    # Failure step: first step where a failure flag is True
    fail_step = None
    for i, (bg, sp) in enumerate(zip(below_ground_flag, spill_flag)):
        if bg or sp:
            fail_step = i
            break

    return dict(
        cup_pos           = np.array(cup_pos),          # (T, 3)
        slosh_rad         = np.array(slosh_rad),        # (T,)
        dist_goal         = np.array(dist_goal),        # (T,)
        actions           = np.array(actions),          # (T, 3)
        below_ground_flag = np.array(below_ground_flag),# (T,)
        spill_flag        = np.array(spill_flag),       # (T,)
        link_pts          = np.array(link_pts_all),     # (T, 4, 3)
        fail_step         = fail_step,
        T                 = t,
        dt                = ENV_KWARGS['dt'],
        goal              = raw_env.goal_pos.copy(),
        seed              = seed,
    )


def _time_axis(ep):
    return np.arange(ep['T']) * ep['dt']


def _failure_band(ax, ep, alpha=0.15):
    """Shade every time window where the arm is in the failure set."""
    flags = ep['below_ground_flag'] | ep['spill_flag']
    t     = _time_axis(ep)
    in_fail = False
    t_start = None
    for i, f in enumerate(flags):
        if f and not in_fail:
            t_start = t[i]
            in_fail = True
        elif not f and in_fail:
            ax.axvspan(t_start, t[i], color='red', alpha=alpha, linewidth=0)
            in_fail = False
    if in_fail:
        ax.axvspan(t_start, t[-1] + ep['dt'], color='red', alpha=alpha, linewidth=0)
    # Vertical dashed line at first failure
    if ep['fail_step'] is not None:
        ax.axvline(t[ep['fail_step']], color='red', lw=1.4, ls='--',
                   label='Failure set entry', zorder=5)


def make_figure(hero_ep, bg_eps, out_path):
    fig = plt.figure(figsize=(12, 9))
    fig.subplots_adjust(
        left=0.07, right=0.97, top=0.92, bottom=0.08,
        hspace=0.42, wspace=0.32,
    )

    # ── Grid: 2 rows × 2 cols, 3D plot spans top row ──────────────────────────
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, figure=fig,
                  hspace=0.48, wspace=0.32,
                  left=0.07, right=0.97, top=0.91, bottom=0.07)

    ax3d = fig.add_subplot(gs[0, :], projection='3d')   # spans both cols
    ax_z   = fig.add_subplot(gs[1, 0])
    ax_sl  = fig.add_subplot(gs[1, 1])
    ax_dg  = fig.add_subplot(gs[2, 0])
    ax_act = fig.add_subplot(gs[2, 1])

    hero_t    = _time_axis(hero_ep)
    hero_cup  = hero_ep['cup_pos']
    goal      = hero_ep['goal']
    SLOSH_MAX = DEFAULT_SLOSH_RAD_MAX

    # ── Color trail: blue → red approaching ground ─────────────────────────────
    # Hue based on z: 1.0 = high (blue), 0.0 = at ground (red)
    z_vals   = hero_cup[:, 2]
    z_safe   = 0.15                            # "comfortable" height
    hue_arr  = np.clip(z_vals / z_safe, 0.0, 1.0)

    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # 3D cup trail, colored
    pts_3d = hero_cup[:, None, :]              # (T, 1, 3)
    segs   = np.concatenate([pts_3d[:-1], pts_3d[1:]], axis=1)  # (T-1, 2, 3)
    cmap   = plt.get_cmap('RdYlBu')
    colors = cmap(hue_arr[:-1])
    lc = Line3DCollection(segs, colors=colors, linewidths=2.0, zorder=3)
    ax3d.add_collection3d(lc)

    # Start marker
    ax3d.scatter(*hero_cup[0], color='royalblue', s=80, zorder=5, label='Start')

    # Failure marker
    if hero_ep['fail_step'] is not None:
        fp = hero_cup[hero_ep['fail_step']]
        ax3d.scatter(*fp, color='red', s=150, marker='X', zorder=6,
                     label='Failure set entry')

    # Goal marker
    ax3d.scatter(*goal, color='gold', s=200, marker='*', zorder=6, label='Goal')

    # Goal sphere wireframe
    u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:8j]
    ax3d.plot_wireframe(
        goal[0] + COMPLETION_DIST*np.cos(u)*np.sin(v),
        goal[1] + COMPLETION_DIST*np.sin(u)*np.sin(v),
        goal[2] + COMPLETION_DIST*np.cos(v),
        color='gold', alpha=0.15, linewidth=0.5,
    )

    # Ground plane
    lim = 0.85
    xx, yy = np.meshgrid(np.linspace(-lim, lim, 3), np.linspace(-lim, lim, 3))
    ax3d.plot_surface(xx, yy, np.zeros_like(xx),
                      alpha=0.08, color='red', zorder=0)

    # Background episodes as thin gray trails
    for ep in bg_eps:
        ax3d.plot(ep['cup_pos'][:, 0], ep['cup_pos'][:, 1], ep['cup_pos'][:, 2],
                  color='gray', lw=0.6, alpha=0.3)

    ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(0, 0.9)
    ax3d.set_xlabel('X (m)', fontsize=8, labelpad=2)
    ax3d.set_ylabel('Y (m)', fontsize=8, labelpad=2)
    ax3d.set_zlabel('Z (m)', fontsize=8, labelpad=2)
    ax3d.set_title('(A)  Cup end-effector trajectory  (trail: blue = safe height, red = near ground)',
                   fontsize=9, fontweight='bold', pad=6)
    ax3d.view_init(elev=18, azim=45)
    ax3d.tick_params(labelsize=7)
    ax3d.legend(fontsize=7, loc='upper left', framealpha=0.7)

    colorbar_ax = fig.add_axes([0.97, 0.62, 0.012, 0.22])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, z_safe))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=colorbar_ax)
    cb.set_label('Cup Z (m)', fontsize=7, rotation=270, labelpad=10)
    cb.ax.tick_params(labelsize=6)

    # ── Panel B: cup Z vs time ─────────────────────────────────────────────────
    for ep in bg_eps:
        ax_z.plot(_time_axis(ep), ep['cup_pos'][:, 2],
                  color='gray', lw=0.7, alpha=0.3)

    ax_z.plot(hero_t, hero_cup[:, 2], color='steelblue', lw=1.8, zorder=4)
    ax_z.axhline(GROUND_Z, color='red', lw=1.5, ls='--', label='Ground (failure boundary)', zorder=3)
    ax_z.axhspan(-0.05, GROUND_Z, color='red', alpha=0.12, linewidth=0, zorder=2)

    _failure_band(ax_z, hero_ep)

    if hero_ep['fail_step'] is not None:
        t_f = hero_t[hero_ep['fail_step']]
        ax_z.annotate(
            'Cup below\nground!',
            xy=(t_f, hero_cup[hero_ep['fail_step'], 2]),
            xytext=(t_f - 1.0, 0.15),
            fontsize=7, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
        )

    ax_z.set_xlabel('Time (s)', fontsize=8)
    ax_z.set_ylabel('Cup Z (m)', fontsize=8)
    ax_z.set_title('(B)  Cup height', fontsize=9, fontweight='bold')
    ax_z.legend(fontsize=7, framealpha=0.7)
    ax_z.set_ylim(-0.05, max(hero_cup[:, 2].max() * 1.1, 0.5))
    ax_z.grid(True, alpha=0.3)
    ax_z.tick_params(labelsize=7)

    # ── Panel C: slosh radius vs time ──────────────────────────────────────────
    for ep in bg_eps:
        ax_sl.plot(_time_axis(ep), ep['slosh_rad'],
                   color='gray', lw=0.7, alpha=0.3)

    ax_sl.plot(hero_t, hero_ep['slosh_rad'], color='darkorange', lw=1.8, zorder=4)
    ax_sl.axhline(SLOSH_MAX, color='red', lw=1.5, ls='--',
                  label=f'Spill limit ({SLOSH_MAX*1000:.1f} mm)', zorder=3)
    ax_sl.axhspan(SLOSH_MAX, SLOSH_MAX * 2.5, color='red', alpha=0.10,
                  linewidth=0, zorder=2)

    _failure_band(ax_sl, hero_ep)

    ax_sl.set_xlabel('Time (s)', fontsize=8)
    ax_sl.set_ylabel('Slosh radius (m)', fontsize=8)
    ax_sl.set_title('(C)  Sloshing displacement', fontsize=9, fontweight='bold')
    ax_sl.legend(fontsize=7, framealpha=0.7)
    ax_sl.set_ylim(0, SLOSH_MAX * 2.5)
    ax_sl.grid(True, alpha=0.3)
    ax_sl.tick_params(labelsize=7)

    # ── Panel D: distance to goal ──────────────────────────────────────────────
    for ep in bg_eps:
        ax_dg.plot(_time_axis(ep), ep['dist_goal'],
                   color='gray', lw=0.7, alpha=0.3)

    ax_dg.plot(hero_t, hero_ep['dist_goal'], color='mediumseagreen', lw=1.8, zorder=4)
    ax_dg.axhline(COMPLETION_DIST, color='green', lw=1.5, ls='--',
                  label=f'Goal threshold ({COMPLETION_DIST} m)', zorder=3)

    _failure_band(ax_dg, hero_ep)

    ax_dg.set_xlabel('Time (s)', fontsize=8)
    ax_dg.set_ylabel('Distance to goal (m)', fontsize=8)
    ax_dg.set_title('(D)  Progress toward goal', fontsize=9, fontweight='bold')
    ax_dg.legend(fontsize=7, framealpha=0.7)
    ax_dg.set_ylim(0, None)
    ax_dg.grid(True, alpha=0.3)
    ax_dg.tick_params(labelsize=7)

    # ── Panel E: joint actions ─────────────────────────────────────────────────
    colors_act = ['#e07b54', '#6fa8dc', '#93c47d']
    labels_act = ['Joint 1', 'Joint 2', 'Joint 3']
    for j, (c, lbl) in enumerate(zip(colors_act, labels_act)):
        ax_act.plot(hero_t, hero_ep['actions'][:, j],
                    color=c, lw=1.2, alpha=0.85, label=lbl)

    _failure_band(ax_act, hero_ep)

    ax_act.axhline(0, color='black', lw=0.6, alpha=0.4)
    ax_act.set_xlabel('Time (s)', fontsize=8)
    ax_act.set_ylabel('Action (rad/s²)', fontsize=8)
    ax_act.set_title('(E)  Control inputs', fontsize=9, fontweight='bold')
    ax_act.legend(fontsize=7, framealpha=0.7, ncol=3, loc='upper right')
    ax_act.grid(True, alpha=0.3)
    ax_act.tick_params(labelsize=7)

    # ── Failure legend patch ───────────────────────────────────────────────────
    fail_patch = mpatches.Patch(color='red', alpha=0.3, label='Failure set  (below ground)')
    fig.legend(handles=[fail_patch], loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=8, framealpha=0.8)

    # Title
    ckpt_label = args_ckpt if 'args_ckpt' in dir() else '100k-step checkpoint'
    n_fail = int(hero_ep['below_ground_flag'].sum() + hero_ep['spill_flag'].sum())
    title = (f'PPO Baseline (no safety filter) — {ckpt_label}  |  '
             f'Seed {hero_ep["seed"]}  |  '
             f'{n_fail} failure-set step{"s" if n_fail != 1 else ""}')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.975)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    if out_path.endswith('.pdf') or True:
        pdf_path = out_path.replace('.png', '.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    if out_path.endswith('.png'):
        print(f'Saved: {pdf_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',  default='100000',
                        help='Checkpoint steps ("100000") or "final"')
    parser.add_argument('--seed',  type=int, default=1,
                        help='Hero episode seed (default: 1, 969-step ground failure)')
    parser.add_argument('--bg_seeds', type=str, default='0,2,3,4,5',
                        help='Comma-separated background episode seeds')
    parser.add_argument('--out',   default='figures/ppo_baseline_rollout.png',
                        help='Output path (.png — PDF also saved)')
    args = parser.parse_args()

    global args_ckpt
    args_ckpt = f'{args.ckpt}-step checkpoint' if args.ckpt != 'final' else 'final model'

    print(f'Loading {args.ckpt} checkpoint...')
    model, vec_env = load_model(args.ckpt)

    print(f'Recording hero episode (seed={args.seed})...')
    hero_ep = record_episode(model, vec_env, seed=args.seed)
    fs = hero_ep['fail_step']
    n_bg = int(hero_ep['below_ground_flag'].sum())
    print(f'  Hero: {hero_ep["T"]} steps, fail_step={fs}, '
          f'below_ground_count={n_bg}')

    bg_seeds = [int(s) for s in args.bg_seeds.split(',')]
    bg_eps   = []
    for s in bg_seeds:
        ep = record_episode(model, vec_env, seed=s)
        bg_eps.append(ep)
        print(f'  Background seed={s}: {ep["T"]} steps, '
              f'failed={ep["fail_step"] is not None}')

    print('Rendering figure...')
    make_figure(hero_ep, bg_eps, args.out)


if __name__ == '__main__':
    main()
