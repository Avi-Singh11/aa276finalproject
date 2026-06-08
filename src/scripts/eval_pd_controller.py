"""PD controller vs PD + BRT safety filter evaluation.

Baseline: a joint-space PD controller that drives q → IK(goal).
          It knows nothing about obstacles, slosh, or ground clearance.
Safety layer: BRT-based CBF-QP filter applied on top of u_nom.

This is the cleanest demonstration of what the safety filter contributes:
the PD controller is correct about *where to go* but wrong about *safety*,
and the filter corrects exactly that gap.

Usage:
    # PD baseline only, normal + hard starts:
    python -m src.scripts.eval_pd_controller --mode both

    # PD vs PD+BRT, 50 seeds each:
    python -m src.scripts.eval_pd_controller --mode both --use_filter --n_seeds 50

    # Quick smoke test:
    python -m src.scripts.eval_pd_controller --mode both --use_filter --n_seeds 10
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.envs.base_env import CoffeePouringEnv, CoffeeArmEnv, inverse_kinematics
from src.core.arm_dynamics import position_cup
from src.config.constants import DEFAULT_L, DEFAULT_SLOSH_RAD_MAX


# ── Environment wrappers ───────────────────────────────────────────────────────

class LowStartEnv(CoffeeArmEnv):
    """Hard-start variant: low cup elevation + random initial joint velocity."""
    EL_LOW_DEG  = 2.0
    EL_HIGH_DEG = 12.0
    VEL_SCALE   = 2.0  # rad/s

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        while True:
            r  = self.np_random.uniform(0.5, 0.7)
            az = np.deg2rad(self.np_random.uniform(60, 90))
            el = np.deg2rad(self.np_random.uniform(self.EL_LOW_DEG, self.EL_HIGH_DEG))
            x  = r * np.cos(el) * np.cos(az)
            y  = r * np.cos(el) * np.sin(az)
            z  = r * np.sin(el)
            try:
                theta_init = inverse_kinematics(x, y, z, self.L, elbow="up")
                break
            except ValueError:
                continue

        dq = self.np_random.uniform(-self.VEL_SCALE, self.VEL_SCALE, 3).astype(np.float32)
        self.state = np.concatenate([theta_init, dq, np.zeros(4)]).astype(np.float32)
        self.step_count = 0
        self._prev_dist = float(
            np.linalg.norm(position_cup(self.state[:6], self.L) - self.goal_pos)
        )
        return self._make_obs(), {}


# ── PD controller ──────────────────────────────────────────────────────────────

class PDController:
    """Joint-space PD controller: u = Kp*(q_goal - q) + Kd*(0 - dq).

    q_goal is the IK solution for goal_pos. The controller has no knowledge of
    obstacles, slosh, or ground clearance — the BRT filter handles all of that.
    """

    def __init__(self, goal_pos, L, Kp: float = 5.0, Kd: float = 3.0, u_max: float = 15.0):
        self.q_goal = inverse_kinematics(
            float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]),
            L, elbow="up",
        )
        self.Kp    = float(Kp)
        self.Kd    = float(Kd)
        self.u_max = float(u_max)

    def compute(self, state: np.ndarray) -> np.ndarray:
        q  = state[:3]
        dq = state[3:6]
        u  = self.Kp * (self.q_goal - q) + self.Kd * (0.0 - dq)
        return np.clip(u, -self.u_max, self.u_max).astype(np.float32)


# ── BRT loader ─────────────────────────────────────────────────────────────────

DEEPREACH_PATH = os.path.dirname(PROJECT_ROOT)
if DEEPREACH_PATH not in sys.path:
    sys.path.insert(0, DEEPREACH_PATH)

BRT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'brt_model', 'model_final.pth')


def load_brt(brt_model_path: str = BRT_MODEL_PATH):
    """Load trained DeepReach BRT model. Returns (brt_nn, brt_dynamics)."""
    import torch
    from dynamics.dynamics import CoffeeArmDynamics
    from deepreach.utils import modules
    from src.reachability.compute_brt import CFG as BRT_CFG

    dyn = CoffeeArmDynamics()
    sd = torch.load(brt_model_path, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    # infer hidden size from checkpoint so fast (128) and full (256) both work
    first_key = next(k for k in sd if 'weight' in k)
    hidden_features = sd[first_key].shape[0]
    num_hidden_layers = sum(1 for k in sd if k.endswith('.weight')) - 2
    brt = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1,
        type='sine', mode='mlp', final_layer_factor=1,
        hidden_features=hidden_features,
        num_hidden_layers=num_hidden_layers,
    )
    brt.load_state_dict(sd)
    brt.eval()
    print(f'  BRT loaded from {brt_model_path}')
    return brt, dyn


# ── Rollout recording ──────────────────────────────────────────────────────────

def record_rollout(env, pd_ctrl: PDController, seed: int, brt_bundle=None):
    """Run one episode and return per-step data + outcome flags."""
    from src.reachability.safety_filter import safety_filter as brt_safety_filter
    from src.reachability.compute_brt import CFG as BRT_CFG

    obs, _ = env.reset(seed=seed)

    cup_z, slosh_r, dist_g, actions = [], [], [], []
    brt_overrides = []
    ground_flag, spill_flag, obs_flag, joint_flag = [], [], [], []

    brt_nn, brt_dyn = brt_bundle if brt_bundle is not None else (None, None)

    done = False
    while not done:
        u_nom = pd_ctrl.compute(env.state)

        if brt_nn is not None:
            u_exe, did_intervene = brt_safety_filter(
                brt_nn, brt_dyn, env.state.copy(), u_nom, t=BRT_CFG['tMax']
            )
            brt_overrides.append(bool(did_intervene))
        else:
            u_exe = u_nom
            brt_overrides.append(False)

        obs, _, terminated, truncated, info = env.step(u_exe)
        done = terminated or truncated

        cup_z.append(float(info['cup_pos'][2]))
        slosh_r.append(float(info['slosh_rad']))
        dist_g.append(float(info['dist_to_goal']))
        actions.append(u_exe.copy())
        ground_flag.append(bool(info.get('below_ground',    False)))
        spill_flag.append( bool(info.get('spill_slosh',     False)))
        obs_flag.append(   bool(info.get('obstacle_hit',    False)))
        joint_flag.append( bool(info.get('joint_violation', False)))

    ground_arr = np.array(ground_flag)
    spill_arr  = np.array(spill_flag)
    obs_arr    = np.array(obs_flag)
    joint_arr  = np.array(joint_flag)

    failed    = bool(np.any(ground_arr) or np.any(spill_arr) or
                     np.any(obs_arr)    or np.any(joint_arr))
    completed = bool(dist_g[-1] < 0.1)
    truncated_ep = bool(not failed and not completed)

    return dict(
        cup_z         = np.array(cup_z),
        slosh_r       = np.array(slosh_r),
        dist_g        = np.array(dist_g),
        actions       = np.array(actions),
        brt_overrides = np.array(brt_overrides),
        below_ground  = ground_arr,
        spill         = spill_arr,
        obstacle      = obs_arr,
        joint         = joint_arr,
        failed        = failed,
        completed     = completed,
        truncated     = truncated_ep,
        T             = len(cup_z),
        dt            = env.dt,
        init_z        = cup_z[0],
        max_slosh     = float(np.max(slosh_r)),
        seed          = seed,
    )


# ── Batch evaluation & summary ─────────────────────────────────────────────────

def evaluate_batch(env_cls, env_kwargs, pd_ctrl, seeds, brt_bundle=None, label='PD'):
    results = []
    for seed in seeds:
        env = env_cls(**env_kwargs)
        ep  = record_rollout(env, pd_ctrl, seed, brt_bundle=brt_bundle)
        results.append(ep)

        if ep['failed']:
            cause = ('ground'   if np.any(ep['below_ground']) else
                     'spill'    if np.any(ep['spill'])        else
                     'obstacle' if np.any(ep['obstacle'])     else 'joint')
        else:
            cause = 'done' if ep['completed'] else 'trunc'
        brt_pct = 100 * ep['brt_overrides'].mean() if brt_bundle else 0.0
        print(f"  [{label}] seed={seed:3d}  z0={ep['init_z']:.3f}m  "
              f"steps={ep['T']:4d}  {cause:8s}"
              + (f"  BRT={brt_pct:.1f}%" if brt_bundle else ""))
    return results


def summarise(results, label='', verbose=True):
    N = len(results)
    if N == 0:
        return {}

    fail_rate  = np.mean([r['failed']    for r in results])
    comp_rate  = np.mean([r['completed'] for r in results])
    trunc_rate = np.mean([r['truncated'] for r in results])

    ground_rate = np.mean([np.any(r['below_ground']) for r in results])
    spill_rate  = np.mean([np.any(r['spill'])        for r in results])
    obs_rate    = np.mean([np.any(r['obstacle'])     for r in results])
    joint_rate  = np.mean([np.any(r['joint'])        for r in results])

    mean_len   = np.mean([r['T'] for r in results])
    mean_slosh = np.mean([r['max_slosh'] for r in results])
    brt_rate   = np.mean([r['brt_overrides'].mean() for r in results])

    stats = dict(
        fail_rate=fail_rate, comp_rate=comp_rate, trunc_rate=trunc_rate,
        ground_rate=ground_rate, spill_rate=spill_rate,
        obs_rate=obs_rate, joint_rate=joint_rate,
        mean_len=mean_len, mean_slosh=mean_slosh, brt_rate=brt_rate,
    )

    if verbose:
        print(f"\n{'─'*56}")
        print(f"  {label}  (n={N} episodes)")
        print(f"  ── Outcomes ──────────────────────────────────────────")
        print(f"  Failure rate       : {fail_rate*100:5.1f}%")
        print(f"    ↳ below ground   : {ground_rate*100:5.1f}%")
        print(f"    ↳ slosh/spill    : {spill_rate*100:5.1f}%")
        print(f"    ↳ obstacle hit   : {obs_rate*100:5.1f}%")
        print(f"    ↳ joint limit    : {joint_rate*100:5.1f}%")
        print(f"  Completion rate    : {comp_rate*100:5.1f}%")
        print(f"  Truncation rate    : {trunc_rate*100:5.1f}%")
        print(f"  ── Dynamics ──────────────────────────────────────────")
        print(f"  Mean episode len   : {mean_len:6.1f} steps  ({mean_len*results[0]['dt']:.1f}s)")
        print(f"  Mean peak slosh    : {mean_slosh:.4f} m")
        if any(r['brt_overrides'].any() for r in results):
            print(f"  BRT override rate  : {brt_rate*100:5.1f}% of steps")
        print(f"{'─'*56}")

    return stats


# ── Figures ────────────────────────────────────────────────────────────────────

def _t(ep):
    return np.arange(ep['T']) * ep['dt']


def _failure_band(ax, ep, alpha=0.15):
    flags = ep['below_ground'] | ep['spill'] | ep['obstacle'] | ep['joint']
    t = _t(ep)
    in_f = False
    t0 = None
    for i, f in enumerate(flags):
        if f and not in_f:
            t0 = t[i]; in_f = True
        elif not f and in_f:
            ax.axvspan(t0, t[i], color='red', alpha=alpha, linewidth=0)
            in_f = False
    if in_f:
        ax.axvspan(t0, t[-1] + ep['dt'], color='red', alpha=alpha, linewidth=0)


def plot_results(
    norm_bl=None, norm_ft=None,
    hard_bl=None, hard_ft=None,
    kp: float = 5.0,
    kd: float = 3.0,
    out_path: str = 'figures/eval_pd_comparison.png',
):
    """6-panel paper figure: PD baseline vs PD + BRT safety filter."""
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.52, wspace=0.35,
                            left=0.07, right=0.97, top=0.91, bottom=0.07)

    ax_rates = fig.add_subplot(gs[0, 0])
    ax_ftype = fig.add_subplot(gs[0, 1])
    ax_zbl   = fig.add_subplot(gs[1, 0])
    ax_zft   = fig.add_subplot(gs[1, 1])
    ax_dist  = fig.add_subplot(gs[2, 0])
    ax_ovrd  = fig.add_subplot(gs[2, 1])

    BLUE  = 'steelblue'
    GREEN = 'mediumseagreen'
    RED   = 'tomato'
    GRAY  = '#888888'

    # ── Panel A: Outcome rates ────────────────────────────────────────────────
    categories = ['Completion', 'Failure', 'Truncation']
    keys       = ['comp_rate', 'fail_rate', 'trunc_rate']
    conditions = []
    if norm_bl is not None:
        conditions.append(('Normal / PD',      BLUE,           summarise(norm_bl, verbose=False)))
    if norm_ft is not None:
        conditions.append(('Normal / PD+BRT',  GREEN,          summarise(norm_ft, verbose=False)))
    if hard_bl is not None:
        conditions.append(('Hard / PD',        'darkorange',   summarise(hard_bl, verbose=False)))
    if hard_ft is not None:
        conditions.append(('Hard / PD+BRT',    'mediumorchid', summarise(hard_ft, verbose=False)))

    x   = np.arange(len(categories))
    n   = len(conditions)
    w   = 0.18
    offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

    for i, (lbl, col, st) in enumerate(conditions):
        vals = [st[k] * 100 for k in keys]
        bars = ax_rates.bar(x + offsets[i], vals, w, color=col, label=lbl,
                            zorder=3, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v >= 3:
                ax_rates.text(bar.get_x() + bar.get_width()/2,
                              bar.get_height() + 0.8,
                              f'{v:.0f}%', ha='center', fontsize=6)

    ax_rates.set_xticks(x); ax_rates.set_xticklabels(categories, fontsize=8)
    ax_rates.set_ylabel('Rate (%)', fontsize=8)
    ax_rates.set_ylim(0, 118)
    ax_rates.set_title('(A)  Episode outcomes', fontsize=9, fontweight='bold')
    ax_rates.legend(fontsize=6.5, framealpha=0.85, ncol=2)
    ax_rates.grid(axis='y', alpha=0.3); ax_rates.tick_params(labelsize=7)

    # ── Panel B: Failure type breakdown (hard starts) ─────────────────────────
    if hard_bl is not None or hard_ft is not None:
        ftypes   = ['Ground', 'Spill', 'Obstacle', 'Joint']
        fkeys    = ['ground_rate', 'spill_rate', 'obs_rate', 'joint_rate']
        fx       = np.arange(len(ftypes))
        ft_conds = []
        if hard_bl is not None:
            ft_conds.append(('PD',      'darkorange',   summarise(hard_bl, verbose=False)))
        if hard_ft is not None:
            ft_conds.append(('PD+BRT',  'mediumorchid', summarise(hard_ft, verbose=False)))

        fn    = len(ft_conds)
        fw    = 0.3
        foffs = np.linspace(-(fn-1)*fw/2, (fn-1)*fw/2, fn)
        for i, (lbl, col, st) in enumerate(ft_conds):
            fvals = [st[k]*100 for k in fkeys]
            ax_ftype.bar(fx + foffs[i], fvals, fw, color=col, label=lbl,
                         zorder=3, alpha=0.85)
        ax_ftype.set_xticks(fx); ax_ftype.set_xticklabels(ftypes, fontsize=8)
        ax_ftype.set_ylabel('Rate (%)', fontsize=8)
        ax_ftype.set_ylim(0, None)
        ax_ftype.set_title('(B)  Failure type breakdown (hard starts)', fontsize=9, fontweight='bold')
        ax_ftype.legend(fontsize=7); ax_ftype.grid(axis='y', alpha=0.3)
        ax_ftype.tick_params(labelsize=7)
    else:
        ax_ftype.set_visible(False)

    # ── Panels C & D: cup height over time ────────────────────────────────────
    def _draw_traces(ax, eps, color, title):
        if eps is None:
            ax.text(0.5, 0.5, 'not evaluated', transform=ax.transAxes,
                    ha='center', va='center', color=GRAY, style='italic')
            ax.set_title(title, fontsize=9, fontweight='bold')
            return
        for ep in eps:
            lc = RED if ep['failed'] else color
            ax.plot(_t(ep), ep['cup_z'], color=lc, lw=0.7, alpha=0.45)
            if ep['failed']:
                _failure_band(ax, ep)
        ax.axhline(0.0, color='red', lw=1.5, ls='--', zorder=5)
        ax.axhspan(-0.05, 0.0, color='red', alpha=0.10, linewidth=0)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Cup Z (m)', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylim(-0.05, 0.55)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)

    _draw_traces(ax_zbl, hard_bl, BLUE,  '(C)  Hard starts — PD baseline (cup height)')
    _draw_traces(ax_zft, hard_ft, GREEN, '(D)  Hard starts — PD + BRT filter (cup height)')

    # ── Panel E: distance to goal ─────────────────────────────────────────────
    plotted = False
    for eps, col, lbl in [
        (hard_bl, BLUE,  'PD baseline'),
        (hard_ft, GREEN, 'PD + BRT'),
    ]:
        if eps is None:
            continue
        for ep in eps:
            ax_dist.plot(_t(ep), ep['dist_g'], color=col, lw=0.7, alpha=0.35)
        plotted = True

    ax_dist.axhline(0.1, color='green', lw=1.5, ls='--', zorder=5,
                    label='Goal threshold (0.1 m)')
    if plotted:
        handles = [
            Line2D([0], [0], color=BLUE,  lw=1.5, label='PD baseline'),
            Line2D([0], [0], color=GREEN, lw=1.5, label='PD + BRT filter'),
            Line2D([0], [0], color='green', lw=1.5, ls='--', label='Goal (0.1 m)'),
        ]
        ax_dist.legend(handles=handles, fontsize=7, framealpha=0.8)
    ax_dist.set_xlabel('Time (s)', fontsize=8)
    ax_dist.set_ylabel('Distance to goal (m)', fontsize=8)
    ax_dist.set_title('(E)  Progress toward goal (hard starts)', fontsize=9, fontweight='bold')
    ax_dist.set_ylim(0, None)
    ax_dist.grid(alpha=0.3); ax_dist.tick_params(labelsize=7)

    # ── Panel F: BRT intervention per episode ─────────────────────────────────
    if hard_ft is not None and any(r['brt_overrides'].any() for r in hard_ft):
        pcts = [100 * r['brt_overrides'].mean() for r in hard_ft]
        ax_ovrd.bar(range(len(hard_ft)), pcts, color='darkorange', alpha=0.8, zorder=3)
        mean_ov = np.mean(pcts)
        ax_ovrd.axhline(mean_ov, color='red', lw=1.2, ls='--',
                        label=f'Mean {mean_ov:.1f}%')
        ax_ovrd.set_xlabel('Episode index', fontsize=8)
        ax_ovrd.set_ylabel('BRT override (%)', fontsize=8)
        ax_ovrd.set_title('(F)  BRT filter activity (hard starts)', fontsize=9, fontweight='bold')
        ax_ovrd.legend(fontsize=7); ax_ovrd.grid(axis='y', alpha=0.3)
        ax_ovrd.tick_params(labelsize=7)
    else:
        ax_ovrd.text(0.5, 0.5, 'BRT filter not evaluated\n(run with --use_filter)',
                     transform=ax_ovrd.transAxes, ha='center', va='center',
                     fontsize=9, color=GRAY, style='italic')
        ax_ovrd.set_title('(F)  BRT filter activity', fontsize=9, fontweight='bold', color=GRAY)
        ax_ovrd.set_xticks([]); ax_ovrd.set_yticks([])

    # ── Super-title ───────────────────────────────────────────────────────────
    parts = []
    if hard_bl is not None:
        s = summarise(hard_bl, verbose=False)
        parts.append(f'PD hard-start failure {s["fail_rate"]*100:.0f}%')
    if hard_ft is not None:
        s = summarise(hard_ft, verbose=False)
        parts.append(f'PD+BRT hard-start failure {s["fail_rate"]*100:.0f}%')
    suptitle = (f'Safety evaluation: PD controller vs PD + BRT safety filter  '
                f'(Kp={kp}, Kd={kd})\n' + '  |  '.join(parts))
    fig.suptitle(suptitle, fontsize=9, fontweight='bold', y=0.975)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    pdf = out_path.replace('.png', '.pdf')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved: {out_path}')
    print(f'Figure saved: {pdf}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['normal', 'hard', 'both'], default='hard')
    parser.add_argument('--use_filter', action='store_true',
                        help='Also run PD + BRT filter comparison')
    parser.add_argument('--brt_model', default=None)
    parser.add_argument('--n_seeds', type=int, default=50)
    parser.add_argument('--kp', type=float, default=5.0,
                        help='Proportional gain (default 5.0)')
    parser.add_argument('--kd', type=float, default=3.0,
                        help='Derivative gain (default 3.0)')
    parser.add_argument('--out', default='figures/eval_pd_comparison.png')
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))
    env_kwargs = dict(u_max=15.0, T=10.0, dt=0.01)

    brt_bundle = None
    if args.use_filter:
        brt_path = args.brt_model or BRT_MODEL_PATH
        print(f'Loading BRT model from {brt_path}...')
        brt_bundle = load_brt(brt_path)

    # Build a single PDController — goal must match the env's default goal_pos
    goal_pos = np.array([0.55, 0.10, 0.10], dtype=np.float32)
    pd_ctrl  = PDController(goal_pos, DEFAULT_L, Kp=args.kp, Kd=args.kd)
    print(f'\nPD controller: Kp={args.kp}, Kd={args.kd}')
    print(f'  q_goal = {pd_ctrl.q_goal.round(4)} rad '
          f'({np.rad2deg(pd_ctrl.q_goal).round(2)} deg)')

    norm_bl = norm_ft = hard_bl = hard_ft = None

    # ── Normal starts ──────────────────────────────────────────────────────────
    if args.mode in ('normal', 'both'):
        print(f'\n{"="*56}')
        print(f'NORMAL STARTS  (n={args.n_seeds})')
        print(f'{"="*56}')

        print('\nPD baseline:')
        norm_bl = evaluate_batch(CoffeeArmEnv, env_kwargs, pd_ctrl,
                                 seeds, brt_bundle=None, label='PD')
        summarise(norm_bl, label='Normal starts — PD baseline')

        if brt_bundle is not None:
            print('\nPD + BRT filter:')
            norm_ft = evaluate_batch(CoffeeArmEnv, env_kwargs, pd_ctrl,
                                     seeds, brt_bundle=brt_bundle, label='PD+BRT')
            summarise(norm_ft, label='Normal starts — PD + BRT filter')

    # ── Hard starts ───────────────────────────────────────────────────────────
    if args.mode in ('hard', 'both'):
        print(f'\n{"="*56}')
        print(f'HARD STARTS  el∈[{LowStartEnv.EL_LOW_DEG}°,{LowStartEnv.EL_HIGH_DEG}°]  '
              f'vel_scale={LowStartEnv.VEL_SCALE} rad/s  (n={args.n_seeds})')
        print(f'{"="*56}')

        print('\nPD baseline:')
        hard_bl = evaluate_batch(LowStartEnv, env_kwargs, pd_ctrl,
                                 seeds, brt_bundle=None, label='PD')
        summarise(hard_bl, label='Hard starts — PD baseline')

        if brt_bundle is not None:
            print('\nPD + BRT filter:')
            hard_ft = evaluate_batch(LowStartEnv, env_kwargs, pd_ctrl,
                                     seeds, brt_bundle=brt_bundle, label='PD+BRT')
            summarise(hard_ft, label='Hard starts — PD + BRT filter')

    # ── Figure ─────────────────────────────────────────────────────────────────
    print('\nRendering figure...')
    plot_results(norm_bl=norm_bl, norm_ft=norm_ft,
                 hard_bl=hard_bl, hard_ft=hard_ft,
                 kp=args.kp, kd=args.kd,
                 out_path=args.out)


if __name__ == '__main__':
    main()
