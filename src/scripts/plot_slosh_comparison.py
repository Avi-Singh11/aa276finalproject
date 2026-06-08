"""Static paper figure: slosh displacement vs. time for baseline vs. BRT filter.

Both episodes run for the full episode duration (T seconds). Failure events are
annotated on the plot rather than stopping the simulation — this gives a complete
apples-to-apples comparison across the same time window.

Two vertically-stacked panels sharing the x-axis:
  Top    — PPO baseline: shaded blocks where slosh exceeds the limit,
            triangle markers for non-slosh failures (obstacle, joint, ground)
  Bottom — PPO + BRT filter: amber shading during interventions,
            same failure markers if any occur

Output: slosh_comparison.png  (saved in aa276finalproject/)

Run from aa276finalproject/:
    python -m src.scripts.plot_slosh_comparison
    python -m src.scripts.plot_slosh_comparison --seed 7 --out my_figure.png
"""

from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
from src.config.constants import DEFAULT_SLOSH_RAD_MAX
from src.reachability.safety_filter import safety_filter as brt_safety_filter
from src.reachability.compute_brt import CFG as BRT_CFG

MODEL_PATH   = os.path.join(PROJECT_ROOT, 'ppo_baseline_final.zip')
VECNORM_PATH = os.path.join(PROJECT_ROOT, 'ppo_baseline_vecnormalize.pkl')
BRT_PATH     = os.path.join(PROJECT_ROOT, 'brt_model', 'brt_model_10t.pth')
OUT_PATH     = os.path.join(PROJECT_ROOT, 'slosh_comparison.png')

ENV_KWARGS = dict(u_max=15.0, T=10.0, dt=0.01)
DT = ENV_KWARGS['dt']

# Failure type labels and their marker styles for the plot
FAILURE_STYLES = {
    'obstacle_hit':    dict(marker='^', color='purple',  label='Obstacle hit'),
    'joint_violation': dict(marker='s', color='navy',    label='Joint limit'),
    'below_ground':    dict(marker='v', color='sienna',  label='Below ground'),
}


def _load_vec_norm():
    """Load VecNormalize purely for obs normalisation stats (training=False)."""
    dummy = DummyVecEnv([lambda: CoffeeArmEnv(**ENV_KWARGS)])
    vec = VecNormalize.load(VECNORM_PATH, dummy)
    vec.training = False
    vec.norm_reward = False
    return vec


def load_brt(ckpt_path=BRT_PATH):
    from dynamics.dynamics import CoffeeArmDynamics
    from deepreach.utils import modules
    dyn = CoffeeArmDynamics()
    brt = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1,
        type='sine', mode='mlp', final_layer_factor=1,
        hidden_features=BRT_CFG['hidden_features'],
        num_hidden_layers=BRT_CFG['num_hidden_layers'],
    )
    sd = torch.load(ckpt_path, map_location='cpu')
    brt.load_state_dict(sd)
    brt.eval()
    return brt, dyn


def record_episode(seed, use_filter=False, brt=None, brt_dyn=None):
    """Run a full episode without stopping at failure.

    Uses the raw CoffeeArmEnv so we can ignore `terminated` and keep stepping.
    Obs normalisation is applied via a frozen VecNormalize instance so PPO
    still receives properly scaled observations.

    Returns
    -------
    dict with:
      slosh_rad      : (T,) float array
      intervened     : (T,) bool array
      spill_steps    : sorted list of steps where slosh_rad > limit
      other_failures : list of (step, type_str) for non-slosh failures
      T              : total steps recorded
    """
    vec = _load_vec_norm()
    model = PPO.load(MODEL_PATH, device='cpu')

    env = CoffeeArmEnv(**ENV_KWARGS)
    raw_obs, _ = env.reset(seed=seed)
    norm_obs = vec.normalize_obs(raw_obs.reshape(1, -1))

    slosh_rad  = []
    intervened = []
    other_failures = []   # (step, label) for non-slosh failures

    step = 0
    truncated = False

    while not truncated:
        action_arr, _ = model.predict(norm_obs, deterministic=True)
        u_nom = action_arr[0].copy()

        if use_filter and brt is not None:
            u_exe, did_intervene = brt_safety_filter(
                brt, brt_dyn, env.state.copy(), u_nom, t=BRT_CFG['tMax']
            )
        else:
            u_exe  = u_nom.copy()
            did_intervene = False

        raw_obs, _, terminated, truncated, info = env.step(u_exe)
        norm_obs = vec.normalize_obs(raw_obs.reshape(1, -1))

        slosh_rad.append(float(info['slosh_rad']))
        intervened.append(did_intervene)

        # Log non-slosh failures as point events
        for key, style in FAILURE_STYLES.items():
            if info.get(key):
                other_failures.append((step, style['label']))

        # Do NOT break on terminated — continue to record the full episode.
        # truncated fires when step_count >= max_steps (natural episode end).
        step += 1

    slosh_rad = np.array(slosh_rad)
    spill_steps = list(np.where(slosh_rad > DEFAULT_SLOSH_RAD_MAX)[0])

    return dict(
        slosh_rad=slosh_rad,
        intervened=np.array(intervened),
        spill_steps=spill_steps,
        other_failures=other_failures,
        T=step,
    )


def _shade_blocks(ax, mask, time, color, alpha, zorder=1):
    """Shade contiguous True-blocks of `mask` along the time axis."""
    in_block = False
    t0 = None
    for i, val in enumerate(mask):
        if val and not in_block:
            in_block = True
            t0 = time[i]
        elif not val and in_block:
            ax.axvspan(t0, time[i], alpha=alpha, color=color,
                       linewidth=0, zorder=zorder)
            in_block = False
    if in_block:
        ax.axvspan(t0, time[-1] + (time[1] - time[0]), alpha=alpha,
                   color=color, linewidth=0, zorder=zorder)


def plot(traj_base, traj_filt, out_path):
    limit  = DEFAULT_SLOSH_RAD_MAX
    t_base = np.arange(traj_base['T']) * DT
    t_filt = np.arange(traj_filt['T']) * DT
    t_max  = max(t_base[-1], t_filt[-1]) + DT

    # y-limits: tight to data, cap baseline at 2.5× limit to keep threshold visible
    y_top_base = min(max(traj_base['slosh_rad'].max(), limit) * 1.15, limit * 2.5)
    y_top_filt = max(traj_filt['slosh_rad'].max(), limit) * 1.15

    C_BASE    = '#E74C3C'
    C_FILT    = '#2ECC71'
    C_LIMIT   = '#E67E22'
    C_SHADE_F = '#F39C12'   # amber — filter intervention
    C_SHADE_B = '#E74C3C'   # red   — slosh-over-limit

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw=dict(hspace=0.38),
    )

    # ── top panel: baseline ─────────────────────────────────────────────────
    spill_mask = traj_base['slosh_rad'] > limit
    _shade_blocks(ax_top, spill_mask, t_base, C_SHADE_B, alpha=0.20, zorder=1)

    ax_top.plot(t_base, traj_base['slosh_rad'], color=C_BASE, lw=2.0,
                label='Slosh displacement', zorder=3)
    ax_top.axhline(limit, color=C_LIMIT, lw=1.8, ls='--', zorder=2,
                   label=f'Safety limit  ({limit*1000:.1f} mm)')

    # Non-slosh failure markers (obstacle, joint, ground)
    seen_labels = set()
    for step_i, label in traj_base['other_failures']:
        style = next(s for s in FAILURE_STYLES.values() if s['label'] == label)
        kw = dict(color=style['color'], marker=style['marker'],
                  s=60, zorder=5)
        if label not in seen_labels:
            kw['label'] = label
            seen_labels.add(label)
        ax_top.scatter([t_base[step_i]], [traj_base['slosh_rad'][step_i]], **kw)

    legend_handles = [
        plt.Line2D([0], [0], color=C_BASE, lw=2.0, label='Slosh displacement'),
        plt.Line2D([0], [0], color=C_LIMIT, lw=1.8, ls='--',
                   label=f'Safety limit  ({limit*1000:.1f} mm)'),
        mpatches.Patch(color=C_SHADE_B, alpha=0.45,
                       label=f'Slosh violation  ({int(spill_mask.sum())} steps)'),
    ]
    for label in seen_labels:
        style = next(s for s in FAILURE_STYLES.values() if s['label'] == label)
        legend_handles.append(
            plt.Line2D([0], [0], color=style['color'], marker=style['marker'],
                       ls='None', ms=7, label=label)
        )

    ax_top.set_title('(a)  PPO baseline — no safety filter', fontsize=10, loc='left')
    ax_top.set_ylabel('Slosh radius (m)', fontsize=9)
    ax_top.set_xlim(0, t_max)
    ax_top.set_ylim(0, y_top_base)
    ax_top.legend(handles=legend_handles, fontsize=8,
                  loc='upper right', framealpha=0.85)
    ax_top.grid(True, alpha=0.3, lw=0.6)
    ax_top.tick_params(labelsize=8)

    # ── bottom panel: filter ────────────────────────────────────────────────
    interv = traj_filt['intervened']
    _shade_blocks(ax_bot, interv, t_filt, C_SHADE_F, alpha=0.28, zorder=1)

    filt_spill_mask = traj_filt['slosh_rad'] > limit
    if filt_spill_mask.any():
        _shade_blocks(ax_bot, filt_spill_mask, t_filt, C_SHADE_B,
                      alpha=0.25, zorder=2)

    ax_bot.plot(t_filt, traj_filt['slosh_rad'], color=C_FILT, lw=2.0,
                label='Slosh displacement', zorder=3)
    ax_bot.axhline(limit, color=C_LIMIT, lw=1.8, ls='--', zorder=2,
                   label=f'Safety limit  ({limit*1000:.1f} mm)')

    n_int      = int(interv.sum())
    int_rate   = 100 * n_int / max(traj_filt['T'], 1)
    n_filt_vio = int(filt_spill_mask.sum())

    filt_handles = [
        plt.Line2D([0], [0], color=C_FILT, lw=2.0, label='Slosh displacement'),
        plt.Line2D([0], [0], color=C_LIMIT, lw=1.8, ls='--',
                   label=f'Safety limit  ({limit*1000:.1f} mm)'),
        mpatches.Patch(color=C_SHADE_F, alpha=0.55,
                       label=f'Filter active  ({n_int} steps, {int_rate:.0f}%)'),
    ]
    if n_filt_vio:
        filt_handles.append(
            mpatches.Patch(color=C_SHADE_B, alpha=0.45,
                           label=f'Slosh violation  ({n_filt_vio} steps)')
        )

    seen_labels_filt = set()
    for step_i, label in traj_filt['other_failures']:
        style = next(s for s in FAILURE_STYLES.values() if s['label'] == label)
        kw = dict(color=style['color'], marker=style['marker'], s=60, zorder=5)
        ax_bot.scatter([t_filt[step_i]], [traj_filt['slosh_rad'][step_i]], **kw)
        if label not in seen_labels_filt:
            filt_handles.append(
                plt.Line2D([0], [0], color=style['color'], marker=style['marker'],
                           ls='None', ms=7, label=label)
            )
            seen_labels_filt.add(label)

    ax_bot.set_title('(b)  PPO + BRT safety filter', fontsize=10, loc='left')
    ax_bot.set_xlabel('Time (s)', fontsize=9)
    ax_bot.set_ylabel('Slosh radius (m)', fontsize=9)
    ax_bot.set_ylim(0, y_top_filt)
    ax_bot.legend(handles=filt_handles, fontsize=8,
                  loc='upper right', framealpha=0.85)
    ax_bot.grid(True, alpha=0.3, lw=0.6)
    ax_bot.tick_params(labelsize=8)

    fig.suptitle('Slosh Displacement: Baseline vs. BRT Safety Filter',
                 fontsize=11, fontweight='bold', y=1.01)

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── TEMP: scripted PD controller + per-constraint signal recording ──────────────

from src.core.arm_dynamics import position_cup as _position_cup, get_link_positions as _get_link_pts
from src.core.obstacles import dist_point_to_segment as _seg_dist
from src.config.constants import DEFAULT_OBSTACLES as _OBSTACLES, DEFAULT_JOINT_LIMITS as _JLIMS

_L     = np.array([0.30, 0.30, 0.25], dtype=np.float64)
_Q_A   = np.array([0.650, -0.400, 0.600], dtype=np.float32)
_Q_B   = np.array([2.400, -0.400, 0.600], dtype=np.float32)
_Q_MID = (_Q_A + _Q_B) / 2
_GOAL_B = _position_cup(np.concatenate([_Q_B, np.zeros(3)]), _L).astype(np.float32)
_T1, _T2 = 0.8, 0.8
_KP, _KD = 40.0, 5.0
_MAX_OBS_R = max(o['radius'] for o in _OBSTACLES)   # normalisation reference


def _scripted_u(state: np.ndarray, t: float) -> np.ndarray:
    """Two-phase joint-space PD: A → mid → B (fast enough to cause slosh)."""
    if t < _T1:
        q_ref, dq_ref = _Q_A + (t / _T1) * (_Q_MID - _Q_A), (_Q_MID - _Q_A) / _T1
    elif t < _T1 + _T2:
        a = (t - _T1) / _T2
        q_ref, dq_ref = _Q_MID + a * (_Q_B - _Q_MID), (_Q_B - _Q_MID) / _T2
    else:
        q_ref, dq_ref = _Q_B, np.zeros(3, dtype=np.float32)
    q, dq = state[:3].astype(np.float32), state[3:6].astype(np.float32)
    return np.clip(_KP * (q_ref - q) + _KD * (dq_ref - dq), -15.0, 15.0).astype(np.float32)


def _compute_signals(state):
    """Return per-constraint safety signals for one state vector."""
    pts  = _get_link_pts(state[:6], _L)
    segs = [(pts[i], pts[i+1]) for i in range(len(pts)-1)]

    # obstacle clearance: min signed clearance (dist_to_surface) across all segs × obs
    obs_clr = min(
        _seg_dist(np.array(obs['center']), a, b) - obs['radius']
        for obs in _OBSTACLES for a, b in segs
    )
    # joint margin: min(limit - |q|) for each joint
    joint_margin = float(np.min(_JLIMS - np.abs(state[:3])))
    # cup z-height above ground
    cup_z = float(_position_cup(state[:6], _L)[2])

    return obs_clr, joint_margin, cup_z


def record_episode_pd(use_filter=False, brt=None, brt_dyn=None):
    """Run the scripted PD scenario for the full episode, continuing past failures.
    Records slosh_rad, obstacle_clearance, joint_margin, and cup_z at every step.
    """
    env = CoffeeArmEnv(**ENV_KWARGS)
    env.reset(seed=0)
    env.state    = np.concatenate([_Q_A, np.zeros(7)]).astype(np.float32)
    env.goal_pos = _GOAL_B.copy()

    slosh_rad, obs_clr, joint_margin, cup_z, intervened = [], [], [], [], []
    step, truncated = 0, False

    while not truncated:
        t     = step * DT
        u_nom = _scripted_u(env.state, t)
        if use_filter and brt is not None:
            u_exe, did_intervene = brt_safety_filter(
                brt, brt_dyn, env.state.copy(), u_nom, t=BRT_CFG['tMax'])
        else:
            u_exe, did_intervene = u_nom.copy(), False

        _, _, _, truncated, info = env.step(u_exe)

        oc, jm, cz = _compute_signals(env.state)
        slosh_rad.append(float(info['slosh_rad']))
        obs_clr.append(oc)
        joint_margin.append(jm)
        cup_z.append(cz)
        intervened.append(did_intervene)
        step += 1

    slosh_rad    = np.array(slosh_rad)
    obs_clr      = np.array(obs_clr)
    joint_margin = np.array(joint_margin)
    cup_z        = np.array(cup_z)

    return dict(
        slosh_rad=slosh_rad,
        obs_clr=obs_clr,
        joint_margin=joint_margin,
        cup_z=cup_z,
        intervened=np.array(intervened),
        spill_steps=list(np.where(slosh_rad > DEFAULT_SLOSH_RAD_MAX)[0]),
        T=step,
    )


# ── Individual failure-set plots ────────────────────────────────────────────────

# Config for each constraint: (signal_key, safety_limit, y_label, title_label, color)
_CONSTRAINT_CFG = [
    ('slosh_rad',    DEFAULT_SLOSH_RAD_MAX, 'Slosh radius (m)',        'Slosh displacement',      '#E74C3C', '#2ECC71'),
    ('obs_clr',      0.0,                   'Obstacle clearance (m)',  'Obstacle clearance',       '#8E44AD', '#9B59B6'),
    ('joint_margin', 0.0,                   'Joint margin (rad)',      'Joint limit margin',       '#2C3E50', '#5D6D7E'),
    ('cup_z',        0.0,                   'Cup height (m)',          'Ground clearance (cup z)', '#795548', '#A1887F'),
]


def plot_individual(traj_base, traj_filt, signal_key, limit, y_label,
                    title_label, c_base, c_filt, out_path):
    """Two-panel (baseline / filter) plot for a single constraint signal."""
    sig_b  = traj_base[signal_key]
    sig_f  = traj_filt[signal_key]
    t_b    = np.arange(traj_base['T']) * DT
    t_f    = np.arange(traj_filt['T']) * DT
    t_max  = max(t_b[-1], t_f[-1]) + DT

    viol_b = sig_b < limit if signal_key != 'slosh_rad' else sig_b > limit
    viol_f = sig_f < limit if signal_key != 'slosh_rad' else sig_f > limit

    # y range: show the limit line clearly; pad 15% above and below data range
    all_vals = np.concatenate([sig_b, sig_f])
    y_lo = min(all_vals.min(), limit) * (1.15 if min(all_vals.min(), limit) < 0 else 0.85)
    y_hi = max(all_vals.max(), limit) * 1.15

    C_LIMIT   = '#E67E22'
    C_SHADE_F = '#F39C12'
    C_VIOL    = '#C0392B'

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                          gridspec_kw=dict(hspace=0.40))

    for ax, sig, t, viol, interv, c, panel, label in [
        (ax_top, sig_b, t_b, viol_b, None,                   c_base, '(a) Scripted PD baseline', 'no safety filter'),
        (ax_bot, sig_f, t_f, viol_f, traj_filt['intervened'], c_filt, '(b) Scripted PD + BRT safety filter', None),
    ]:
        _shade_blocks(ax, viol, t, C_VIOL, alpha=0.20, zorder=1)
        if interv is not None:
            _shade_blocks(ax, interv, t, C_SHADE_F, alpha=0.22, zorder=1)
        ax.plot(t, sig, color=c, lw=2.0, zorder=3, label=title_label)
        ax.axhline(limit, color=C_LIMIT, lw=1.8, ls='--', zorder=2,
                   label=f'Safety limit  ({limit:.4g})')
        ax.set_title(f'{panel}' + (f' — {label}' if label else ''),
                     fontsize=10, loc='left')
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_xlim(0, t_max)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.tick_params(labelsize=8)

        handles = [
            plt.Line2D([0], [0], color=c, lw=2.0, label=title_label),
            plt.Line2D([0], [0], color=C_LIMIT, lw=1.8, ls='--',
                       label=f'Safety limit  ({limit:.4g})'),
            mpatches.Patch(color=C_VIOL, alpha=0.45,
                           label=f'Violation  ({int(viol.sum())} steps)'),
        ]
        if interv is not None:
            n_int = int(interv.sum())
            handles.append(mpatches.Patch(
                color=C_SHADE_F, alpha=0.55,
                label=f'Filter active  ({n_int} steps, {100*n_int/max(traj_filt["T"],1):.0f}%)'))
        ax.legend(handles=handles, fontsize=8, loc='upper right', framealpha=0.85)

    ax_bot.set_xlabel('Time (s)', fontsize=9)
    fig.suptitle(f'{title_label}: Baseline vs. BRT Safety Filter',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Combined plot ───────────────────────────────────────────────────────────────

def plot_combined(traj_base, traj_filt, out_path):
    """4-row × 2-column grid: one row per constraint, left=baseline, right=filter."""
    n_rows = len(_CONSTRAINT_CFG)
    t_b   = np.arange(traj_base['T']) * DT
    t_f   = np.arange(traj_filt['T']) * DT
    t_max = max(t_b[-1], t_f[-1]) + DT

    C_LIMIT   = '#E67E22'
    C_SHADE_F = '#F39C12'
    C_VIOL    = '#C0392B'

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.2 * n_rows),
                              sharex=True,
                              gridspec_kw=dict(hspace=0.55, wspace=0.30))

    col_titles = ['Scripted PD baseline (no filter)', 'Scripted PD + BRT safety filter']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold', pad=8)

    for row, (key, limit, y_label, title_label, c_base, c_filt) in enumerate(_CONSTRAINT_CFG):
        sig_b = traj_base[key]
        sig_f = traj_filt[key]
        viol_b = sig_b < limit if key != 'slosh_rad' else sig_b > limit
        viol_f = sig_f < limit if key != 'slosh_rad' else sig_f > limit

        all_vals = np.concatenate([sig_b, sig_f])
        y_lo = min(all_vals.min(), limit) * (1.15 if min(all_vals.min(), limit) < 0 else 0.85)
        y_hi = max(all_vals.max(), limit) * 1.15

        for col, (ax, sig, t, viol, interv, c) in enumerate([
            (axes[row, 0], sig_b, t_b, viol_b, None,                    c_base),
            (axes[row, 1], sig_f, t_f, viol_f, traj_filt['intervened'], c_filt),
        ]):
            _shade_blocks(ax, viol, t, C_VIOL, alpha=0.22, zorder=1)
            if interv is not None:
                _shade_blocks(ax, interv, t, C_SHADE_F, alpha=0.18, zorder=1)
            ax.plot(t, sig, color=c, lw=1.8, zorder=3)
            ax.axhline(limit, color=C_LIMIT, lw=1.5, ls='--', zorder=2)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlim(0, t_max)
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=7)

            n_viol = int(viol.sum())
            color_patch = mpatches.Patch(color=C_VIOL, alpha=0.45,
                                          label=f'{title_label}  ({n_viol} viol. steps)')
            handles = [plt.Line2D([0], [0], color=c, lw=1.8, label=title_label),
                       plt.Line2D([0], [0], color=C_LIMIT, lw=1.5, ls='--', label='Limit'),
                       color_patch]
            if interv is not None and interv.any():
                n_int = int(interv.sum())
                handles.append(mpatches.Patch(
                    color=C_SHADE_F, alpha=0.55,
                    label=f'Filter active ({n_int} steps)'))
            ax.legend(handles=handles, fontsize=6.5, loc='upper right', framealpha=0.85)

    for col in range(2):
        axes[-1, col].set_xlabel('Time (s)', fontsize=9)

    fig.suptitle('All Safety Constraints: Baseline vs. BRT Safety Filter',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── END TEMP ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pd', action='store_true',
                        help='Use scripted PD controller instead of PPO (recommended for slosh demo)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Fix a single seed for both episodes (skips sweep)')
    parser.add_argument('--seeds', type=int, default=40,
                        help='Number of seeds to sweep when searching (default 40)')
    parser.add_argument('--brt', default=BRT_PATH,
                        help='Path to BRT model checkpoint')
    parser.add_argument('--out', default=OUT_PATH)
    args = parser.parse_args()

    print(f"Loading BRT safety filter from {args.brt}...")
    brt, brt_dyn = load_brt(args.brt)

    if args.pd:
        print("\nUsing scripted PD controller (TEMP mode).")
        traj_base = record_episode_pd(use_filter=False)
        print(f"Baseline: {traj_base['T']} steps, {len(traj_base['spill_steps'])} spill steps")
        traj_filt = record_episode_pd(use_filter=True, brt=brt, brt_dyn=brt_dyn)
        print(f"Filter  : {traj_filt['T']} steps, "
              f"interventions={int(traj_filt['intervened'].sum())}, "
              f"{len(traj_filt['spill_steps'])} spill steps")

        base_out = args.out.replace('.png', '_pd') if args.out == OUT_PATH \
                   else args.out.replace('.png', '')

        # Individual plot per constraint
        for key, limit, y_label, title_label, c_base, c_filt in _CONSTRAINT_CFG:
            slug = key.replace('_', '-')
            plot_individual(traj_base, traj_filt, key, limit, y_label,
                            title_label, c_base, c_filt,
                            f"{base_out}_{slug}.png")

        # Combined 4×2 grid
        plot_combined(traj_base, traj_filt, f"{base_out}_combined.png")
        return

    if args.seed is not None:
        best_seed = args.seed
        print(f"\nUsing fixed seed={best_seed}.")
        traj_base = record_episode(best_seed, use_filter=False)
    else:
        print(f"\nSweeping {args.seeds} seeds to find best baseline slosh violation...")
        best_seed, best_traj, best_score = None, None, -1
        for s in range(args.seeds):
            t = record_episode(s, use_filter=False)
            score = len(t['spill_steps'])
            print(f"  seed={s:3d}:  spill_steps={score:4d}  "
                  f"other_failures={len(t['other_failures'])}")
            if score > best_score:
                best_score, best_seed, best_traj = score, s, t
        print(f"\nBest seed: {best_seed}  ({best_score} spill steps)")
        traj_base = best_traj

    print(f"Baseline: {traj_base['T']} steps, "
          f"{len(traj_base['spill_steps'])} spill steps, "
          f"{len(traj_base['other_failures'])} other failures")

    print(f"\nRecording filter episode (seed={best_seed})...")
    traj_filt = record_episode(best_seed, use_filter=True, brt=brt, brt_dyn=brt_dyn)
    print(f"Filter  : {traj_filt['T']} steps, "
          f"interventions={int(traj_filt['intervened'].sum())}, "
          f"{len(traj_filt['spill_steps'])} spill steps")

    plot(traj_base, traj_filt, args.out)


if __name__ == '__main__':
    main()
