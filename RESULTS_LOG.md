# AA276 Final Project — Results Log

## System Overview

**Task**: 3-DOF robot arm carrying a coffee cup. Reach a goal position without spilling.

**State (10D)**: `[q1, q2, q3, dq1, dq2, dq3, x_slosh, y_slosh, vx_slosh, vy_slosh]`
- Joint angles (q) in rad, joint velocities (dq) in rad/s
- Slosh displacement (xs, ys) and velocity (vxs, vys) in meters / m/s

**Arm geometry**:
- Link lengths: L = [0.30, 0.50, 0.30] m (total = 1.10 m)
- Control: joint acceleration u ∈ [-15, +15] rad/s² per joint
- Cup pendulum: l_eff = 0.025 m, damping = 0.2

**Safety constraints** (failure margin l(x) = min of all):
| Constraint | Normalization | Safe value |
|---|---|---|
| Ground clearance | cup_z / l_total | > 0 |
| Slosh spill | (slosh_rad_max − slosh_disp) / (10 × slosh_rad_max) | > 0, max = 0.1 |
| Joint limits ± π | slack / π | > 0 |
| Obstacle clearance | clearance / l_total | > 0 |
| Self-collision | link_dist / l_total | > 0 |

- **slosh_rad_max** = l_eff × sin(0.30 rad) ≈ **0.00739 m (7.39 mm)**
- **3 spherical obstacles**: [0.24,−0.3,0.08] r=0.08m, [0.39,0,0.30] r=0.15m, [0.21,0.21,0.08] r=0.08m

---

## Experiment Conditions

### Condition 1 — PD Baseline (no safety filter)

**Controller**: Joint-space PD
```
u = Kp * (q_goal − q) + Kd * (0 − dq)
Kp = 5.0, Kd = 3.0, u clipped to ±15 rad/s²
```
Goal position: [0.55, 0.10, 0.10] m → q_goal ≈ [0.18, −0.87, 1.53] rad (IK, elbow up)

**Run command**:
```bash
cd /home/avisingh/Documents/Stanford/Spring2026/AA276/aa276finalproject
python src/scripts/eval_pd_controller.py --mode normal --n_seeds 50
python src/scripts/eval_pd_controller.py --mode hard --n_seeds 50
```

**Normal starts** (random IK targets, no initial velocity):
- Failure rate: **100%** (all slosh/spill)
- Mean episode length: **9.3 steps (0.09 s)**
- Mean peak slosh: 0.0078 m

**Hard starts** (low elevation IK + random dq ∈ [−2, 2] rad/s):
- Failure rate: **90%** (all slosh/spill)
- Mean episode length: **18.3 steps (0.18 s)**

---

### Condition 2 — PD + BRT Safety Filter (fast model, 50k epochs)

**BRT model**: `brt_model_fast/checkpoints/model_epoch_050000.pth`
- Architecture: SIREN SingleBVPNet, 128 hidden features, 3 hidden layers
- Training: 50k epochs, lr=2×10⁻⁵, pretrain 8k iters
- Time horizon: tMax = 10.0 s

**Safety filter**: CBF-QP at every step
```
a_cbf @ u ≥ b_cbf
a_cbf = dvds @ g(x)          (BRT gradient × control matrix)
b_cbf = −γ·V + dvdt − drift_term
γ = 1.0,  dt_num = 1e-4,  ε_num = 1.0
```
QP solver: SLSQP; fallback to directional max-u on infeasibility.

**Run command**:
```bash
python src/scripts/eval_pd_controller.py \
    --mode normal --n_seeds 50 --use_filter \
    --brt_model brt_model_fast/checkpoints/model_epoch_050000.pth
```

**Normal starts**:
- Failure rate: **100%** (all slosh/spill)
- Mean episode length: **62.2 steps (0.62 s)** — 6.7× longer than PD baseline
- BRT override rate: **72.5% of steps**
- Mean peak slosh: 0.0078 m

**Hard starts** (from prior run):
- Failure rate: **100%** (all slosh/spill)
- Mean episode length: **35.4 steps (0.35 s)**
- BRT override rate: **82.0% of steps**

**Why the filter fails despite overriding 72% of steps**:
1. The 50k fast model **overestimates safety**: V > 0 throughout episodes, but the physical slosh still crosses the 7.39 mm limit. The model's V=0 contour sits at ~11 mm in slosh space (1.5× the true limit). The CBF never gets tight before the physical spill happens.
2. V stays in [0.024, 0.098] the whole episode even while slosh grows to 7+ mm — the model says "safe" when it's not.
3. The BRT gradient (a_cbf) direction is noisy with only 50k epochs, causing u_safe to randomly differ from u_pd rather than specifically reduce slosh excitation.

**What the filter does accomplish**:
- Reduces mean control magnitude (u_safe_norm << u_pd_norm in many steps)
- Significantly slows slosh growth → **7× longer episodes**
- All failures remain slosh/spill (no new constraint types introduced by the filter)

---

## BRT Training — Key Diagnostics

### Fast model probe values (safe config: q=[0.2, −0.5, 0.8], vx=vy=0)

| Checkpoint | slosh=0% | slosh=50% | slosh=90% | slosh=101% (unsafe) |
|---|---|---|---|---|
| 10k | −0.010 | −0.017 | −0.032 | −0.036 |
| 30k | +0.050 | +0.046 | +0.020 | **+0.009** ← should be < 0 |
| 50k | +0.055 | +0.048 | +0.024 | **+0.012** ← should be < 0 |

The model is training in the right direction but hasn't converged: V is still positive for states that have already crossed the slosh limit. The 200k/256-feature run should push V(slosh=101%) negative.

### Visualization of fast model (brt_visualizations/brt_model_fast/brt_value_slices.png)

- **Top-left (slosh position slice)**: V=0 contour (black) correctly circular and centered, but shifted outward to ~11 mm vs actual limit 7.39 mm (dashed line). Model has not fully learned the terminal condition yet.
- **Top-right (slosh velocity at zero displacement)**: All green (V > 0 everywhere). Model doesn't yet know that high slosh velocity at zero displacement is dangerous. Should correct with more training.
- **Bottom-left (slosh phase portrait)**: Looks physically correct — states with large displacement + outward velocity are red (unsafe), returning states are green.
- **Bottom-right (joint angle plane)**: Correctly shows obstacle holes in joint space.

---

## Next Steps

### Condition 3 — PD + BRT (full model, 200k epochs)

**Start command**:
```bash
cd /home/avisingh/Documents/Stanford/Spring2026/AA276/aa276finalproject
python src/reachability/compute_brt.py 2>&1 | tee brt_full_train.log
```
Saves to `brt_model/` every 5,000 epochs. Architecture: 256 features, 3 layers.
Expected wall time: several hours on GPU.

**Evaluate (once trained)**:
```bash
python src/scripts/eval_pd_controller.py \
    --mode both --n_seeds 50 --use_filter \
    --brt_model brt_model/model_final.pth
```

### Condition 4 — PPO (no BRT)
```bash
# Train:
python src/scripts/train_ppo.py   # (or equivalent)
# Evaluate:
python src/scripts/eval_ppo.py --n_seeds 50 --mode both
```

### Condition 5 — PPO + BRT
Apply the same CBF-QP filter on top of the PPO policy's action.
```bash
python src/scripts/eval_ppo.py \
    --n_seeds 50 --mode both --use_filter \
    --brt_model brt_model/model_final.pth
```

---

## Key File Locations

| File | Purpose |
|---|---|
| `dynamics/dynamics.py` | DeepReach adapter — failure_margin, boundary_fn, hamiltonian |
| `src/reachability/compute_brt.py` | BRT training (CFG=200k/256, CFG_FAST=50k/128) |
| `src/reachability/safety_filter.py` | CBF-QP safety filter implementation |
| `src/scripts/eval_pd_controller.py` | PD and PD+BRT evaluation |
| `brt_model_fast/checkpoints/model_epoch_050000.pth` | Fast 50k checkpoint |
| `brt_model/model_final.pth` | Full 200k model (after training) |
| `brt_visualizations/brt_model_fast/brt_value_slices.png` | Fast model visualization |
| `figures/eval_pd_comparison.png` | Last eval figure |

---

## Critical Bug Fixes Applied

| Bug | Symptom | Fix |
|---|---|---|
| `boundary_fn` used `l_total` before definition | Training crash | Moved `l_total = l1+l2+l3` to top of `boundary_fn` |
| Slosh gradient 135 m⁻¹ >> H_MAX=25 | BRT blind to slosh (all V=−1.7) | Normalize slosh margin by 10×slosh_rad_max; raise H_MAX→50 |
| Static safe pool infinite loop | `l(x) > 0.05` threshold never met after normalization fix | New max l(x)=0.1 with 10×normalization; original threshold restored |
| Checkpoint dict format | `load_brt()` crash | Extract `sd['model']` if dict |
| Architecture inference | Fast (128) vs full (256) model mismatch | Infer hidden_features from checkpoint keys |
| `dvdt` sign in b_cbf | `b_cbf = −γV − dvdt − drift` (wrong) | Changed to `b_cbf = −γV + dvdt − drift` (correct per chain rule) |
