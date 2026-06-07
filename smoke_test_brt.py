#!/usr/bin/env python3
"""
Smoke test for the two BRT fixes:
  1. compute_brt.py: states passed to hamiltonian are now normalized (not physical)
  2. safety_filter.py: dvds divided by state_scale before CBF construction

Run from the aa276finalproject directory:
  python smoke_test_brt.py
"""

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEEPREACH_PARENT = os.path.dirname(PROJECT_ROOT)
for p in [PROJECT_ROOT, DEEPREACH_PARENT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


# ── 1. Hamiltonian scaling ─────────────────────────────────────────────────────
def test_hamiltonian_scaling():
    print("\n=== Test 1: Hamiltonian scaling ===")
    from dynamics.dynamics import CoffeeArmDynamics
    dyn = CoffeeArmDynamics()

    # At normalized zero state with zero gradient → H = 0
    z = torch.zeros(1, 10)
    p = torch.zeros(1, 10)
    H0 = dyn.hamiltonian(z, p).item()
    ok0 = abs(H0) < 1e-5
    print(f"  H(z=0, p=0) = {H0:.8f}  (expect 0)  [{PASS if ok0 else FAIL}]")

    # e_dq1 gradient in normalized space: p[3]=1 means ∂V/∂z_dq1 = 1.
    # Hamiltonian converts: p_phys = p / scale[3] = 1/5 = 0.2 rad/s
    # Control term: u_max[0] * |p_phys[3] * K[0,0]| = 15 * 0.2 = 3.0
    p1 = torch.zeros(1, 10); p1[0, 3] = 1.0
    H1 = dyn.hamiltonian(z, p1).item()
    expected_H1 = dyn.u_max[0] / dyn.state_scale[3]  # 15 / 5 = 3.0
    ok1 = abs(H1 - expected_H1) < 0.01
    print(f"  H(z=0, e_dq1_norm) = {H1:.4f}  (expect {expected_H1:.1f} = u_max/scale_dq)  [{PASS if ok1 else FAIL}]")

    # BUG CHECK: the old bug passed physical state (e.g. state*scale) to hamiltonian,
    # which then multiplied by scale again → double scaling.
    # Demonstrate: at a non-zero joint angle, slosh control terms depend on J(q).
    # With q2=π/6 (normalized), J is different from q2=π/6*scale[1] (double-scaled).
    # Use p[8]=0.1 (vx_slosh gradient) — small enough that neither value hits the
    # H_MAX=25 clamp, so the double-scaling bug remains detectable.
    z_norm = torch.zeros(1, 10); z_norm[0, 1] = np.pi/6  # q2 = π/6 in normalized space
    z_norm[0, 6] = 0.5  # xs = 0.5 in normalized space
    p_slosh = torch.zeros(1, 10); p_slosh[0, 8] = 0.1

    H_correct = dyn.hamiltonian(z_norm, p_slosh).item()

    # Simulate the OLD BUG: pass physical state = normalized * scale
    z_phys_bug = z_norm * torch.as_tensor(dyn.state_scale)
    H_bug = dyn.hamiltonian(z_phys_bug, p_slosh).item()

    ok2 = abs(H_correct - H_bug) > 0.01
    print(f"  H(correct normalized q2)   = {H_correct:.6f}")
    print(f"  H(old bug: physical state) = {H_bug:.6f}")
    print(f"  These differ (old bug is wrong)  [{PASS if ok2 else FAIL}]")

    return ok0 and ok1 and ok2


# ── 2. Safety filter gradient scaling ─────────────────────────────────────────
def test_safety_filter_gradient():
    print("\n=== Test 2: Safety filter gradient scaling ===")
    from dynamics.dynamics import CoffeeArmDynamics
    try:
        from deepreach.utils import modules
    except ImportError as e:
        print(f"  DeepReach unavailable ({e}), skipping")
        return None

    dyn = CoffeeArmDynamics()
    # Use CPU to match safety_filter's device handling
    model = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1,
        type='sine', mode='mlp', final_layer_factor=1,
        hidden_features=32, num_hidden_layers=2,
    ).cpu()
    model.eval()

    from src.reachability.safety_filter import _brt_value_and_gradient

    state_np = np.zeros(10, dtype=np.float32)
    V, dvdt, dvds = _brt_value_and_gradient(model, dyn, state_np, t=3.0)

    print(f"  V(origin) = {V:.6f}")
    print(f"  dvdt      = {dvdt:.6f}")
    print(f"  dvds norm = {np.linalg.norm(dvds):.6f}")

    ok = np.all(np.isfinite(dvds)) and np.linalg.norm(dvds) > 0
    print(f"  gradient finite and nonzero  [{PASS if ok else FAIL}]")

    # Verify the fix: dvds should equal raw_grad / state_scale
    # Build the raw (pre-fix) gradient for comparison:
    coord = torch.cat([
        torch.tensor([3.0], dtype=torch.float32),
        torch.tensor(state_np, dtype=torch.float32),
    ]).unsqueeze(0)
    inp = dyn.coord_to_input(coord)  # stays on CPU
    result = model({'coords': inp})
    dv_raw = dyn.io_to_dv(result['model_in'], result['model_out'].squeeze(-1))
    dvds_raw = dv_raw[0, 1:].detach().cpu().numpy().reshape(-1)

    dvds_expected = dvds_raw / dyn.state_scale
    ok2 = np.allclose(dvds, dvds_expected, rtol=1e-5)
    print(f"  dvds == raw_grad / state_scale  [{PASS if ok2 else FAIL}]")

    # Spot-check the slosh dimension: state_scale[6]=0.025 so ratio should be 1/0.025=40
    ratio_slosh = abs(dvds[6]) / (abs(dvds_raw[6]) + 1e-12)
    expected_ratio = 1.0 / dyn.state_scale[6]  # 40.0
    ok3 = abs(ratio_slosh - expected_ratio) < 1.0
    print(f"  slosh-x ratio (fixed/raw) = {ratio_slosh:.1f}  (expect ~{expected_ratio:.0f})  [{PASS if ok3 else FAIL}]")

    return ok and ok2 and ok3


# ── 3. Short end-to-end training loop ─────────────────────────────────────────
def test_short_training():
    print("\n=== Test 3: Short training loop (200 epochs) ===")
    try:
        from deepreach.utils import modules, dataio, losses
    except ImportError as e:
        print(f"  DeepReach unavailable ({e}), skipping")
        return None

    from dynamics.dynamics import CoffeeArmDynamics
    from torch.utils.data import DataLoader

    dyn = CoffeeArmDynamics()
    train_device = 'cpu'  # keep on CPU for speed in smoke test
    model = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1,
        type='sine', mode='mlp', final_layer_factor=1,
        hidden_features=64, num_hidden_layers=2,
    ).to(train_device)

    loss_fn = losses.init_brt_hjivi_loss(dyn, minWith='target', dirichlet_loss_divisor=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Use curriculum (counter_start=0): training starts near t=0 and expands.
    # This is essential so the PDE propagates spatial structure forward in time.
    dataset = dataio.ReachabilityDataset(
        dynamics=dyn, numpoints=1000, pretrain=False, pretrain_iters=0,
        tMin=0.0, tMax=3.0, counter_start=0, counter_end=200,
        num_src_samples=100, num_target_samples=0,
    )
    loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)

    # --- Pretrain phase (validates pretrain_lr fix) ---
    scale_t = torch.tensor(dyn.state_scale, dtype=torch.float32)
    pretrain_opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    model.train()
    for i in range(500):
        xn = torch.zeros(500, dyn.state_dim).uniform_(-1, 1)
        tt = torch.zeros(500, 1).uniform_(0.0, 3.0)
        coords = torch.cat([tt, xn], dim=1)
        out = model({'coords': coords})
        V_pt = dyn.io_to_value(out['model_in'].detach(), out['model_out'].squeeze(-1))
        xp = (xn * scale_t).numpy()
        lx = torch.tensor(dyn.boundary_fn(xp), dtype=torch.float32)
        pt_loss = torch.abs(V_pt - lx).mean()
        pretrain_opt.zero_grad(); pt_loss.backward(); pretrain_opt.step()

    pt_vstd = V_pt.detach().std().item()
    print(f"  pretrain done: loss={pt_loss.item():.4f}  V_std={pt_vstd:.4f}")
    ok_pt = pt_vstd > 0.15  # should be ~0.3-0.5; old 2e-5 lr gave <0.001
    print(f"  pretrain V_std > 0.15 (spatial structure learned)  [{PASS if ok_pt else FAIL}]")

    # After pretrain: V ≈ l(x), so V(safe,t=0) > V(unsafe,t=0) should hold immediately.
    model.eval()
    with torch.no_grad():
        def probe_pretrain(state_np, t=0.0):
            coord = torch.cat([torch.tensor([t]), torch.tensor(state_np, dtype=torch.float32)]).unsqueeze(0)
            inp = dyn.coord_to_input(coord).to(train_device)
            out = model({'coords': inp})
            return float(dyn.io_to_value(out['model_in'].detach(), out['model_out'].squeeze(-1).detach()).item())
        v_safe_pt   = probe_pretrain(np.zeros(10))
        unsafe_st   = np.zeros(10); unsafe_st[6] = dyn.slosh_rad_max * 1.5
        v_unsafe_pt = probe_pretrain(unsafe_st)
    print(f"  V(safe,  t=0) after pretrain = {v_safe_pt:+.4f}  (l ≈ +0.35)")
    print(f"  V(unsafe,t=0) after pretrain = {v_unsafe_pt:+.4f}  (l ≈ -0.20)")
    ok_sep = v_safe_pt > v_unsafe_pt
    print(f"  V(safe) > V(unsafe) after pretrain  [{PASS if ok_sep else FAIL}]")

    # --- HJB training phase ---
    losses_log = []
    model.train()
    for epoch in range(200):
        for model_input, gt in loader:
            model_input = {k: v.to(train_device) for k, v in model_input.items()}
            gt = {k: v.to(train_device) for k, v in gt.items()}

            results = model({'coords': model_input['model_coords']})
            states = results['model_in'].detach()[..., 1:]
            values = dyn.io_to_value(results['model_in'].detach(), results['model_out'].squeeze(-1))
            dvs = dyn.io_to_dv(results['model_in'], results['model_out'].squeeze(-1))

            step_losses = loss_fn(
                states, values,
                dvs[..., 0], dvs[..., 1:],
                gt['boundary_values'], gt['dirichlet_masks'],
                results['model_out'],
            )
            n_pde = 1000 - 100
            total_loss = step_losses['dirichlet'] + step_losses['diff_constraint_hom'] / n_pde
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        losses_log.append(total_loss.item())

    first_avg = np.mean(losses_log[:10])
    last_avg  = np.mean(losses_log[-10:])
    ok = last_avg < first_avg
    print(f"  HJB loss [0-9]    = {first_avg:.6f}")
    print(f"  HJB loss [190-199] = {last_avg:.6f}")
    print(f"  loss decreased  [{PASS if ok else FAIL}]")

    return ok_pt and ok_sep and ok


# ── 4. Obstacle boundary function sanity check ────────────────────────────────
def test_obstacle_boundary():
    print("\n=== Test 4: Obstacle boundary function ===")
    from dynamics.dynamics import CoffeeArmDynamics
    dyn = CoffeeArmDynamics()

    print(f"  Obstacles loaded: {len(dyn.obstacles)}")
    for i, obs in enumerate(dyn.obstacles):
        print(f"    [{i}] center={obs['center']}  radius={obs['radius']}")

    all_ok = True

    # For each obstacle: build a state where the arm's elbow (link 2 endpoint)
    # sits inside that obstacle sphere, then check boundary_fn < 0.
    for i, obs in enumerate(dyn.obstacles):
        cx, cy, cz = obs['center']
        r = float(obs['radius'])

        # q1 points the arm toward the obstacle in XY
        q1 = float(np.arctan2(cy, cx))
        # q2 tilts the arm to put the elbow near the obstacle height
        l1 = float(dyn.L[0])
        l2 = float(dyn.L[1])
        elev = np.clip((cz - l1) / l2, -0.98, 0.98)
        q2 = float(np.arcsin(elev))
        q3 = 0.0
        state_inside = np.array([q1, q2, q3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        lx = float(dyn.boundary_fn(state_inside))
        inside_obs = lx < 0.0
        print(f"  Obs {i}: state aimed at center → l(x)={lx:+.4f}  (expect < 0)  [{PASS if inside_obs else FAIL}]")
        all_ok = all_ok and inside_obs

    # A clearly safe state: q1=π/2 points the arm in +y, clearing all obstacles.
    # (q=[0,0,0] sends link 2 through obstacle 2 at [0.39,0,0.30] — not safe.)
    safe_state = np.array([np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    lx_safe = float(dyn.boundary_fn(safe_state))
    ok_safe = lx_safe > 0.0
    print(f"  Safe state (q1=π/2, arm in +y, zero slosh) → l(x)={lx_safe:+.4f}  (expect > 0)  [{PASS if ok_safe else FAIL}]")

    # A slosh-violated state must be < 0
    spill_state = np.zeros(10, dtype=np.float32)
    spill_state[6] = dyn.slosh_rad_max * 1.5
    lx_spill = float(dyn.boundary_fn(spill_state))
    ok_spill = lx_spill < 0.0
    print(f"  Spill state (slosh 1.5× max)  → l(x)={lx_spill:+.4f}  (expect < 0)  [{PASS if ok_spill else FAIL}]")

    return all_ok and ok_safe and ok_spill


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = []
    results.append(test_hamiltonian_scaling())
    results.append(test_safety_filter_gradient())
    results.append(test_short_training())
    results.append(test_obstacle_boundary())

    passed  = [r for r in results if r is True]
    failed  = [r for r in results if r is False]
    skipped = [r for r in results if r is None]

    print(f"\n{'='*50}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print("SMOKE TEST FAILED")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED")
