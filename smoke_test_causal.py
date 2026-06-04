#!/usr/bin/env python3
"""Smoke tests for the causal PDE weighting introduced in losses.py.

Tests:
  1. Weight monotonicity       — bins in time order get non-increasing weights
  2. Gradient flow             — weighted loss backprops through the residuals
  3. Shape variants            — (N,), (1,N), (B,N) all handled by flatten
  4. Device consistency        — no CPU/CUDA tensor mixing when run on GPU
  5. Bypass when disabled      — causal_eps=0 gives plain abs sum (no regression)
  6. Pretrain bypass           — all-dirichlet mask skips causal path entirely
  7. Edge case: n < n_bins     — doesn't crash when batch < n_bins
  8. Integration: 20-step loop — loss finite, no NaN, grad_fn present

Run from the aa276finalproject directory:
  python smoke_test_causal.py
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


def check(cond, msg):
    print(f"  {msg}  [{PASS if cond else FAIL}]")
    return cond


# ── Test 1: Weight monotonicity ───────────────────────────────────────────────
def test_weight_monotonicity():
    print("\n=== Test 1: Weight monotonicity ===")
    from deepreach.utils.losses import _causal_pde_loss

    torch.manual_seed(0)
    n = 1000
    residuals = torch.ones(n) * 2.0
    times = torch.linspace(0.0, 3.0, n)

    # With uniform nonzero residuals, bins 1..k accumulate cumulative residual, so
    # their weight = exp(-eps * cumulative) < 1.  Causal loss < plain sum.
    eps = 1.0
    loss_causal = _causal_pde_loss(residuals, times, eps=eps, n_bins=5).item()
    loss_plain  = residuals.abs().sum().item()  # = 2000

    ok1 = check(loss_causal < loss_plain,
                f"causal < plain sum: {loss_causal:.3f} < {loss_plain:.3f}")

    # eps=0 should give the plain sum
    loss_zero_eps = _causal_pde_loss(residuals, times, eps=0.0, n_bins=5).item()
    ok2 = check(abs(loss_zero_eps - loss_plain) < 1.0,
                f"eps=0 ≈ plain sum: {loss_zero_eps:.1f} ≈ {loss_plain:.1f}")

    # Causal weighting means LATE large residuals are more suppressed than EARLY ones.
    # Construct two scenarios with equal magnitude |residuals| but different positions:
    #   A: first half t ∈ [0, 1.5] has R=5, second half t ∈ [1.5, 3] has R=0.1
    #   B: first half t ∈ [0, 1.5] has R=0.1, second half t ∈ [1.5, 3] has R=5
    # In A: early bins get w=1 and large R → large contribution; late bins suppressed → R_late suppressed
    # In B: early bins get w=1 but small R; late bins get w≈exp(-small)≈1 but large R
    # Net: loss(A) > loss(B) because big residuals at w=1 vs big residuals at w≈exp(-small)<1
    half = n // 2
    res_A = torch.cat([torch.ones(half) * 5.0, torch.ones(n - half) * 0.1])
    res_B = torch.cat([torch.ones(half) * 0.1, torch.ones(n - half) * 5.0])

    loss_A = _causal_pde_loss(res_A, times, eps=eps, n_bins=2).item()
    loss_B = _causal_pde_loss(res_B, times, eps=eps, n_bins=2).item()

    # A: early bins are at full weight w=1 and have R=5 → dominate
    # B: late bins have large R but early R is small so their weight ≈ exp(-0.1) < 1
    # A > B confirms early bins get higher effective weight than late bins
    ok3 = check(loss_A > loss_B,
                f"early high-R > late high-R (A={loss_A:.1f}, B={loss_B:.1f}): "
                f"early bins at weight=1, late bins downweighted")

    return ok1 and ok2 and ok3


# ── Test 2: Gradient flow ─────────────────────────────────────────────────────
def test_gradient_flow():
    print("\n=== Test 2: Gradient flow ===")
    from deepreach.utils.losses import _causal_pde_loss

    torch.manual_seed(1)
    n = 500
    # residuals must have requires_grad=True to test backprop
    residuals = torch.randn(n, requires_grad=True)
    times = torch.rand(n) * 3.0

    loss = _causal_pde_loss(residuals, times, eps=1.0, n_bins=10)

    ok1 = check(loss.requires_grad, "loss.requires_grad is True")
    ok2 = check(loss.grad_fn is not None, "loss.grad_fn is not None")

    loss.backward()
    ok3 = check(residuals.grad is not None, "residuals.grad computed")
    ok4 = check(torch.all(torch.isfinite(residuals.grad)), "gradients are finite")
    ok5 = check(residuals.grad.abs().sum().item() > 0, "gradients are nonzero")

    return ok1 and ok2 and ok3 and ok4 and ok5


# ── Test 3: Shape variants ────────────────────────────────────────────────────
def test_shape_variants():
    print("\n=== Test 3: Shape variants (N), (1,N), (B,N) ===")
    from deepreach.utils.losses import _causal_pde_loss

    torch.manual_seed(2)
    n = 200

    res_1d   = torch.randn(n)
    res_2d   = res_1d.unsqueeze(0)        # (1, n) — DataLoader batch_size=1
    res_3d   = res_1d.unsqueeze(0).unsqueeze(0)  # (1, 1, n)

    t_1d = torch.rand(n) * 3.0
    t_2d = t_1d.unsqueeze(0)
    t_3d = t_1d.unsqueeze(0).unsqueeze(0)

    eps = 1.0
    ref = _causal_pde_loss(res_1d, t_1d, eps, n_bins=5).item()

    try:
        v2 = _causal_pde_loss(res_2d, t_2d, eps, n_bins=5).item()
        ok1 = check(abs(v2 - ref) < 1e-4, f"(1,N) matches (N): {v2:.5f} ≈ {ref:.5f}")
    except Exception as e:
        ok1 = check(False, f"(1,N) raised: {e}")

    try:
        v3 = _causal_pde_loss(res_3d, t_3d, eps, n_bins=5).item()
        ok2 = check(abs(v3 - ref) < 1e-4, f"(1,1,N) matches (N): {v3:.5f} ≈ {ref:.5f}")
    except Exception as e:
        ok2 = check(False, f"(1,1,N) raised: {e}")

    return ok1 and ok2


# ── Test 4: Device consistency ────────────────────────────────────────────────
def test_device_consistency():
    print("\n=== Test 4: Device consistency ===")
    from deepreach.utils.losses import _causal_pde_loss

    if not torch.cuda.is_available():
        print("  (no CUDA, skipping GPU half)")
        return None

    torch.manual_seed(3)
    n = 300
    res_gpu = torch.randn(n, device='cuda', requires_grad=True)
    t_gpu   = torch.rand(n, device='cuda') * 3.0

    try:
        loss = _causal_pde_loss(res_gpu, t_gpu, eps=1.0, n_bins=10)
        ok1 = check(loss.device.type == 'cuda', f"loss on CUDA: {loss.device}")
        loss.backward()
        ok2 = check(res_gpu.grad is not None and res_gpu.grad.device.type == 'cuda',
                    "grad on CUDA")
        ok3 = check(torch.all(torch.isfinite(res_gpu.grad)), "CUDA grads finite")
    except Exception as e:
        ok1 = ok2 = ok3 = check(False, f"CUDA raised: {e}")

    return ok1 and ok2 and ok3


# ── Test 5: eps=0 bypass (no regression) ─────────────────────────────────────
def test_eps_zero_bypass():
    print("\n=== Test 5: eps=0 gives plain abs sum (regression guard) ===")
    from deepreach.utils.losses import init_brt_hjivi_loss
    from dynamics.dynamics import CoffeeArmDynamics

    dyn = CoffeeArmDynamics()
    loss_fn_plain   = init_brt_hjivi_loss(dyn, minWith='target', dirichlet_loss_divisor=100, causal_eps=0.0)
    loss_fn_causal  = init_brt_hjivi_loss(dyn, minWith='target', dirichlet_loss_divisor=100, causal_eps=1.0)

    torch.manual_seed(4)
    n = 200
    state = torch.randn(n, 10) * 0.1
    value = torch.randn(n)
    dvdt  = torch.randn(n)
    dvds  = torch.randn(n, 10) * 0.01
    bv    = torch.randn(n) * 0.3
    mask  = torch.zeros(n, dtype=torch.bool)   # no dirichlet points
    mask[-10:] = True
    out   = torch.randn(n, 1)
    times = torch.rand(n) * 3.0

    l_plain  = loss_fn_plain(state, value, dvdt, dvds, bv, mask, out)
    l_causal = loss_fn_causal(state, value, dvdt, dvds, bv, mask, out, times=times)

    # Without times, causal path is also bypassed
    l_notimes = loss_fn_causal(state, value, dvdt, dvds, bv, mask, out, times=None)

    ok1 = check(abs(l_plain['diff_constraint_hom'].item() -
                    l_notimes['diff_constraint_hom'].item()) < 1e-4,
                "times=None → same as causal_eps=0")

    # With causal, loss ≤ plain (some weights < 1)
    ok2 = check(l_causal['diff_constraint_hom'].item() <=
                l_plain['diff_constraint_hom'].item() + 1e-4,
                f"causal ≤ plain: {l_causal['diff_constraint_hom'].item():.4f} ≤ {l_plain['diff_constraint_hom'].item():.4f}")

    return ok1 and ok2


# ── Test 6: Pretrain bypass (all-dirichlet mask) ──────────────────────────────
def test_pretrain_bypass():
    print("\n=== Test 6: Pretrain bypass (all-dirichlet mask) ===")
    from deepreach.utils.losses import init_brt_hjivi_loss
    from dynamics.dynamics import CoffeeArmDynamics

    dyn = CoffeeArmDynamics()
    loss_fn = init_brt_hjivi_loss(dyn, minWith='target', dirichlet_loss_divisor=100, causal_eps=5.0)

    torch.manual_seed(5)
    n = 100
    state = torch.randn(n, 10) * 0.1
    value = torch.randn(n)
    dvdt  = torch.randn(n)
    dvds  = torch.randn(n, 10) * 0.01
    bv    = torch.randn(n) * 0.3
    mask  = torch.ones(n, dtype=torch.bool)   # ALL dirichlet (pretrain)
    out   = torch.randn(n, 1)
    times = torch.zeros(n)  # all t=0 during pretrain

    result = loss_fn(state, value, dvdt, dvds, bv, mask, out, times=times)

    ok1 = check('dirichlet' in result, "dirichlet key present during pretrain")
    ok2 = check('diff_constraint_hom' not in result or
                result.get('diff_constraint_hom', torch.tensor(0.0)).item() == 0.0,
                "diff_constraint_hom is 0 during pretrain (PDE not evaluated)")

    return ok1 and ok2


# ── Test 7: Edge case n < n_bins ──────────────────────────────────────────────
def test_small_batch():
    print("\n=== Test 7: Edge case n < n_bins ===")
    from deepreach.utils.losses import _causal_pde_loss

    torch.manual_seed(6)
    # 5 points, 10 bins
    res   = torch.randn(5, requires_grad=True)
    times = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0])

    try:
        loss = _causal_pde_loss(res, times, eps=1.0, n_bins=10)
        ok1 = check(torch.isfinite(loss), f"finite loss with n=5, bins=10: {loss.item():.4f}")
        loss.backward()
        ok2 = check(res.grad is not None and torch.all(torch.isfinite(res.grad)),
                    "finite gradients with n=5, bins=10")
    except Exception as e:
        ok1 = ok2 = check(False, f"raised: {e}")

    return ok1 and ok2


# ── Test 8: Integration — 20-step training loop ───────────────────────────────
def test_integration():
    print("\n=== Test 8: Integration (20 training steps with causal_eps=1.0) ===")
    try:
        from deepreach.utils import modules, dataio, losses
    except ImportError as e:
        print(f"  DeepReach unavailable ({e}), skipping")
        return None

    from dynamics.dynamics import CoffeeArmDynamics
    from torch.utils.data import DataLoader

    dyn = CoffeeArmDynamics()
    model = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1,
        type='sine', mode='mlp', final_layer_factor=1,
        hidden_features=32, num_hidden_layers=2,
    ).cpu()

    loss_fn = losses.init_brt_hjivi_loss(
        dyn, minWith='target', dirichlet_loss_divisor=50,
        causal_eps=1.0, n_causal_bins=10,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    dataset = dataio.ReachabilityDataset(
        dynamics=dyn, numpoints=500, pretrain=False, pretrain_iters=0,
        tMin=0.0, tMax=3.0, counter_start=50, counter_end=200,
        num_src_samples=50, num_target_samples=0,
    )
    loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)

    all_ok = True
    model.train()
    for step, (model_input, gt) in enumerate(loader):
        if step >= 20:
            break
        results = model({'coords': model_input['model_coords']})
        states = results['model_in'].detach()[..., 1:]
        values = dyn.io_to_value(results['model_in'].detach(), results['model_out'].squeeze(-1))
        dvs    = dyn.io_to_dv(results['model_in'], results['model_out'].squeeze(-1))

        step_losses = loss_fn(
            states, values,
            dvs[..., 0], dvs[..., 1:],
            gt['boundary_values'], gt['dirichlet_masks'],
            results['model_out'],
            times=model_input['model_coords'][..., 0],
        )

        n_pde = 500 - 50
        total_loss = step_losses['dirichlet'] + step_losses['diff_constraint_hom'] / n_pde

        ok_finite = torch.isfinite(total_loss)
        ok_grad   = total_loss.grad_fn is not None
        if not (ok_finite and ok_grad):
            print(f"  step {step}: loss={total_loss.item():.5f}  finite={ok_finite}  grad_fn={ok_grad}")
            all_ok = False

        optimizer.zero_grad()
        total_loss.backward()

        # Check gradients didn't explode
        max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
        ok_grads = max_grad < 1e6
        if not ok_grads:
            print(f"  step {step}: gradient explosion max_grad={max_grad:.2e}")
            all_ok = False

        optimizer.step()

    ok1 = check(all_ok, "all 20 steps: finite loss, grad_fn present, no gradient explosion")

    # Check that the dirichlet loss is small (boundary condition honored)
    dir_loss = step_losses['dirichlet'].item()
    ok2 = check(dir_loss < 50.0, f"dirichlet loss finite: {dir_loss:.4f}")

    return ok1 and ok2


# ── Test 9: Static trajectory loss ───────────────────────────────────────────
def test_static_traj_loss():
    print("\n=== Test 9: Static safe trajectory loss ===")
    try:
        from deepreach.utils import modules, dataio, losses
    except ImportError as e:
        print(f"  DeepReach unavailable ({e}), skipping")
        return None

    from dynamics.dynamics import CoffeeArmDynamics
    from torch.utils.data import DataLoader

    dyn = CoffeeArmDynamics()
    model = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1,
        type='sine', mode='mlp', final_layer_factor=1,
        hidden_features=32, num_hidden_layers=2,
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    model.train()

    # Build a tiny static safe pool
    rng = np.random.default_rng(42)
    chunks, collected = [], 0
    while collected < 2000:
        q  = rng.uniform(-1, 1, (1024, 3)).astype(np.float32) * dyn.joint_limits
        z  = np.zeros((1024, 7), dtype=np.float32)
        c  = np.concatenate([q, z], axis=1)
        lx = dyn.boundary_fn(c)
        keep = c[lx > 0.05]
        chunks.append(keep)
        collected += len(keep)
    static_phys = np.concatenate(chunks)[:2000]
    static_lx   = dyn.boundary_fn(static_phys).astype(np.float32)
    static_norm = (static_phys / dyn.state_scale).astype(np.float32)
    static_norm_t = torch.tensor(static_norm, dtype=torch.float32)
    static_lx_t   = torch.tensor(static_lx,   dtype=torch.float32)
    N_st = len(static_phys)

    ok1 = check(N_st >= 100, f"pool built: {N_st} static safe states")
    ok2 = check(float(static_lx.min()) > 0.04, f"all l(x) > 0.04: min={static_lx.min():.4f}")

    # Simulate what the training loop does: compute trajectory loss and backprop
    loss_fn = losses.init_brt_hjivi_loss(dyn, minWith='target', dirichlet_loss_divisor=50,
                                          causal_eps=1.0)
    dataset = dataio.ReachabilityDataset(
        dynamics=dyn, numpoints=200, pretrain=False, pretrain_iters=0,
        tMin=0.0, tMax=3.0, counter_start=100, counter_end=200,
        num_src_samples=20, num_target_samples=0,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    all_ok = True
    for step, (model_input, gt) in enumerate(loader):
        if step >= 5:
            break

        results = model({'coords': model_input['model_coords']})
        states = results['model_in'].detach()[..., 1:]
        values = dyn.io_to_value(results['model_in'].detach(), results['model_out'].squeeze(-1))
        dvs    = dyn.io_to_dv(results['model_in'], results['model_out'].squeeze(-1))

        step_losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:],
                               gt['boundary_values'], gt['dirichlet_masks'],
                               results['model_out'],
                               times=model_input['model_coords'][..., 0])
        n_pde = 200 - 20
        total_loss = step_losses['dirichlet'] + step_losses['diff_constraint_hom'] / n_pde

        # Add trajectory loss
        n_st = 64
        idx  = torch.randint(0, N_st, (n_st,))
        norm = static_norm_t[idx]
        lx   = static_lx_t[idx]
        t_samp = torch.empty(n_st, 1).uniform_(0.0, 1.5)
        coords = torch.cat([t_samp, norm], dim=1)
        st_out = model({'coords': coords})
        V_st   = dyn.io_to_value(st_out['model_in'].detach(), st_out['model_out'].squeeze(-1))
        traj_loss = torch.abs(V_st - lx).mean()

        ok_finite_traj = torch.isfinite(traj_loss)
        ok_grad_traj   = traj_loss.grad_fn is not None
        if not (ok_finite_traj and ok_grad_traj):
            all_ok = False

        total_loss = total_loss + 5.0 * traj_loss
        optimizer.zero_grad()
        total_loss.backward()

        max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
        if max_grad >= 1e6:
            print(f"  step {step}: gradient explosion {max_grad:.2e}")
            all_ok = False

        optimizer.step()

    ok3 = check(all_ok, "5 steps with trajectory loss: finite, grad_fn, no explosion")
    ok4 = check(traj_loss.item() > 0, f"trajectory loss is nonzero (not trivially zero): {traj_loss.item():.4f}")

    return ok1 and ok2 and ok3 and ok4


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = {
        'weight_monotonicity':   test_weight_monotonicity(),
        'gradient_flow':         test_gradient_flow(),
        'shape_variants':        test_shape_variants(),
        'device_consistency':    test_device_consistency(),
        'eps_zero_bypass':       test_eps_zero_bypass(),
        'pretrain_bypass':       test_pretrain_bypass(),
        'small_batch':           test_small_batch(),
        'integration':           test_integration(),
        'static_traj_loss':      test_static_traj_loss(),
    }

    passed  = [k for k, v in results.items() if v is True]
    failed  = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]

    print(f"\n{'='*50}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if skipped:
        print(f"Skipped: {skipped}")
    if failed:
        print(f"Failed:  {failed}")
        print("SMOKE TEST FAILED")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED")
