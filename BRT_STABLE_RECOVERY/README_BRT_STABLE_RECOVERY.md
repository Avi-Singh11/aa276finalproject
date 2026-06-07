# BRT_STABLE_RECOVERY

This directory recovers from the last finite exact-boundary checkpoint without
modifying any earlier training files or checkpoints.

```bash
/home/avisingh/miniconda3/bin/python \
  BRT_STABLE_RECOVERY/BRT_STABLE_RECOVERY_compute.py \
  --recover BRT_EXACT_BOUNDARY_model_pilot/checkpoints/model_epoch_006000.pth
```

Recovery uses learning rate `2e-6`, gradient clipping at norm `1.0`, finite
loss/gradient/parameter checks, and checkpoints every 500 epochs.

This directory is a separate Stable-recovery BRT pipeline. It does not modify or
overwrite the existing project files or model checkpoints.

## Files

- `BRT_STABLE_RECOVERY_dynamics.py`: Stable-recovery target function plus audited
  vector field and exact box-control Hamiltonian.
- `BRT_STABLE_RECOVERY_compute.py`: GPU-resident sampling and training entry point.
- `BRT_STABLE_RECOVERY_preflight.py`: equivalence and correctness checks.

## Important differences

- The boundary condition is pretrained only at `t=0`.
- No nonphysical "static trajectory" labels are imposed at positive times.
- The Hamiltonian is not clipped.
- Joint damping is included everywhere, including cup/slosh acceleration.
- The joint-velocity domain covers `[-15, 15]` rad/s.
- Targeted sample counts account for the actual number of obstacles.
- Boundary labels and uniform/safe pools stay on the GPU.
- Output goes to `BRT_STABLE_RECOVERY_model*`, never existing model directories.

## Commands

```bash
/home/avisingh/miniconda3/bin/python BRT_STABLE_RECOVERY/BRT_STABLE_RECOVERY_preflight.py
/home/avisingh/miniconda3/bin/python BRT_STABLE_RECOVERY/BRT_STABLE_RECOVERY_compute.py --benchmark
/home/avisingh/miniconda3/bin/python BRT_STABLE_RECOVERY/BRT_STABLE_RECOVERY_compute.py --pilot
/home/avisingh/miniconda3/bin/python BRT_STABLE_RECOVERY/BRT_STABLE_RECOVERY_compute.py
```

Run the preflight first. The training script also runs it automatically and
refuses to train if the Hamiltonian, vector field, or boundary implementation
is inconsistent.
