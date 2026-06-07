# BRT_VERIFIED

This directory is a separate, audited BRT pipeline. It does not modify or
overwrite the existing project files or model checkpoints.

## Files

- `BRT_VERIFIED_dynamics.py`: consistent damped arm, cup acceleration, slosh
  vector field, target function, and exact box-control Hamiltonian.
- `BRT_VERIFIED_compute.py`: DeepReach training entry point.
- `BRT_VERIFIED_preflight.py`: cheap checks that run before any training.

## Important differences

- The boundary condition is pretrained only at `t=0`.
- No nonphysical "static trajectory" labels are imposed at positive times.
- The Hamiltonian is not clipped.
- Joint damping is included everywhere, including cup/slosh acceleration.
- The joint-velocity domain covers `[-15, 15]` rad/s.
- Targeted sample counts account for the actual number of obstacles.
- Output goes to `BRT_VERIFIED_model*`, never `brt_model`.

## Commands

```bash
/home/avisingh/miniconda3/bin/python BRT_VERIFIED/BRT_VERIFIED_preflight.py
/home/avisingh/miniconda3/bin/python BRT_VERIFIED/BRT_VERIFIED_compute.py --benchmark
/home/avisingh/miniconda3/bin/python BRT_VERIFIED/BRT_VERIFIED_compute.py --pilot
/home/avisingh/miniconda3/bin/python BRT_VERIFIED/BRT_VERIFIED_compute.py --fast
/home/avisingh/miniconda3/bin/python BRT_VERIFIED/BRT_VERIFIED_compute.py
```

Run the preflight first. The training script also runs it automatically and
refuses to train if the Hamiltonian, vector field, or boundary implementation
is inconsistent.
