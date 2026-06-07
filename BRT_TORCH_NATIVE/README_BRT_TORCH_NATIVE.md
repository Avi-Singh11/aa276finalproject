# BRT_TORCH_NATIVE

This directory is a separate Torch-native BRT pipeline. It does not modify or
overwrite the existing project files or model checkpoints.

## Files

- `BRT_TORCH_NATIVE_dynamics.py`: Torch-native target function plus audited
  vector field and exact box-control Hamiltonian.
- `BRT_TORCH_NATIVE_compute.py`: GPU-resident sampling and training entry point.
- `BRT_TORCH_NATIVE_preflight.py`: equivalence and correctness checks.

## Important differences

- The boundary condition is pretrained only at `t=0`.
- No nonphysical "static trajectory" labels are imposed at positive times.
- The Hamiltonian is not clipped.
- Joint damping is included everywhere, including cup/slosh acceleration.
- The joint-velocity domain covers `[-15, 15]` rad/s.
- Targeted sample counts account for the actual number of obstacles.
- Boundary labels and uniform/safe pools stay on the GPU.
- Output goes to `BRT_TORCH_NATIVE_model*`, never existing model directories.

## Commands

```bash
/home/avisingh/miniconda3/bin/python BRT_TORCH_NATIVE/BRT_TORCH_NATIVE_preflight.py
/home/avisingh/miniconda3/bin/python BRT_TORCH_NATIVE/BRT_TORCH_NATIVE_compute.py --benchmark
/home/avisingh/miniconda3/bin/python BRT_TORCH_NATIVE/BRT_TORCH_NATIVE_compute.py --pilot
/home/avisingh/miniconda3/bin/python BRT_TORCH_NATIVE/BRT_TORCH_NATIVE_compute.py
```

Run the preflight first. The training script also runs it automatically and
refuses to train if the Hamiltonian, vector field, or boundary implementation
is inconsistent.
