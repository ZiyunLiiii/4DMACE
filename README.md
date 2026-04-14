# 4DCT MACE Recon

This repository is a storage and execution home for stable 4D CT reconstructions using MACE, with a separate area for work in progress.

## Layout

- `4DCT_serial.py`: stable serial reconstruction entry point.
- `under_dev/4DMACE_multi_gpu.py`: experimental multi-GPU implementation.
- `configs/`: checked-in run configurations.
- `stable_versions/`: place to store release notes, manifests, and exported stable artifacts metadata.

## Intended use

Use this repo to keep reproducible versions of the reconstruction code and the configuration used to generate each stable result. Large raw datasets and large output arrays should usually live outside git unless you explicitly want them versioned with Git LFS.

## Quick start

1. Create a Python environment with `mbirjax`, `jax`, `numpy`, and their runtime dependencies installed.
2. Copy [configs/serial_example.json](/Users/a124601/Desktop/Research_Purdue/4DCT/configs/serial_example.json) and edit the dataset and output paths for your machine.
3. Run:

```bash
python 4DCT_serial.py --config configs/serial_example.json
```

## Stable version workflow

For each version you want to keep stable:

1. Commit the code that produced it.
2. Save the exact config used for that run under `configs/` or `stable_versions/`.
3. Add a short note in `stable_versions/README.md` with the commit hash, dataset identifier, and output location.

## Notes

- The current stable path is the serial implementation.
- The multi-GPU script is kept under `under_dev/` because it still assumes a 4-GPU environment and hardcoded runtime behavior.
