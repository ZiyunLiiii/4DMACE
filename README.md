# 4DCT MACE Recon

This repository is a storage and execution home for stable 4D CT reconstructions using MACE, with a separate area for work in progress.

## Layout

- `4DCT_serial.py`: stable serial reconstruction entry point.
- `under_dev/4DMACE_multi_gpu.py`: experimental multi-GPU implementation.

## Intended use

Use this repo to keep reproducible versions of the reconstruction code and the configuration used to generate each stable result. Large raw datasets and large output arrays should usually live outside git unless you explicitly want them versioned with Git LFS.
