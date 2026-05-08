# 4DCT MACE Recon

This repository is a storage and execution home for stable 4D CT reconstructions using MACE, with a separate area for work in progress.

## Layout

- `4DMACE_serial/4DCT_serial.py`: serial reconstruction script.
- `4DMACE_multi_threads/4DMACE_multi_threads.py`: multi-threaded 4D MACE script.
- `4DMACE_multi_threads/utilities_4D_multi_threads.py`: multi-GPU 4D MACE utilities.


## Multi-GPU MACE

The multi-threaded implementation runs four MACE agents concurrently with `ThreadPoolExecutor(4)`:

- Agent 0: cone-beam `prox_map`
- Agent 1: qGGMRF denoiser on XY-t hyperplanes
- Agent 2: qGGMRF denoiser on YZ-t hyperplanes
- Agent 3: qGGMRF denoiser on XZ-t hyperplanes

GPU assignment is controlled in `utilities_4D_multi_threads.py` by:

```python
agent_device_indices = [0, 1, 2, 3]
```

The code discovers devices with `jax.devices("gpu")` and passes the selected device index into each agent. The current implementation requires at least four JAX-visible GPUs.

## Timing Log

`4DMACE_multi_threads/4DMACE_multi_threads.py` writes a per-iteration timing CSV to:

```text
./output/timing_log.csv
```

The CSV columns are:

```text
iteration
agent_0_forward_sec
agent_1_prior_xyt_sec
agent_2_prior_yzt_sec
agent_3_prior_xzt_sec
iteration_total_sec
```

This log is useful on HPC systems where runtime can vary between runs because of node assignment, shared filesystem load, GPU contention, JAX/XLA compilation, CPU thread contention, or data-transfer overhead.

If calling the utility function directly, pass:

```python
recon_4d = mace4d_from_cone_beam_params(
    ...,
    timing_log_path="./output/timing_log.csv",
)
```
