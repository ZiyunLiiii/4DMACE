"""
4D MACE reconstruction with multi(>=4)-GPU support.

GPU assignment is controlled by agent_device_indices:
  - Agent 0: cone-beam prox_map (serial over time bins, original logic)
  - Agent 1: qGGMRF denoiser, XY-t hyperplanes
  - Agent 2: qGGMRF denoiser, YZ-t hyperplanes
  - Agent 3: qGGMRF denoiser, XZ-t hyperplanes

All 4 agents are dispatched concurrently via ThreadPoolExecutor(4).
Each prior-agent thread uses jax.default_device() to pin all JAX ops (including
QGGMRFDenoiser) to its assigned GPU.
init_image (large) lives on CPU as a NumPy array throughout.
Per-thread denoiser caching avoids cross-thread JAX state sharing.
"""

from __future__ import annotations

import os
import sys

# Ensure JAX does not inherit incompatible system CUDA/cuDNN from LD_LIBRARY_PATH.
if "LD_LIBRARY_PATH" in os.environ and not os.environ.get("_JAX_CLEAN_REEXEC"):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["_JAX_CLEAN_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

import concurrent.futures
import csv
import threading
import time

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import mbirjax as mj
import numpy as np

# ---------------------------------------------------------------------------
# Thread-local denoiser cache
# ---------------------------------------------------------------------------

_THREAD_LOCAL = threading.local()


def get_qggmrf_denoiser(shape):
    """Return a per-thread cached QGGMRFDenoiser to avoid cross-thread state sharing."""
    cache = getattr(_THREAD_LOCAL, "denoiser_cache", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.denoiser_cache = cache
    if shape not in cache:
        cache[shape] = mj.QGGMRFDenoiser(shape)
    return cache[shape]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_prior_weights(prior_weight):
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        prior = list(prior_weight)
        return [1.0 - sum(prior)] + prior
    w = prior_weight
    return [1.0 - w, w / 3.0, w / 3.0, w / 3.0]


def estimate_sigma_per_hyperplane(x, sigma_noise_floor=1e-6):
    """
    Estimate one sigma value per hyperplane.
    x shape: (num_hyperplanes, dim1, dim2)
    """
    denoiser = get_qggmrf_denoiser(x.shape[1:])
    sigma_list = np.empty(x.shape[0], dtype=np.float32)
    for i in range(x.shape[0]):
        sigma_use = denoiser.estimate_image_noise_std(x[i][:, ::4, ::4])
        if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
            sigma_use = 0.0
        sigma_list[i] = sigma_use
    return sigma_list


def qggmrf_hyperplane_denoise(x, sigma_list, device, sigma_noise_floor=1e-6):
    """
    Denoise a stack of hyperplanes on the given JAX device.
    x shape: (num_hyperplanes, dim1, dim2)
    All JAX ops inside QGGMRFDenoiser run on `device` via jax.default_device().
    """
    y = np.empty_like(x)
    with jax.default_device(device):
        denoiser = get_qggmrf_denoiser(x.shape[1:])

        for i in range(x.shape[0]):
            sigma_use = sigma_list[i]
            if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
                y[i] = x[i]
            else:
                image_i = jax.device_put(jnp.asarray(x[i]), device)
                y_i, _ = denoiser.denoise(image=image_i, sigma_noise=sigma_use)
                y[i] = np.asarray(y_i)
    return y


def denoiser_wrapper(x, permute_vector, sigma_list, device):
    """
    Permute 4D volume -> denoise hyperplane stack on device -> permute back.
    x shape: (nt, nx, ny, nz)
    """
    x_perm = np.transpose(x, permute_vector)
    y_perm = qggmrf_hyperplane_denoise(x_perm, sigma_list=sigma_list, device=device)
    inv_perm = np.argsort(permute_vector)
    return np.transpose(y_perm, inv_perm)


# ---------------------------------------------------------------------------
# Multi-GPU MACE core
# ---------------------------------------------------------------------------

def run_mace_with_models_multigpu(
    models,
    sino_list,
    weights_list,
    beta,
    max_admm_itr=10,
    rho=0.5,
    forward_num_iterations=3,
    stop_threshold=0.02,
    init_image=None,
    sigma_p=None,
    verbose=1,
    init_save_dir=None,
    timing_log_path=None,
):
    nt = len(sino_list)

    # ── GPU device discovery ───────────────────────────────────────────────
    devices = jax.devices("gpu")
    n_gpu = len(devices)
    if n_gpu == 0:
        raise RuntimeError("No GPU devices found by JAX.")
    if n_gpu < 4:
        raise RuntimeError(f"Need at least 4 GPUs, found {n_gpu}.")
    agent_device_indices = [0, 1, 2, 3]
    if verbose:
        print(f"[MACE] Found {n_gpu} GPU(s): {devices}")
        print(
            "[MACE] GPU assignment: "
            + ", ".join(f"Agent{k}->GPU{idx}" for k, idx in enumerate(agent_device_indices))
        )
        print(f"[MACE] Start 4D reconstruction with {nt} time bins.")

    # ── Initialisation — serial on Agent 0's device, identical to original ─
    init_device_index = agent_device_indices[0]
    init_device = devices[init_device_index]
    if init_image is None:
        if verbose:
            print(
                f"[MACE] Computing initial MBIR recon for each time bin "
                f"on GPU {init_device_index} (serial)..."
            )
        t0 = time.time()
        init_image = np.stack([
            np.asarray(
                models[t].recon(
                    jax.device_put(jnp.asarray(sino_list[t]), init_device),
                    weights=jax.device_put(jnp.asarray(weights_list[t]), init_device),
                    max_iterations=20,
                    stop_threshold_change_pct=stop_threshold,
                )[0]
            )
            for t in range(nt)
        ])
        if init_save_dir is not None:
            os.makedirs(init_save_dir, exist_ok=True)
            np.save(os.path.join(init_save_dir, "init_image.npy"), init_image)
        if verbose:
            print(f"[MACE] Initialisation done in {time.time() - t0:.2f} sec.")
    else:
        init_image = np.asarray(init_image)
        if verbose:
            print("[MACE] Using provided init_image.")

    # ── Pre-compute sigma lists (CPU, one-time) ────────────────────────────
    if verbose:
        print("[MACE] Precomputing sigma lists...")
    sigma_xyt = estimate_sigma_per_hyperplane(np.transpose(init_image, (3, 0, 1, 2)))
    sigma_yzt = estimate_sigma_per_hyperplane(np.transpose(init_image, (1, 0, 2, 3)))
    sigma_xzt = estimate_sigma_per_hyperplane(np.transpose(init_image, (2, 0, 1, 3)))
    if verbose:
        print(
            "[MACE] Nonzero sigma counts: "
            f"XY-t={np.count_nonzero(sigma_xyt > 1e-6)}/{sigma_xyt.size}, "
            f"YZ-t={np.count_nonzero(sigma_yzt > 1e-6)}/{sigma_yzt.size}, "
            f"XZ-t={np.count_nonzero(sigma_xzt > 1e-6)}/{sigma_xzt.size}"
        )
        print("[MACE] Sigma precomputation done.")

    # ── ADMM state (all on CPU / NumPy) ───────────────────────────────────
    W = [np.copy(init_image) for _ in range(4)]
    X = [np.copy(init_image) for _ in range(4)]  # warm-start for X[0]
    timing_log = []

    if timing_log_path is not None:
        timing_log_dir = os.path.dirname(timing_log_path)
        if timing_log_dir:
            os.makedirs(timing_log_dir, exist_ok=True)
        with open(timing_log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "agent_0_forward_sec",
                    "agent_1_prior_xyt_sec",
                    "agent_2_prior_yzt_sec",
                    "agent_3_prior_xzt_sec",
                    "iteration_total_sec",
                ],
            )
            writer.writeheader()

    # ── Agent definitions ──────────────────────────────────────────────────

    def run_forward_agent(W_k, X_prev, device_index):
        """
        Agent 0: cone-beam prox_map, serial over time bins, pinned to device_index.
        """
        device = devices[device_index]
        agent_t0 = time.time()
        out = np.stack([
            np.asarray(
                models[t].prox_map(
                    prox_input=jax.device_put(jnp.asarray(W_k[t]), device),
                    sinogram=jax.device_put(jnp.asarray(sino_list[t]), device),
                    sigma_prox=sigma_p,
                    weights=jax.device_put(jnp.asarray(weights_list[t]), device),
                    init_recon=jax.device_put(jnp.asarray(X_prev[t]), device),
                    max_iterations=forward_num_iterations,
                    stop_threshold_change_pct=stop_threshold,
                )[0]
            )
            for t in range(nt)
        ])
        agent_sec = time.time() - agent_t0
        if verbose:
            print(f"[MACE]  Agent 0 ran on {device} in {agent_sec:.2f} sec.")
        return out, agent_sec

    def run_prior_agent_1(W_k, device_index):
        """Agent 1: qGGMRF XY-t (fixed z slabs)."""
        device = devices[device_index]
        agent_t0 = time.time()
        out = denoiser_wrapper(W_k, permute_vector=(3, 0, 1, 2), sigma_list=sigma_xyt, device=device)
        agent_sec = time.time() - agent_t0
        if verbose:
            print(f"[MACE]  Agent 1 ran on {device} in {agent_sec:.2f} sec.")
        return out, agent_sec

    def run_prior_agent_2(W_k, device_index):
        """Agent 2: qGGMRF YZ-t (fixed row slabs)."""
        device = devices[device_index]
        agent_t0 = time.time()
        out = denoiser_wrapper(W_k, permute_vector=(1, 0, 2, 3), sigma_list=sigma_yzt, device=device)
        agent_sec = time.time() - agent_t0
        if verbose:
            print(f"[MACE]  Agent 2 ran on {device} in {agent_sec:.2f} sec.")
        return out, agent_sec

    def run_prior_agent_3(W_k, device_index):
        """Agent 3: qGGMRF XZ-t (fixed col slabs)."""
        device = devices[device_index]
        agent_t0 = time.time()
        out = denoiser_wrapper(W_k, permute_vector=(2, 0, 1, 3), sigma_list=sigma_xzt, device=device)
        agent_sec = time.time() - agent_t0
        if verbose:
            print(f"[MACE]  Agent 3 ran on {device} in {agent_sec:.2f} sec.")
        return out, agent_sec

    # ── Main MACE loop ─────────────────────────────────────────────────────
    for itr in range(max_admm_itr):
        itr_t0 = time.time()
        if verbose:
            print(f"\n[MACE] ── Iteration {itr + 1}/{max_admm_itr} ──")

        # Snapshot W so all agents see a consistent state for this iteration.
        W_snap = [np.copy(W[k]) for k in range(4)]
        agent_times = {}

        # All 4 agents run concurrently on the indexed devices.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(run_forward_agent, W_snap[0], X[0], agent_device_indices[0]): (0, "forward"),
                pool.submit(run_prior_agent_1, W_snap[1], agent_device_indices[1]): (1, "prior XY-t"),
                pool.submit(run_prior_agent_2, W_snap[2], agent_device_indices[2]): (2, "prior YZ-t"),
                pool.submit(run_prior_agent_3, W_snap[3], agent_device_indices[3]): (3, "prior XZ-t"),
            }

            for fut in concurrent.futures.as_completed(futures):
                agent_id, agent_name = futures[fut]
                done_t0 = time.time()
                X[agent_id], agent_times[agent_id] = fut.result()
                if verbose:
                    print(
                        f"[MACE]  Agent {agent_id} ({agent_name}) done "
                        f"at +{done_t0 - itr_t0:.2f} sec."
                    )

        if verbose:
            print("[MACE]  All agents done. Running consensus update...")

        # MACE consensus update (CPU)
        z = sum(beta[k] * (2.0 * X[k] - W[k]) for k in range(4))
        for k in range(4):
            W[k] = W[k] + 2.0 * rho * (z - X[k])

        iteration_sec = time.time() - itr_t0
        timing_row = {
            "iteration": itr + 1,
            "agent_0_forward_sec": agent_times[0],
            "agent_1_prior_xyt_sec": agent_times[1],
            "agent_2_prior_yzt_sec": agent_times[2],
            "agent_3_prior_xzt_sec": agent_times[3],
            "iteration_total_sec": iteration_sec,
        }
        timing_log.append(timing_row)

        if timing_log_path is not None:
            with open(timing_log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=timing_row.keys())
                writer.writerow(timing_row)

        if verbose:
            print(
                "[MACE] Timing summary: "
                f"itr={itr + 1}, "
                f"agent0={agent_times[0]:.2f}s, "
                f"agent1={agent_times[1]:.2f}s, "
                f"agent2={agent_times[2]:.2f}s, "
                f"agent3={agent_times[3]:.2f}s, "
                f"total={iteration_sec:.2f}s"
            )
            print(f"[MACE] Iteration {itr + 1} done in {iteration_sec:.2f} sec.")

    if verbose:
        print("\n[MACE] Reconstruction complete.")

    return sum(beta[k] * X[k] for k in range(4))

def mace4d_from_cone_beam_params(
    sino_list,
    cone_beam_params_list,
    optional_params_list,
    weight_type="transmission_root",
    prior_weight=0.5,
    max_admm_itr=10,
    rho=0.5,
    forward_num_iterations=3,
    stop_threshold=0.02,
    init_image=None,
    sigma_p=None,
    sharpness=1.0,
    verbose=1,
    init_save_dir=None,
    timing_log_path=None,
):
    if verbose:
        print("[MACE] Building weights and per-bin cone-beam models...")

    weights_list = [
        mj.gen_weights(jnp.asarray(s), weight_type=weight_type)
        for s in sino_list
    ]

    models = []
    for cone_t, opt_t in zip(cone_beam_params_list, optional_params_list):
        ct_model = mj.ConeBeamModel(**cone_t)
        ct_model.set_params(**opt_t)
        ct_model.set_params(positivity_flag=True, sharpness=sharpness, verbose=verbose)
        models.append(ct_model)

    if verbose:
        print(f"[MACE] Built {len(models)} cone-beam models.")

    recon_4d = run_mace_with_models_multigpu(
        models=models,
        sino_list=sino_list,
        weights_list=weights_list,
        beta=normalize_prior_weights(prior_weight),
        max_admm_itr=max_admm_itr,
        rho=rho,
        forward_num_iterations=forward_num_iterations,
        stop_threshold=stop_threshold,
        init_image=init_image,
        sigma_p=sigma_p,
        verbose=verbose,
        init_save_dir=init_save_dir,
        timing_log_path=timing_log_path,
    )
    return recon_4d
