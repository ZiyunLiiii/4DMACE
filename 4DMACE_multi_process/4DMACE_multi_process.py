"""
4D MACE reconstruction with 4-GPU support.

GPU assignment:
  - GPU 0  ->  Agent 0: cone-beam prox_map (serial over time bins, original logic)
  - GPU 1  ->  Agent 1: qGGMRF denoiser, XY-t hyperplanes
  - GPU 2  ->  Agent 2: qGGMRF denoiser, YZ-t hyperplanes
  - GPU 3  ->  Agent 3: qGGMRF denoiser, XZ-t hyperplanes

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
import threading
import time

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
    denoiser = get_qggmrf_denoiser(x.shape[1:])
    y = np.empty_like(x)
    with jax.default_device(device):
        for i in range(x.shape[0]):
            sigma_use = sigma_list[i]
            if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
                y[i] = x[i]
            else:
                y_i, _ = denoiser.denoise(image=x[i], sigma_noise=sigma_use)
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
# Multi-threads MACE
# ---------------------------------------------------------------------------

def run_mace_with_models_multi_threads(
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
):
    nt = len(sino_list)

    # ── GPU device discovery ───────────────────────────────────────────────
    devices = jax.devices("gpu")
    n_gpu = len(devices)

    device0, device1, device2, device3 = devices[0], devices[0], devices[0], devices[0]

    # Switch to the following settings if you have 4 GPUs
    # device0, device1, device2, device3 = devices[0], devices[1], devices[2], devices[3]

    if verbose:
        print(f"[MACE] Found {n_gpu} GPU(s): {devices}")
        print(f"[MACE] GPU assignment: Agent0->{device0}, Agent1->{device1}, Agent2->{device2}, Agent3->{device3}")
        print(f"[MACE] Start 4D reconstruction with {nt} time bins.")

    # ── Initialisation — serial on GPU 0, identical to original ───────────
    if init_image is None:
        if verbose:
            print("[MACE] Computing initial MBIR recon for each time bin on GPU 0 (serial)...")
        t0 = time.time()
        init_image = np.stack([
            np.asarray(
                models[t].recon(
                    jax.device_put(jnp.asarray(sino_list[t]), device0),
                    weights=jax.device_put(jnp.asarray(weights_list[t]), device0),
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
        print("[MACE] Sigma precomputation done.")

    # ── ADMM state (all on CPU / NumPy) ───────────────────────────────────
    W = [np.copy(init_image) for _ in range(4)]
    X = [np.copy(init_image) for _ in range(4)]  # warm-start for X[0]

    # ── Agent definitions ──────────────────────────────────────────────────

    def run_forward_agent(W_k, X_prev):
        """
        Agent 0: cone-beam prox_map, serial over time bins, pinned to GPU 0.
        Identical logic to the original single-GPU version.
        """
        return np.stack([
            np.asarray(
                models[t].prox_map(
                    prox_input=jax.device_put(jnp.asarray(W_k[t]), device0),
                    sinogram=jax.device_put(jnp.asarray(sino_list[t]), device0),
                    sigma_prox=sigma_p,
                    weights=jax.device_put(jnp.asarray(weights_list[t]), device0),
                    init_recon=jax.device_put(jnp.asarray(X_prev[t]), device0),
                    max_iterations=forward_num_iterations,
                    stop_threshold_change_pct=stop_threshold,
                )[0]
            )
            for t in range(nt)
        ])

    def run_prior_agent_1(W_k):
        """Agent 1: qGGMRF XY-t (fixed z slabs), device 1."""
        return denoiser_wrapper(W_k, permute_vector=(3, 0, 1, 2), sigma_list=sigma_xyt, device=device1)

    def run_prior_agent_2(W_k):
        """Agent 2: qGGMRF YZ-t (fixed row slabs), device 2."""
        return denoiser_wrapper(W_k, permute_vector=(1, 0, 2, 3), sigma_list=sigma_yzt, device=device2)

    def run_prior_agent_3(W_k):
        """Agent 3: qGGMRF XZ-t (fixed col slabs), device 3."""
        return denoiser_wrapper(W_k, permute_vector=(2, 0, 1, 3), sigma_list=sigma_xzt, device=device3)

    # ── Main loop ─────────────────────────────────────────────────────
    for itr in range(max_admm_itr):
        itr_t0 = time.time()
        if verbose:
            print(f"\n[MACE] ── Iteration {itr + 1}/{max_admm_itr} ──")

        # Snapshot W so all agents see a consistent state for this iteration.
        W_snap = [np.copy(W[k]) for k in range(4)]

        # All 4 agents run concurrently:
        #   Agent 0  -> Device 0 (serial over time bins)
        #   Agent 1  -> Device 1 (qGGMRF XY-t)
        #   Agent 2  -> Device 2 (qGGMRF YZ-t)
        #   Agent 3  -> Device 3 (qGGMRF XZ-t)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fut0 = pool.submit(run_forward_agent, W_snap[0], X[0])
            fut1 = pool.submit(run_prior_agent_1, W_snap[1])
            fut2 = pool.submit(run_prior_agent_2, W_snap[2])
            fut3 = pool.submit(run_prior_agent_3, W_snap[3])

            X[0] = fut0.result()
            if verbose:
                print("[MACE]  Agent 0 (forward) done.")
            X[1] = fut1.result()
            if verbose:
                print("[MACE]  Agent 1 (prior XY-t) done.")
            X[2] = fut2.result()
            if verbose:
                print("[MACE]  Agent 2 (prior YZ-t) done.")
            X[3] = fut3.result()
            if verbose:
                print("[MACE]  Agent 3 (prior XZ-t) done.")

        if verbose:
            print("[MACE]  All agents done. Running consensus update...")

        # Consensus / ADMM update (CPU)
        z = sum(beta[k] * (2.0 * X[k] - W[k]) for k in range(4))
        for k in range(4):
            W[k] = W[k] + 2.0 * rho * (z - X[k])

        if verbose:
            print(f"[MACE] Iteration {itr + 1} done in {time.time() - itr_t0:.2f} sec.")

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

    recon_4d = run_mace_with_models_multi_threads(
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
    )
    return recon_4d

