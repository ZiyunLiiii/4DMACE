"""
4D MACE reconstruction with multi(>=4)-GPU support.

GPU assignment is controlled by agent_device_indices:
  - Agent 0: cone-beam prox_map (serial over time bins, original logic)
  - Agent 1: ViDNet denoiser, fixed-z XY-t hyperplanes
  - Agent 2: ViDNet denoiser, fixed-x YZ-t hyperplanes
  - Agent 3: ViDNet denoiser, fixed-y XZ-t hyperplanes

All 4 agents are dispatched concurrently via ThreadPoolExecutor(4).
Each prior-agent thread runs ViDNet with PyTorch on its assigned CUDA device.
init_image (large) lives on CPU as a NumPy array throughout.
"""

from __future__ import annotations

import os
import sys

# Let JAX preallocate a smaller GPU-memory pool so PyTorch ViDNet agents still have room.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.35"

# Ensure JAX does not inherit incompatible system CUDA/cuDNN from LD_LIBRARY_PATH.
if "LD_LIBRARY_PATH" in os.environ and not os.environ.get("_JAX_CLEAN_REEXEC"):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["_JAX_CLEAN_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

import concurrent.futures
import csv
import time

import jax
import jax.numpy as jnp
import mbirjax as mj
import numpy as np
import torch

STRIVER_ROOT = "/home/li5273/PycharmProjects/STRIVER-deep"
VIDNET_MODEL_PATH = f"{STRIVER_ROOT}/models/vidnet/vidnet.pth"
VIDNET_SIGMAS = (0.018, 0.018, 0.008)
VIDNET_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_prior_weights(prior_weight):
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        prior = list(prior_weight)
        return [1.0 - sum(prior)] + prior
    w = prior_weight
    return [1.0 - w, w / 3.0, w / 3.0, w / 3.0]


def import_vidnet_wrapper(striver_root=STRIVER_ROOT):
    if striver_root not in sys.path:
        sys.path.insert(0, striver_root)

    from models.vidnet.wrapper import VideoDenoiserViDNet

    return VideoDenoiserViDNet


def normalize_hyperplane_batch(video_bthw):
    """
    Normalize each hyperplane video independently to [0, 1].

    video_bthw shape: (B, T, H, W)
    """
    video_bthw = video_bthw.astype(np.float32, copy=False)
    mins = video_bthw.min(axis=(1, 2, 3), keepdims=True)
    maxs = video_bthw.max(axis=(1, 2, 3), keepdims=True)
    ranges = maxs - mins

    video_norm = np.zeros_like(video_bthw, dtype=np.float32)
    np.divide(
        video_bthw - mins,
        ranges,
        out=video_norm,
        where=ranges > 0,
    )
    video_norm = np.clip(video_norm, 0.0, 1.0).astype(np.float32, copy=False)
    return video_norm, mins.astype(np.float32), maxs.astype(np.float32)


def denormalize_hyperplane_batch(video_norm_bthw, mins, maxs):
    ranges = maxs - mins
    return (video_norm_bthw * ranges + mins).astype(np.float32, copy=False)


def run_vidnet_inference(denoiser, video_norm_bthw, sigma_norm, torch_device):
    """
    Run ViDNet on a normalized batch.

    Preferred tensor shape is (B, T, C, H, W). If the STRIVER wrapper only
    supports one video at a time, fall back to (T, C, H, W) per hyperplane.
    """
    vid = torch.from_numpy(video_norm_bthw[:, :, None, :, :]).float().to(torch_device)

    with torch.no_grad():
        try:
            den = denoiser.inference(vid, sig=sigma_norm)
            den = den.detach().cpu().numpy()
            if den.ndim == 5:
                den = den[:, :, 0, :, :]
            elif den.ndim == 4:
                den = den[:, 0, :, :][None, ...]
            else:
                raise ValueError(f"Unexpected ViDNet output shape: {den.shape}")
        except (RuntimeError, ValueError, IndexError):
            del vid
            torch.cuda.empty_cache()
            den_list = []
            for video_norm_thw in video_norm_bthw:
                vid_one = torch.from_numpy(video_norm_thw[:, None, :, :]).float().to(torch_device)
                den_one = denoiser.inference(vid_one, sig=sigma_norm)
                den_one = den_one.detach().cpu().numpy()[:, 0, :, :]
                den_list.append(den_one)
            den = np.stack(den_list, axis=0)

    return np.clip(den, 0.0, 1.0).astype(np.float32, copy=False)


def vidnet_hyperplane_denoise(
    x_perm,
    sigma_norm,
    torch_device,
    model_path=VIDNET_MODEL_PATH,
    striver_root=STRIVER_ROOT,
    batch_size=VIDNET_BATCH_SIZE,
    verbose=1,
):
    """
    Denoise a hyperplane stack with ViDNet.

    x_perm shape: (num_hyperplanes, T, H, W)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available; ViDNet prior agents require CUDA.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ViDNet model not found: {model_path}")

    torch.cuda.set_device(torch_device)
    VideoDenoiserViDNet = import_vidnet_wrapper(striver_root)
    denoiser = VideoDenoiserViDNet(model_path=model_path, device=torch_device)

    y_perm = np.empty_like(x_perm, dtype=np.float32)
    for start in range(0, x_perm.shape[0], batch_size):
        end = min(start + batch_size, x_perm.shape[0])
        batch = x_perm[start:end]
        batch_norm, mins, maxs = normalize_hyperplane_batch(batch)
        den_norm = run_vidnet_inference(denoiser, batch_norm, sigma_norm, torch_device)
        y_perm[start:end] = denormalize_hyperplane_batch(den_norm, mins, maxs)

        if verbose:
            print(
                f"[MACE]    ViDNet {torch_device} batch {start}:{end}/"
                f"{x_perm.shape[0]}, sigma={sigma_norm:.6f}"
            )

    return y_perm


def denoiser_wrapper(
    x,
    permute_vector,
    sigma_norm,
    torch_device,
    model_path=VIDNET_MODEL_PATH,
    striver_root=STRIVER_ROOT,
    batch_size=VIDNET_BATCH_SIZE,
    verbose=1,
):
    """
    Permute 4D volume -> denoise hyperplane stack -> permute back.
    x shape: (nt, nx, ny, nz)
    """
    x_perm = np.ascontiguousarray(np.transpose(x, permute_vector), dtype=np.float32)
    y_perm = vidnet_hyperplane_denoise(
        x_perm,
        sigma_norm=sigma_norm,
        torch_device=torch_device,
        model_path=model_path,
        striver_root=striver_root,
        batch_size=batch_size,
        verbose=verbose,
    )
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
    vidnet_model_path=VIDNET_MODEL_PATH,
    striver_root=STRIVER_ROOT,
    vidnet_sigmas=VIDNET_SIGMAS,
    vidnet_batch_size=VIDNET_BATCH_SIZE,
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
        print(f"[MACE] ViDNet sigmas: {tuple(vidnet_sigmas)}")
        print(f"[MACE] ViDNet model: {vidnet_model_path}")

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
        """Agent 1: ViDNet XY-t, fixed z slabs. Batch shape (z, t, x, y)."""
        torch_device = f"cuda:{device_index}"
        agent_t0 = time.time()
        out = denoiser_wrapper(
            W_k,
            permute_vector=(3, 0, 1, 2),
            sigma_norm=vidnet_sigmas[0],
            torch_device=torch_device,
            model_path=vidnet_model_path,
            striver_root=striver_root,
            batch_size=vidnet_batch_size[0],
            verbose=verbose,
        )
        agent_sec = time.time() - agent_t0
        if verbose:
            print(f"[MACE]  Agent 1 ran ViDNet on {torch_device} in {agent_sec:.2f} sec.")
        return out, agent_sec

    def run_prior_agent_2(W_k, device_index):
        """Agent 2: ViDNet YZ-t, fixed x slabs. Batch shape (x, t, y, z)."""
        torch_device = f"cuda:{device_index}"
        agent_t0 = time.time()
        out = denoiser_wrapper(
            W_k,
            permute_vector=(1, 0, 2, 3),
            sigma_norm=vidnet_sigmas[1],
            torch_device=torch_device,
            model_path=vidnet_model_path,
            striver_root=striver_root,
            batch_size=vidnet_batch_size[1],
            verbose=verbose,
        )
        agent_sec = time.time() - agent_t0
        if verbose:
            print(f"[MACE]  Agent 2 ran ViDNet on {torch_device} in {agent_sec:.2f} sec.")
        return out, agent_sec

    def run_prior_agent_3(W_k, device_index):
        """Agent 3: ViDNet XZ-t, fixed y slabs. Batch shape (y, t, x, z)."""
        torch_device = f"cuda:{device_index}"
        agent_t0 = time.time()
        out = denoiser_wrapper(
            W_k,
            permute_vector=(2, 0, 1, 3),
            sigma_norm=vidnet_sigmas[2],
            torch_device=torch_device,
            model_path=vidnet_model_path,
            striver_root=striver_root,
            batch_size=vidnet_batch_size[2],
            verbose=verbose,
        )
        agent_sec = time.time() - agent_t0
        if verbose:
            print(f"[MACE]  Agent 3 ran ViDNet on {torch_device} in {agent_sec:.2f} sec.")
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
    vidnet_model_path=VIDNET_MODEL_PATH,
    striver_root=STRIVER_ROOT,
    vidnet_sigmas=VIDNET_SIGMAS,
    vidnet_batch_size=VIDNET_BATCH_SIZE,
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
        vidnet_model_path=vidnet_model_path,
        striver_root=striver_root,
        vidnet_sigmas=vidnet_sigmas,
        vidnet_batch_size=vidnet_batch_size,
    )
    return recon_4d
