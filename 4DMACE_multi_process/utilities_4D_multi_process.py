from __future__ import annotations

import os
import time
import multiprocessing as mp
import numpy as np


# -----------------------------------------------------------------------------
# Utility helpers used by the main process
# -----------------------------------------------------------------------------

def normalize_prior_weights(prior_weight):
    """Convert a scalar or 3-list prior weight into 4 MACE agent weights."""
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        prior = [w for w in prior_weight]
        return [1.0 - sum(prior)] + prior
    w = prior_weight
    return [1.0 - w, w / 3.0, w / 3.0, w / 3.0]


# -----------------------------------------------------------------------------
# Worker-side helpers
# These functions are imported/executed inside subprocesses after GPU binding.
# -----------------------------------------------------------------------------

def get_qggmrf_denoiser_process_local(shape, denoiser_cache, mj_module):
    """Return a process-local cached QGGMRFDenoiser."""
    if shape not in denoiser_cache:
        denoiser_cache[shape] = mj_module.QGGMRFDenoiser(shape)
    return denoiser_cache[shape]


def estimate_sigma_per_hyperplane_worker(x, denoiser_cache, mj_module, sigma_noise_floor=1e-6):
    """
    Estimate one sigma value per hyperplane.
    x shape: (num_hyperplanes, dim1, dim2)
    """
    denoiser = get_qggmrf_denoiser_process_local(x.shape[1:], denoiser_cache, mj_module)
    sigma_list = np.empty(x.shape[0], dtype=np.float32)
    for i in range(x.shape[0]):
        sigma_use = denoiser.estimate_image_noise_std(x[i][:, ::4, ::4])
        if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
            sigma_use = 0.0
        sigma_list[i] = sigma_use
    return sigma_list


def qggmrf_hyperplane_denoise_worker(x, sigma_list, denoiser_cache, mj_module, sigma_noise_floor=1e-6):
    """
    Denoise a stack of hyperplanes serially using a precomputed sigma_list.
    x shape: (num_hyperplanes, dim1, dim2)
    """
    denoiser = get_qggmrf_denoiser_process_local(x.shape[1:], denoiser_cache, mj_module)
    y = np.empty_like(x)

    for i in range(x.shape[0]):
        sigma_use = sigma_list[i]
        if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
            y[i] = x[i]
        else:
            y_i, _ = denoiser.denoise(image=x[i], sigma_noise=sigma_use)
            y[i] = np.asarray(y_i)
    return y


def denoiser_wrapper_worker(x, permute_vector, sigma_list, denoiser_cache, mj_module):
    """
    Permute a 4D volume, denoise the hyperplane stack, then permute back.
    """
    x_perm = np.transpose(x, permute_vector)
    y_perm = qggmrf_hyperplane_denoise_worker(x_perm, sigma_list=sigma_list,
                                              denoiser_cache=denoiser_cache, mj_module=mj_module)
    inv_perm = np.argsort(permute_vector)
    return np.transpose(y_perm, inv_perm)


# -----------------------------------------------------------------------------
# Forward agent subprocess
# -----------------------------------------------------------------------------

def forward_agent_worker(physical_gpu_id: int, task_queue, result_queue,
                         weight_type, cone_beam_params_list, optional_params_list, sino_list, sharpness, verbose):
    """
    Persistent forward-agent subprocess.

    This process:
      1) binds to one physical GPU,
      2) imports JAX/MBIRJAX after binding,
      3) builds all cone-beam models locally,
      4) repeatedly executes the forward prox_map agent.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax
    import jax.numpy as jnp
    import mbirjax as mj

    visible_gpus = jax.devices("gpu")
    if len(visible_gpus) == 0:
        raise RuntimeError(f"[Forward worker] No visible GPU for physical GPU {physical_gpu_id}.")
    device = visible_gpus[0]

    # Build weights and models inside this subprocess.
    weights_list = [mj.gen_weights(jnp.asarray(s), weight_type=weight_type) for s in sino_list]

    models = []
    for cone_t, opt_t in zip(cone_beam_params_list, optional_params_list):
        ct_model = mj.ConeBeamModel(**cone_t)
        ct_model.set_params(**opt_t)
        ct_model.set_params(positivity_flag=True, sharpness=sharpness, verbose=verbose)
        models.append(ct_model)

    nt = len(sino_list)

    if verbose:
        print(f"[Forward worker PID {os.getpid()}] Bound to physical GPU {physical_gpu_id}, visible device {device}")

    while True:
        task = task_queue.get()

        if task["cmd"] == "stop":
            if verbose:
                print(f"[Forward worker PID {os.getpid()}] Stopping.")
            break

        if task["cmd"] == "recon_init":
            stop_threshold = task["stop_threshold"]

            init_image = np.asarray([
                np.asarray(
                    models[t].recon(
                        jax.device_put(jnp.asarray(sino_list[t]), device),
                        weights=jax.device_put(jnp.asarray(weights_list[t]), device),
                        max_iterations=20,
                        stop_threshold_change_pct=stop_threshold,
                    )[0]
                )
                for t in range(nt)
            ])

            result_queue.put({
                "agent_id": 0,
                "cmd": "recon_init_done",
                "init_image": init_image,
            })

        elif task["cmd"] == "forward_step":
            W0 = task["W"]
            X0_prev = task["X_prev"]
            sigma_p = task["sigma_p"]
            forward_num_iterations = task["forward_num_iterations"]
            stop_threshold = task["stop_threshold"]
            itr = task["itr"]

            X0 = np.asarray([
                np.asarray(
                    models[t].prox_map(
                        prox_input=jax.device_put(jnp.asarray(W0[t]), device),
                        sinogram=jax.device_put(jnp.asarray(sino_list[t]), device),
                        sigma_prox=sigma_p,
                        weights=jax.device_put(jnp.asarray(weights_list[t]), device),
                        init_recon=jax.device_put(jnp.asarray(X0_prev[t]), device),
                        max_iterations=forward_num_iterations,
                        stop_threshold_change_pct=stop_threshold,
                    )[0]
                )
                for t in range(nt)
            ])

            result_queue.put({
                "agent_id": 0,
                "cmd": "forward_step_done",
                "itr": itr,
                "X": X0,
            })

        else:
            raise ValueError(f"[Forward worker] Unknown command: {task['cmd']}")


# -----------------------------------------------------------------------------
# Prior agent subprocess
# -----------------------------------------------------------------------------

def prior_agent_worker(agent_id, physical_gpu_id, task_queue, result_queue,
                       permute_vector, sigma_list, verbose):
    """
    Persistent prior-agent subprocess.

    Each prior worker owns one GPU and repeatedly applies one directional
    qGGMRF denoiser agent.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax
    import mbirjax as mj

    visible_gpus = jax.devices("gpu")
    if len(visible_gpus) == 0:
        raise RuntimeError(f"[Prior worker {agent_id}] No visible GPU for physical GPU {physical_gpu_id}.")
    device = visible_gpus[0]

    denoiser_cache = {}

    if verbose:
        print(
            f"[Prior worker {agent_id} PID {os.getpid()}] "
            f"Bound to physical GPU {physical_gpu_id}, visible device {device}"
        )

    while True:
        task = task_queue.get()

        if task["cmd"] == "stop":
            if verbose:
                print(f"[Prior worker {agent_id} PID {os.getpid()}] Stopping.")
            break

        if task["cmd"] == "prior_step":
            Wk = task["W"]
            itr = task["itr"]

            Xk = denoiser_wrapper_worker(
                Wk,
                permute_vector=permute_vector,
                sigma_list=sigma_list,
                denoiser_cache=denoiser_cache,
                mj_module=mj,
            )

            result_queue.put({
                "agent_id": agent_id,
                "cmd": "prior_step_done",
                "itr": itr,
                "X": Xk,
            })

        else:
            raise ValueError(f"[Prior worker {agent_id}] Unknown command: {task['cmd']}")


# -----------------------------------------------------------------------------
# Main MACE orchestration
# -----------------------------------------------------------------------------

def run_mace_multigpu(
    sino_list,
    cone_beam_params_list,
    optional_params_list,
    weight_type,
    beta,
    gpu_ids=(0, 1, 2, 3),
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
    """
    Multi-process / multi-GPU MACE driver.

    Agent-to-GPU mapping:
      agent 0 -> forward prox_map      -> gpu_ids[0]
      agent 1 -> prior XY-t            -> gpu_ids[1]
      agent 2 -> prior YZ-t            -> gpu_ids[2]
      agent 3 -> prior XZ-t            -> gpu_ids[3]
    """
    if len(gpu_ids) != 4:
        raise ValueError("gpu_ids must contain exactly 4 GPU ids.")

    nt = len(sino_list)

    if verbose:
        print(f"[MACE] Start 4D reconstruction with {nt} time bins.")
        print(
            f"[MACE] GPU assignment: "
            f"forward->{gpu_ids[0]}, XY-t->{gpu_ids[1]}, YZ-t->{gpu_ids[2]}, XZ-t->{gpu_ids[3]}"
        )

    # -------------------------------------------------------------------------
    # Step 1: obtain or compute initialization
    # -------------------------------------------------------------------------
    # The initialization is produced by the forward worker so that the expensive
    # reconstruction runs directly on its assigned GPU.
    forward_task_queue = mp.Queue()
    result_queue = mp.Queue()

    forward_proc = mp.Process(
        target=forward_agent_worker,
        args=(
            gpu_ids[0],
            forward_task_queue,
            result_queue,
            weight_type,
            cone_beam_params_list,
            optional_params_list,
            sino_list,
            sharpness,
            verbose,
        ),
    )
    forward_proc.start()

    if init_image is None:
        if verbose:
            print("[MACE] Computing initial MBIR recon in the forward worker...")

        t0 = time.time()
        forward_task_queue.put({
            "cmd": "recon_init",
            "stop_threshold": stop_threshold,
        })

        msg = result_queue.get()
        if msg["cmd"] != "recon_init_done":
            raise RuntimeError("Initialization failed: unexpected message from forward worker.")

        init_image = np.asarray(msg["init_image"])

        if init_save_dir is not None:
            os.makedirs(init_save_dir, exist_ok=True)
            np.save(os.path.join(init_save_dir, "init_image.npy"), init_image)

        if verbose:
            print(f"[MACE] Initialization done in {time.time() - t0:.2f} sec.")
    else:
        init_image = np.asarray(init_image)
        if verbose:
            print("[MACE] Using provided init_image.")

    # -------------------------------------------------------------------------
    # Step 2: precompute sigma lists in the main process
    # -------------------------------------------------------------------------
    # This is done once from the initialization. We use MBIRJAX here on CPU/main
    # only for sigma estimation. If you prefer, these could also be computed in
    # separate subprocesses, but doing it once here is simpler.
    if verbose:
        print("[MACE] Precomputing sigma lists for each denoising direction...")

    import mbirjax as mj

    sigma_cache = {}
    sigma_xyt = estimate_sigma_per_hyperplane_worker(
        np.transpose(init_image, (3, 0, 1, 2)),
        sigma_cache,
        mj,
    )
    sigma_yzt = estimate_sigma_per_hyperplane_worker(
        np.transpose(init_image, (1, 0, 2, 3)),
        sigma_cache,
        mj,
    )
    sigma_xzt = estimate_sigma_per_hyperplane_worker(
        np.transpose(init_image, (2, 0, 1, 3)),
        sigma_cache,
        mj,
    )

    if verbose:
        print("[MACE] Sigma precomputation done.")

    # -------------------------------------------------------------------------
    # Step 3: start the three prior workers
    # -------------------------------------------------------------------------
    prior_task_queues = {
        1: mp.Queue(),
        2: mp.Queue(),
        3: mp.Queue(),
    }

    prior_proc_1 = mp.Process(
        target=prior_agent_worker,
        args=(1, gpu_ids[1], prior_task_queues[1], result_queue, (3, 0, 1, 2), sigma_xyt, verbose),
    )
    prior_proc_2 = mp.Process(
        target=prior_agent_worker,
        args=(2, gpu_ids[2], prior_task_queues[2], result_queue, (1, 0, 2, 3), sigma_yzt, verbose),
    )
    prior_proc_3 = mp.Process(
        target=prior_agent_worker,
        args=(3, gpu_ids[3], prior_task_queues[3], result_queue, (2, 0, 1, 3), sigma_xzt, verbose),
    )

    prior_proc_1.start()
    prior_proc_2.start()
    prior_proc_3.start()

    # -------------------------------------------------------------------------
    # Step 4: ADMM / MACE loop
    # -------------------------------------------------------------------------
    W = [np.copy(init_image) for _ in range(4)]
    X = [np.copy(init_image) for _ in range(4)]

    try:
        for itr in range(max_admm_itr):
            itr_t0 = time.time()
            if verbose:
                print(f"\n[MACE] Iteration {itr + 1}/{max_admm_itr}")

            # Submit all four agent jobs.
            forward_task_queue.put({
                "cmd": "forward_step",
                "itr": itr,
                "W": W[0],
                "X_prev": X[0],
                "sigma_p": sigma_p,
                "forward_num_iterations": forward_num_iterations,
                "stop_threshold": stop_threshold,
            })

            prior_task_queues[1].put({
                "cmd": "prior_step",
                "itr": itr,
                "W": W[1],
            })

            prior_task_queues[2].put({
                "cmd": "prior_step",
                "itr": itr,
                "W": W[2],
            })

            prior_task_queues[3].put({
                "cmd": "prior_step",
                "itr": itr,
                "W": W[3],
            })

            # Collect all four agent outputs.
            received = 0
            X_new = [None, None, None, None]

            while received < 4:
                msg = result_queue.get()

                if msg["itr"] != itr:
                    raise RuntimeError("Received result from an unexpected iteration.")

                agent_id = msg["agent_id"]
                X_new[agent_id] = msg["X"]
                received += 1

                if verbose:
                    if agent_id == 0:
                        print("[MACE]  Forward agent done.")
                    elif agent_id == 1:
                        print("[MACE]  Prior agent XY-t done.")
                    elif agent_id == 2:
                        print("[MACE]  Prior agent YZ-t done.")
                    elif agent_id == 3:
                        print("[MACE]  Prior agent XZ-t done.")

            X = X_new

            if verbose:
                print("[MACE]  Consensus / ADMM update...")

            z = sum(beta[k] * (2.0 * X[k] - W[k]) for k in range(4))
            for k in range(4):
                W[k] = W[k] + 2.0 * rho * (z - X[k])

            if verbose:
                print(f"[MACE] Iteration {itr + 1} done in {time.time() - itr_t0:.2f} sec.")

    finally:
        # Always stop all subprocesses cleanly, even if something fails.
        forward_task_queue.put({"cmd": "stop"})
        prior_task_queues[1].put({"cmd": "stop"})
        prior_task_queues[2].put({"cmd": "stop"})
        prior_task_queues[3].put({"cmd": "stop"})

        forward_proc.join()
        prior_proc_1.join()
        prior_proc_2.join()
        prior_proc_3.join()

    if verbose:
        print("\n[MACE] Reconstruction complete.")

    return sum(beta[k] * X[k] for k in range(4))


def mace4d_from_cone_beam_params_multigpu(
    sino_list,
    cone_beam_params_list,
    optional_params_list,
    weight_type='transmission_root',
    gpu_ids=(0, 1, 2, 3),
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
    """Public API for multi-GPU 4D MACE."""
    beta = normalize_prior_weights(prior_weight)

    return run_mace_multigpu(
        sino_list=sino_list,
        cone_beam_params_list=cone_beam_params_list,
        optional_params_list=optional_params_list,
        weight_type=weight_type,
        beta=beta,
        gpu_ids=gpu_ids,
        max_admm_itr=max_admm_itr,
        rho=rho,
        forward_num_iterations=forward_num_iterations,
        stop_threshold=stop_threshold,
        init_image=init_image,
        sigma_p=sigma_p,
        sharpness=sharpness,
        verbose=verbose,
        init_save_dir=init_save_dir,
    )
