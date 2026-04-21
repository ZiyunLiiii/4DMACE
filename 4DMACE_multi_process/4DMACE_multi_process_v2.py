from __future__ import annotations
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
import sys

# Ensure JAX does not inherit incompatible system CUDA/cuDNN from LD_LIBRARY_PATH.
if "LD_LIBRARY_PATH" in os.environ and not os.environ.get("_JAX_CLEAN_REEXEC"):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["_JAX_CLEAN_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np


# -----------------------------------------------------------------------------
# Shared-memory helpers
# -----------------------------------------------------------------------------

def create_shared_numpy(shape, dtype=np.float32, name=None):
    """Create a shared-memory NumPy array and return (shm, array_view)."""
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr


def attach_shared_numpy(name, shape, dtype=np.float32):
    """Attach to an existing shared-memory NumPy array and return (shm, array_view)."""
    dtype = np.dtype(dtype)
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr


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

def forward_agent_worker(
    physical_gpu_id,
    task_queue,
    result_queue,
    weight_type,
    cone_beam_params_list,
    optional_params_list,
    sino_list,
    sharpness,
    verbose,
    shared_W0_name,
    shared_X0_name,
    shared_shape,
    shared_dtype_str,
):
    """
    Persistent forward-agent subprocess.

    This process:
      1) binds to one physical GPU,
      2) imports JAX/MBIRJAX after binding,
      3) builds all cone-beam models locally,
      4) repeatedly executes the forward prox_map agent.

    The forward agent reads W[0] and X[0] from shared memory and writes its
    output X[0] back to shared memory, so that large 4D arrays are not sent
    through multiprocessing.Queue every iteration.
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

    shared_dtype = np.dtype(shared_dtype_str)
    shm_W0, W0_shared = attach_shared_numpy(shared_W0_name, shared_shape, shared_dtype)
    shm_X0, X0_shared = attach_shared_numpy(shared_X0_name, shared_shape, shared_dtype)

    # Keep weights on CPU as NumPy arrays. Transfer only the current bin to GPU.
    weights_list = [np.asarray(mj.gen_weights(np.asarray(s), weight_type=weight_type)) for s in sino_list]

    models = []
    for cone_t, opt_t in zip(cone_beam_params_list, optional_params_list):
        ct_model = mj.ConeBeamModel(**cone_t)
        ct_model.set_params(**opt_t)
        ct_model.set_params(positivity_flag=True, sharpness=sharpness, verbose=verbose)
        models.append(ct_model)

    nt = len(sino_list)

    if verbose:
        print(
            f"[Forward worker PID {os.getpid()}] Bound to physical GPU {physical_gpu_id}, "
            f"visible device {device}",
            flush=True,
        )

    try:
        while True:
            task = task_queue.get()

            if task["cmd"] == "stop":
                if verbose:
                    print(f"[Forward worker PID {os.getpid()}] Stopping.", flush=True)
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

                # Also store the initialization in shared X0 so the first iteration
                # can reuse it without extra queue transfer.
                X0_shared[...] = init_image

                result_queue.put({
                    "agent_id": 0,
                    "cmd": "recon_init_done",
                    "init_image": init_image,
                })

            elif task["cmd"] == "forward_step":
                sigma_p = task["sigma_p"]
                forward_num_iterations = task["forward_num_iterations"]
                stop_threshold = task["stop_threshold"]
                itr = task["itr"]

                # Read the latest W[0] and X[0] directly from shared memory.
                W0 = np.asarray(W0_shared)
                X0_prev = np.asarray(X0_shared)

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

                # Write the result back to shared memory.
                X0_shared[...] = X0

                # Send only a small completion message.
                result_queue.put({
                    "agent_id": 0,
                    "cmd": "forward_step_done",
                    "itr": itr,
                })

            else:
                raise ValueError(f"[Forward worker] Unknown command: {task['cmd']}")
    finally:
        shm_W0.close()
        shm_X0.close()


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
            f"Bound to physical GPU {physical_gpu_id}, visible device {device}",
            flush=True,
        )

    while True:
        task = task_queue.get()

        if task["cmd"] == "stop":
            if verbose:
                print(f"[Prior worker {agent_id} PID {os.getpid()}] Stopping.", flush=True)
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

    forward_task_queue = mp.Queue()
    result_queue = mp.Queue()

    # -------------------------------------------------------------------------
    # Step 1: obtain or compute initialization
    # -------------------------------------------------------------------------
    if init_image is None:
        init_shape = (nt,) + tuple(cone_beam_params_list[0]["recon_shape"])
        init_dtype = np.float32
    else:
        init_image = np.asarray(init_image)
        init_shape = init_image.shape
        init_dtype = init_image.dtype

    shm_W0, W0_shared = create_shared_numpy(init_shape, init_dtype)
    shm_X0, X0_shared = create_shared_numpy(init_shape, init_dtype)

    if init_image is not None:
        W0_shared[...] = init_image
        X0_shared[...] = init_image

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
            shm_W0.name,
            shm_X0.name,
            init_shape,
            np.dtype(init_dtype).str,
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
        W0_shared[...] = init_image
        X0_shared[...] = init_image

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

            # Update shared buffers for the forward worker.
            W0_shared[...] = W[0]
            X0_shared[...] = X[0]

            # Submit all four agent jobs.
            forward_task_queue.put({
                "cmd": "forward_step",
                "itr": itr,
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

                if agent_id == 0:
                    X_new[0] = np.copy(X0_shared)
                else:
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
        forward_task_queue.put({"cmd": "stop"})
        prior_task_queues[1].put({"cmd": "stop"})
        prior_task_queues[2].put({"cmd": "stop"})
        prior_task_queues[3].put({"cmd": "stop"})

        forward_proc.join()
        prior_proc_1.join()
        prior_proc_2.join()
        prior_proc_3.join()

        shm_W0.close()
        shm_W0.unlink()
        shm_X0.close()
        shm_X0.unlink()

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


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import mbirjax as mj
    import mbirjax.preprocess as mjp
    output_path = "/home/li5273/Desktop/data/output/2026/0421/4DMACE_multigpu"
    os.makedirs(output_path, exist_ok=True)

    USE_SAVED_INIT_IMAGE = True

    dataset_url = "/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024.tgz"
    download_dir = "/home/li5273/PycharmProjects/lilly_exp/nsi/demo_data/"
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Preprocessing parameters
    downsample_rate = [1, 1]
    subsample_view_factor = 1

    # 4D split parameters
    views_per_bin = 48
    stride = 24

    print("\n************** NSI dataset preprocessing **************")

    sino, cone_beam_params, optional_params = mjp.nsi.compute_sino_and_params(
        dataset_dir,
        downsample_factor=downsample_rate,
        subsample_view_factor=subsample_view_factor,
    )

    print("\n************** Split into time bins **************")
    start = 0
    end = -1
    time_range = slice(start, end)

    bins = mjp.truncate_sino_into_time_bins(
        sino=sino,
        cone_beam_params=cone_beam_params,
        optional_params=optional_params,
        views_per_bin=views_per_bin,
        stride=stride,
    )[time_range]

    print(f"Total bins: {len(bins)}")

    sino_list = []
    cone_beam_params_list = []
    optional_params_list = []

    print("\n***************** Reconstruct each bin ****************")
    for t, (sino_t, cone_t, opt_t, sl) in enumerate(bins):
        sino_list.append(sino_t)
        cone_beam_params_list.append(cone_t)
        optional_params_list.append(opt_t)

    if USE_SAVED_INIT_IMAGE:
        init_image_path = "/home/li5273/Desktop/data/output/2026/0402/4DMACE/init/init_image.npy"
        init_image = np.load(init_image_path)
    else:
        init_image = None

    time0 = time.time()

    recon_4d = mace4d_from_cone_beam_params_multigpu(
        sino_list=sino_list,
        cone_beam_params_list=cone_beam_params_list,
        optional_params_list=optional_params_list,
        weight_type="transmission_root",
        gpu_ids=(0, 1, 2, 3),
        init_image=init_image,
        prior_weight=0.5,
        max_admm_itr=10,
        rho=0.5,
        forward_num_iterations=3,
        stop_threshold=0.02,
        sigma_p=None,
        sharpness=1.0,
        verbose=1,
        init_save_dir=os.path.join(output_path, "init"),
    )

    time1 = time.time()
    run_time = (time1 - time0) / 60 / 60

    np.save(os.path.join(output_path, f"recon_4d_{run_time:.2f}h.npy"), recon_4d)
    print(f"[MACE] Total wall time: {run_time:.2f} hours.")