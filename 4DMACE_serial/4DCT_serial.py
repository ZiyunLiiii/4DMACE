"""Simple 4D MACE demo using MBIRJAX cone-beam prox_map and qGGMRF prior denoisers."""

from __future__ import annotations

import time
import threading
import jax.numpy as jnp
import numpy as np
import mbirjax as mj
import os
import mbirjax.preprocess as mjp

_THREAD_LOCAL = threading.local()

def get_qggmrf_denoiser(shape):
    """Return a per-thread cached denoiser to avoid cross-thread state sharing."""
    cache = getattr(_THREAD_LOCAL, "denoiser_cache", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.denoiser_cache = cache
    if shape not in cache:
        cache[shape] = mj.QGGMRFDenoiser(shape)
    return cache[shape]


def normalize_prior_weights(prior_weight):
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        prior = [w for w in prior_weight]
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
        sigma_use = denoiser.estimate_image_noise_std(x[i][:,::4,::4])
        if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
            sigma_use = 0.0
        sigma_list[i] = sigma_use

    return sigma_list


def qggmrf_hyperplane_denoise(x, sigma_list, sigma_noise_floor=1e-6):
    """
    Denoise a stack of hyperplanes serially using precomputed sigma_list.
    x shape: (num_hyperplanes, dim1, dim2)
    """
    denoiser = get_qggmrf_denoiser(x.shape[1:])
    y = np.empty_like(x)

    for i in range(x.shape[0]):
        sigma_use = sigma_list[i]

        if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
            y[i] = x[i]
        else:
            y_i, _ = denoiser.denoise(image=x[i], sigma_noise=sigma_use)
            y[i] = np.asarray(y_i)

    return y


def denoiser_wrapper(x, permute_vector, sigma_list):
    """
    Permute 4D volume, denoise the hyperplane stack, then permute back.
    """
    x_perm = np.transpose(x, permute_vector)
    y_perm = qggmrf_hyperplane_denoise(x_perm, sigma_list=sigma_list)
    inv_perm = np.argsort(permute_vector)
    y = np.transpose(y_perm, inv_perm)
    return y


def run_mace_with_models(
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
    init_save_dir=None
):
    nt = len(sino_list)

    if verbose:
        print(f"[MACE] Start 4D reconstruction with {nt} time bins.")

    if init_image is None:
        if verbose:
            print("[MACE] Computing initial MBIR recon for each time bin...")
        t0 = time.time()
        init_image = np.asarray([
            np.asarray(
                models[t].recon(
                    jnp.asarray(sino_list[t]),
                    weights=jnp.asarray(weights_list[t]),
                    max_iterations=20,
                    stop_threshold_change_pct=stop_threshold,
                )[0])
            for t in range(nt)
        ])
        os.makedirs(init_save_dir, exist_ok=True)
        np.save(os.path.join(init_save_dir, "init_image.npy"), init_image)
        if verbose:
            print(f"[MACE] Initialization done in {time.time() - t0:.2f} sec.")
    else:
        init_image = np.asarray(init_image)
        if verbose:
            print("[MACE] Using provided init_image.")

    # Precompute sigma lists from the initialization
    if verbose:
        print("[MACE] Precomputing sigma_list for each denoising direction...")

    sigma_xyt = estimate_sigma_per_hyperplane(np.transpose(init_image, (3, 0, 1, 2)))

    sigma_yzt = estimate_sigma_per_hyperplane(np.transpose(init_image, (1, 0, 2, 3)))

    sigma_xzt = estimate_sigma_per_hyperplane(np.transpose(init_image, (2, 0, 1, 3)))

    if verbose:
        print("[MACE] sigma_list precomputation done.")

    W = [np.copy(init_image) for _ in range(4)]
    X = [np.copy(init_image) for _ in range(4)]

    for itr in range(max_admm_itr):
        if verbose:
            itr_t0 = time.time()
            print(f"[MACE] Iteration {itr + 1}/{max_admm_itr}")

        if verbose:
            print("[MACE]  Forward agent: cone-beam prox_map for all time bins...")
        X[0] = np.asarray([
            np.asarray(
                models[t].prox_map(
                    prox_input=jnp.asarray(W[0][t]),
                    sinogram=jnp.asarray(sino_list[t]),
                    sigma_prox=sigma_p,
                    weights=jnp.asarray(weights_list[t]),
                    init_recon=jnp.asarray(X[0][t]),
                    max_iterations=forward_num_iterations,
                    stop_threshold_change_pct=stop_threshold,
                )[0]
            )
            for t in range(nt)
        ])

        if verbose:
            print("[MACE]  Prior agent XY-t (fixed z slabs)...")
        X[1] = denoiser_wrapper(W[1], permute_vector=(3, 0, 1, 2), sigma_list=sigma_xyt)

        if verbose:
            print("[MACE]  Prior agent YZ-t (fixed row slabs)...")
        X[2] = denoiser_wrapper(W[2], permute_vector=(1, 0, 2, 3), sigma_list=sigma_yzt)

        if verbose:
            print("[MACE]  Prior agent XZ-t (fixed col slabs)...")
        X[3] = denoiser_wrapper(W[3], permute_vector=(2, 0, 1, 3), sigma_list=sigma_xzt,)

        if verbose:
            print("[MACE]  Consensus / ADMM update...")
        z = sum(beta[k] * (2.0 * X[k] - W[k]) for k in range(4))

        for k in range(4):
            W[k] = W[k] + 2.0 * rho * (z - X[k])

        if verbose:
            print(f"[MACE] Iteration {itr + 1} done in {time.time() - itr_t0:.2f} sec.")

    if verbose:
        print("[MACE] Reconstruction complete.")

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
    init_save_dir=None):
    if verbose:
        print("[MACE] Building weights and per-bin cone-beam models...")

    weights_list = [mj.gen_weights(jnp.asarray(s), weight_type=weight_type) for s in sino_list]

    models = []
    for cone_t, opt_t in zip(cone_beam_params_list, optional_params_list):
        ct_model = mj.ConeBeamModel(**cone_t)
        ct_model.set_params(**opt_t)

        ct_model.set_params(positivity_flag=True, sharpness=sharpness, verbose=verbose,)
        models.append(ct_model)
    if verbose:
        print(f"[MACE] Built {len(models)} cone-beam models.")

    recon_4d = run_mace_with_models(
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
        init_save_dir=init_save_dir
    )
    return recon_4d

if __name__ == "__main__":

    output_path = "/home/li5273/Desktop/data/output/2026/0423/4DMACE_serial"
    os.makedirs(output_path, exist_ok=True)

    # Set to True if you want to use the saved init recons
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
    # Select a range of the time stamps for faster execution
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
        # Specify init recon path
        init_image_path = "/home/li5273/Desktop/data/output/2026/0423/4DMACE_multi_gpu/init/init_image.npy"
        init_image = np.load(init_image_path)[time_range]

    else:
        init_image = None

    weight_type = "transmission_root"
    # Set this weight to balance the weights of forward and priors
    prior_weight = 0.5
    max_admm_itr = 10
    # Parameter rho in the Mann iterations
    rho = 0.5
    forward_num_iterations = 3
    stop_threshold = 0.02
    sigma_p = None
    sharpness = 1.0
    verbose = 1
    init_save_dir = os.path.join(output_path, "init")

    time0 = time.time()
    # Call MACE 4D
    recon_4d = mace4d_from_cone_beam_params(
        sino_list,
        cone_beam_params_list,
        optional_params_list,
        init_image=init_image,
        weight_type=weight_type,
        prior_weight=prior_weight,
        max_admm_itr=max_admm_itr,
        rho=rho,
        forward_num_iterations=forward_num_iterations,
        stop_threshold=stop_threshold,
        sigma_p=sigma_p,
        sharpness=sharpness,
        verbose=verbose,
        init_save_dir=init_save_dir,
    )
    time1 = time.time()
    run_time = (time1-time0)/60/60

    np.save(os.path.join(output_path, f"recon_4d_{run_time:2f}.npy"), recon_4d)
