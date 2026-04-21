import os
import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
import time

from utilities_4D_multi_threads import mace4d_from_cone_beam_params

if __name__ == "__main__":

    output_path = "/home/li5273/Desktop/data/output/2026/0423/4DMACE_multi_gpu"
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
        init_image = np.load(init_image_path)

    else:
        init_image = None

    time0 = time.time()

    recon_4d = mace4d_from_cone_beam_params(
        sino_list,
        cone_beam_params_list,
        optional_params_list,
        init_image=init_image,
        weight_type="transmission_root",
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
    print(f"\n[MACE] Total wall time: {run_time:.2f} hours. Saved to {output_path}")