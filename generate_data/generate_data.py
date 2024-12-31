import os

import cv2
import numpy as np
import pandas as pd
from generate_data import utils_generate
import yaml
from tqdm import tqdm


# main function to build the datasets based on config
def gen_patches(config, dir_structure, purpose="train"):
    gen_image, gen_noise = utils_generate.get_function_config(config)
    gen_shading = utils_generate.get_shading(config)
    shading = config["shading"]
    status_shading = shading["status"]

    glob_dir = dir_structure  # A directory : see utils_generate.get_directory_paths to get a grip

    datasets = glob_dir["children"]["datasets"]
    datasets_path = datasets["path"]
    # Where will be stored the data

    info = glob_dir["children"]["info"]
    # Where will be stored statistical information about the dataset

    conf_path = info["children"]["config"]["path"]
    # Path to store the config in order to reproduce the dataset

    conf_file = os.path.join(conf_path, "config.yaml")

    with open(conf_file, "w") as file:
        yaml.dump(config, file)  # write the config

    dataframe = {
        "real": [],
        "noisy": [],
        "snr": [],
        "ps": [],
        "pn": [],
        "mean": [],
    }  # dataframe to store values of interest to compute SNR
    k = 0

    n_samples = config["n_samples"]  # number of pair of images generated
    split = config["train_split"]  # percentage of train images compared to test images

    if purpose == "train":
        n = int(split * n_samples)

    else:
        n = int((1 - split) * n_samples)

    for _ in tqdm(range(n)):

        real_arr = gen_image()  # real images
        if status_shading : 
            real_arr = gen_shading(real_arr)
        noisy_arr = gen_noise(real_arr)  # synthetic noise added

        real_arr = np.clip(real_arr, 0, 1)
        noisy_arr = np.clip(noisy_arr, 0, 1)

        real_image = (real_arr * 255.0).astype(int)
        noisy_image = (noisy_arr * 255.0).astype(int)

        result = utils_generate.SNR(real_image, noisy_image)
        mean = np.mean(real_image.astype(float) / 255.0).item()

        if result is None:
            k = k - 1
            continue

        snr, ps, pn = result

        number = str(k)

        real_name = number.zfill(10) + "-real.png"
        noisy_name = number.zfill(10) + "-noisy.png"

        relative_real_path = os.path.join(purpose, real_name)
        relative_noisy_path = os.path.join(purpose, noisy_name)

        real_path = os.path.join(datasets_path, relative_real_path)
        noisy_path = os.path.join(datasets_path, relative_noisy_path)

        cv2.imwrite(real_path, real_image)
        cv2.imwrite(noisy_path, noisy_image)

        dataframe["real"].append(relative_real_path)
        dataframe["noisy"].append(relative_noisy_path)
        dataframe["snr"].append(snr)
        dataframe["ps"].append(ps)
        dataframe["pn"].append(pn)
        dataframe["mean"].append(mean)
        k = k + 1

    dataframe_name = purpose + ".csv"
    df = pd.DataFrame(dataframe)
    df.to_csv(os.path.join(datasets_path, dataframe_name))
