import os

import numpy as np

from generate_data import generators


# Compute SNR between tow images
# Get images with int values 0-255
def SNR(clean_arr0, noisy_arr0):
    clean_arr = clean_arr0.astype("float") / 255.0
    noisy_arr = noisy_arr0.astype("float") / 255.0
    ps = np.mean(clean_arr**2)
    pn = np.mean((clean_arr - noisy_arr) ** 2)

    if ps != 0 and pn != 0:
        snr = 10 * np.log10(ps / pn)
        return snr, ps, pn

    else:
        return None


# ==================== Get Function ====================

# Get the parametrized functions based on config


def get_function_config(conf):
    gen_noise_name = conf["generator_noise"]
    # noise generator name
    gen_image_name = conf["generator_images"]
    # image generator name

    noise_parameters = conf["generator_noise_parameters"]
    image_parameters = conf["generator_image_parameters"]

    gen_function_image = getattr(generators, gen_image_name)(image_parameters)
    gen_function_noise = getattr(generators, gen_noise_name)(noise_parameters)

    return gen_function_image, gen_function_noise


def get_shading(conf):
    shading = conf["shading"]
    parameters = shading["parameters"]

    gen_shading = getattr(generators, "illuminate_image")(parameters)

    return gen_shading


# ===================== Directories utils ====================


def sub(parent, L):
    path = parent["path"]
    for name in L:
        sub = {}
        child_path = os.path.join(path, name)
        child = {"path": child_path, "children": sub, "parent": parent}
        parent["children"][name] = child
        # Add a child


# helper to format the structure of a folder with its children


def get_directory_paths(location, name0, name):
    path0 = os.path.join(location, name0)
    path = os.path.join(path0, name)

    glob_dir = {"path": path, "children": {}, "parent": {}}

    sub_directories = ["info", "datasets"]
    current_parent = glob_dir

    sub(current_parent, sub_directories)

    current_parent = current_parent["children"]["datasets"]
    sub_directories = ["train", "val", "test"]

    sub(current_parent, sub_directories)

    current_parent = current_parent["parent"]["children"]["info"]
    sub_directories = ["config", "plots", "stats"]

    sub(current_parent, sub_directories)

    current_parent = current_parent["children"]["stats"]
    sub_directories = ["mean", "snr", "dep"]

    sub(current_parent, sub_directories)

    return glob_dir


# data structure of the directory


def create_directory(directory):
    children = directory["children"]

    if not children:
        return

    for child in children.values():
        child_path = child["path"]
        os.makedirs(child_path, exist_ok=True)

        create_directory(child)


# Create a directory with the appropriate dictionnary
# Be careful, ~/ is not treated by os.makedirs
# You have to replace it with /home/usrname/path_to_data
