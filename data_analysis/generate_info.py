import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_analysis import utils_plot
from data_analysis import utils_stats
import yaml

# create some info about the data 
def generate_info_snr(glob_dir, purpose="train"):
    datasets_path = glob_dir["children"]["datasets"]["path"]
    info = glob_dir["children"]["info"]
    stats = info["children"]["stats"]

    path_snr = stats["children"]["snr"]["path"]

    csv_file = purpose + ".csv"
    csv_path = os.path.join(datasets_path, csv_file)

    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    snr = dataframe["snr"]

    title = "SNR distribution for " + purpose + " datatset"
    utils_plot.plot_hist(snr, title)

    fig_name = "histogram_snr-" + purpose + ".png"
    fig_path = os.path.join(path_snr, fig_name)

    plt.savefig(fig_path, dpi=300)

    snr = np.array(snr)
    stats_info = utils_stats.stats(snr, n_quant=10)

    file_name = "stats-snr-" + purpose + ".yaml"
    stats_path = os.path.join(path_snr, file_name)

    with open(stats_path, "w") as file:
        yaml.dump(stats_info, file)


def generate_info_ps_pn(glob_dir, purpose):
    datasets_path = glob_dir["children"]["datasets"]["path"]
    info = glob_dir["children"]["info"]
    stats = info["children"]["stats"]

    path_mean = stats["children"]["mean"]["path"]
    path_dep = stats["children"]["dep"]["path"]
    csv_file = purpose + ".csv"
    csv_path = os.path.join(datasets_path, csv_file)

    dataframe = pd.read_csv(csv_path)

    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    ps = dataframe["ps"]
    pn = dataframe["pn"]
    snr = dataframe["snr"]
    mean = dataframe["mean"]

    title = "mean distribution for " + purpose + " dataset"
    utils_plot.plot_hist(mean, title)

    fig_name = "histogram_mean-" + purpose + ".png"
    fig_path = os.path.join(path_mean, fig_name)

    plt.savefig(fig_path, dpi=300)

    mean = np.array(mean)
    stats_info = utils_stats.stats(mean, n_quant=10)

    file_name = "stats-mean-" + purpose + ".yaml"
    stats_path = os.path.join(path_mean, file_name)

    with open(stats_path, "w") as file:
        yaml.dump(stats_info, file)

    title = "noise/signal"
    utils_plot.plot_scatter(ps, pn, title)
    file_name = "scatter_signal_noise"
    file_path = os.path.join(path_dep, file_name)
    plt.savefig(file_path, dpi=300)

    title = "snr/signal"
    utils_plot.plot_scatter(ps, snr, title)
    file_name = "scatter_signal_snr"
    file_path = os.path.join(path_dep, file_name)
    plt.savefig(file_path, dpi=300)


def plot(glob_dir, purpose, n_files=25):
    datasets_path = glob_dir["children"]["datasets"]["path"]
    info = glob_dir["children"]["info"]

    plots = info["children"]["plots"]
    path_plot = plots["path"]
    csv_file = purpose + ".csv"
    csv_path = os.path.join(datasets_path, csv_file)

    dataframe = pd.read_csv(csv_path)

    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    real_files = dataframe["real"].values

    noisy_files = dataframe["noisy"].values

    real = []
    noisy = []

    for i in range(n_files):
        real_file = real_files[i]
        noisy_file = noisy_files[i]

        real_path = os.path.join(datasets_path, real_file)
        noisy_path = os.path.join(datasets_path, noisy_file)

        real_arr = cv2.imread(real_path)
        noisy_arr = cv2.imread(noisy_path)

        real.append(real_arr)
        noisy.append(noisy_arr)

    real = np.array(real)
    noisy = np.array(noisy)

    utils_plot.plot_grid(real, "real images")
    real_name = "real.png"
    real_path = os.path.join(path_plot, real_name)
    plt.savefig(real_path, dpi=300)

    utils_plot.plot_grid(noisy, "noisy images")
    noisy_name = "noisy.png"
    noisy_path = os.path.join(path_plot, noisy_name)
    plt.savefig(noisy_path, dpi=300)

# copy into cwd
def copy(glob_dir, dir_name):
    cwd = os.path.dirname(__file__)
    cwd = os.path.dirname(cwd)

    info_path = glob_dir["children"]["info"]["path"]
    source_dir = info_path
    destination_dir = os.path.join(cwd, "info", dir_name)

    # Copy the directory
    try:
        shutil.copytree(source_dir, destination_dir)
        print(f"Directory copied from {source_dir} to {destination_dir}")
    except FileExistsError:
        print("Destination directory already exists.")
    except FileNotFoundError:
        print("Source directory does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


