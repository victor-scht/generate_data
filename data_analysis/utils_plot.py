from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

# pre-configured plots

def plot_hist(arr, title):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.hist(
        arr, bins=40, label="number of bins = 40", color="skyblue", edgecolor="black"
    )
    plt.legend()
    plt.grid(linewidth=0.5)


def plot_grid(image_tensors, title):
    n_images = np.array(image_tensors.shape[0])
    side = int(sqrt(n_images))

    plt.figure(figsize=(15, 15))
    plt.title(title)
    for i in range(side):
        for j in range(side):
            plt.subplot(side, side, i * side + j + 1)
            plt.imshow(image_tensors[i * side + j], cmap="gray", vmin=0, vmax=1)
    plt.tight_layout()


def plot_scatter(x, y,title):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.scatter(x, y, alpha=0.5, s=1)
    plt.grid(linewidth=0.5)
