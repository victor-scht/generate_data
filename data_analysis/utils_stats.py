import numpy as np

# stats info 
def stats(arr, n_quant=10):
    # Calculate the mean
    n=len(arr)
    mean = np.mean(arr).item()

    # Calculate the standard deviation
    std_dev = np.std(arr).item()

    quantiles = dict()

    for i in range(1, n_quant):
        quant = int(i * 100.0 * (1.0 / n_quant))
        perc = np.percentile(arr, quant).item()
        quantiles[f"q{quant}%"] = perc

    stats_info = {"mean": mean,
                  "std": std_dev,
                  "quantiles": quantiles,
                  "n_samples":n}

    return stats_info
