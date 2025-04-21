import csv
import math
import random
import numpy as np

# Try to import matplotlib, but ignore if not available
try:
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib or scipy not available. Plotting functions disabled.")
    MATPLOTLIB_AVAILABLE = False

import BoundingBox
from BoundingBox import *

import Grid
from Grid import *

SHOW_PLOTS = False
PRINT_OUTPUT = False

# function that reads data from csv
# assumes data contains two columns, x and y
def read_data(filename):
    data = []
    with open(filename, newline='') as processedcsvfile:
        reader = csv.reader(processedcsvfile, delimiter=',')
        for row in reader:
            data.append([float(row[0]), float(row[1])])

    return data

# Algorithm to find the value of the Hopkins Statistic H
# function expects dataset in the form of a Grid class object
def hopkins_stat(D, m=10):
    # Each point 𝑃𝑖 has coordinates (𝑥𝑖, 𝑦𝑖). A data set 𝐷 contains n points 𝑃1, ... , 𝑃𝑛.
    # Choose 𝑚 << 𝑛 random points in 𝐷. Call this subset 𝐷𝑚 ⊂ 𝐷.
    n = len(D.points)
    Dm = [D.points[math.floor(random.random()*n)] for i in range(m)]

    # Generate 𝑚 random position points 𝑅 over the same range of points in 𝐷.
    minX = D.bbox.x_min
    maxX = D.bbox.x_max
    minY = D.bbox.y_min
    maxY = D.bbox.y_max
    R = [[minX + random.random()*(maxX - minX), minY + random.random()*(maxY - minY)] for i in range(m)]

    # Find the nearest neighbor in 𝐷 to each 𝑅𝑖 ∈ 𝑅, calculate this distance 𝑟𝑖.
    r = [D.nearest_neighbor(Ri)[1] for Ri in R]

    # Find the nearest neighbor in 𝐷 to each 𝑃𝑖 ∈ 𝐷𝑚 , calculate this distance 𝑝𝑖.
    p = [D.nearest_neighbor(Pi)[1] for Pi in Dm]

    # Compute the Hopkins statistic as:
    # 𝐻 = (∑[1≤𝑖≤m] 𝑟𝑖^2) / (∑[1≤𝑖≤m] 𝑝𝑖^2 + ∑[1≤𝑖≤m] 𝑟𝑖^2)
    sum_ri_2 = sum([math.pow(ri, 2) for ri in r])
    sum_pi_2 = sum([math.pow(pi, 2) for pi in p])
    H = sum_ri_2 / (sum_pi_2 + sum_ri_2)
    return H


def get_p_vals(H, mean, std):
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: scipy not available. Skipping p-value calculation.")
        return None
    Z_scores = (H - mean) / std
    p_values = 2 * (1 - stats.norm.cdf(np.abs(Z_scores)))
    return p_values


def get_p_vals_v2(H):
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: scipy not available. Skipping p-value calculation.")
        return None
    Z_scores = stats.zscore(H)
    p_values_twoTail =  stats.norm.sf(abs(Z_scores))*2
    return p_values_twoTail


def pivotal_method(bootstrap_means):
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    return lower, upper


def normal_dis_method(bootstrap_means):
    mean_bootstrap = np.mean(bootstrap_means)
    std_bootstrap = np.std(bootstrap_means)
    lower = mean_bootstrap - 1.96 * std_bootstrap
    upper = mean_bootstrap + 1.96 * std_bootstrap
    return lower, upper


def get_alpha_hats (bootstrap_means, var):
    alpha_hats = bootstrap_means * (((bootstrap_means * (1-bootstrap_means)) / var) - 1)
    return alpha_hats


def get_beta_hats (bootstrap_means, var):
    beta_hats = (1-bootstrap_means) * (((bootstrap_means * (1-bootstrap_means)) / var) - 1)
    return beta_hats


def plot_histogram(samples, mean, std, file_name):
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Skipping histogram plotting.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.hist(
        samples,
        bins=21,
        density=True,
        alpha=0.4,
        color='yellow',
        edgecolor='green',
        label=f"{file_name}"
    )
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label=f'µ: {mean:.3f}')
    plt.axvline(mean + std, color='blue', linestyle='dashed', linewidth=1.5, label=f'σ: {std:.3f}')
    plt.axvline(mean - std, color='blue', linestyle='dashed', linewidth=1.5)

    x_values = np.linspace(mean - 3 * std, mean + 3 * std, m)
    normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std) ** 2)
    plt.plot(x_values, normal_pdf, color='green', linewidth=1, label='Normal PDF')

    plt.title('Hopkins Statistic for k=1000 samples of m=100 points')
    plt.xlim([0, 1])
    plt.xlabel('Hopkins Statistic')
    plt.ylabel('Density')
    plt.grid(True)

    plt.legend()
    plt.show()


def plot_p_vals(p_vals, file_name):
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Skipping p-values plotting.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.hist(p_vals, 
            bins=21,
            alpha=0.4, 
            density=True,
            color='yellow',
            edgecolor='green',
            label=f"{file_name}")
    plt.xlim([0, 1])
    plt.xlabel("P-values")
    plt.ylabel("Density")
    plt.title(f"P-value Histogram for {file_name}")

    plt.legend()
    plt.show()


def plot_bootstrap(bootstrap_means_H, bootstrap_means_alpha_hat, bootstrap_means_beta_hat, file_name):
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Skipping bootstrap plotting.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.hist(bootstrap_means_H, 
            bins=21,
            alpha=0.4, 
            density=True,
            color='orange',
            edgecolor='red',
            label=f"Mean (H̅)")

    plt.xlabel('Value')
    plt.ylabel("Density")
    plt.title(f"Bootstrap H̅ Histogram for {file_name}")

    plt.legend()
    plt.show()


    plt.figure(figsize=(12, 6))
    plt.hist(bootstrap_means_alpha_hat, 
            bins=21,
            alpha=0.4, 
            density=True,
            color='yellow',
            edgecolor='green',
            label=f"Alpha_hat")

    plt.xlabel('Value')
    plt.ylabel("Density")
    plt.title(f"Bootstrap α̂  Histogram for {file_name}")

    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    
    plt.hist(bootstrap_means_beta_hat, 
            bins=21,
            alpha=0.4, 
            density=True,
            color='blue',
            edgecolor='purple',
            label=f"Beta_hat")
    plt.xlabel('Value')
    plt.ylabel("Density")
    plt.title(f"Bootstrap β̂  Histogram for {file_name}")

    plt.legend()
    plt.show()


def bootstrap(samples, B):
    bootstrap_means_H = []
    bootstrap_var_H = []
    for i in range(B):
        samples_rand = np.random.choice(np.array(samples), size=B, replace=True)
        bootstrap_means_H.append(np.mean(samples_rand))
        bootstrap_var_H.append(np.var(samples_rand, ddof=1))

    bootstrap_means_alpha_hat = get_alpha_hats(np.array(bootstrap_means_H), np.array(bootstrap_var_H))
    bootstrap_means_beta_hat = get_beta_hats(np.array(bootstrap_means_H), np.array(bootstrap_var_H))


    # Pivotal Method 
    lower_H_P, upper_H_P = pivotal_method(bootstrap_means_H)
    lower_alpha_hat_P, upper_alpha_hat_P = pivotal_method(bootstrap_means_alpha_hat)
    lower_beta_hat_P, upper_beta_hat_P = pivotal_method(bootstrap_means_beta_hat)

    # Normal distribution Method 
    lower_H_ND, upper_H_ND = normal_dis_method(bootstrap_means_H)
    lower_alpha_hat_ND, upper_alpha_hat_ND = normal_dis_method(bootstrap_means_alpha_hat)
    lower_beta_hat_ND, upper_beta_hat_ND = normal_dis_method(bootstrap_means_beta_hat)

    if PRINT_OUTPUT and MATPLOTLIB_AVAILABLE:
        print(f"(H̅) Pivotal method: lower- {lower_H_P}  upper {upper_H_P}")
        print(f"(H̅) Normal Distribution method: lower- {lower_H_ND}  upper {upper_H_ND}")

        print(f"(α̂ ) Pivotal method: lower- {lower_alpha_hat_P}  upper {upper_alpha_hat_P}")
        print(f"(α̂ ) Normal Distribution method: lower- {lower_alpha_hat_ND}  upper {upper_alpha_hat_ND}")

        print(f"(β̂ ) Pivotal method: lower- {lower_beta_hat_P}  upper {upper_beta_hat_P}")
        print(f"(β̂ ) Normal Distribution method: lower- {lower_beta_hat_ND}  upper {upper_beta_hat_ND}")

    return bootstrap_means_H, bootstrap_means_alpha_hat, bootstrap_means_beta_hat


if __name__=="__main__":

    # Change this to get the correct file 
    NUMBER_OF_POINTS = 2000


    filename1 = "DS-1/dataset-1-1-" + str(NUMBER_OF_POINTS) + ".csv"
    filename2 = "DS-2/dataset-2-1-" + str(NUMBER_OF_POINTS) + ".csv"
    filename3 = "DS-3/dataset-3-1-" + str(NUMBER_OF_POINTS) + ".csv"
    filename4 = "DS-4/dataset-4-1-" + str(NUMBER_OF_POINTS) + ".csv"

    # CHANGE FILE USED FOR TESTING HERE 
    FILE_IN_USE = filename1

    # get dataset for testing
    dataset = read_data(FILE_IN_USE)

    # create a Grid class object to make searching for nearest neighbor more efficient
    bbox = BoundingBox(x_min=-10000, x_max=10000, y_min=-10000, y_max=10000)
    D_Grid = Grid(dataset, bbox=bbox, partition_width=1000)

    # Computing Hopkins Statistics for B bootstraps
    m = 100
    B = 1000
    Hs = [hopkins_stat(D_Grid, m) for b in range(B)]

    # Statistics about the Samples
    mean = np.mean(Hs)
    std = np.std(Hs, ddof=1)
    alpha_hat = mean * (((mean * (1 - mean)) / np.var(Hs, ddof=1)) - 1)
    beta_hat = (1 - mean) * (((mean * (1 - mean)) / np.var(Hs, ddof=1)) - 1)
    
    if PRINT_OUTPUT:
        print("(H̅ ) Sample mean:", mean)
        print("(H̅ ) Sample standard deviation:", std)
        print("(α̂ ) Alpha hat:", alpha_hat)
        print("(β̂ ) Beta hat:", beta_hat)
    
    # Compute P-values
    if MATPLOTLIB_AVAILABLE:
        p_vals = get_p_vals_v2(Hs)
        
        # Bootstrapping
        bootstrap_means_H, bootstrap_means_alpha_hat, bootstrap_means_beta_hat = bootstrap(Hs, B)
        
        if SHOW_PLOTS:
            plot_histogram(Hs, mean, std, FILE_IN_USE)
            plot_p_vals(p_vals, FILE_IN_USE)
            plot_bootstrap(bootstrap_means_H, bootstrap_means_alpha_hat, bootstrap_means_beta_hat, FILE_IN_USE)