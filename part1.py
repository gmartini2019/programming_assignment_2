#import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  
from sklearn.cluster import KMeans
# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clusters):
    data, labels = dataset
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init="random", random_state=12)
    kmeans.fit(data_standardized)
    kmeans_predictions = kmeans.predict(data_standardized)
    return kmeans_predictions

def fit_kmeans_random_init(data_label_pair, n_clusters):
    data, _ = data_label_pair  
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=12)
    kmeans.fit(data_standardized)
    return kmeans.predict(data_standardized)

def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)

    n_samples = 100
    seed = 42

    nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
    
    nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    
    b = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    
    bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)

    datasets_dict = {"nc": nc, "nm": nm, "bvv": bvv, "add": add, "b": b}

    answers = {"1A: datasets": datasets_dict}

    dct = answers["1A: datasets"] = datasets_dict

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 16))
    k_values = [2, 3, 5, 10]

    for i, k in enumerate(k_values):
        for j, (dataset_name, dataset) in enumerate(datasets_dict.items()):
            predicted_labels = fit_kmeans(dataset, k)
            axes[i, j].scatter(dataset[0][:, 0], dataset[0][:, 1], c=predicted_labels, cmap='viridis', s=10)
            if i == 0:
                axes[i, j].set_title(dataset_name)
            if j == 0:
                axes[i, j].set_ylabel(f'k={k}')

    plt.tight_layout()

    #plt.savefig("/home/gmartini2019/DATA_MINING/kmeans_clustering_plots.pdf")
    plt.close()

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    dct =  {"bvv": [3], "add": [3], "b": [3]} 
    answers["1C: cluster successes"] =  dct
    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = ["nc", "nm"]
    answers["1C: cluster failures"] = dct

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.

    k_values_sensitivity = [2, 3]
    dataset_names = list(datasets_dict.keys())

    init_sensitivity_results = {k: {name: [] for name in dataset_names} for k in k_values_sensitivity}

    repetitions = 5

    fig, axes = plt.subplots(nrows=5, ncols=len(k_values_sensitivity) * repetitions, figsize=(20, 16))


    for i, dataset_name in enumerate(dataset_names):
        dataset = datasets_dict[dataset_name]
        for j, k in enumerate(k_values_sensitivity):
            for rep in range(repetitions):
                predicted_labels = fit_kmeans_random_init(dataset, k)
                col = j * repetitions + rep
                axes[i, col].scatter(dataset[0][:, 0], dataset[0][:, 1], c=predicted_labels, cmap='viridis', s=10)
                if i == 0:
                    axes[i, col].set_title(f'k={k}, Rep={rep+1}')
                if rep == 0:
                    axes[i, col].set_ylabel(f'{dataset_name}\nk={k}')

    plt.tight_layout()

    #plt.savefig("/home/gmartini2019/DATA_MINING/kmeans_clustering_plots_part_D.pdf")
    plt.close()

    dct = answers["1D: datasets sensitive to initialization"] = ["nc", "nm", "bvv", "add"]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
