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
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
#import utils as u
from scipy.cluster.hierarchy import fcluster

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(dataset, n_clusters, linkage_type='ward'):
    data= dataset  
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    hierarchical.fit(data_standardized)
    
    return hierarchical.labels_

def fit_modified(dataset, linkage_method):

    data,_ = dataset  
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    Z = linkage(data_standardized, method=linkage_method)

    distance_deltas = np.diff(Z[:, 2])
    max_delta_index = np.argmax(distance_deltas)
    
    cutoff_distance = Z[max_delta_index, 2] + distance_deltas[max_delta_index] / 2

    cluster_labels = fcluster(Z, cutoff_distance, criterion='distance')
    
    cluster_labels = np.array(cluster_labels) - min(cluster_labels)
    
    return cluster_labels


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)



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

    dct = answers["4A: datasets"] = datasets_dict

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    linkage_methods = ['average', 'single', 'ward', 'complete']
    fig, axes = plt.subplots(nrows=len(linkage_methods), ncols=len(datasets_dict), figsize=(20, 16))

    for i, linkage in enumerate(linkage_methods):
        for j, (dataset_name, dataset) in enumerate(datasets_dict.items()):
            data, labels = dataset
            predicted_labels = fit_hierarchical_cluster(data, n_clusters=2, linkage_type=linkage)
            axes[i, j].scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis', s=10)
            if i == 0:
                axes[i, j].set_title(dataset_name)
            if j == 0:
                axes[i, j].set_ylabel(f'Linkage: {linkage}')

    plt.tight_layout()
    pdf_file_path = 'hierarchical_clustering_plots.pdf'
    plt.savefig(pdf_file_path)
    plt.close()
    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc", "nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    fig, axes = plt.subplots(nrows=1, ncols=len(datasets_dict), figsize=(20, 5))  

    for j, (dataset_name, dataset) in enumerate(datasets_dict.items()):
        predicted_labels = fit_modified(dataset, linkage_method='single')
        axes[j].scatter(dataset[0][:, 0], dataset[0][:, 1], c=predicted_labels, cmap='viridis', s=10)
        axes[j].set_title(dataset_name)

    plt.tight_layout()
    pdf_filename_with_cutoff = "hierarchical_clustering_with_cutoff_plots.pdf"
    plt.savefig(pdf_filename_with_cutoff)
    plt.close()

    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
