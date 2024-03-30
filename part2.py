
# import plotly.figure_factory as ff
import pickle

import myplots as myplt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
import math
import pickle
from sklearn.metrics import pairwise_distances



# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans_inertia(dataset, n_clusters):
    data, _ = dataset  
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init="random", random_state=42)
    kmeans.fit(data_standardized)
    inertia = kmeans.inertia_
    return inertia

def fit_kmeans(dataset, n_clusters):
    data, _ = dataset  
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init="random", random_state=42)
    kmeans.fit(data_standardized)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    sse = sum(np.min(pairwise_distances(data_standardized, centers, metric='euclidean')**2, axis=1))
    return sse


def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    data= make_blobs(center_box=(-20, 20), n_samples=20, centers=5, random_state=12, return_centers=True)

    answers["2A: blob"] = list(data)
    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    k_values = range(1, 9)
    sse_values = []
    for k in k_values:
        sse = fit_kmeans_inertia([data[0], data[1]], k)  
        sse_values.append([k, sse])
    plt.figure(figsize=(10, 6))
    plt.plot([k for k, _ in sse_values], [sse for _, sse in sse_values], marker='o')
    plt.title('SSE as a function of k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.show()

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    answers["2C: SSE plot"] = sse_values

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    k_values = range(1, 9)
    sse_values_inertia = []
    for k in k_values:
        sse = fit_kmeans_inertia([data[0], data[1]], k)  
        sse_values_inertia.append([k, sse])
    plt.figure(figsize=(10, 6))
    plt.plot([k for k, _ in sse_values_inertia], [sse for _, sse in sse_values_inertia], marker='o')
    plt.title('INERTIA as a function of k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('INERTIA')
    plt.grid(True)
    plt.show()

    # dct value has the same structure as in 2C
    answers["2D: inertia plot"] = sse_values_inertia

    # dct value should be a string, e.g., "yes" or "no"
    answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
