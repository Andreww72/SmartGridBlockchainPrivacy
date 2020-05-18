#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1b) Grid data only, uninformed attacker: clustering

Clustering methods
    Comparing datasets themselves against each other
    i) Euclidean distance, ii) LTSM, iii) Cointegration analysis, iv) Clustering
    Clustering: https://petolau.github.io/Multiple-data-streams-clustering-in-r/
    https://www.researchgate.net/publication/322011057_Time_Series_Analysis_for_Spatial_Node_Selection_in_Environment_Monitoring_Sensor_Networks
    LSTM, Cointegration analysis
"""

import os
import json

import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

os.chdir("../BlockchainData/")

# Euclidean distance
results = {}
results_list = []
comparison_vector = np.zeros(78913-1)

for num in range(300):
    if not num == 2-1:
        print(f"Euclidean distance of {num+1}")
        df = pd.read_csv(f"{num+1}_blockchain.csv", header=0)
        arr = df['Amount'].to_numpy()
        euc_dist = round(np.linalg.norm(comparison_vector - arr), 2)
        results[num+1] = euc_dist
        results_list.append(euc_dist)

json.dump(results, open("results.json", "w"), indent=4)

# Simple cluster of Euclidean distance results for visualisation
np_array = np.array(results_list)
k_means = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                 n_clusters=3, n_init=10, random_state=0, tol=0.0001, verbose=0)

quotient_2d = np_array.reshape(-1, 1)
k_means.fit(quotient_2d)

times = []
for n in range(300):
    times.append(n)

colors = ['r', 'g', 'b']
centroids = k_means.cluster_centers_
for n, y in enumerate(centroids):
    plt.plot(1, y, marker='x', color=colors[n], ms=10)
plt.title('Kmeans cluster centroids')

Z = k_means.predict(quotient_2d)

n_clusters = 3
for n in range(n_clusters):
    # Filter data points to plot each in turn.
    ys = np_array[Z == n]
    xs = times[Z == n]

    plt.scatter(xs, ys, color=colors[n])

plt.title("Points by cluster")

# DTW

# Fancy cluster
# Might use R for this
