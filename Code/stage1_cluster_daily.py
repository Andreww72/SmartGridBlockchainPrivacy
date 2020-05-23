#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1b) Grid data only, uninformed attacker: clustering

Cases (without obfuscation techniques)
    Worst case: Cannot perform this clustering
    TODO Realistic cases: Households change PKs at some interval, those on same PK are linked
    TODO Best case: Household has one PK, all transactions linked

Clustering methods
    ii)  https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    iii) https://www.researchgate.net/publication/322011057_Time_Series_Analysis_for_Spatial_Node_Selection_in_Environment_Monitoring_Sensor_Networks
    iv)  https://petolau.github.io/Multiple-data-streams-clustering-in-r/

Cluster
    Comparing datasets themselves against each other
    Predictions for a) hourly, b) daily, & c) weekly
    i) Euclidean distance??, ii) LTSM, iii) Cointegration analysis, iv) Clustering
"""

import os
import json

import pandas as pd
import numpy as np

os.chdir("../BlockchainData/Daily")

###################################
##     Classify on daily data    ##
###################################
print("Clustering stage 1 daily data")

# Euclidean distance
results = {}
results_list = []
num_customers = 300
length_daily = 3288
comparison_vector = np.zeros(length_daily)

for num in range(num_customers):
    # Customer 2 dataset is missing several months, skip it
    if not num == 2-1:
        print(f"Euclidean distance of {num+1}")
        df = pd.read_csv(f"{num+1}_blockchain.csv", header=0)
        arr = df['Amount'].to_numpy()
        euc_dist = round(np.linalg.norm(comparison_vector - arr), 2)
        results[num+1] = euc_dist
        results_list.append(euc_dist)

json.dump(results, open("0_euclidean_results.json", "w"), indent=4)
