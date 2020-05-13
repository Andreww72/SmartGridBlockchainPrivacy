#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML clustering on initial blockchain data (1a)

1a) Grid data only, uninformed attacker: Euclidean distance, and clustering
https://petolau.github.io/Multiple-data-streams-clustering-in-r/
1b) Grid data only, informed attacker: classification (split and k-fold)

2a) Grid + solar data, uninformed attacker: clustering
2b) Grid + solar data, informed attacker: classification (split and k-fold)

3a) Grid + solar + other data, uninformed attacker: clustering
3b) Grid + solar + other data, informed attacker: classification (split and k-fold)

4) Apply obfuscation methods to grid data, add solar + other data
    4i) Technique 1    4ii) Technique 2    4iii) Technique 3
4a) On all techniques --> Uninformed attacker: clustering
4b) On all techniques --> Informed attacker: classification (split and k-fold)

"""

import os
import glob
import json

import pandas as pd
import numpy as np
from sklearn import datasets, cluster


os.chdir("../BlockchainData/")

# Euclidean distance
results = {}
comparison_vector = np.zeros(78913-1)

for num in range(300):
    if not num == 2-1:
        print(f"Euclidean distance of {num+1}")
        df = pd.read_csv(f"{num+1}_blockchain.csv", header=0)
        arr = df['Amount'].to_numpy()
        results[num+1] = round(np.linalg.norm(comparison_vector - arr), 2)

json.dump(results, open("results.json", "w"), indent=4)
