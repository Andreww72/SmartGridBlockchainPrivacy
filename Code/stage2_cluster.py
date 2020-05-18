#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
2b) Grid + solar data, uninformed attacker: clustering

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

os.chdir("../BlockchainData/")
