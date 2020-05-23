#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1a) Grid data only, informed attacker: classification

Cases (without obfuscation techniques)
    Worst case: Households use a new PK every transaction, no links between transactions
    TODO Realistic cases: Households change PKs at some interval, those on same PK are linked
    TODO Best case: Household has one PK, all transactions linked

Classifiers
    Neural network MPL classification. Investigate different layers and neurons.
    TODO Second method that better classifies and uses time series properly.

Classify
    Include consumer number, generator, and postcode for training set
    Drop those three from the test set
    Predictions for a) hourly, b) daily, & c) weekly
        Predictions for i) consumer number, ii) postcode
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

os.chdir("../BlockchainData/Hourly")

###################################
##    Classify on hourly data    ##
###################################
