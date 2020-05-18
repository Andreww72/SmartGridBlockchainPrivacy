#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1a) Grid data only, informed attacker: classification

This treats all households on a single ledger

Classification methods
    Technique that yields greatest accuracy for each
    Include consumer number, generator, and postcode in for training
    Predictions for i) hourly, ii) daily, iii) weekly, iv) monthly data
        Predictions for i) consumer number, ii) postcode
            i) Split and k-fold training/validation
"""

import os

import pandas as pd

os.chdir("../BlockchainData/")

# Table of customers, postcodes, and generator sizes
extra_info = pd.read_csv(f"../OriginalEnergyData/Solutions.csv", header=0)
generator_col = 1
postcode_col = 2
num_customers = 300

###################################
##    Classify on hourly data    ##
###################################
os.chdir("../BlockchainData/Hourly")


###################################
##    Classify on daily data     ##
###################################
os.chdir("../Daily")


###################################
##    Classify on weekly data    ##
###################################
os.chdir("../Weekly")


###################################
##    Classify on monthly data   ##
###################################
os.chdir("../Monthly")
