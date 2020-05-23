#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML data preparation for analysis
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

hourly_to_combine = []
# Can include customer two for classification
# Loop on remaining files to append to first
for num in range(num_customers):
    print(f"Load and adjust hourly {num+1}")
    df = pd.read_csv(f"{num+1}_blockchain.csv", header=0)

    # Add columns needed
    row_count = df.shape[0]
    postcode = extra_info.iloc[num, postcode_col]
    generator = extra_info.iloc[num, generator_col]
    df.insert(loc=0, column='Customer', value=[num+1] * row_count)
    df.insert(loc=1, column='Postcode', value=[postcode] * row_count)
    df.insert(loc=2, column='Generator', value=[generator] * row_count)

    hourly_to_combine.append(df)

print(f"Concatenate and save hourly")
combined_hourly = pd.concat(hourly_to_combine)
combined_hourly.to_csv('0_1a_combined_hourly.csv', index=False)

###################################
##    Classify on daily data     ##
###################################
os.chdir("../Daily")

daily_to_combine = []
# Can include customer two for classification
# Loop on remaining files to append to first
for num in range(num_customers):
    print(f"Load and adjust daily {num+1}")
    df = pd.read_csv(f"{num+1}_blockchain.csv", header=0)

    # Add columns needed
    row_count = df.shape[0]
    postcode = extra_info.iloc[num, postcode_col]
    generator = extra_info.iloc[num, generator_col]
    df.insert(loc=0, column='Customer', value=[num+1] * row_count)
    df.insert(loc=1, column='Postcode', value=[postcode] * row_count)
    df.insert(loc=2, column='Generator', value=[generator] * row_count)

    daily_to_combine.append(df)

print(f"Concatenate and save daily")
combined_daily = pd.concat(daily_to_combine)
combined_daily.to_csv('0_1a_combined_daily.csv', index=False)

###################################
##    Classify on weekly data    ##
###################################
os.chdir("../Weekly")

weekly_to_combine = []
# Can include customer two for classification
# Loop on remaining files to append to first
for num in range(num_customers):
    print(f"Load and adjust weekly {num+1}")
    df = pd.read_csv(f"{num+1}_blockchain.csv", header=0)

    # Add columns needed
    row_count = df.shape[0]
    postcode = extra_info.iloc[num, postcode_col]
    generator = extra_info.iloc[num, generator_col]
    df.insert(loc=0, column='Customer', value=[num+1] * row_count)
    df.insert(loc=1, column='Postcode', value=[postcode] * row_count)
    df.insert(loc=2, column='Generator', value=[generator] * row_count)

    weekly_to_combine.append(df)

print(f"Concatenate and save weekly")
combined_weekly = pd.concat(weekly_to_combine)
combined_weekly.to_csv('0_1a_combined_weekly.csv', index=False)
