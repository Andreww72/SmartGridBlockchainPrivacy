#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML data preparation for analysis
1a) Grid data only, informed attacker: classification

Use: python ./stage1_classify_prep.py [hourly] [daily] [weekly]
Use a 1 or 0 indicator for each argument

Cases (without obfuscation techniques)
    Worst case: Households use a new PK every transaction, no links between transactions
    TODO Realistic cases: Households change PKs at some interval, those on same PK are linked
    TODO Best case: Household has one PK, all transactions linked

Classifiers
    Neural network MLP classification. TODO Investigate changing parameters
    LSTM single layer network classification. TODO Investigate changing parameters

Classify
    Include consumer number, generator, and postcode for training set
    Drop those three from the test set
    Predictions for a) hourly, b) daily, & c) weekly
        Predictions for i) consumer number, ii) postcode
"""

import os
import sys
import pandas as pd

# Table of customers, postcodes, and generator sizes
generator_col = 1
postcode_col = 2
num_customers = 300


###################################
##      Prepare hourly data      ##
###################################
def hourly():
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
##      Preprare daily data      ##
###################################
def daily():
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
##      Prepare weekly data      ##
###################################
def weekly():

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


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 4:
        print("Use: python ./stage1_classify_prep.py [hourly] [daily] [weekly]")
        print("Use a 1 or 0 indicator for each argument")
        exit()

    extra_info = pd.read_csv(f"../OriginalEnergyData/Solutions.csv", header=0)
    os.chdir("../BlockchainData/Hourly")

    if int(sys.argv[1]):
        print("Preparing hourly data")
        hourly()

    if int(sys.argv[2]):
        os.chdir("../Daily")
        print("Preparing daily data")
        daily()

    if int(sys.argv[3]):
        os.chdir("../Weekly")
        print("Preparing weekly data")
        weekly()
