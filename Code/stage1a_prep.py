#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML data preparation for analysis
1a) Grid data only, informed attacker: classification

Use: python ./stage1a_prep.py [half-hourly] [hourly] [daily] [weekly]
Use a 1 or 0 indicator for each argument

Cases
    Worst case: Households use a new PK every transaction, no links between transactions
    Obfuscation cases: Households change PKs at some interval, those on same PK are linked
    Best case: Household has one PK, all transactions linked

Classifiers
    Neural network MLP classification
    Random forest classification
    KNN classification

Classify
    Include consumer number, generator, and postcode for training set
    Drop those three from the test set
    Predictions for a) half-hourly, b) hourly, c) daily, & d) weekly
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
##   Prepare half hourly data    ##
###################################
def half_hourly():
    half_hourly_to_combine = []
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

        half_hourly_to_combine.append(df)

    print("Concatenate hourly")
    combined_half_hourly = pd.concat(half_hourly_to_combine)
    print("Convert timestamp str to timestamp for spitting")
    combined_half_hourly['Timestamp'] = pd.to_datetime(combined_half_hourly['Timestamp'], dayfirst=True)

    print("Splitting and saving hourly")
    # Hourly data is unmanageable when all together, split into financial years
    split_year = combined_half_hourly[
        (combined_half_hourly['Timestamp'] >= pd.Timestamp(2010, 7, 1)) &
        (combined_half_hourly['Timestamp'] <= pd.Timestamp(2011, 6, 30))]
    split_year.to_csv('0_1a_combined_hourly_2010-11.csv', index=False)

    split_year = combined_half_hourly[
        (combined_half_hourly['Timestamp'] >= pd.Timestamp(2011, 7, 1)) &
        (combined_half_hourly['Timestamp'] <= pd.Timestamp(2012, 6, 30))]
    split_year.to_csv('0_1a_combined_hourly_2011-12.csv', index=False)

    split_year = combined_half_hourly[
        (combined_half_hourly['Timestamp'] >= pd.Timestamp(2012, 7, 1)) &
        (combined_half_hourly['Timestamp'] <= pd.Timestamp(2013, 6, 30))]
    split_year.to_csv('0_1a_combined_hourly_2012-13.csv', index=False)


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

    print("Concatenate hourly")
    combined_hourly = pd.concat(hourly_to_combine)
    print("Convert timestamp str to timestamp for spitting")
    combined_hourly['Timestamp'] = pd.to_datetime(combined_hourly['Timestamp'], dayfirst=True)

    print("Splitting and saving hourly")
    # Hourly data is unmanageable when all together, split into financial years
    split_year = combined_hourly[
        (combined_hourly['Timestamp'] >= pd.Timestamp(2010, 7, 1)) &
        (combined_hourly['Timestamp'] <= pd.Timestamp(2011, 6, 30))]
    split_year.to_csv('0_1a_combined_hourly_2010-11.csv', index=False)

    split_year = combined_hourly[
        (combined_hourly['Timestamp'] >= pd.Timestamp(2011, 7, 1)) &
        (combined_hourly['Timestamp'] <= pd.Timestamp(2012, 6, 30))]
    split_year.to_csv('0_1a_combined_hourly_2011-12.csv', index=False)

    split_year = combined_hourly[
        (combined_hourly['Timestamp'] >= pd.Timestamp(2012, 7, 1)) &
        (combined_hourly['Timestamp'] <= pd.Timestamp(2013, 6, 30))]
    split_year.to_csv('0_1a_combined_hourly_2012-13.csv', index=False)


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
    if not len(sys.argv) == 5:
        print("Use: python ./stage1a_prep.py [half-hourly] [hourly] [daily] [weekly]")
        print("Use a 1 or 0 indicator for each argument")
        exit()

    extra_info = pd.read_csv(f"../OriginalEnergyData/Solutions.csv", header=0)
    os.chdir("../BlockchainData/Hourly")

    if int(sys.argv[1]):
        print("Preparing hourly data")
        half_hourly()

    if int(sys.argv[2]):
        print("Preparing hourly data")
        hourly()

    if int(sys.argv[3]):
        os.chdir("../Daily")
        print("Preparing daily data")
        daily()

    if int(sys.argv[4]):
        os.chdir("../Weekly")
        print("Preparing weekly data")
        weekly()
