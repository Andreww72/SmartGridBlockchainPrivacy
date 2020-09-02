#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
Grid + solar data, informed attacker: classification
Use: python ./solar -h for usage
Cases
    Ledger per customer: Households have one PK and their transactions on a unique ledger
    Ledger per postcode: Households have one PK and their transactions on a ledger by postcode
    One mixed ledger: Households use a new PK every transaction, no links between transactions
Classifiers
    Neural network - CNN classification
    Decision tree - Random forest classification
Classify
    Drop consumer number, generator, and postcode for training and test sets
    Predictions for a) weekly, & b) daily
        Predictions for i) consumer number, ii) postcode
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts


def rearrange_data():
    for postcode in os.listdir():
        for item in os.listdir(postcode):
            if item.endswith(".txt"):
                os.remove(f"{postcode}/{item}")
            else:
                station = re.search(r'[0-9]{6}', item).group(0)
                new_name = f"{postcode}_{station}.csv"
                os.rename(f"{postcode}/{item}", new_name)


def crop_years():
    # Only want 1 July 2010 to 30 June 2013
    for postcode in os.listdir():
        data = pd.read_csv(postcode, header=0)
        print(postcode)

        try:
            data.drop(['Product code', 'Bureau of Meteorology station number'], axis=1, inplace=True)
        except KeyError:
            pass

        data = data[data['Year'].isin([2010, 2011, 2012, 2013])]
        data = data.loc[~((data['Year'] == 2010) & (data['Month'].isin([1, 2, 3, 4, 5, 6]))), :]
        data = data.loc[~((data['Year'] == 2013) & (data['Month'].isin([7, 8, 9, 10, 11, 12]))), :]

        data.to_csv(postcode, index=False)


def solar_add_weekly():
    # Add in solar data
    os.chdir("../BlockchainData/weekly")
    for data in ["0_customer_weekly.csv", "0_postcode_weekly.csv"]:
        print(data)
        energy_data = pd.read_csv(data, header=0)
        os.chdir("../../WeatherData/")

        energy_data['Solar'] = 0
        current_solar_open = None
        solar_data = None

        for index, row in energy_data.iterrows():
            postcode = row['Postcode']

            if not postcode == current_solar_open:
                current_solar_open = postcode
                for file in glob.glob(f"{postcode}_*.csv"):
                    solar_data = pd.read_csv(file, header=0)

            timestamp = row['Timestamp']
            year = int(timestamp.split('-')[0])
            month = int(timestamp.split('-')[1])
            day = int(timestamp.split('-')[2].split(" ")[0])

            if year == 2010 and month == 7 and day == 5:
                # Find corresponding date in solar_data
                solar_row = solar_data.loc[((solar_data['Year'] == year) &
                                            (solar_data['Month'] == month) &
                                            (solar_data['Day'] == day)), :]

                # Add dates back to 2010-07-01
                week_end_index = solar_row.index.values.astype(int)[0]
                range_wanted = range(week_end_index-4, week_end_index+1, 1)

            elif year == 2013 and month == 7 and day == 1:
                month = 6
                day = 30
                # Find corresponding date in solar_data
                solar_row = solar_data.loc[((solar_data['Year'] == year) &
                                            (solar_data['Month'] == month) &
                                            (solar_data['Day'] == day)), :]

                # Add dates back a week but ignore 2013-07-01
                week_end_index = solar_row.index.values.astype(int)[0]
                range_wanted = range(week_end_index-5, week_end_index+1, 1)

            else:
                # Find corresponding date in solar_data
                solar_row = solar_data.loc[((solar_data['Year'] == year) &
                                            (solar_data['Month'] == month) &
                                            (solar_data['Day'] == day)), :]
                # Add dates back a week
                week_end_index = solar_row.index.values.astype(int)[0]
                range_wanted = range(week_end_index-6, week_end_index+1, 1)

            solar_value = 0
            solar_col = 3
            for thing in range_wanted:
                solar_value += solar_data.iloc[thing, solar_col]
            solar_value = round(solar_value, 2)

            energy_data.iloc[index, energy_data.columns.get_loc('Solar')] = solar_value

        os.chdir("../BlockchainData/weekly")
        energy_data.to_csv(f"{data}_solar.csv", index=False)
        os.chdir("../../BlockchainData/weekly")


def solar_add_daily():
    # Add in solar data
    os.chdir("../BlockchainData/daily")
    for data in ["0_postcode_daily.csv"]:
        print(data)
        energy_data = pd.read_csv(data, header=0)
        os.chdir("../../WeatherData/")

        energy_data['Solar'] = 0
        current_solar_open = None
        solar_data = None

        for index, row in energy_data.iterrows():
            postcode = row['Postcode']

            if not postcode == current_solar_open:
                current_solar_open = postcode
                for file in glob.glob(f"{postcode}_*.csv"):
                    solar_data = pd.read_csv(file, header=0)

            timestamp = row['Timestamp']
            year = int(timestamp.split('/')[2])
            month = int(timestamp.split('/')[1])
            day = int(timestamp.split('/')[0])

            # Find corresponding date in solar_data
            solar_row = solar_data.loc[((solar_data['Year'] == year) &
                                        (solar_data['Month'] == month) &
                                        (solar_data['Day'] == day)), :]

            solar_value = solar_row['Daily global solar exposure (MJ/m*m)'].values[0]
            energy_data.iloc[index, energy_data.columns.get_loc('Solar')] = solar_value

        os.chdir("../BlockchainData/daily")
        energy_data.to_csv(f"{data}_solar.csv", index=False)
        os.chdir("../../BlockchainData/daily")


def compare_data(coint, correl):
    # For daily data
    os.chdir("../OriginalEnergyData/")
    solutions = pd.read_csv("Solutions.csv")
    solutions = dict(zip(solutions['Customer'], solutions['Postcode']))

    os.chdir("../BlockchainData/daily/")
    corrs, coints = [], []
    all_results = {}

    for customer in os.listdir():
        customer_num = customer.split("_")[0]
        if customer[5] == 'b' or "daily" in customer:
            continue
        print(customer)
        data = pd.read_csv(customer)
        data = data[data['Type'] == "GG"]
        data = data['Amount'].round(3)
        data = data.to_list()

        os.chdir("../../WeatherData/")
        corr_results = []
        coint_results = []

        postcodes = os.listdir()
        postcode_nums = []
        for postcode in postcodes:
            postcode_nums.append(postcode.split("_")[0])
            solar = pd.read_csv(postcode)
            solar = solar['Daily global solar exposure (MJ/m*m)'].round(1)
            solar = solar.to_list()

            corr_results.append(np.corrcoef(data, solar)[0][1])
            coint_results.append(ts.coint(data, solar)[0])

        # Order them
        df_corr = pd.DataFrame({'Postcode': postcode_nums, 'Correlation': corr_results})
        df_coint = pd.DataFrame({'Postcode': postcode_nums, 'Cointegration': coint_results})
        df_corr.sort_values(by=["Correlation"], ascending=False, inplace=True)
        df_coint.sort_values(by=["Cointegration"], ascending=True, inplace=True)
        df_corr.reset_index(drop=True, inplace=True)
        df_coint.reset_index(drop=True, inplace=True)

        # Find position of correct postcode
        position_corr = df_corr[df_corr['Postcode'] == str(solutions[int(customer_num)])].index[0]
        position_coint = df_coint[df_coint['Postcode'] == str(solutions[int(customer_num)])].index[0]

        all_results[customer_num] = [position_corr, position_coint]
        corrs.append(position_corr)
        coints.append(position_coint)

        os.chdir("../BlockchainData/daily/")

    # Average results
    corrs = np.array(corrs)
    average_corr = corrs.mean()
    spread_corr = corrs.std()
    coints = np.array(coints)
    average_coint = coints.mean()
    spread_coint = coints.std()

    print(corrs)
    print(coints)
    print(f"Average correlation position: {average_corr}")
    print(f"Spread correlation position: {spread_corr}")
    print(f"Average cointegration position: {average_coint}")
    print(f"Spread correlation position: {spread_coint}")


if __name__ == '__main__':
    compare_data(False, True)
