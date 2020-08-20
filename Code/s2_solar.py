#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
Grid + solar data, informed attacker: classification
Use: python ./s1_stage -h for usage
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
import pandas as pd


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
            if row['Type'] == "GG":
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
    for data in ["0_customer_daily.csv"]:
        print(data)
        energy_data = pd.read_csv(data, header=0)
        os.chdir("../../WeatherData/")

        energy_data['Solar'] = 0
        current_solar_open = None
        solar_data = None

        for index, row in energy_data.iterrows():
            if row['Type'] == "GG":
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


if __name__ == '__main__':
    os.chdir("../WeatherData/")

    solar_add_weekly()
