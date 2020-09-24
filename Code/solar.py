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
import sys
import glob
import numpy as np
import pandas as pd
import multiprocessing


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
    for data in ["0_customer_daily.csv", "0_postcode_daily.csv"]:
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


def solar_add_hourly(users, az):
    # Get postcodes
    os.chdir("../OriginalEnergyData/")
    postcodes = pd.read_csv("Solutions.csv")
    postcodes = dict(zip(postcodes['Customer'], postcodes['Postcode']))

    # Add in solar data
    os.chdir("../BlockchainData/hourly")

    # Calculate each hour as a portion of the total day
    # y = -(x-12.5)^2/19 + 1.6
    t = (0.0193 + 0.7535 + 1.1219 + 1.3851 + 1.5430) * 2 + 1.5956
    map_portion = {"7": 0.0193/t, "8": 0.7535/t, "9": 1.1219/t, "10": 1.3851/t, "11": 1.5430/t, "12": 1.5956/t,
                   "13": 1.5430/t, "14": 1.3851/t, "15": 1.1219/t, "16": 0.7535/t, "17": 0.0193/t}

    # Open solar data in advance
    for file in glob.glob("*_blockchain.csv"):
        customer = int(file.split("_")[0])
        if customer not in users or customer > 999:
            continue
        print(customer)

        df = pd.read_csv(file, header=0)
        df = df[df['Timestamp'].str.contains("2012") | df['Timestamp'].str.contains("2013")]
        df = df.loc[~((df['Timestamp'].str.contains("2012")) &
                      (df['Timestamp'].str.contains("/01/|/02/|/03/|/04/|/05/|/06/"))), :]
        df.reset_index(drop=True, inplace=True)
        df['Solar'] = 0

        os.chdir("../../WeatherData")
        for weather in glob.glob(f"{str(postcodes[customer])}_*.csv"):
            sol = pd.read_csv(weather, header=0)
        os.chdir("../BlockchainData/hourly")

        for index, timestamp in zip(df.index, df['Timestamp']):
            date = timestamp.split(' ')[0]
            time = timestamp.split(' ')[1]
            year = int(date.split('/')[2])
            month = int(date.split('/')[1])
            day = int(date.split('/')[0])
            hour = time.split(":")[0]
            try:
                portion = map_portion[hour]
                value = sol.loc[((sol['Year'] == year) & (sol['Month'] == month) & (sol['Day'] == day)),
                                'Daily global solar exposure (MJ/m*m)'].values[0] * portion
            except KeyError:
                value = 0

            df.iloc[index, 5] = round(value, 3)
        df['Amount'].round(3)
        df.to_csv(f"{file}_solar.csv", index=False)


def combine_hourly_solar():
    os.chdir("../BlockchainData/hourly")
    for data in ["0_customer_hourly_2012-13.csv", "0_postcode_hourly_2012-13.csv"]:
        print(data)
        energy_data = pd.read_csv(data, header=0)
        energy_data['Solar'] = 0.0
        count = 0

        for user in range(1, 300):
            print(user)
            df = pd.read_csv(f"{user}_blockchain.csv_solar.csv", header=0)

            for solar in zip(df['Solar']):
                energy_data.at[count, 'Solar'] = solar[0]
                count += 1

        print("Saving file")
        energy_data.to_csv(f"{data}_solar.csv", index=False)


def solar_add_half_hourly(users, az):
    # Get postcodes
    os.chdir("../OriginalEnergyData/")
    postcodes = pd.read_csv("Solutions.csv")
    postcodes = dict(zip(postcodes['Customer'], postcodes['Postcode']))

    # Add in solar data
    os.chdir("../BlockchainData/half_hourly")

    # Calculate each hour as a portion of the total day
    # y = -(x-12.5)^2/19 + 1.6
    t = (0.0741 + 0.2057 + 0.3241 + 0.4294 + 0.5215 + 0.6004 + 0.6662 + 0.71886 + 0.7583 + 0.78465 + 0.7978) * 2
    map_portion = {"7:00": 0.0741/t, "7:30": 0.2057/t, "8:00": 0.3241/t, "8:30": 0.4294/t, "9:00": 0.5215/t,
                   "9:30": 0.6004/t, "10:00": 0.6662/t, "10:30": 0.71886/t, "11:00": 0.7583/t, "11:30": 0.78465/t,
                   "12:00": 0.7978/t, "12:30": 0.7978/t, "13:00": 0.78465/t, "13:30": 0.7583/t, "14:00": 0.71886/t,
                   "14:30": 0.6662/t, "15:00": 0.6004/t, "15:30": 0.5215/t, "16:00": 0.4294/t, "16:30": 0.3241/t,
                   "17:00": 0.2057/t, "17:30": 0.0741/t}

    # Open solar data in advance
    for file in glob.glob("*_blockchain.csv"):
        customer = int(file.split("_")[0])
        if customer not in users or customer > 999:
            continue
        print(customer)

        df = pd.read_csv(file, header=0)
        df = df[df['Timestamp'].str.contains("2012")]
        df = df.loc[~((df['Timestamp'].str.contains("2012")) &
                      (df['Timestamp'].str.contains("/01/|/02/|/03/|/04/|/05/|/06/"))), :]
        df.reset_index(drop=True, inplace=True)
        df['Solar'] = 0

        os.chdir("../../WeatherData")
        for weather in glob.glob(f"{str(postcodes[customer])}_*.csv"):
            sol = pd.read_csv(weather, header=0)
        os.chdir("../BlockchainData/half_hourly")

        for index, timestamp in zip(df.index, df['Timestamp']):
            date = timestamp.split(' ')[0]
            time = timestamp.split(' ')[1]
            year = int(date.split('/')[2])
            month = int(date.split('/')[1])
            day = int(date.split('/')[0])
            try:
                portion = map_portion[time]
                value = sol.loc[((sol['Year'] == year) & (sol['Month'] == month) & (sol['Day'] == day)),
                                'Daily global solar exposure (MJ/m*m)'].values[0] * portion
            except KeyError:
                value = 0

            df.iloc[index, 5] = round(value, 3)
        df['Amount'].round(3)
        df.to_csv(f"{file}_solar.csv", index=False)


def combine_half_hourly_solar():
    os.chdir("../BlockchainData/half_hourly")
    for data in ["0_customer_half_hourly_2012-13a.csv", "0_postcode_half_hourly_2012-13a.csv"]:
        print(data)
        energy_data = pd.read_csv(data, header=0)
        energy_data['Solar'] = 0.0
        count = 0

        for user in range(1, 300-1):
            print(user)
            df = pd.read_csv(f"{user}_blockchain.csv_solar.csv", header=0)

            for solar in zip(df['Solar']):
                energy_data.at[count, 'Solar'] = solar[0]
                count += 1

        print("Saving file")
        energy_data.to_csv(f"{data}_solar.csv", index=False)


def compare_data():
    import statsmodels.tsa.stattools as ts

    # For daily data
    os.chdir("../OriginalEnergyData/")
    solutions = pd.read_csv("Solutions.csv")
    solutions = dict(zip(solutions['Customer'], solutions['Postcode']))

    os.chdir("../BlockchainData/daily/")
    corrs, coints = [], []
    all_results = {}

    for customer in os.listdir():
        customer_num = customer.split("_")[0]
        if customer[5] == 'b' or "daily" in customer or "2_blockchain" in customer:
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


def reconstruct_usage():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.exceptions import DataConversionWarning
    import warnings
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # Goal: Take household net energy (post meter so generation-usage) and predict usage
    os.chdir("../BlockchainData/half_hourly")

    rmses = []
    r2s = []
    # For each households transactions (files contain one year of data)
    for house in glob.glob(f"*csv_solar.csv"):
        df = pd.read_csv(house, header=0)

        # Separate into three dataframes by type
        df_cl = df[df['Type'] == "CL"]
        df_gc = df[df['Type'] == "GC"]
        df_gg = df[df['Type'] == "GG"]

        # Drop times without solar generation otherwise accuracy is massive cause usage = net
        frame = {'CL': df_cl['Amount'].values, 'GC': df_gc['Amount'].values, 'GG': df_gg['Amount'].values}
        df_comb = pd.DataFrame(frame)
        df_comb = df_comb[df_comb['GG'] > 0]

        # Calculate net energy consumption (CL + GC - GG) per time period
        se_use = df_comb['CL'].values + df_comb['GC'].values
        se_net = se_use - df_comb['GG'].values

        # Learn mapping of net energy to usage
        x = se_net.reshape(-1, 1)
        y = se_use.reshape(-1, 1)
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
        except ValueError:
            print(f"ERROR ON {house}")
            continue

        rf_model = RandomForestRegressor()
        rf_model.fit(x_train, y_train)

        # Test reconstruction
        y_pred = rf_model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = round(np.sqrt(mse), 4)
        r2 = round(r2_score(y_test, y_pred), 4)
        rmses.append(rmse)
        r2s.append(r2)
        print(f"{house.split('_')[0]}: RMSE {rmse} and R2 {r2}")
    print(f"Overall RMSE {sum(rmses)/len(rmses)} and R2 {sum(r2s)/len(r2s)}")


if __name__ == '__main__':

    # Check usage
    if not len(sys.argv) == 9:
        print("Use: python ./graphs.py [weekly] [daily] [hourly] [combine hourly] [half_hourly] "
              "[combine half_hourly] [stats] [reconstruct]")
        print("Use a 1 or 0 indicator for each argument")
        exit()

    if int(sys.argv[1]):
        print("Solar add weekly")
        solar_add_weekly()

    if int(sys.argv[2]):
        print("Solar add daily")
        solar_add_daily()

    if int(sys.argv[3]):
        print("Solar add hourly")
        processes = [
            multiprocessing.Process(target=solar_add_hourly, name="Number1",
                                    args=(list(range(1, 26)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number2",
                                    args=(list(range(26, 51)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number3",
                                    args=(list(range(51, 76)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number4",
                                    args=(list(range(76, 101)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number5",
                                    args=(list(range(101, 126)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number6",
                                    args=(list(range(126, 151)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number7",
                                    args=(list(range(151, 176)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number8",
                                    args=(list(range(176, 201)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number9",
                                    args=(list(range(201, 226)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number10",
                                    args=(list(range(226, 251)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number11",
                                    args=(list(range(251, 276)), 1)),
            multiprocessing.Process(target=solar_add_hourly, name="Number12",
                                    args=(list(range(276, 301)), 1)),
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    if int(sys.argv[4]):
        print("Combine hourly")
        combine_hourly_solar()

    if int(sys.argv[5]):
        print("Solar add half hourly")
        processes = [
            multiprocessing.Process(target=solar_add_half_hourly, name="Number1",
                                    args=(list(range(1, 26)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number2",
                                    args=(list(range(26, 51)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number3",
                                    args=(list(range(51, 76)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number4",
                                    args=(list(range(76, 101)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number5",
                                    args=(list(range(101, 126)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number6",
                                    args=(list(range(126, 151)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number7",
                                    args=(list(range(151, 176)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number8",
                                    args=(list(range(176, 201)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number9",
                                    args=(list(range(201, 226)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number10",
                                    args=(list(range(226, 251)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number11",
                                    args=(list(range(251, 276)), 1)),
            multiprocessing.Process(target=solar_add_half_hourly, name="Number12",
                                    args=(list(range(276, 301)), 1)),
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    if int(sys.argv[6]):
        print("Combine half hourly")
        combine_half_hourly_solar()

    if int(sys.argv[7]):
        print("Correlation and cointegraton")
        compare_data()

    if int(sys.argv[8]):
        print("Reconstruct usage from generation")
        reconstruct_usage()
