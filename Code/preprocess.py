#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocessing(data_freq, class_type, case, year, solar, net_export, pk, per_ledger):
    """Preprocess fully setup blockchain data for the ML analysis
    :parameter data_freq --> 'weekly', 'daily', 'hourly', or 'half_hourly' time data resolution.
    :parameter class_type --> 'customer', or 'postcode' are the target for classification.
    :parameter case --> 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger' analysis.
    :parameter year --> 0 (2010-11), 1 (2011-12), 2 (2012-13 or 1st half if hourly), or 3 (2012-13 2nd half).
    :parameter solar --> Boolean if to load data file with solar attribute
    :parameter strip_zeros --> bool to remove rows with 0 amounts
    :returns return x_train, x_test, y_train, y_test
    """

    ledger = "postcode" if case == "lpp" else "customer"
    solar_n = "_solar" if solar else ""
    datafile = f"0_{ledger}_{data_freq}{solar_n}.csv"

    if data_freq == "hourly" or data_freq == "half_hourly":
        if year == 0:
            datafile = f"0_{ledger}_{data_freq}{solar_n}_2010-11.csv"
        elif year == 1:
            datafile = f"0_{ledger}_{data_freq}_2011-12{solar_n}.csv"
        elif year == 2 and data_freq == 'hourly':
            datafile = f"0_{ledger}_{data_freq}_2012-13{solar_n}.csv"
        elif year == 2 and data_freq == 'half_hourly':
            datafile = f"0_{ledger}_{data_freq}_2012-13a{solar_n}.csv"
        elif year == 3:
            datafile = f"0_{ledger}_{data_freq}_2012-13b{solar_n}.csv"

    if pk or per_ledger:
        datafile = f"0_pk{pk}_ledger{per_ledger}_{data_freq}.csv"

    data = None
    try:
        data = pd.read_csv(datafile, header=0)
    except FileNotFoundError:
        print(f"{datafile} file not found")
        exit()

    if net_export:
        df_cl = data[data['Type'] == "CL"]
        df_gc = data[data['Type'] == "GC"]
        df_gg = data[data['Type'] == "GG"]
        frame = {'CL': df_cl['Amount'].values, 'GC': df_gc['Amount'].values, 'GG': df_gg['Amount'].values}
        df_comb = pd.DataFrame(frame)
        data = df_cl
        data['Amount'] = df_comb['CL'].values + df_comb['GC'].values - df_comb['GG'].values
        data.drop(['Type'], axis=1, inplace=True)
    else:
        data['Type'] = data['Type'].astype('category').cat.codes
    # Convert categorical columns to numeric
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True)
    data['Timestamp'] = (data.Timestamp - pd.to_datetime('1970-01-01')).dt.total_seconds()

    data.sort_values(['Ledger', 'Timestamp'], ascending=[True, True])

    if not per_ledger or not per_ledger > 1:
        # If only one ledger each then all PKs on separate ledger and thus useless
        data.drop(['Ledger'], axis=1, inplace=True)
    if case == "aol":
        # Drop the PKs as all would be unique and thus useless
        data.drop(['PK'], axis=1, inplace=True)

    # Strip zeros
    data = data[data['Amount'] != 0]

    x = data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    y = None
    if class_type == 'customer':
        y = data['Customer']
    elif class_type == 'postcode':
        y = data['Postcode']

    # Preprocess
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    if case == "obfs" and per_ledger > 1:
        x_test['Ledger'] = x_test['Ledger'] / 2

    scaler_num = StandardScaler()
    scaler_num.fit(x_train) # Fit only to training data
    StandardScaler(copy=True, with_mean=True, with_std=True)

    # Apply the transformations to the data:
    x_train = scaler_num.transform(x_train)
    x_test = scaler_num.transform(x_test)

    return x_train, x_test, y_train, y_test
