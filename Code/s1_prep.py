#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocessing(data_freq, class_type, case, year, strip_zeros=True):
    """Preprocess fully setup blockchain data for the ML analysis
    :parameter data_freq --> 'weekly', 'daily', 'hourly', or 'half_hourly' time data resolution.
    :parameter class_type --> 'customer', or 'postcode' are the target for classification.
    :parameter case --> 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger' analysis.
    :parameter year --> 0 (2010-11), 1 (2011-12), 2 (2012-13 or 1st half if hourly), or 3 (2012-13 2nd half).
    :parameter strip_zeros --> bool to remove rows with 0 amounts
    :returns return x_train, x_test, y_train, y_test
    """

    print(f"Preprocessing {data_freq} {case} data")
    datafile = f"0_combined_{data_freq}.csv"
    if data_freq == 'hourly' or data_freq == 'half_hourly':
        if year == 0:
            datafile = f"0_combined_{data_freq}_2010-11.csv"
        elif year == 1:
            datafile = f"0_combined_{data_freq}_2011-12.csv"
        elif year == 2 and data_freq == 'hourly':
            datafile = f"0_combined_{data_freq}_2012-13.csv"
        elif year == 2 and data_freq == 'half_hourly':
            datafile = f"0_combined_{data_freq}_2012-13a.csv"
        elif year == 3:
            datafile = f"0_combined_{data_freq}_2012-13b.csv"
    data = pd.read_csv(datafile, header=0)

    # Convert categorical columns to numeric
    data['Type'] = data['Type'].astype('category').cat.codes
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True)
    data['Timestamp'] = (data.Timestamp - pd.to_datetime('1970-01-01')).dt.total_seconds()

    if case == 'one_ledger':
        # Drop the PK and hash information
        data.drop(['Hash', 'PHash', 'PK'], axis=1, inplace=True)
        # Structure: Customer | Postcode | Generator | Timestamp | Type | Amount
    else:
        pass
        # Structure: Customer | Postcode | Generator | Hash | PHash | PK | Timestamp | Type | Amount

    if strip_zeros:
        data = data[data['Amount'] != 0]

    if class_type == 'customer':
        x = data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
        y = data['Customer']
    elif class_type == 'postcode':
        x = data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
        y = data['Postcode']

    # Preprocess
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    scaler_num = StandardScaler()
    scaler_num.fit(x_train) # Fit only to training data
    StandardScaler(copy=True, with_mean=True, with_std=True)

    # Apply the transformations to the data:
    x_train = scaler_num.transform(x_train)
    x_test = scaler_num.transform(x_test)

    return x_train, x_test, y_train, y_test
