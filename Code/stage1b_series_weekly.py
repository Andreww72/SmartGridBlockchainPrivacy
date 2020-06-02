#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1b) Grid data only, uninformed attacker: clustering

Use: python ./stage1a_points_weekly.py [case] [m1] [m2]
Use a 0 for worst case, 1 for best case for case argument
Use a 1 or 0 indicator for method arguments

Cases (without obfuscation techniques)
    TODO Worst case: Cannot perform this clustering
    TODO Realistic cases: Households change PKs at some interval, those on same PK are linked. Train on first two years, test on third. Instead of random split.
    TODO Best case: Household has one PK, all transactions linked

Clustering methods
    ii)  https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
        https://www.analyticsvidhya.com/blog/2019/01/introduction-time-series-classification/
    iii) https://www.researchgate.net/publication/322011057_Time_Series_Analysis_for_Spatial_Node_Selection_in_Environment_Monitoring_Sensor_Networks
    iv)  https://petolau.github.io/Multiple-data-streams-clustering-in-r/

https://tsfresh.readthedocs.io/en/latest/text/introduction.html
http://alexminnaar.com/2014/04/16/Time-Series-Classification-and-Clustering-with-Python.html

Cluster
    Comparing datasets themselves against each other
    Predictions for a) hourly, b) daily, & c) weekly
    i) LTSM, iii) Cointegration analysis, iv) Clustering
"""

import os
import sys
import multiprocessing
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


###################################
##         Preprocessing         ##
###################################
def preprocessing(case=1, strip_zeros=False):
    print("Preprocessing stage 1 weekly data")

    weekly_data = pd.read_csv('0_1a_combined_weekly.csv', header=0)

    # Convert categorical columns to numeric
    weekly_data['Type'] = weekly_data['Type'].astype('category').cat.codes

    weekly_data['Timestamp'] = pd.to_datetime(weekly_data['Timestamp'], dayfirst=True)
    weekly_data['Timestamp'] = (weekly_data.Timestamp - pd.to_datetime('1970-01-01')).dt.total_seconds()

    if case == 0:
        print("Preprocessing data for worst case")
        # Drop the PK and hash information
        weekly_data.drop(['Hash', 'PHash', 'PK'], axis=1, inplace=True)
        # Structure: Customer | Postcode | Generator | Timestamp | Type | Amount
    elif case == 1:
        print("Preprocessing data for best case")
        # Don't remove anything lol
        # Structure: Customer | Postcode | Generator | Hash | PHash | PK | Timestamp | Type | Amount
    else:
        print("Invalid case selected")
        print("Invalid usage: python ./stage1a_points_weekly.py [case] [MLP] [KNN] [KMS]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for method arguments")

    if strip_zeros:
        weekly_data = weekly_data[weekly_data['Amount'] != 0]

    x_num = weekly_data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    y_num = weekly_data['Customer']
    x_post = weekly_data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    y_post = weekly_data['Postcode']

    global X_train_num, X_test_num, Y_train_num, Y_test_num
    global X_train_post, X_test_post, Y_train_post, Y_test_post
    X_train_num, X_test_num, Y_train_num, Y_test_num = train_test_split(x_num, y_num)
    X_train_post, X_test_post, Y_train_post, Y_test_post = train_test_split(x_post, y_post)

    # Preprocess
    scaler_num = StandardScaler()
    scaler_post = StandardScaler()

    # Fit only to the training data
    scaler_num.fit(X_train_num)
    scaler_post.fit(X_train_post)

    StandardScaler(copy=True, with_mean=True, with_std=True)

    # Now apply the transformations to the data:
    X_train_num = scaler_num.transform(X_train_num)
    X_test_num = scaler_num.transform(X_test_num)
    X_train_post = scaler_post.transform(X_train_post)
    X_test_post = scaler_post.transform(X_test_post)


###################################
##         Classify 111          ##
###################################
def method1(case, customer, postcode):
    preprocessing(case, True)

    if customer:
        print("Applying method1 for customer")
        test = [1, 1]
        predictions = [1, 1]
        print("method1 number weekly accuracy: ", accuracy_score(test, predictions))

    if postcode:
        print("Applying method1 for postcode")
        test = [1, 1]
        predictions = [1, 1]
        print("method1 postcode weekly accuracy: ", accuracy_score(test, predictions))


###################################
##         Classify 222          ##
###################################
def method2(case, customer, postcode):
    preprocessing(case, True)

    if customer:
        print("Applying method2 for customer")
        test = [1, 1]
        predictions = [1, 1]
        print("method2 number weekly accuracy: ", accuracy_score(test, predictions))

    if postcode:
        print("Applying method2 for postcode")
        test = [1, 1]
        predictions = [1, 1]
        print("method2 postcode weekly accuracy: ", accuracy_score(test, predictions))


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 4:
        print("Invalid usage: python ./stage1a_series_weekly.py [case] [m1] [m2]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for method arguments")
        exit()

    case = int(sys.argv[1])

    os.chdir("../BlockchainData/Weekly")

    if int(sys.argv[2]):
        print("Classifying stage 1 weekly data with method1")

        print("Creating 2 processes for method1 analysis")
        processes = [
            multiprocessing.Process(target=method1,
                                    name="Process Customer",
                                    args=(case, True, False)),
            multiprocessing.Process(target=method1,
                                    name="Process Postcode",
                                    args=(case, False, True))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    if int(sys.argv[3]):
        print("Classifying stage 1 weekly data with method2")
        method2(case, True, True)
