#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1a) Grid data only, informed attacker: classification

Use: python ./stage1_classify_weekly.py [case] [MLP] [LSTM]
Use a 0 for worst case, 1 for best case for case argument
Use a 1 or 0 indicator for MLP and LSTM arguments

Cases (without obfuscation techniques)
    Worst case: Households use a new PK every transaction, no links between transactions
    TODO Realistic cases: Households change PKs at some interval, those on same PK are linked
    Best case: Household has one PK, all transactions linked

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
import multiprocessing
import pandas as pd

# Classify MLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Classify KNN


case = None


###################################
##         Preprocessing         ##
###################################
def preprocessing(strip_zeros=False):
    print("Preprocessing stage 1 weekly data")

    weekly_data = pd.read_csv('0_1a_combined_weekly.csv', header=0)

    # Convert categorical columns to numeric
    weekly_data['Type'] = weekly_data['Type'].astype('category').cat.codes

    weekly_data['Timestamp'] = pd.to_datetime(weekly_data['Timestamp'], dayfirst=True)
    weekly_data['Timestamp'] = (weekly_data.Timestamp - pd.to_datetime('1970-01-01')).dt.total_seconds()

    if int(case) == 0:
        print("Preprocessing data for worst case")
        # Drop the PK and hash information
        weekly_data.drop(['Hash', 'PHash', 'PK'], axis=1)
        # Structure: Customer | Postcode | Generator | Timestamp | Type | Amount
    elif int(case) == 1:
        print("Preprocessing data for best case")
        # Don't remove anything lol
        # Structure: Customer | Postcode | Generator | Hash | PHash | PK | Timestamp | Type | Amount
    else:
        print("Invalid case selected")
        print("Invalid usage: python ./stage1_classify_weekly.py [case] [MLP] [LSTM]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for MLP and LSTM arguments")

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
##         Classify MLP          ##
###################################
def mlp(customer, postcode):
    if customer:
        print("Applying MLP neural network for customer number")
        mlp_num = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
        mlp_num.fit(X_train_num, Y_train_num)
        mlp_predictions_num = mlp_num.predict(X_test_num)
        print("NN number weekly accuracy: ", accuracy_score(Y_test_num, mlp_predictions_num, normalize=True))

    if postcode:
        print("Applying MLP neural network for postcode")
        mlp_post = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
        mlp_post.fit(X_train_post, Y_train_post)
        mlp_predictions_post = mlp_post.predict(X_test_post)
        print("NN postcode weekly accuracy: ", accuracy_score(Y_test_post, mlp_predictions_post, normalize=True))


###################################
##         Classify LSTM         ##
###################################
def knn():
    preprocessing(False)


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 4:
        print("Invalid usage: python ./stage1_classify_weekly.py [case] [MLP] [LSTM]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for MLP and LSTM arguments")
        exit()

    case = sys.argv[1]

    os.chdir("../BlockchainData/Weekly")

    if int(sys.argv[2]):

        print("Classifying stage 1 weekly data with MLP neural network")
        preprocessing(True)

        print("Creating 2 processes for MLP analysis")
        processes = [
            multiprocessing.Process(target=mlp,
                                    name="Process Customer",
                                    args=(True, False)),
            multiprocessing.Process(target=mlp,
                                    name="Process Postcode",
                                    args=(False, True))]
        for p in processes:
            p.start()

        # Wait for completion
        for p in processes:
            p.join()

    if int(sys.argv[3]):
        print("Classifying stage 1 weekly data with LSTM")
        knn()

# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(Y_test_num, predictions_num_nn))
# print(classification_report(Y_test_num, predictions_num_nn))
# print(confusion_matrix(Y_test_post, predictions_post_nn))
# print(classification_report(Y_test_post, predictions_post_nn))
