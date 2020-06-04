#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1a) Grid data only, informed attacker: classification

Use: python ./stage1a_points_weekly.py [case] [MLP] [FOR] [KNN]
Use a 0 for worst case, 1 for best case for case argument
Use a 1 or 0 indicator for method arguments

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
import multiprocessing
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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
        print("Invalid usage: python ./stage1a_points_weekly.py [case] [MLP] [FOR] [KNN]")
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
##         Classify MLP          ##
###################################
def mlp(case, customer, postcode):
    preprocessing(case, True)

    if customer:
        print("Applying MLP for customer")
        mlp_num = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
        mlp_num.fit(X_train_num, Y_train_num)
        mlp_predictions_num = mlp_num.predict(X_test_num)
        print("MLP number weekly accuracy: ", accuracy_score(Y_test_num, mlp_predictions_num))

    if postcode:
        print("Applying MLP for postcode")
        mlp_post = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
        mlp_post.fit(X_train_post, Y_train_post)
        mlp_predictions_post = mlp_post.predict(X_test_post)
        print("MLP postcode weekly accuracy: ", accuracy_score(Y_test_post, mlp_predictions_post, normalize=True))


###################################
##        Classify Forest        ##
###################################
def forest(case, customer, postcode):
    preprocessing(case, True)

    if customer:
        print("Applying forest for customer")
        forest_num = RandomForestRegressor(n_estimators=20, random_state=0)
        forest_num.fit(X_train_num, Y_train_num)
        forest_predictions_num = np.round(forest_num.predict(X_test_num))

        print("Forest customer weekly accuracy information")
        #print(confusion_matrix(Y_test_num, forest_predictions_num))
        #print(classification_report(Y_test_num, forest_predictions_num))
        print(accuracy_score(Y_test_num, forest_predictions_num, normalize=True))

    if postcode:
        print("Applying forest for postcode")
        forest_post = RandomForestRegressor(n_estimators=20, random_state=0)
        forest_post.fit(X_train_post, Y_train_post)
        forest_predictions_post = np.round(forest_post.predict(X_test_post))
        print(forest_predictions_post)
        print("Forest postcode weekly accuracy information")
        #print(confusion_matrix(Y_test_post, forest_predictions_post))
        #print(classification_report(Y_test_post, forest_predictions_post))
        print(accuracy_score(Y_test_post, forest_predictions_post, normalize=True))


###################################
##         Classify KNN          ##
###################################
def knn(case, customer, postcode):
    ks = [1, 3, 5, 10, 20, 50]
    results_num = []
    results_post = []

    print(f"KNN with k values: {ks}")
    preprocessing(case, True)

    if customer:
        print("Applying KNN for customer")
        for k in ks:
            knn_num = KNeighborsClassifier(n_neighbors=k)
            knn_num.fit(X_train_num, Y_train_num)
            knn_predictions_num = knn_num.predict(X_test_num)
            result = accuracy_score(Y_test_num, knn_predictions_num)
            results_num.append(result)
        best_k = ks[results_num.index(max(results_num))]
        print(f"Best KNN number weekly accuracy (k={best_k}:", max(results_num))

    if postcode:
        print("Applying KNN for postcode")
        for k in ks:
            knn_post = KNeighborsClassifier(n_neighbors=k)
            knn_post.fit(X_train_post, Y_train_post)
            knn_predictions_post = knn_post.predict(X_test_post)
            result = accuracy_score(Y_test_post, knn_predictions_post)
            results_post.append(result)
        best_k = ks[results_post.index(max(results_post))]
        print(f"KNN postcode weekly accuracy (k={best_k}:", max(results_post))


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 5:
        print("Invalid usage: python ./stage1a_points_weekly.py [case] [MLP] [FOR] [KNN]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for method arguments")
        exit()

    case = int(sys.argv[1])

    os.chdir("../BlockchainData/Weekly")

    if int(sys.argv[2]):
        print("Classifying stage 1 weekly data with MLP")
        processes = [
            multiprocessing.Process(target=mlp,
                                    name="MLP Customer",
                                    args=(case, True, False)),
            multiprocessing.Process(target=mlp,
                                    name="MLP Postcode",
                                    args=(case, False, True))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    if int(sys.argv[3]):
        print("Clustering stage 1 weekly data with random forest")
        processes = [
            multiprocessing.Process(target=forest,
                                    name="Forest Customer",
                                    args=(case, True, False)),
            multiprocessing.Process(target=forest,
                                    name="Forest Postcode",
                                    args=(case, False, True))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    if int(sys.argv[4]):
        print("Classifying stage 1 weekly data with KNN")
        processes = [
            multiprocessing.Process(target=knn,
                                    name="KNN Customer",
                                    args=(case, True, False)),
            multiprocessing.Process(target=knn,
                                    name="Process Postcode",
                                    args=(case, False, True))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
