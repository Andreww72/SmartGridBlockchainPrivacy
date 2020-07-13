#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1a) Grid data only, informed attacker: classification

Use: python ./stage1_half_hourly.py [case] [MLP] [FOR] [KNN]
Use a 0 for worst case, 1 for best case for case argument
Use a 1 or 0 indicator for method arguments

Cases
    Worst case: Households use a new PK every transaction, no links between transactions
    Obfuscation cases: Households change PKs at some interval, those on same PK are linked
    Best case: Household has one PK, all transactions linked

Classifiers
    Neural network - MLP classification
    Decision tree - Random forest classification
    KNN classification

Classify
    Include consumer number, generator, and postcode for training set
    Drop those three from the test set
    Predictions for a) half-hourly, b) hourly, c) daily, & d) weekly
        Predictions for i) consumer number, ii) postcode
"""

import os
import sys
import random
import multiprocessing
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


###################################
##         Preprocessing         ##
###################################
def preprocessing(case=1, strip_zeros=False, year=0):
    print("Preprocessing stage 1 half hourly data")
    half_hourly_data = []

    if year == 0:
        half_hourly_data = pd.read_csv('0_1a_combined_half_hourly_2010-11.csv', header=0)
    elif year == 1:
        half_hourly_data = pd.read_csv('0_1a_combined_half_hourly_2011-12.csv', header=0)
    elif year == 2:
        half_hourly_data = pd.read_csv('0_1a_combined_half_hourly_2012-13a.csv', header=0)
    elif year == 3:
        half_hourly_data = pd.read_csv('0_1a_combined_half_hourly_2012-13b.csv', header=0)

    # Convert categorical columns to numeric
    half_hourly_data['Type'] = half_hourly_data['Type'].astype('category').cat.codes

    half_hourly_data['Timestamp'] = pd.to_datetime(half_hourly_data['Timestamp'], dayfirst=True)
    half_hourly_data['Timestamp'] = (half_hourly_data.Timestamp - pd.to_datetime('1970-01-01')).dt.total_seconds()

    if case == 0:
        print("Preprocessing data for worst case")
        # Drop the PK and hash information
        half_hourly_data.drop(['Hash', 'PHash', 'PK'], axis=1, inplace=True)
        # Structure: Customer | Postcode | Generator | Timestamp | Type | Amount
    elif case == 1:
        print("Preprocessing data for best case")
        # Don't remove anything lol
        # Structure: Customer | Postcode | Generator | Hash | PHash | PK | Timestamp | Type | Amount
    else:
        print("Invalid case selected")
        print("Invalid usage: python ./stage1_half_hourly.py [case] [year] [MLP] [FOR] [KNN]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for method arguments")

    if strip_zeros:
        half_hourly_data = half_hourly_data[half_hourly_data['Amount'] != 0]

    x_num = half_hourly_data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    y_num = half_hourly_data['Customer']
    x_post = half_hourly_data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    y_post = half_hourly_data['Postcode']

    global X_train_num, X_test_num, Y_train_num, Y_test_num
    global X_train_post, X_test_post, Y_train_post, Y_test_post
    X_train_num, X_test_num, Y_train_num, Y_test_num = train_test_split(x_num, y_num)
    X_train_post, X_test_post, Y_train_post, Y_test_post = train_test_split(x_post, y_post)

    # Make test set PKs differ from training set so random forest can't cheat
    if case == 1:
        X_test_num.PK = X_test_num.PK + random.randint(0, 10000000)

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
def mlp(case, year, customer, postcode):
    preprocessing(case, True, year)

    if customer:
        print("Applying MLP for customer")
        mlp_num = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
        mlp_num.fit(X_train_num, Y_train_num)
        mlp_predictions_num = mlp_num.predict(X_test_num)
        print("MLP customer half hourly accuracy information")
        print("MLP number half hourly accuracy: ", accuracy_score(Y_test_num, mlp_predictions_num))
        print(classification_report(Y_test_num, mlp_predictions_num))

    if postcode:
        print("Applying MLP for postcode")
        mlp_post = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
        mlp_post.fit(X_train_post, Y_train_post)
        mlp_predictions_post = mlp_post.predict(X_test_post)
        print("MLP postcode half hourly accuracy information")
        print("MLP postcode half hourly accuracy: ", accuracy_score(Y_test_post, mlp_predictions_post))
        print(classification_report(Y_test_post, mlp_predictions_post))


###################################
##        Classify Forest        ##
###################################
def forest(case, year, customer, postcode):
    preprocessing(case, True, year)
    features = []
    if case == 0:
        features = ['Timestamp', 'Type', 'Amount']
    elif case == 1:
        features = ['Hash', 'PHash', 'PK', 'Timestamp', 'Type', 'Amount']

    if customer:
        print("Applying forest for customer")
        forest_num = RandomForestClassifier(n_jobs=1, max_depth=6, random_state=0)
        forest_num.fit(X_train_num, Y_train_num)
        forest_predictions_num = np.round(forest_num.predict(X_test_num))

        print("Forest customer half hourly accuracy information")
        print(accuracy_score(Y_test_num, forest_predictions_num, normalize=True))
        print(classification_report(Y_test_num, forest_predictions_num))
        feature_imp = pd.Series(forest_num.feature_importances_, index=features).sort_values(ascending=False)

        # Creating a bar plot
        sns.barplot(x=feature_imp, y=feature_imp.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("RF Half Hourly Customer")
        plt.legend()
        if case == 0:
            plt.savefig('C:\\results\\half_hourly_worst_customer_rf.png')
        elif case == 1:
            plt.savefig('C:\\results\\half_hourly_best_customer_rf.png')

    if postcode:
        print("Applying forest for postcode")
        forest_post = RandomForestClassifier(n_jobs=1, max_depth=6, random_state=0)
        forest_post.fit(X_train_post, Y_train_post)
        forest_predictions_post = np.round(forest_post.predict(X_test_post))

        print("Forest postcode half hourly accuracy information")
        print(accuracy_score(Y_test_post, forest_predictions_post, normalize=True))
        print(classification_report(Y_test_post, forest_predictions_post))
        feature_imp = pd.Series(forest_post.feature_importances_, index=features).sort_values(ascending=False)

        # Creating a bar plot
        sns.barplot(x=feature_imp, y=feature_imp.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("RF Half Hourly Postcode")
        plt.legend()
        if case == 0:
            plt.savefig('C:\\results\\half_hourly_worst_postcode_rf.png')
        elif case == 1:
            plt.savefig('C:\\results\\half_hourly_best_postcode_rf.png')


###################################
##         Classify KNN          ##
###################################
def knn(case, year, customer, postcode):
    preprocessing(case, False, year)

    if customer:
        k = 1
        print("Applying KNN for customer")
        knn_num = KNeighborsClassifier(n_neighbors=k)
        knn_num.fit(X_train_num, Y_train_num)
        knn_predictions_num = knn_num.predict(X_test_num)
        print("KNN customer half hourly accuracy information")
        print("KNN customer half hourly accuracy: ", accuracy_score(Y_test_num, knn_predictions_num))
        print(classification_report(Y_test_num, knn_predictions_num))

    if postcode:
        k = 50
        print("Applying KNN for postcode")
        knn_post = KNeighborsClassifier(n_neighbors=k)
        knn_post.fit(X_train_post, Y_train_post)
        knn_predictions_post = knn_post.predict(X_test_post)
        print("KNN postcode half hourly accuracy information")
        print("KNN postcode half hourly accuracy: ", accuracy_score(Y_test_post, knn_predictions_post))
        print(classification_report(Y_test_post, knn_predictions_post))


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 6:
        print("Invalid usage: python ./stage1_half_hourly.py [case] [year] [MLP] [FOR] [KNN]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for method arguments")
        exit()

    case = int(sys.argv[1])
    year = int(sys.argv[2])

    os.chdir("../BlockchainData/HalfHourly")

    if int(sys.argv[3]):
        # Classifying stage 1 half hourly data with MLP
        processes = [
            multiprocessing.Process(target=mlp,
                                    name="Process Customer",
                                    args=(case, year, True, False)),
            multiprocessing.Process(target=mlp,
                                    name="Process Postcode",
                                    args=(case, year, False, True))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    if int(sys.argv[4]):
        # Classifying stage 1 half hourly data with random forest")
        processes = [
            multiprocessing.Process(target=forest,
                                    name="Forest Customer",
                                    args=(case, year, True, False)),
            multiprocessing.Process(target=forest,
                                    name="Forest Postcode",
                                    args=(case, year, False, True))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    if int(sys.argv[5]):
        # Classifying stage 1 daily data with KNN
        processes = [
            multiprocessing.Process(target=knn,
                                    name="KNN Customer",
                                    args=(case, year, True, False)),
            multiprocessing.Process(target=knn,
                                    name="Process Postcode",
                                    args=(case, year, False, True))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
