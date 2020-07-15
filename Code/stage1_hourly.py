#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1a) Grid data only, informed attacker: classification

Use: python ./stage1_hourly.py [case] [MLP] [CNN] [FOR] [KNN]
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
import multiprocessing
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import seaborn as sns


###################################
##         Preprocessing         ##
###################################
def preprocessing(case=1, strip_zeros=False, year=0):
    print("Preprocessing stage 1 hourly data")
    hourly_data = []

    if year == 0:
        hourly_data = pd.read_csv('0_1a_combined_hourly_2010-11.csv', header=0)
    elif year == 1:
        hourly_data = pd.read_csv('0_1a_combined_hourly_2011-12.csv', header=0)
    elif year == 2:
        hourly_data = pd.read_csv('0_1a_combined_hourly_2012-13.csv', header=0)

    # Convert categorical columns to numeric
    hourly_data['Type'] = hourly_data['Type'].astype('category').cat.codes

    hourly_data['Timestamp'] = pd.to_datetime(hourly_data['Timestamp'], dayfirst=True)
    hourly_data['Timestamp'] = (hourly_data.Timestamp - pd.to_datetime('1970-01-01')).dt.total_seconds()

    if case == 0:
        print("Preprocessing data for worst case")
        # Drop the PK and hash information
        hourly_data.drop(['Hash', 'PHash', 'PK'], axis=1, inplace=True)
        # Structure: Customer | Postcode | Generator | Timestamp | Type | Amount
    elif case == 1:
        print("Preprocessing data for best case")
        # Don't remove anything lol
        # Structure: Customer | Postcode | Generator | Hash | PHash | PK | Timestamp | Type | Amount
    else:
        print("Invalid case selected")
        print("Invalid usage: python ./stage1_hourly.py [case] [year] [MLP] [CNN] [FOR] [KNN]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for method arguments")

    if strip_zeros:
        hourly_data = hourly_data[hourly_data['Amount'] != 0]

    x_num = hourly_data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    y_num = hourly_data['Customer']
    x_post = hourly_data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    y_post = hourly_data['Postcode']

    global x_train_num, x_test_num, y_train_num, y_test_num
    global x_train_post, x_test_post, y_train_post, y_test_post
    x_train_num, x_test_num, y_train_num, y_test_num = train_test_split(x_num, y_num)
    x_train_post, x_test_post, y_train_post, y_test_post = train_test_split(x_post, y_post)

    # Preprocess
    scaler_num = StandardScaler()
    scaler_post = StandardScaler()

    # Fit only to the training data
    scaler_num.fit(x_train_num)
    scaler_post.fit(x_train_post)

    StandardScaler(copy=True, with_mean=True, with_std=True)

    # Now apply the transformations to the data:
    x_train_num = scaler_num.transform(x_train_num)
    x_test_num = scaler_num.transform(x_test_num)
    x_train_post = scaler_post.transform(x_train_post)
    x_test_post = scaler_post.transform(x_test_post)


###################################
##         Classify MLP          ##
###################################
def mlp(case, year, customer, postcode):
    preprocessing(case, True, year)

    if customer:
        print("Applying MLP for customer")
        mlp_num = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
        mlp_num.fit(x_train_num, y_train_num)
        mlp_predictions_num = mlp_num.predict(x_test_num)
        print("MLP customer hourly accuracy information")
        print("MLP number hourly accuracy: ", accuracy_score(y_test_num, mlp_predictions_num))
        print(classification_report(y_test_num, mlp_predictions_num))

    if postcode:
        print("Applying MLP for postcode")
        mlp_post = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
        mlp_post.fit(x_train_post, y_train_post)
        mlp_predictions_post = mlp_post.predict(x_test_post)
        print("MLP postcode hourly accuracy information")
        print("MLP postcode hourly accuracy: ", accuracy_score(y_test_post, mlp_predictions_post))
        print(classification_report(y_test_post, mlp_predictions_post))


###################################
##         Classify CNN          ##
###################################
def cnn(case, year, customer, postcode):
    preprocessing(case, True, year)
    filter_size = 128
    batch_size = 128

    if customer:
        print("Applying CNN for customer")
        x_train_num_cnn = np.expand_dims(x_train_num, axis=2)
        x_test_num_cnn = np.expand_dims(x_test_num, axis=2)
        n_timesteps, n_features = x_train_num_cnn.shape[0], x_train_num_cnn.shape[1]
        y_train_num_cnn = to_categorical(y_train_num)
        y_test_num_cnn = to_categorical(y_test_num)

        inp = Input(shape=(n_features, 1))
        t = Conv1D(filters=filter_size, kernel_size=1)(inp)
        t = Conv1D(filters=filter_size, kernel_size=1)(t)
        t = Flatten()(t)
        t = Dense(301, activation='relu')(t)
        t = Dense(301, activation='softmax')(t)
        model = Model(inp, t)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train_num_cnn, y_train_num_cnn, batch_size=batch_size, epochs=10, validation_split=0.2)
        print(model.evaluate(x_test_num_cnn, y_test_num_cnn)[1])

    if postcode:
        print("Applying CNN for postcode")
        x_train_post_cnn = np.expand_dims(x_train_post, axis=2)
        x_test_post_cnn = np.expand_dims(x_test_post, axis=2)
        n_timesteps, n_features = x_train_post_cnn.shape[0], x_train_post_cnn.shape[1]

        # Subtract 2000 from every postcode to reduce to_categorical output size
        y_train_post_cnn = to_categorical(np.subtract(y_train_post, [2000] * len(y_train_post)))
        y_test_post_cnn = to_categorical(np.subtract(y_test_post, [2000] * len(y_test_post)))

        inp = Input(shape=(n_features, 1))
        t = Conv1D(filters=filter_size, kernel_size=1)(inp)
        t = Conv1D(filters=filter_size, kernel_size=1)(t)
        t = Flatten()(t)
        t = Dense(331, activation='relu')(t)
        t = Dense(331, activation='softmax')(t)
        model = Model(inp, t)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train_post_cnn, y_train_post_cnn, batch_size=batch_size, epochs=10, validation_split=0.2)
        print(model.evaluate(x_test_post_cnn, y_test_post_cnn)[1])


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
        forest_num.fit(x_train_num, y_train_num)
        forest_predictions_num = np.round(forest_num.predict(x_test_num))

        print("Forest customer hourly accuracy information")
        print(accuracy_score(y_test_num, forest_predictions_num, normalize=True))
        print(classification_report(y_test_num, forest_predictions_num))
        feature_imp = pd.Series(forest_num.feature_importances_, index=features).sort_values(ascending=False)

        # Creating a bar plot
        sns.barplot(x=feature_imp, y=feature_imp.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("RF Hourly Customer")
        plt.legend()
        if case == 0:
            plt.savefig('C:\\results\\hourly_worst_customer_rf.png')
        elif case == 1:
            plt.savefig('C:\\results\\hourly_best_customer_rf.png')

    if postcode:
        print("Applying forest for postcode")
        forest_post = RandomForestClassifier(n_jobs=1, max_depth=6, random_state=0)
        forest_post.fit(x_train_post, y_train_post)
        forest_predictions_post = np.round(forest_post.predict(x_test_post))

        print("Forest postcode hourly accuracy information")
        print(accuracy_score(y_test_post, forest_predictions_post, normalize=True))
        print(classification_report(y_test_post, forest_predictions_post))
        feature_imp = pd.Series(forest_post.feature_importances_, index=features).sort_values(ascending=False)

        # Creating a bar plot
        sns.barplot(x=feature_imp, y=feature_imp.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("RF Hourly Postcode")
        plt.legend()
        if case == 0:
            plt.savefig('C:\\results\\hourly_worst_postcode_rf.png')
        elif case == 1:
            plt.savefig('C:\\results\\hourly_best_postcode_rf.png')


###################################
##         Classify KNN          ##
###################################
def knn(case, year, customer, postcode):
    preprocessing(case, False, year)

    if customer:
        k = 1
        print("Applying KNN for customer")
        knn_num = KNeighborsClassifier(n_neighbors=k)
        knn_num.fit(x_train_num, y_train_num)
        knn_predictions_num = knn_num.predict(x_test_num)
        print("KNN customer hourly accuracy information")
        print("KNN customer hourly accuracy: ", accuracy_score(y_test_num, knn_predictions_num))
        print(classification_report(y_test_num, knn_predictions_num))

    if postcode:
        k = 50
        print("Applying KNN for postcode")
        knn_post = KNeighborsClassifier(n_neighbors=k)
        knn_post.fit(x_train_post, y_train_post)
        knn_predictions_post = knn_post.predict(x_test_post)
        print("KNN postcode hourly accuracy information")
        print("KNN postcode hourly accuracy: ", accuracy_score(y_test_post, knn_predictions_post))
        print(classification_report(y_test_post, knn_predictions_post))


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 7:
        print("Invalid usage: python ./stage1_hourly.py [case] [year] [MLP] [CNN] [FOR] [KNN]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for method arguments")
        exit()

    case = int(sys.argv[1])
    year = int(sys.argv[2])

    os.chdir("../BlockchainData/Hourly")

    if int(sys.argv[3]):
        # Classifying stage 1 hourly data with MLP
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
        # Classifying stage 1 hourly data with CNN
        cnn(case, year, True, True)

    if int(sys.argv[5]):
        # Classifying stage 1 hourly data with random forest
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

    if int(sys.argv[6]):
        # Classifying stage 1 hourly data with KNN
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
