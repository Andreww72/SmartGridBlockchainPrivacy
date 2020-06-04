#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1a) Grid data only, informed attacker: classification

Use: python ./stage1_points_weekly.py [case] [MLP] [FOR] [KNN] [LSTM]
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
    LSTM classification
    Cointegration analysis

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
from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


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
        weekly_data.drop(['PK'], axis=1, inplace=True)
        # Structure: Customer | Postcode | Generator | Timestamp | Type | Amount
    elif case == 1:
        print("Preprocessing data for best case")
        # Don't remove anything lol
        # Structure: Customer | Postcode | Generator | Hash | PHash | PK | Timestamp | Type | Amount
    else:
        print("Invalid case selected")
        print("Invalid usage: python ./stage1_points_weekly.py [case] [MLP] [FOR] [KNN] [LSTM]")
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
    features = []

    if case == 0:
        features = ['Timestamp', 'Type', 'Amount']
    elif case == 1:
        features = ['Hash', 'PHash', 'PK', 'Timestamp', 'Type', 'Amount']

    if customer:
        print("Applying forest for customer")
        forest_num = RandomForestRegressor(n_estimators=20, random_state=0)
        forest_num.fit(X_train_num, Y_train_num)
        forest_predictions_num = np.round(forest_num.predict(X_test_num))

        print("Forest customer weekly accuracy information")
        print(classification_report(Y_test_num, forest_predictions_num))
        print(accuracy_score(Y_test_num, forest_predictions_num, normalize=True))
        print(forest_num.feature_importances_)
        feature_imp = pd.Series(forest_num.feature_importances_, index=features).sort_values(ascending=False)

        # Creating a bar plot
        sns.barplot(x=feature_imp, y=feature_imp.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("RF Weekly Customer")
        plt.legend()
        plt.show()

    if postcode:
        print("Applying forest for postcode")
        forest_post = RandomForestRegressor(n_estimators=20, random_state=0)
        forest_post.fit(X_train_post, Y_train_post)
        forest_predictions_post = np.round(forest_post.predict(X_test_post))

        print("Forest postcode weekly accuracy information")
        print(classification_report(Y_test_post, forest_predictions_post))
        print(accuracy_score(Y_test_post, forest_predictions_post, normalize=True))
        feature_imp = pd.Series(forest_post.feature_importances_, index=features).sort_values(ascending=False)

        # Creating a bar plot
        sns.barplot(x=feature_imp, y=feature_imp.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("RF Weekly Postcode")
        plt.legend()
        plt.show()


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
        print(f"Best KNN postcode weekly accuracy (k={best_k}:", max(results_post))


###################################
##         Classify LSTM         ##
###################################
def lstm(case, customer, postcode):
    # LSTM neural network
    preprocessing(case, True)

    if customer:
        print("Applying LSTM for customer")

        X_reshaped = X_train_num.reshape(-1, X_train_num.shape[0], X_train_num.shape[1])
        Y_reshaped = Y_train_num.reshape(-1, Y_train_num.shape[0], Y_train_num.shape[1])

        model = Sequential()
        model.add(LSTM(100, input_shape=(X_train_num.shape[0], X_train_num.shape[1]), return_sequences=True))
        model.add(LSTM(5, input_shape=(X_train_num.shape[0], X_train_num.shape[1]), return_sequences=True))

        # model = Sequential()
        # model.add(SpatialDropout1D(0.2))
        # model.add(LSTM(100, dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
        # model.add(Dense(13, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X_reshaped, Y_reshaped, epochs=3, batch_size=64, validation_split=0.1,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        accr = model.evaluate(X_test_num, Y_test_num)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    # if postcode:
    #     print("Applying LSTM for postcode")
    #     lstm_post = Sequential()
    #     lstm_post.add(LSTM(units=50, input_shape=(X_train_post.shape[1], 1, 1)))
    #     lstm_post.add(LSTM(units=50))
    #     lstm_post.add(Dense(1))
    #     lstm_post.compile(loss='mean_squared_error', optimizer='adadelta')
    #     lstm_post.fit(X_train_post, Y_train_post, epochs=3, batch_size=1, verbose=2)
    #     lstm_predictions_post = lstm_post.predict(X_test_post)
    #     print(accuracy_score(Y_test_post, lstm_predictions_post, normalize=True))


###################################
##         Cointegration         ##
###################################
def coint(case, customer, postcode):
    import statsmodels.tsa.stattools as ts

    data1 = web.DataReader('FB', data_source='yahoo',start='4/4/2015', end='4/4/2016')
    data2 = web.DataReader('AAPL', data_source='yahoo',start='4/4/2015', end='4/4/2016')

    data1['key'] = data1.index
    data2['key'] = data2.index
    result = pd.merge(data1, data2, on='key')

    x1 = result['Close_x']
    y1 = result['Close_y']
    coin_result = ts.coint(x1, y1)
    print(coin_result)


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 6:
        print("Invalid usage: python ./stage1_points_weekly.py [case] [MLP] [FOR] [KNN] [LSTM]")
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

    if int(sys.argv[5]):
        print("Classifying stage 1 weekly data with LSTM")
        processes = [
            multiprocessing.Process(target=lstm,
                                    name="KNN Customer",
                                    args=(case, True, False)),
            multiprocessing.Process(target=lstm,
                                    name="Process Postcode",
                                    args=(case, False, True))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
