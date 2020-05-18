#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1a) Grid data only, informed attacker: classification

This treats all households on a single ledger

Neural network classification
    Include consumer number, generator, and postcode in for training
    Predictions for i) hourly, ii) daily, iii) weekly, iv) monthly data
        Predictions for i) consumer number, ii) postcode
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report, confusion_matrix

os.chdir("../BlockchainData/Daily")

###################################
##    Classify on daily data     ##
###################################
print("Classifying stage 1 daily data")

daily_data = pd.read_csv('0_1a_combined_daily_allnumeric.csv', header=0)
# May want to consider removing rows with amount = 0
# Have manually converted categorical to numeric, may consider coding this, ex:
# https://www.dataquest.io/blog/sci-kit-learn-tutorial/

X_num = daily_data.drop('Customer', axis=1)
Y_num = daily_data['Customer']
X_post = daily_data.drop('Postcode', axis=1)
Y_post = daily_data['Postcode']

X_train_num, X_test_num, Y_train_num, Y_test_num = train_test_split(X_num, Y_num)
X_train_post, X_test_post, Y_train_post, Y_test_post = train_test_split(X_post, Y_post)

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

# Neural network
print("Applying neural network for customer number")
mlp_num = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
mlp_num.fit(X_train_num, Y_train_num)
predictions_num_nn = mlp_num.predict(X_test_num)

print("Applying neural network for postcode")
mlp_post = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
mlp_post.fit(X_train_post, Y_train_post)
predictions_post_nn = mlp_post.predict(X_test_post)

print("NN number daily accuracy: ", accuracy_score(Y_test_num, predictions_num_nn, normalize=True))
print("NN postcode daily accuracy: ", accuracy_score(Y_test_post, predictions_post_nn, normalize=True))
# print(confusion_matrix(Y_test_num, predictions_num_nn))
# print(classification_report(Y_test_num, predictions_num_nn))
# print(confusion_matrix(Y_test_post, predictions_post_nn))
# print(classification_report(Y_test_post, predictions_post_nn))
