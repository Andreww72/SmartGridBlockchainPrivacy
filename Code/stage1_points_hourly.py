#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1a) Grid data only, informed attacker: classification

Use: python ./stage1_points_hourly.py [case] [MLP] [LSTM]
Use a 0 for worst case, 1 for best case for case argument
Use a 1 or 0 indicator for MLP and LSTM arguments

Cases (without obfuscation techniques)
    Worst case: Households use a new PK every transaction, no links between transactions
    TODO Realistic cases: Households change PKs at some interval, those on same PK are linked
    TODO Best case: Household has one PK, all transactions linked

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
import pandas as pd

# Classify MLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Classify LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

case = None


###################################
##         Preprocessing         ##
###################################
def preprocessing():
    print("Preprocessing stage 1 daily data")

    hourly_data = pd.read_csv('0_1a_combined_hourly.csv', header=0)
    # May want to consider removing rows with amount = 0
    # Have manually converted categorical to numeric, may consider coding this, ex:
    # https://www.dataquest.io/blog/sci-kit-learn-tutorial/

    global X_num, Y_num, X_post, Y_post
    X_num = hourly_data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    Y_num = hourly_data['Customer']
    X_post = hourly_data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    Y_post = hourly_data['Postcode']

    global X_train_num, X_test_num, Y_train_num, Y_test_num
    global X_train_post, X_test_post, Y_train_post, Y_test_post
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


###################################
##         Classify MLP          ##
###################################
def mlp():
    print("Applying MLP neural network for customer number")
    mlp_num = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
    mlp_num.fit(X_train_num, Y_train_num)
    mlp_predictions_num = mlp_num.predict(X_test_num)
    print("NN number hourly accuracy: ", accuracy_score(Y_test_num, mlp_predictions_num, normalize=True))

    print("Applying MLP neural network for postcode")
    mlp_post = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
    mlp_post.fit(X_train_post, Y_train_post)
    mlp_predictions_post = mlp_post.predict(X_test_post)
    print("NN postcode hourly accuracy: ", accuracy_score(Y_test_post, mlp_predictions_post, normalize=True))


###################################
##         Classify LSTM         ##
###################################
def lstm():
    # Note this is a single layer LSTM network
    # https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/
    print("Applying MLP neural network for customer number")
    lstm_num = Sequential()
    lstm_num.add(LSTM(256, input_shape=(seq_len, 4)))
    lstm_num.add(Dense(1, activation='sigmoid'))
    lstm_num.summary()

    adam = Adam(lr=0.001)
    chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
    lstm_num.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    lstm_num.fit(X_train_num, Y_num, epochs=200, batch_size=128, callbacks=[chk], validation_data=(validation,validation_target))

    #loading the model and checking accuracy on the test data
    model = load_model('best_model.pkl')

    lstm_predictions_num = model.predict_classes(X_test_num)
    accuracy_score(Y_test_num, lstm_predictions_num)
    print("NN number hourly accuracy: ", accuracy_score(Y_test_num, lstm_predictions_num, normalize=True))

    print("Applying MLP neural network for postcode")
    #print("NN postcode hourly accuracy: ", accuracy_score(Y_test_post, lstm_predictions_post, normalize=True))


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 4:
        print("Invalid usage: python ./stage1_points_hourly.py [case] [MLP] [LSTM]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for MLP and LSTM arguments")
        exit()

    case = sys.argv[1]

    os.chdir("../BlockchainData/Hourly")
    preprocessing()

    if int(sys.argv[2]):
        print("Classifying stage 1 hourly data with MLP neural network")
        mlp()

    if int(sys.argv[3]):
        print("Classifying stage 1 hourly data with LSTM")
        lstm()

# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(Y_test_num, predictions_num_nn))
# print(classification_report(Y_test_num, predictions_num_nn))
# print(confusion_matrix(Y_test_post, predictions_post_nn))
# print(classification_report(Y_test_post, predictions_post_nn))
