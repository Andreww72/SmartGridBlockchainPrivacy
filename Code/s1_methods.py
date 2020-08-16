#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from s1_prep import preprocessing

mlp_layers = (10, 10, 10)
mlp_iterations = 500
cnn_filter_size = 128
cnn_batch_size = 128
cnn_epochs = 50
knn_k_customer = 3
knn_k_postcode = 2


def mlp(data_freq, class_type, case, year):
    """Perform multilayer perceptron ML classification
    :parameter data_freq --> 'weekly', 'daily', 'hourly', or 'half_hourly' time data resolution.
    :parameter class_type --> 'customer', or 'postcode' are the target for classification.
    :parameter case --> 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger' analysis.
    :parameter year --> 0 (2010-11), 1 (2011-12), 2 (2012-13 or 1st half if hourly), or 3 (2012-13 2nd half).
    """
    from sklearn.neural_network import MLPClassifier

    print(f"MLP for {case} {data_freq} {class_type}")
    x_train, x_test, y_train, y_test = preprocessing(data_freq, class_type, case, year)

    mlp_num = MLPClassifier(hidden_layer_sizes=mlp_layers, max_iter=mlp_iterations)
    mlp_num.fit(x_train, y_train)
    mlp_predictions_num = mlp_num.predict(x_test)

    print(f"MLP {case} {data_freq} {class_type} accuracy: {accuracy_score(y_test, mlp_predictions_num)}")
    # print(classification_report(y_test, mlp_predictions_num))


def cnn(data_freq, class_type, case, year):
    """Perform convolutional neural network ML classification
    :parameter data_freq --> 'weekly', 'daily', 'hourly', or 'half_hourly' time data resolution.
    :parameter class_type --> 'customer', or 'postcode' are the target for classification.
    :parameter case --> 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger' analysis.
    :parameter year --> 0 (2010-11), 1 (2011-12), 2 (2012-13 or 1st half if hourly), or 3 (2012-13 2nd half).
    """
    from keras.models import Model
    from keras.layers import Input, Conv1D, Flatten, Dense
    from keras.utils import to_categorical

    print(f"CNN for {case} {data_freq} {class_type}")
    x_train, x_test, y_train, y_test = preprocessing(data_freq, class_type, case, year)

    x_train_cnn = np.expand_dims(x_train, axis=2)
    x_test_cnn = np.expand_dims(x_test, axis=2)
    n_timesteps, n_features = x_train_cnn.shape[0], x_train_cnn.shape[1]

    if class_type == 'customer':
        elements = 301
        y_train_cnn = to_categorical(y_train)
        y_test_cnn = to_categorical(y_test)
    else:
        elements = 331
        y_train_cnn = to_categorical(np.subtract(y_train, [2000] * len(y_train)))
        y_test_cnn = to_categorical(np.subtract(y_test, [2000] * len(y_test)))

    inp = Input(shape=(n_features, 1))
    t = Conv1D(filters=cnn_filter_size, kernel_size=1)(inp)
    t = Conv1D(filters=cnn_filter_size, kernel_size=1)(t)
    t = Flatten()(t)
    t = Dense(elements, activation='relu')(t)
    t = Dense(elements, activation='softmax')(t)
    model = Model(inp, t)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy'])
    model.fit(x_train_cnn, y_train_cnn, batch_size=cnn_batch_size, epochs=cnn_epochs, validation_split=0.2)
    print(f"CNN {case} {data_freq} {class_type} accuracy: {model.evaluate(x_test_cnn, y_test_cnn)[1]}")
    print(f"CNN {case} {data_freq} {class_type} top-5 ac: {model.evaluate(x_test_cnn, y_test_cnn)[2]}")


def rfc(data_freq, class_type, case, year):
    """Perform random forest ML classification
    :parameter data_freq --> 'weekly', 'daily', 'hourly', or 'half_hourly' time data resolution.
    :parameter class_type --> 'customer', or 'postcode' are the target for classification.
    :parameter case --> 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger' analysis.
    :parameter year --> 0 (2010-11), 1 (2011-12), 2 (2012-13 or 1st half if hourly), or 3 (2012-13 2nd half).
    """
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    import seaborn as sns

    if case == 'one_ledger':
        features = ['Timestamp', 'Type', 'Amount']
    else:
        features = ['Ledger', 'PK', 'Timestamp', 'Type', 'Amount']

    print(f"RFC for {case} {data_freq} {class_type}")
    x_train, x_test, y_train, y_test = preprocessing(data_freq, class_type, case, year)

    forest_num = RandomForestClassifier(n_jobs=1, max_depth=8, random_state=0)
    forest_num.fit(x_train, y_train)
    forest_predictions_num = np.round(forest_num.predict(x_test))

    print(f"RFC {case} {data_freq} {class_type} accuracy: "
          f"{accuracy_score(y_test, forest_predictions_num, normalize=True)}")
    # print(classification_report(y_test, forest_predictions_num))
    feature_imp = pd.Series(forest_num.feature_importances_, index=features).sort_values(ascending=False)

    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(f"RFC {data_freq} {class_type}")
    plt.legend()
    plt.savefig(f"/home/andrew/results/{data_freq}_{case}_{class_type}_rfc.png")


def knn(data_freq, class_type, case, year):
    """Perform K-nearest neighbours ML classification
    :parameter data_freq --> 'weekly', 'daily', 'hourly', or 'half_hourly' time data resolution.
    :parameter class_type --> 'customer', or 'postcode' are the target for classification.
    :parameter case --> 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger' analysis.
    :parameter year --> 0 (2010-11), 1 (2011-12), 2 (2012-13 or 1st half if hourly), or 3 (2012-13 2nd half).
    """
    from sklearn.neighbors import KNeighborsClassifier

    k = knn_k_customer if class_type == 'customer' else knn_k_postcode

    print(f"KNN for {case} {data_freq} {class_type}")
    x_train, x_test, y_train, y_test = preprocessing(data_freq, class_type, case, year)

    knn_num = KNeighborsClassifier(n_neighbors=k)
    knn_num.fit(x_train, y_train)
    knn_predictions = knn_num.predict(x_test)
    print(f"KNN {case} {data_freq} {class_type} accuracy: ", accuracy_score(y_test, knn_predictions))
    # print(classification_report(y_test, knn_predictions))
