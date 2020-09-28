#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from preprocess import preprocessing

mlp_layers = (10, 10, 10)
mlp_iterations = 1000
cnn_filter_size = 128
cnn_batch_size = 128
cnn_epochs = 100
knn_k_customer = 3
knn_k_postcode = 2


def mlp(data_freq, class_type, case, year, solar, net_export, pk, ledger):
    """Perform multilayer perceptron ML classification
    :parameter data_freq --> 'weekly', 'daily', 'hourly', or 'half_hourly' time data resolution.
    :parameter class_type --> 'customer', or 'postcode' are the target for classification.
    :parameter case --> 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger' analysis.
    :parameter year --> 0 (2010-11), 1 (2011-12), 2 (2012-13 or 1st half if hourly), or 3 (2012-13 2nd half).
    :parameter solar --> Boolean if to load data file with solar attribute
    """
    from sklearn.neural_network import MLPClassifier

    print(f"MLP for {case} {data_freq} {class_type} solar {solar}")
    x_train, x_test, y_train, y_test = preprocessing(data_freq, class_type, case, year, solar, net_export, pk, ledger)

    mlp_num = MLPClassifier(hidden_layer_sizes=mlp_layers, max_iter=mlp_iterations)
    mlp_num.fit(x_train, y_train)
    mlp_predictions_num = mlp_num.predict(x_test)

    print(f"MLP {case} {data_freq} {class_type} solar {solar} accuracy: {accuracy_score(y_test, mlp_predictions_num)}")
    # print(classification_report(y_test, mlp_predictions_num))


def cnn(data_freq, class_type, case, year, solar, net_export, pk, ledger):
    """Perform convolutional neural network ML classification
    :parameter data_freq --> 'weekly', 'daily', 'hourly', or 'half_hourly' time data resolution.
    :parameter class_type --> 'customer', or 'postcode' are the target for classification.
    :parameter case --> 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger' analysis.
    :parameter year --> 0 (2010-11), 1 (2011-12), 2 (2012-13 or 1st half if hourly), or 3 (2012-13 2nd half).
    :parameter solar --> Boolean if to load data file with solar attribute
    """
    from keras.models import Sequential
    from keras.layers import Input, Conv1D, Flatten, Dense
    from keras.utils import to_categorical

    print(f"CNN for {case} {data_freq} {class_type} solar {solar}")
    x_train, x_test, y_train, y_test = preprocessing(data_freq, class_type, case, year, solar, net_export, pk, ledger)

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

    model = Sequential()
    model.add(Input(shape=(n_features, 1)))
    model.add(Conv1D(filters=cnn_filter_size, kernel_size=1))
    model.add(Conv1D(filters=cnn_filter_size, kernel_size=1))
    model.add(Flatten())
    model.add(Dense(elements, activation='relu'))
    model.add(Dense(elements, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy'])
    history = model.fit(x_train_cnn, y_train_cnn, batch_size=cnn_batch_size, epochs=cnn_epochs, validation_split=0.2)
    print(f"CNN {case} {data_freq} {class_type} solar {solar} accuracy: {model.evaluate(x_test_cnn, y_test_cnn)[1]}")
    print(f"CNN {case} {data_freq} {class_type} solar {solar} top-5 ac: {model.evaluate(x_test_cnn, y_test_cnn)[2]}")

    # plot loss during training
    # from matplotlib import pyplot
    # pyplot.subplot(211)
    # pyplot.title('Loss')
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # # plot accuracy during training
    # pyplot.subplot(212)
    # pyplot.title('Accuracy')
    # pyplot.plot(history.history['accuracy'], label='train')
    # pyplot.plot(history.history['val_accuracy'], label='test')
    # pyplot.legend()
    # pyplot.show()


def rfc(data_freq, class_type, case, year, solar, net_export, pk, ledger):
    """Perform random forest ML classification
    :parameter data_freq --> 'weekly', 'daily', 'hourly', or 'half_hourly' time data resolution.
    :parameter class_type --> 'customer', or 'postcode' are the target for classification.
    :parameter case --> 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger' analysis.
    :parameter year --> 0 (2010-11), 1 (2011-12), 2 (2012-13 or 1st half if hourly), or 3 (2012-13 2nd half).
    :parameter solar --> Boolean if to load data file with solar attribute
    """
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    import seaborn as sns

    if case == 'aol':
        features = ['Timestamp', 'Type', 'Amount']
    else:
        features = ['PK', 'Timestamp', 'Type', 'Amount']
    if solar:
        features.append('Solar')

    print(f"RFC for {case} {data_freq} {class_type} solar {solar}")
    x_train, x_test, y_train, y_test = preprocessing(data_freq, class_type, case, year, solar, net_export, pk, ledger)

    forest_num = RandomForestClassifier(max_depth=12, random_state=0)
    forest_num.fit(x_train, y_train)
    forest_predictions_num = np.round(forest_num.predict(x_test))

    print(f"RFC {case} {data_freq} {class_type} solar {solar} accuracy: "
          f"{accuracy_score(y_test, forest_predictions_num, normalize=True)}")
    print(classification_report(y_test, forest_predictions_num))
    feature_imp = pd.Series(forest_num.feature_importances_, index=features).sort_values(ascending=False)

    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(f"RFC {data_freq} {class_type}")
    plt.legend()
    plt.savefig(f"/home/andrew/results/{data_freq}_{case}_{class_type}_rfc.png")


def knn(data_freq, class_type, case, year, solar, net_export, pk, ledger):
    """Perform K-nearest neighbours ML classification
    :parameter data_freq --> 'weekly', 'daily', 'hourly', or 'half_hourly' time data resolution.
    :parameter class_type --> 'customer', or 'postcode' are the target for classification.
    :parameter case --> 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger' analysis.
    :parameter year --> 0 (2010-11), 1 (2011-12), 2 (2012-13 or 1st half if hourly), or 3 (2012-13 2nd half).
    :parameter solar --> Boolean if to load data file with solar attribute
    """
    from sklearn.neighbors import KNeighborsClassifier

    k = knn_k_customer if class_type == 'customer' else knn_k_postcode

    print(f"KNN for {case} {data_freq} {class_type} solar {solar}")
    x_train, x_test, y_train, y_test = preprocessing(data_freq, class_type, case, year, solar, net_export, pk, ledger)

    knn_num = KNeighborsClassifier(n_neighbors=k)
    knn_num.fit(x_train, y_train)
    knn_predictions = knn_num.predict(x_test)
    print(f"KNN {case} {data_freq} {class_type} solar {solar} accuracy: ", accuracy_score(y_test, knn_predictions))
    # print(classification_report(y_test, knn_predictions))
