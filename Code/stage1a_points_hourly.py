#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
1a) Grid data only, informed attacker: classification

Use: python ./stage1a_points_hourly.py [case] [MLP] [KNN] [KMS]
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
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


###################################
##         Preprocessing         ##
###################################
def preprocessing(case=1, strip_zeros=False):
    print("Preprocessing stage 1 hourly data")

    hourly_data = pd.read_csv('0_1a_combined_hourly.csv', header=0)

    # Convert categorical columns to numeric
    hourly_data['Type'] = hourly_data['Type'].astype('category').cat.codes

    hourly_data['Timestamp'] = pd.to_datetime(hourly_data['Timestamp'], dayfirst=True)
    hourly_data['Timestamp'] = (hourly_data.Timestamp - pd.to_datetime('1970-01-01')).dt.total_seconds()

    if case == 0:
        print("Preprocessing data for worst case")
        # Drop the PK and hash information
        hourly_data.drop(['Hash', 'PHash', 'PK'], axis=1)
        # Structure: Customer | Postcode | Generator | Timestamp | Type | Amount
    elif case == 1:
        print("Preprocessing data for best case")
        # Don't remove anything lol
        # Structure: Customer | Postcode | Generator | Hash | PHash | PK | Timestamp | Type | Amount
    else:
        print("Invalid case selected")
        print("Invalid usage: python ./stage1a_points_hourly.py [case] [MLP] [KNN] [KMS]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for MLP and LSTM arguments")

    if strip_zeros:
        hourly_data = hourly_data[hourly_data['Amount'] != 0]

    x_num = hourly_data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    y_num = hourly_data['Customer']
    x_post = hourly_data.drop(['Customer', 'Postcode', 'Generator'], axis=1)
    y_post = hourly_data['Postcode']

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
        print("Applying MLP neural network for customer number")
        mlp_num = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
        mlp_num.fit(X_train_num, Y_train_num)
        mlp_predictions_num = mlp_num.predict(X_test_num)
        print("NN number hourly accuracy: ", accuracy_score(Y_test_num, mlp_predictions_num))

    if postcode:
        print("Applying MLP neural network for postcode")
        mlp_post = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
        mlp_post.fit(X_train_post, Y_train_post)
        mlp_predictions_post = mlp_post.predict(X_test_post)
        print("NN postcode hourly accuracy: ", accuracy_score(Y_test_post, mlp_predictions_post, normalize=True))


###################################
##         Classify KNN          ##
###################################
def knn(case, customer, postcode):
    ks = [1, 3, 5, 10, 20, 50]
    results_num = []
    results_post = []

    print(f"KNN with k values: {ks}")
    preprocessing(case, False)

    if customer:
        print("Applying KNN for customer")
        for k in ks:
            knn_num = KNeighborsClassifier(n_neighbors=k)
            knn_num.fit(X_train_num, Y_train_num)
            knn_predictions_num = knn_num.predict(X_test_num)
            result = accuracy_score(Y_test_num, knn_predictions_num)
            results_num.append(result)
        best_k = ks[results_num.index(max(results_num))]
        print(f"Best KNN number hourly accuracy (k={best_k}:", max(results_num))

    if postcode:
        print("Applying KNN for postcode")
        for k in ks:
            knn_post = KNeighborsClassifier(n_neighbors=k)
            knn_post.fit(X_train_post, Y_train_post)
            knn_predictions_post = knn_post.predict(X_test_post)
            result = accuracy_score(Y_test_post, knn_predictions_post)
            results_post.append(result)
        best_k = ks[results_post.index(max(results_post))]
        print(f"KNN postcode hourly accuracy (k={best_k}:", max(results_post))


def kms(case):
    preprocessing(case, False)

    clusters = 100
    reduced_data = PCA(n_components=2).fit_transform(X_train_num)
    kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a red .
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='.', s=169, linewidths=2,
                color='r', zorder=10)
    plt.title('K-means clustering (PCA-reduced data)\n'
              'Centroids are marked with red dot')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 5:
        print("Invalid usage: python ./stage1a_points_hourly.py [case] [MLP] [KNN] [KMS]")
        print("Use a 0 for worst case, 1 for best case for case argument")
        print("Use a 1 or 0 indicator for MLP and LSTM arguments")
        exit()

    case = int(sys.argv[1])

    os.chdir("../BlockchainData/Hourly")

    if int(sys.argv[2]):
        print("Classifying stage 1 hourly data with MLP neural network")

        print("Creating 2 processes for MLP analysis")
        processes = [
            multiprocessing.Process(target=mlp,
                                    name="Process Customer",
                                    args=(case, True, False)),
            multiprocessing.Process(target=mlp,
                                    name="Process Postcode",
                                    args=(case, False, True))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    if int(sys.argv[3]):
        print("Classifying stage 1 hourly data with KNN")
        knn(case, True, True)

    if int(sys.argv[4]):
        print("Clustering stage 1 hourly data with KMeans")
        kms(case)

# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(Y_test_num, predictions_num_nn))
# print(classification_report(Y_test_num, predictions_num_nn))
# print(confusion_matrix(Y_test_post, predictions_post_nn))
# print(classification_report(Y_test_post, predictions_post_nn))
