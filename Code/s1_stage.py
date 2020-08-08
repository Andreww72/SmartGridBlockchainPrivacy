#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML analysis
Grid data only, informed attacker: classification
Use: python ./s1_stage -h for usage
Cases
    Ledger per customer: Households have one PK and their transactions on a unique ledger
    Ledger per postcode: Households have one PK and their transactions on a ledger by postcode
    One mixed ledger: Households use a new PK every transaction, no links between transactions
Classifiers
    Neural network - MLP classification
    Neural network - CNN classification
    Decision tree - Random forest classification
    KNN classification
Classify
    Include consumer number, generator, and postcode for training set
    Drop those three from the test set
    Predictions for a) weekly, b) daily, c) hourly, & d) half_hourly
        Predictions for i) consumer number, ii) postcode
"""

import os
import argparse
import multiprocessing
from s1_methods import mlp, cnn, rfc, knn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process parameters for analysis to perform')
    parser.add_argument("method", type=str, choices=['mlp', 'cnn', 'rfc', 'knn'],
                        help="Analysis methods of 'mlp', 'cnn', 'rfc', or 'knn'")
    parser.add_argument("data_freq", type=str, choices=['weekly', 'daily', 'hourly', 'half_hourly'],
                        help="Data resolution of 'weekly', 'daily', 'hourly', or 'half_hourly'")
    parser.add_argument("class_type", choices=['customer', 'postcode', 'both'],
                        help="Classification target of 'customer' or 'postcode' or 'both'")
    parser.add_argument("case", choices=['ledger_per_customer', 'ledger_per_postcode', 'one_ledger'],
                        help="Security case of 'ledger_per_customer', 'ledger_per_postcode', or 'one_ledger'")
    parser.add_argument("-y", "--year", type=int, choices=[0, 1, 2, 3],
                        help="Year of data to use if hourly or half_hourly chosen. 0, 1, 2, or 3")

    args = parser.parse_args()
    method = args.method
    data_freq = args.data_freq
    class_type = args.class_type
    case = args.case

    if data_freq == 'hourly' or data_freq == 'half_hourly':
        year = args.year
        if not year:
            print("Provide a year when using hourly or half_hourly resolution")
            exit()
    else:
        year = None

    os.chdir(f"../BlockchainData/{data_freq}")

    if method == 'mlp':
        if class_type == 'both':
            processes = [
                multiprocessing.Process(target=mlp, name="MLP Customer",
                                        args=(data_freq, 'customer', case, year)),
                multiprocessing.Process(target=mlp, name="MLP Postcode",
                                        args=(data_freq, 'postcode', case, year))
            ]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
        else:
            mlp(data_freq, class_type, case, year)

    elif method == 'cnn':
        if class_type == 'both':
            cnn(data_freq, 'customer', case, year)
            cnn(data_freq, 'postcode', case, year)
        else:
            cnn(data_freq, class_type, case, year)

    elif method == 'rfc':
        if class_type == 'both':
            processes = [
                multiprocessing.Process(target=rfc, name="Forest Customer",
                                        args=(data_freq, 'customer', case, year)),
                multiprocessing.Process(target=rfc, name="Forest Postcode",
                                        args=(data_freq, 'postcode', case, year))
            ]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
        else:
            rfc(data_freq, class_type, case, year)

    elif method == 'knn':
        if class_type == 'both':
            processes = [
                multiprocessing.Process(target=knn, name="KNN Customer",
                                        args=(data_freq, 'customer', case, year)),
                multiprocessing.Process(target=knn, name="Process Postcode",
                                        args=(data_freq, 'postcode', case, year))
            ]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
        else:
            knn(data_freq, class_type, case, year)

    else:
        print("Pick a valid analysis method")
