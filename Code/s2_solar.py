#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
2a) Grid + solar data, informed attacker: classification

This treats all households on a single ledger

Classification methods
    Technique that yields greatest accuracy for each
    Include consumer number, generator, and postcode in for training
    Predictions for i) hourly, ii) daily, iii) weekly, iv) monthly data
        Predictions for i) consumer number, ii) postcode
            i) Split and k-fold training/validation
"""

import os
import re
import pandas as pd


def rearrange_data():
    for postcode in os.listdir():
        for item in os.listdir(postcode):
            if item.endswith(".txt"):
                os.remove(f"{postcode}/{item}")
            else:
                station = re.search(r'[0-9]{6}', item).group(0)
                new_name = f"{postcode}_{station}.csv"
                os.rename(f"{postcode}/{item}", new_name)


def crop_years():
    # Only want 1 July 2010 to 30 June 2013
    for postcode in os.listdir():
        data = pd.read_csv(postcode, header=0)
        print(postcode)

        try:
            data.drop(['Product code', 'Bureau of Meteorology station number'], axis=1, inplace=True)
        except KeyError:
            pass

        data = data[data['Year'].isin([2010, 2011, 2012, 2013])]
        data = data.loc[~((data['Year'] == 2010) & (data['Month'].isin([1, 2, 3, 4, 5, 6]))), :]
        data = data.loc[~((data['Year'] == 2013) & (data['Month'].isin([7, 8, 9, 10, 11, 12]))), :]

        data.to_csv(postcode, index=False)


if __name__ == '__main__':
    os.chdir("../WeatherData/")

    # Run something
