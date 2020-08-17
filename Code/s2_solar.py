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
    pass


if __name__ == '__main__':
    os.chdir("../WeatherData/")
