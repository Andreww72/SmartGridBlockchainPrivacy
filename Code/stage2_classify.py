#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
2a) Grid + solar data, informed attacker: classification

Classification methods
    Technique that yields greatest accuracy for each
    Include consumer number, generator, and postcode in for training
    Predictions for i) hourly, ii) daily, iii) weekly, iv) monthly data
        Predictions for i) consumer number, ii) postcode
            i) Split and k-fold training/validation
"""

import os
import json

import pandas as pd
import numpy as np

os.chdir("../BlockchainData/")
