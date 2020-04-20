#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrangle the energy data into a format resembling what would be available publicly on a blockchain ledger.
"""

import os
import sys

import pandas as pd
import numpy as np


years = ['Solar30minData_1Jul10-30Jun11', 'Solar30minData_1Jul11-30Jun12', 'Solar30minData_1Jul12-30Jun13']

os.chdir('../EnergyData/MainData/')
for year in years:
    full_data = pd.read_csv(year + '.csv', header=0)
    # test_data = pd.DataFrame(np.full((len(full_data['Time']), 2), np.nan), columns=['Time', 'device'])
    # test_data['Time'] = full_data['Time']
    test_data = pd.DataFrame([1, 2, 3, 3])
    test_data.to_csv(year + '_blockchain.csv', index=False)
