#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is an example setup for project consistency
"""

# Standard libraries
import os
import sys
import math
import random
import multiprocessing

# Third party
import pandas as pd
import numpy as np
import hashlib
from sklearn import datasets, svm

# Local source
import code

# Begin code
today = '16-10-10'

os.chdir('../TCP/' + today + '/')
full_data = pd.read_csv('not_ex_not_dropc.csv', header=0)
test_data = pd.DataFrame(np.full((len(full_data['Time']), 2), np.nan), columns=['Time', 'device'])
test_data['Time'] = full_data['Time']
test_data.to_csv(today + '_test2_ne_nd.csv', index=False)
