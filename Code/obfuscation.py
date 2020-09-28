#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
Grid + solar data, informed attacker: classification
Use: python ./obfuscation -h for usage
"""

import os
import sys
import secrets
import hashlib
import numpy as np
import pandas as pd

num_customers = 300


def create_hash(s):
    """Create a random hash from a string
    :parameter s --> String input to hash
    :return hash string truncated to a reasonable length
    """
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 12


def multi_pks(data_freq, pk_count):
    if "half_hourly" in data_freq:
        data = pd.read_csv(f"0_customer_{data_freq}_2012-13a_solar.csv", header=0)
    elif "hourly" in data_freq:
        data = pd.read_csv(f"0_customer_{data_freq}_2012-13_solar.csv", header=0)
    else:
        data = pd.read_csv(f"0_customer_{data_freq}_solar.csv", header=0)

    # Generate PKs for each customer
    pks = {}
    for i in range(num_customers):
        for j in range(pk_count):
            pks[f"{i+1}_{j}"] = create_hash(f"{i+1}{j}")

    # Create list of random selections
    pk_rands = []
    for i in range(len(data.index)):
        pk_rands.append(secrets.randbelow(pk_count))

    # Apply random selection list to choose a PK per transaction with a map operation
    map_series = data['Customer'].astype(str) + "_" + pd.Series(pk_rands).astype(str)
    data['PK'] = map_series.map(pks)
    data['Ledger'] = data['PK']

    # Save final ledger
    data.to_csv(f"0_pk_{pk_count}_{data_freq}.csv", index=False)


def fixed_ledger_groups():
    pass


def net_export():
    pass


if __name__ == '__main__':

    # Check usage
    if not len(sys.argv) == 3:
        print("Use: python ./obfuscation.py [data_freq] [pk_count]")
        print("Use a 1 or 0 indicator for each argument")
        exit()

    data_freq = str(sys.argv[1])
    pk_count = int(sys.argv[2])

    print(f"Multi PK for {data_freq} and {i} PKs")
    os.chdir(f"../BlockchainData/{data_freq}")
    multi_pks(data_freq, i)
