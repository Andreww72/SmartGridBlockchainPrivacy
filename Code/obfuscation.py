#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML analysis
Grid + solar data, informed attacker: classification
Use: python ./obfuscation -h for usage
"""

import os
import sys
import glob
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


def multi_pks_ledgers(data_freq, pk_count, per_ledger):
    os.chdir(f"../BlockchainData/{data_freq}")

    if "half_hourly" in data_freq:
        data = pd.read_csv(f"0_customer_{data_freq}_2012-13a_solar.csv", header=0)
    elif "hourly" in data_freq:
        data = pd.read_csv(f"0_customer_{data_freq}_2012-13_solar.csv", header=0)
    else:
        data = pd.read_csv(f"0_customer_{data_freq}_solar.csv", header=0)

    # Create PK and ledger mappings
    pks = {}
    ledgers = {}
    group, count = 0, 0
    for i in range(num_customers):
        for j in range(pk_count):
            pks[f"{i+1}_{j}"] = create_hash(f"pk{i+1}{j}")
            ledgers[f"{i+1}_{j}"] = create_hash(f"ledger{group}")
            count += 1
            if count == per_ledger:
                group += 1
                count = 0

    # Create list of random PK selections
    pk_rands = []
    for i in range(len(data.index)):
        pk_rands.append(secrets.randbelow(pk_count))

    # Apply random selection list to choose a PK per transaction with a map operation
    map_series = data['Customer'].astype(str) + "_" + pd.Series(pk_rands).astype(str)
    data['PK'] = map_series.map(pks)
    data['Ledger'] = map_series.map(ledgers)

    # Save final ledgers
    data.to_csv(f"0_pk_{pk_count}_ledger_{per_ledger}_{data_freq}.csv", index=False)


if __name__ == '__main__':

    # Check usage
    if not len(sys.argv) == 4:
        print("Use: python ./obfuscation.py [data_freq] [pk_count] [ledger_count]")
        exit()

    data_freq = str(sys.argv[1])
    pk_count = int(sys.argv[2])
    per_ledger = int(sys.argv[3])

    if not 300 % per_ledger == 0:
        print("Use a ledger count (if unsure use 1) factor of 300 (e.g. 1, 2, 3, 4, 5, 6, 10, 12...")
        exit()

    print(f"{data_freq} data with {pk_count} PKs / customer and {per_ledger} PKs / ledger")
    multi_pks_ledgers(data_freq, pk_count, per_ledger)
