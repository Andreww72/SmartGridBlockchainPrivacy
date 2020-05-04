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
cols = ['Block', 'Tid', 'P_Tid', 'Timestamp', 'Type', 'Amount', 'Output', 'PK', 'Sign']
block_size = 10

os.chdir('../EnergyData/MainData/')
for year in years:
    # Load original energy data
    energy_data = pd.read_csv(year + '.csv', header=0)

    transactions_list = []
    block_transactions = 1
    block_count = 1

    # Loop on each entry in energy data (this doesn't quite work cause want blockchain in timestamp order)
    for index, row in energy_data.iterrows():

        transaction = [block_count, block_transactions, 0, 0, 0, 0, 0, 0, 0]
        # Create PK for each transaction
        # Structure of transaction: Tid | Prev Tid | timestamp | type | amount | output | PK | sign
        # Tid is unique transaction identifier, hash of transaction content
        # Prev Tid is identity of previous transaction in the same ledger
        # Timestamp of when transaction packets generated in the database
        # Type is GC, CL, or GG
        # Amount is kWH amount
        # Output is hash of PK device use in next transaction.
        # PK of transaction generator and corresponding signature
        # If data, transaction generator signs hash of data in sign field. Otherwise hash of transaction is signed.

        transactions_list.append(transaction)

        # 10 transactions per block
        block_transactions += 1
        if block_transactions > block_size:
            block_count += 1
            block_transactions = 1

    # Single node acts as miner and collects all home transactions.
    blockchain_data = pd.DataFrame(transactions_list, columns=cols)

    # Save resulting blockchain as csv
    os.chdir('../Blockchained/')
    blockchain_data.to_csv(year + '_blockchain.csv', index=False)
    os.chdir('../MainData/')
