#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrangle the energy data into a format resembling what would be available publicly on a blockchain ledger.
"""

import os
import sys
import math

import pandas as pd
import numpy as np
import hashlib

#years = ['Solar30minData_1Jul10-30Jun11', 'Solar30minData_1Jul11-30Jun12', 'Solar30minData_1Jul12-30Jun13']
years = ['Tester']
block_size = 10


def create_hash(hash_string):
    sha_signature = hashlib.sha256(hash_string.encode()).hexdigest()[0:8]
    return sha_signature


os.chdir('../EnergyData/MainData/')
for year in years:
    # Load original energy data
    energy_data = pd.read_csv(year + '.csv', header=0)
    energy_data_header = energy_data.columns.values.tolist()

    #################################
    # First, wrangle this idiotic Ausgrid data which used timestamps across columns, into rows using datetime.
    wrangled_cols = ['Customer', 'Capacity', 'Timestamp', 'Type', 'Amount']
    wrangled_list = []
    first_kwh_col = 5

    # Loop on input data, wrangling cols into rows of datetime.
    for index, row in energy_data.iterrows():
        print(index)
        # For each column of times during a day (48 half hour periods)
        for i in range(48): # 0 to 47

            # If amount used/generated for consumer, type, and time period is 0 then skip
            if math.ceil(float(kwh_amount := row[first_kwh_col + i])) <= 0:
                break

            datetime = f"{row['Date']} {energy_data_header[first_kwh_col + i]}"
            eng_type = row['Consumption Category']

            wrangled_list.append([
                row['Customer'],
                row['Generator Capacity'],
                datetime,
                eng_type,
                kwh_amount,
            ])

    wrangled_data = pd.DataFrame(wrangled_list, columns=wrangled_cols)
    wrangled_data.sort_values(by=['Timestamp'], inplace=True)

    # Save wrangled data as csv
    os.chdir('../Blockchained/')
    wrangled_data.to_csv(year + '_wrangled.csv', index=False)

    #################################
    # Second, populate the blockchain
    blockchain_cols = ['Customer', 'Capacity', 'Block', 'Tid', 'P_Tid', 'Timestamp', 'Type', 'Amount', 'PK']
    transaction_list = []
    next_pk = 'PK'
    block_count = 1
    block_transacts = 1
    prev_hash = ''

    # Loop on wrangled data to create blockchain transaction format.
    for index, row in wrangled_data.iterrows():
        print(index)

        # Structure of transaction: Block | Tid | Prev Tid | timestamp | type | amount | PK
        datetime = row['Timestamp']
        eng_type = row['Type']
        kwh_amount = row['Amount']
        curr_hash = create_hash(''.join([datetime, eng_type, str(kwh_amount)]))

        transaction_list.append([
            row['Customer'],
            row['Capacity'],
            block_count,
            curr_hash,
            prev_hash,
            datetime,
            eng_type,
            kwh_amount,
            next_pk
        ])

        # Updates for next loop
        prev_hash = curr_hash
        block_transacts += 1
        if block_transacts > block_size:
            block_count += 1
            block_transacts = 1

    # Single node acts as miner and collects all home transactions.
    blockchain_data = pd.DataFrame(transaction_list, columns=blockchain_cols)
    #blockchain_data.sort_values(by=['Timestamp'], inplace=True)

    # Save resulting blockchain as csv
    blockchain_data.to_csv(year + '_blockchain.csv', index=False)
    os.chdir('../MainData/')
