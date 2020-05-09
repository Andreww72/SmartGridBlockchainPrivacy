#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrangle the energy data into a format resembling what would be available publicly on a blockchain ledger.
Warning this will take while despite running in parallel :)
"""

import os
import math
import csv
import multiprocessing

import pandas as pd
import hashlib


block_size = 10
years = ['Jul10-Jun11', 'Jul11-Jun12', 'Jul12-Jun13']
datasets = ['Solar30minData_1Jul10-30Jun11.csv',
            'Solar30minData_1Jul11-30Jun12.csv',
            'Solar30minData_1Jul12-30Jun13.csv']


def create_hash(hash_string):
    sha_signature = hashlib.sha256(hash_string.encode()).hexdigest()[0:8]
    return sha_signature


def wrangle_blockchain_data(set_num, dataset):
    # Load original energy data
    print(f"Process {os.getpid()} loading {dataset}")
    pid = os.getpid()
    energy_data = pd.read_csv(dataset, header=0)

    # Make pandas interpret date column as dates
    #energy_data['Date'] = pd.to_datetime(energy_data['Date'], format="%d-%m-%Y")

    # Adjust headings for better sorting later
    # Originally times denote the end of period block, change to start of block
    energy_data_header = energy_data.columns.values.tolist()
    add_pos = energy_data_header.index("0:30")
    energy_data_header.insert(add_pos, "0:00")
    energy_data_header.pop()
    energy_data.columns = energy_data_header

    # Split by customer, convert back to basic lists for speed
    g = energy_data.groupby(pd.Grouper(key='Customer'))
    houses = [group.values.tolist() for _, group in g]

    #################################
    # First, wrangle this idiotic Ausgrid data which used timestamps across columns, into rows using datetime.
    wrangled_ledgers = []
    type_col, date_col, first_kwh_col = 3, 4, 5

    # Loop on input data, wrangling cols into rows of datetime.
    for num, house in enumerate(houses):
        wrangled_ledger = []

        for row in house:
            # For each column of times during a day (48 half hour periods)
            # Combine half hourly into hourly data
            for i in range(24): # 0 to 24

                # If amount used/generated for consumer, type, and time period is 0 then skip
                kwh_amount = round(row[first_kwh_col + 2 * i] + row[first_kwh_col + 2 * i + 1], 3)
                if math.ceil(kwh_amount) <= 0:
                    break
                datetime = f"{row[date_col]} {energy_data_header[first_kwh_col + 2*i]}"
                eng_type = row[type_col]

                wrangled_ledger.append([
                    datetime,
                    eng_type,
                    kwh_amount
                ])

        wrangled_ledgers.append(wrangled_ledger)
        print(f"Process {pid} wrangled {num}")

    #################################
    # Second, populate the blockchain
    os.chdir('../../Blockchained/')

    # Loop on wrangled data to create blockchain transaction format.
    for num, ledger in enumerate(wrangled_ledgers):
        blockchain_ledger = []
        prev_hash = ''
        block_count = 1
        block_transacts = 1

        for row in ledger:
            # Structure of transaction: Block | Tid | Prev Tid | timestamp | type | amount | PK
            datetime = row[0]
            eng_type = row[1]
            kwh_amount = row[2]
            curr_hash = create_hash(''.join([str(datetime), eng_type, str(kwh_amount)]))

            blockchain_ledger.append([
                block_count,
                curr_hash,
                prev_hash,
                datetime,
                eng_type,
                kwh_amount,
            ])

            # Updates for next loop
            prev_hash = curr_hash
            block_transacts += 1
            if block_transacts > block_size:
                block_count += 1
                block_transacts = 1

        # Save blockchain!
        header = ['Block', 'Hash', 'P_Hash', 'Timestamp', 'Type', 'Amount']
        with open(f'{years[set_num]}_{num}_blockchain.csv', 'w', newline='') as csv_out:
            writer = csv.writer(csv_out, delimiter=',')
            writer.writerow(header)
            writer.writerows(blockchain_ledger)
        print(f"Process {pid} blockchain {num}")

    print(f"Process {pid} complete")
    os.chdir('../MainData/OriginalSolarData')


if __name__ == '__main__':
    os.chdir('../EnergyData/MainData/OriginalSolarData')

    # # Run sequentially if computer can't handle parallel code below
    # for inum, d in enumerate(datasets):
    #     wrangle_blockchain_data(inum, d)

    # Python's Global Interpreter Lock means threads cannot run in parallel, but processes can!
    print(f"Creating {len(datasets)} processes to handle the {len(datasets)} datasets")
    processes = []
    for inum, d in enumerate(datasets):
        p = multiprocessing.Process(target=wrangle_blockchain_data, name=f"Process {inum}", args=(inum, d,))
        processes.append(p)
        p.start()

    # Wait for completion
    for p in processes:
        p.join()

    print(f"Speedy boi now :)")
