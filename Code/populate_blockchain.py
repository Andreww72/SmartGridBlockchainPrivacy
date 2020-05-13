#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrangle the energy data into a format resembling what would be available publicly on a blockchain ledger.
Warning this will take while despite running in parallel :)
"""

import os
import math
import csv
import glob
import multiprocessing

import pandas as pd


years = ['Jul10-Jun11', 'Jul11-Jun12', 'Jul12-Jun13']
datasets = ['EnergyData_1Jul10-30Jun11.csv',
            'EnergyData_1Jul11-30Jun12.csv',
            'EnergyData_1Jul12-30Jun13.csv']


def wrangle_blockchain_data(set_num, dataset):
    # Load original energy data
    print(f"Process {os.getpid()} loading {dataset}")
    pid = os.getpid()
    energy_data = pd.read_csv(dataset, header=0)

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
        prev_type = ''

        for row in house:
            eng_type = row[type_col]

            # If house has no solar production data, add blanks to create same length data sets
            if eng_type == 'GC' and prev_type == 'GG' or (not prev_type and not eng_type == 'CL'):
                for i in range(24):  # 0 to 24
                    datetime = f"{row[date_col]} {energy_data_header[first_kwh_col + 2 * i]}"
                    wrangled_ledger.append([
                        datetime,
                        'CL',
                        0
                    ])
            prev_type = row[type_col]

            # For each column of times during a day (48 half hour periods)
            # Combine half hourly into hourly data
            for i in range(24): # 0 to 24

                # If amount used/generated for consumer, type, and time period is 0 then skip
                kwh_amount = round(row[first_kwh_col + 2 * i] + row[first_kwh_col + 2 * i + 1], 3)
                datetime = f"{row[date_col]} {energy_data_header[first_kwh_col + 2*i]}"

                wrangled_ledger.append([
                    datetime,
                    eng_type,
                    kwh_amount
                ])

        wrangled_ledgers.append(wrangled_ledger)
        print(f"Process {pid} wrangled {num+1}")

    #################################
    # Second, populate the blockchain
    os.chdir('../BlockchainData/')

    # Loop on wrangled data to create blockchain transaction format.
    for num, ledger in enumerate(wrangled_ledgers):
        blockchain_ledger = []

        for row in ledger:
            # Structure of transaction: Block | Tid | Prev Tid | timestamp | type | amount | PK
            datetime = row[0]
            eng_type = row[1]
            kwh_amount = row[2]

            blockchain_ledger.append([
                datetime,
                eng_type,
                kwh_amount,
            ])

        # Save blockchain!
        header = ['Timestamp', 'Type', 'Amount']
        with open(f'{years[set_num]}_{num+1}_blockchain.csv', 'w', newline='') as csv_out:
            writer = csv.writer(csv_out, delimiter=',')
            writer.writerow(header)
            writer.writerows(blockchain_ledger)
        print(f"Process {pid} blockchain {num+1}")

    print(f"Process {pid} complete")


#################################
# Third, combine the separately processed years into one ledger per household.
def combine_years():
    num_customers = 300
    for num in range(0, num_customers):
        print(f"Combining customer {num+1} ledgers")

        # Find the ledger from each of the three data sets per customer
        same_ledger_files = [j for j in glob.glob(f'*_{num+1}_blockchain.csv')]

        if same_ledger_files:
            # Open output files
            with open(f"{num+1}_blockchain.csv", "w", newline='') as combined_out:
                csv_write_file = csv.writer(combined_out, delimiter=',')

                # Start with first file to get header write
                with open(same_ledger_files[0], mode='rt') as read_file:
                    csv_read_file = csv.reader(read_file, delimiter=',')
                    csv_write_file.writerows(csv_read_file)
                os.remove(same_ledger_files[0])

                # Loop on remaining files to append to first
                for i in range(1, len(same_ledger_files)):
                    with open(same_ledger_files[i], mode='rt') as read_file:
                        csv_read_file = csv.reader(read_file, delimiter=',')
                        next(csv_read_file) # ignore header
                        csv_write_file.writerows(csv_read_file)

                    # Remove original files as now combined
                    os.remove(same_ledger_files[i])


if __name__ == '__main__':
    os.chdir('../OriginalEnergyData/')

    # # Run sequentially if computer can't handle parallel code. Uncomment next two lines, comment parallel setup.
    # for inum, d in enumerate(datasets):
    #     wrangle_blockchain_data(inum, d)

    # Parallel process setup
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

    # Combine the files of the same customer number
    os.chdir('../BlockchainData/')
    combine_years()

    print(f"Speedy boi now :)")
