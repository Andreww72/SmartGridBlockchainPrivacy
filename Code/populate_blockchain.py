#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrangle the energy data into a format resembling what would be available publicly on a blockchain ledger.
Warning this will take while despite running in parallel :)

Use: python ./populate_blockchain.py [hourly] [daily/weekly]
Use a true or false indicator for each argument
"""

import os
import sys
import csv
import glob
import hashlib
import multiprocessing

import pandas as pd
import numpy as np

num_customers = 300
years = ['Jul10-Jun11', 'Jul11-Jun12', 'Jul12-Jun13']
datasets = ['EnergyData_1Jul10-30Jun11.csv',
            'EnergyData_1Jul11-30Jun12.csv',
            'EnergyData_1Jul12-30Jun13.csv']


def create_hash(hash_string):
    sha_signature = hashlib.sha256(hash_string.encode()).hexdigest()[0:8]
    return sha_signature


def wrangle_blockchain_data(set_num, dataset, incl_zeroes=True, daily=False):
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
            if incl_zeroes and (eng_type == 'GC' and prev_type == 'GG' or (not prev_type and not eng_type == 'CL')):
                if daily:
                    datetime = f"{row[date_col]}"
                    wrangled_ledger.append([datetime, 'CL', 0])
                else:
                    for i in range(24):  # 0 to 24
                        datetime = f"{row[date_col]} {energy_data_header[first_kwh_col + 2 * i]}"
                        wrangled_ledger.append([datetime, 'CL', 0])
            prev_type = row[type_col]

            # For each column of times during a day (48 half hour periods)
            kwh_amount = 0
            if daily:
                # Combine half hourly into daily data
                datetime = f"{row[date_col]}"
                for j in range(48):
                    kwh_amount += round(row[first_kwh_col + j], 3)

                if incl_zeroes or (not incl_zeroes and kwh_amount > 0):
                    wrangled_ledger.append([datetime, eng_type, kwh_amount])
            else:
                # Combine half hourly into hourly data
                for i in range(24):  # 0 to 24
                    kwh_amount = round(row[first_kwh_col + 2 * i] + row[first_kwh_col + 2 * i + 1], 3)
                    datetime = f"{row[date_col]} {energy_data_header[first_kwh_col + 2 * i]}"

                    if incl_zeroes or (not incl_zeroes and kwh_amount > 0):
                        wrangled_ledger.append([datetime, eng_type, kwh_amount])

        wrangled_ledgers.append(wrangled_ledger)
        print(f"Process {pid} wrangled {num+1}")

    #################################
    # Second, populate the blockchain
    if daily:
        os.chdir('../BlockchainData/Daily')
    else:
        os.chdir('../BlockchainData/Hourly')
    # Loop on wrangled data to create blockchain transaction format.
    for num, ledger in enumerate(wrangled_ledgers):
        blockchain_ledger = []
        prev_hash = "Genisis"
        pk = create_hash(f"{num}")

        for row in ledger:
            # Structure of wrangled data: Timestamp | Type | Amount
            datetime = row[0]
            eng_type = row[1]
            kwh_amount = row[2]
            curr_hash = create_hash(f"{datetime} {eng_type} {kwh_amount}")

            # Structure of new transaction: Hash | PHash | PK | Timestamp | Type | Amount
            blockchain_ledger.append([
                curr_hash,
                prev_hash,
                pk,
                datetime,
                eng_type,
                kwh_amount,
            ])
            prev_hash = curr_hash

        # Save blockchain!
        header = ['Hash', 'PHash', 'PK', 'Timestamp', 'Type', 'Amount']
        with open(f'{years[set_num]}_{num+1}_blockchain.csv', 'w', newline='') as csv_out:
            writer = csv.writer(csv_out, delimiter=',')
            writer.writerow(header)
            writer.writerows(blockchain_ledger)
        print(f"Process {pid} blockchain {num+1}")

    print(f"Process {pid} complete")


#################################
# Third, combine the separately processed years into one ledger per household.
def combine_years():
    for num in range(0, num_customers):
        print(f"Combining customer {num+1} ledgers")

        # Find the ledger from each of the three data sets per customer
        same_ledger_files = [j for j in glob.glob(f'*_{num+1}_blockchain.csv')]

        if same_ledger_files:
            combine_sets = []
            prev_ending_hash = ""

            for i, ledger_file in enumerate(same_ledger_files):
                df = pd.read_csv(ledger_file)
                if prev_ending_hash:
                    df['PHash'].iloc[0] = prev_ending_hash
                prev_ending_hash = df['Hash'].iloc[-1]
                combine_sets.append(df)
                os.remove(ledger_file)

            combined_year = pd.concat(combine_sets)
            combined_year.to_csv(f"{num+1}_blockchain.csv", index=False)


#################################
# From daily blockchain output, create weekly blockchain.
def create_weekly():
    week_and_type_splits = []

    for num in range(num_customers):
        print(f"Creating weekly {num+1} ledger")
        df = pd.read_csv(f"{num+1}_blockchain.csv", header=0)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)

        # Group by week
        dg = df.groupby(["PK", pd.Grouper(key='Timestamp', freq='W-MON'), "Type"]).sum().reset_index()

        # Recreate new hash and prev hash columns
        dg.insert(loc=0, column='PHash', value=np.nan)
        dg.insert(loc=0, column='Hash', value=np.nan)
        weekly_data = dg.values.tolist()
        prev_hash = "Genisis"

        for row in weekly_data:
            # Structure: Hash | PHash | PK | Timestamp | Type | Amount
            curr_hash = create_hash(f"{row[3]} {row[4]} {row[5]}")
            row[0] = curr_hash
            row[1] = prev_hash
            prev_hash = curr_hash

        week_and_type_splits.append(weekly_data)

    # Save blockchains!
    os.chdir("../Weekly")
    print("Saving weekly files")

    for num, week in enumerate(week_and_type_splits):
        header = ['Hash', 'PHash', 'PK', 'Timestamp', 'Type', 'Amount']
        with open(f'{num+1}_blockchain.csv', 'w', newline='') as csv_out:
            writer = csv.writer(csv_out, delimiter=',')
            writer.writerow(header)
            writer.writerows(week)


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 3:
        print("Invalid usage: python ./populate_blockchain [hourly] [daily/weekly]")
        print("Use a true or false indicator for each argument")
        exit()

    if not os.path.exists('../BlockchainData/Hourly'):
        os.makedirs('../BlockchainData/Hourly')
    if not os.path.exists('../BlockchainData/Daily'):
        os.makedirs('../BlockchainData/Daily')
    if not os.path.exists('../BlockchainData/Weekly'):
        os.makedirs('../BlockchainData/Weekly')

    # Parallel process setup
    # Python's Global Interpreter Lock means threads cannot run in parallel, but processes can!
    os.chdir('../OriginalEnergyData/')

    if int(sys.argv[1]):
        # Hourly data!
        print(f"Creating {len(datasets)} processes to create hourly blockchains")
        processes = []
        for inum, d in enumerate(datasets):
            p = multiprocessing.Process(target=wrangle_blockchain_data, name=f"Process {inum}", args=(inum, d, True, False))
            processes.append(p)
            p.start()

        # Wait for completion
        for p in processes:
            p.join()

        # Combine the files of the same customer number
        os.chdir('../BlockchainData/Hourly/')
        combine_years()

    if int(sys.argv[2]) == 1:
        if int(sys.argv[1]) == 1:
            os.chdir('../../OriginalEnergyData/')

        # Daily data! #
        print(f"Creating {len(datasets)} processes to create daily blockchains")
        processes = []
        for inum, d in enumerate(datasets):
            p = multiprocessing.Process(target=wrangle_blockchain_data, name=f"Process {inum}", args=(inum, d, True, True))
            processes.append(p)
            p.start()

        # Wait for completion
        for p in processes:
            p.join()

        # Combine the files of the same customer number
        os.chdir('../BlockchainData/Daily/')
        combine_years()

        # Take daily and create weekly data
        print(f"Creating weekly blockchains")
        create_weekly()

    print(f"Speedy boi now :)")
