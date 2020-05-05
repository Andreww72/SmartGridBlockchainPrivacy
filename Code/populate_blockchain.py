#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrangle the energy data into a format resembling what would be available publicly on a blockchain ledger.
Warning this will take while :)
Perhaps I could make it run in parallel
"""

import os
import math

import pandas as pd
import hashlib

block_size = 10
years = ['Solar30minData_1Jul10-30Jun11', 'Solar30minData_1Jul11-30Jun12', 'Solar30minData_1Jul12-30Jun13']
months = pd.date_range(start='2010-07', freq='M', periods=36)


def create_hash(hash_string):
    sha_signature = hashlib.sha256(hash_string.encode()).hexdigest()[0:8]
    return sha_signature


os.chdir('../EnergyData/MainData/OriginalSolarData')
for year in years:
    # Load original energy data
    print(f"Loading {year}.csv")
    energy_data = pd.read_csv(year + '.csv', header=0)

    # Make pandas interpret date column as dates
    energy_data['Date'] = pd.to_datetime(energy_data['Date'], infer_datetime_format=True)

    # Adjust headings for better sorting later
    # Originally times denote the end of period block, change to start of block
    energy_data_header = energy_data.columns.values.tolist()
    add_pos = energy_data_header.index("0:30")
    energy_data_header.insert(add_pos, "0:00")
    energy_data_header.pop()
    energy_data.columns = energy_data_header

    # Split each file into months
    os.chdir('../SplitMonthly')
    print(f"Splitting by month {year}")

    # Groupby key (Date) and freq (Month)
    g = energy_data.groupby(pd.Grouper(key='Date', freq='M'))
    monthly_data = [group for _, group in g]

    # Save CSVs
    for num, month in enumerate(monthly_data):
        name = str(months[num]).split('-')
        month.to_csv(f"{name[0]}-{name[1]}_split.csv", index=False)
        print(f"Split {name[0]}-{name[1]}")

    #################################
    # First, wrangle this idiotic Ausgrid data which used timestamps across columns, into rows using datetime.
    print(f"Wrangling {year}")
    os.chdir('../../Blockchained/')
    wrangled_cols = ['Customer', 'Capacity', 'Timestamp', 'Type', 'Amount']
    wrangled_monthly_data = []
    first_kwh_col = 5

    # Loop on input data, wrangling cols into rows of datetime.
    for num, month in enumerate(monthly_data):
        wrangled_list = []

        for index, row in month.iterrows():
            # For each column of times during a day (48 half hour periods)
            # Combine half hourly into hourly data
            for i in range(24): # 0 to 24

                # If amount used/generated for consumer, type, and time period is 0 then skip
                kwh_amount = row[first_kwh_col + 2*i] + row[first_kwh_col + 2*i+1]
                if math.ceil(kwh_amount) <= 0:
                    break

                datetime = f"{row['Date']} {energy_data_header[first_kwh_col + 2*i]}"
                eng_type = row['Consumption Category']

                wrangled_list.append([
                    row['Customer'],
                    row['Generator Capacity'],
                    datetime,
                    eng_type,
                    kwh_amount,
                ])

        # Sort by datetime. As a bonus it somewhat randomises customers within each timestamp too
        # TODO: Properly randomise the order within a timestamp
        wrangled_dataframe = pd.DataFrame(wrangled_list, columns=wrangled_cols)
        wrangled_dataframe['Timestamp'] = pd.to_datetime(wrangled_dataframe['Timestamp'], infer_datetime_format=True)
        wrangled_dataframe.sort_values(by=['Timestamp'], inplace=True)

        # Save wrangled data as csv
        wrangled_monthly_data.append(wrangled_dataframe)
        name = str(months[num]).split('-')
        wrangled_dataframe.to_csv(f"{name[0]}-{name[1]}_wrangled.csv", index=False)
        print(f"Wrangled {name[0]}-{name[1]}")

    #################################
    # Second, populate the blockchain
    print(f"Populating blockchain {year}")
    blockchain_cols = ['Customer', 'Capacity', 'Block', 'Tid', 'P_Tid', 'Timestamp', 'Type', 'Amount', 'PK']
    next_pk = 'PK'
    block_count = 1
    block_transacts = 1
    prev_hash = ''

    # Loop on wrangled data to create blockchain transaction format.
    for num, month in enumerate(wrangled_monthly_data):
        transaction_list = []

        for index, row in month.iterrows():
            # Structure of transaction: Block | Tid | Prev Tid | timestamp | type | amount | PK
            datetime = row['Timestamp']
            eng_type = row['Type']
            kwh_amount = row['Amount']
            curr_hash = create_hash(''.join([str(datetime), eng_type, str(kwh_amount)]))

            transaction_list.append([
                row['Customer'], # Obviously not on chain, but needed for ML training
                row['Capacity'], # As above comment
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

        # Save resulting month's blockchain as csv
        name = str(months[num]).split('-')
        blockchain_data.to_csv(f"{name[0]}-{name[1]}_blockchain.csv", index=False)
        print(f"Blockchained {name[0]}-{name[1]}")

    os.chdir('../MainData/OriginalSolarData/')
    print(f"Finished {year}")
    break

print("All complete, bet you weren't expecting it to take that long eh :)")
