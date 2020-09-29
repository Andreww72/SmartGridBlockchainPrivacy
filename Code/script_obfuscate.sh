#!/bin/bash

# PK varying
python obfuscation.py weekly 2 1
python obfuscation.py weekly 4 1
python obfuscation.py weekly 10 1
python obfuscation.py weekly 20 1
python obfuscation.py weekly 50 1
python obfuscation.py half_hourly 2 1
python obfuscation.py half_hourly 4 1
python obfuscation.py half_hourly 10 1
python obfuscation.py half_hourly 20 1
python obfuscation.py half_hourly 50 1

# Ledger varying
python obfuscation.py weekly 1 2
python obfuscation.py weekly 1 4
python obfuscation.py weekly 1 6
python obfuscation.py weekly 1 10
python obfuscation.py weekly 1 20
python obfuscation.py half_hourly 1 2
python obfuscation.py half_hourly 1 4
python obfuscation.py half_hourly 1 6
python obfuscation.py half_hourly 1 10
python obfuscation.py half_hourly 1 20

# Combined
python obfuscation.py weekly 50 20
python obfuscation.py half_hourly 50 20

