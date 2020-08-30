#!/bin/bash

python s1_stage.py mlp $1 both ledger_per_customer -y 2 &> result_$1.txt
python s1_stage.py mlp $1 both ledger_per_customer -y 2 &>> result_$1.txt
python s1_stage.py mlp $1 both ledger_per_postcode -y 2 &>> result_$1.txt
python s1_stage.py mlp $1 both ledger_per_postcode -y 2 &>> result_$1.txt

python s1_stage.py cnn $1 both ledger_per_customer -y 2 &>> result_$1.txt
python s1_stage.py cnn $1 both ledger_per_customer -y 2 &>> result_$1.txt
python s1_stage.py cnn $1 both ledger_per_postcode -y 2 &>> result_$1.txt
python s1_stage.py cnn $1 both ledger_per_postcode -y 2 &>> result_$1.txt

python s1_stage.py rdf $1 both ledger_per_customer -y 2 &>> result_$1.txt
python s1_stage.py rdf $1 both ledger_per_customer -y 2 &>> result_$1.txt
python s1_stage.py rdf $1 both ledger_per_postcode -y 2 &>> result_$1.txt
python s1_stage.py rdf $1 both ledger_per_postcode &>> result_$1.txt

python s1_stage.py knn $1 both ledger_per_customer &>> result_$1.txt
python s1_stage.py knn $1 both ledger_per_customer &>> result_$1.txt
python s1_stage.py knn $1 both ledger_per_postcode &>> result_$1.txt
python s1_stage.py knn $1 both ledger_per_postcode &>> result_$1.txt

