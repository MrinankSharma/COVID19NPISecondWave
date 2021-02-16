#!/bin/sh

for i in 0 1 2 3 4 5 6 7 8
do
   echo "Running Experiment $i"
   python scripts/model_mvp_exps.py --exp $i --num_chains 2 &
   sleep 5
done