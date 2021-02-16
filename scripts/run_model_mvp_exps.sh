#!/bin/sh

for i in {0..8}
do
   echo "Running Experiment $i"
   python scripts/model_mvp_exps.py --exp $i --num_chains 2 &
   sleep 5
done