#!/bin/sh

# for parallelism
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for i in 0 1 2 3 4 5 6 7 8
do
   echo "Running Experiment $i"
   python scripts/model_mvp_exps.py --exp $i --num_chains 2 &
   sleep 180
done