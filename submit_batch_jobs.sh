#!/bin/bash

# Number of folds
folds=5

# Loop to submit jobs
for ((i=1; i<=folds; i++))
do
    sbatch submit_train.sbatch $i
    # sbatch submit_prediction.sbatch $i
done
