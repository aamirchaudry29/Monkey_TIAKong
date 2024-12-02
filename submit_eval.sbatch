#!/bin/bash
#
#SBATCH --job-name=monkey # Job name for tracking
#SBATCH --partition=cpu  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10      # Number of CPU threads used by your job
#SBATCH --mem=32000             # Memory (mb) to use 
#SBATCH --time=1:00:00      # Job time limit
#
#SBATCH --output=/home/u1910100/logs/eval_%j.out # Standard out from your job
#SBATCH --error=/home/u1910100/logs/eval_%j.err  # Standard error from your job

## Activate Conda Environmnet ##
echo "init env ... "
conda init
source /home/u1910100/.bashrc
cd ~
source activate /home/u1910100/cloud_workspace/conda/tiatoolbox
echo $PATH
echo "done"

## Run Python Program ##
cd /home/u1910100/cloud_workspace/GitHub/Monkey_TIAKong
echo "starting evaluation script"
python3.11 evaluate_FROC.py
echo "finished"