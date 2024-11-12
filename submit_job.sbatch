#!/bin/bash
#
#SBATCH --job-name=monkey # Job name for tracking
#SBATCH --partition=gpu  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=30000             # Memory (mb) to use 
#SBATCH --time=48:00:00      # Job time limit
#
#SBATCH --output=/home/u1910100/logs/joboutput_%j.out # Standard out from your job
#SBATCH --error=/home/u1910100/logs/joboutput_%j.err  # Standard error from your job

## Activate Conda Environmnet ##
echo "init env ... "
conda init
source /home/u1910100/.bashrc
cd ~
source activate /home/u1910100/cloud_workspace/conda/tiatoolbox
echo $PATH
echo "done"

## Run Python Program ##
echo "starting python program"
python3.11 /home/u1910100/cloud_workspace/GitHub/Monkey_TIAKong/train_detection.py
echo "finished"