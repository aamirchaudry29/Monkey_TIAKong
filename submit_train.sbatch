#!/bin/bash
#
#SBATCH --job-name=monkey # Job name for tracking
#SBATCH --partition=dgx2  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=20000             # Memory (mb) to use 
#SBATCH --time=12:00:00      # Job time limit
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
cd /home/u1910100/cloud_workspace/GitHub/Monkey_TIAKong
echo "starting training script"
python3.11 train_detection.py
echo "finished"