#!/bin/bash
#
#SBATCH --job-name=Monkey_train # Job name for tracking
#SBATCH --partition=gpu  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=03:00:00      # Job time limit
#
#SBATCH --output=/home/u1910100/joboutput_%j.out # Standard out from your job
#SBATCH --error=/home/u1910100/joboutput_%j.err  # Standard error from your job

## Initialisation ##
conda init
cd ~
source ~/.bashrc
conda activate /home/u1910100/cloud_workspace/conda/tiatoolbox

## Execute your program(s) ##
python3 /home/u1910100/cloud_workspace/GitHub/Monkey_TIAKong/train_example.py