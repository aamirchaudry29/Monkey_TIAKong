#!/bin/bash
#
#SBATCH --job-name=monkey_train # Job name for tracking
#SBATCH --partition=dgxa100  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=32000             # Memory (mb) to use 
#SBATCH --time=48:00:00      # Job time limit
#
#SBATCH --output=/home/u1910100/logs/train_%j.out # Standard out from your job
#SBATCH --error=/home/u1910100/logs/train_%j.err  # Standard error from your job

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
nvidia-smi

# python3.11 train_multiclass_segmentation.py 
python3.11 train_multiclass_detection.py --fold=$1

echo "finished"