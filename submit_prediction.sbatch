#!/bin/bash
#
#SBATCH --job-name=monkey # Job name for tracking
#SBATCH --partition=dgx2  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=32000            # Memory (mb) to use 
#SBATCH --time=2:00:00      # Job time limit
#
#SBATCH --output=/home/u1910100/logs/pred_%j.out # Standard out from your job
#SBATCH --error=/home/u1910100/logs/pred_%j.err  # Standard error from your job

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
echo "starting prediction script"

# python3.11 prediction_singular.py
python3.11 prediction_multiheadUnet.py
# python3.11 prediction_lymph_mono.py

echo "finished"