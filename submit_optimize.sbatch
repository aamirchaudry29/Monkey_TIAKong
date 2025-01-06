#!/bin/bash
#
#SBATCH --job-name=process # Job name for tracking
#SBATCH --partition=cpu  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      # Number of CPU threads used by your job
#SBATCH --mem=64000            # Memory (mb) to use 
#SBATCH --time=2:00:00      # Job time limit
#
#SBATCH --output=/home/u1910100/logs/opti_%j.out # Standard out from your job
#SBATCH --error=/home/u1910100/logs/opti_%j.err  # Standard error from your job

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
echo "starting script"

# python3.11 optimization/run_prediction.py
# python3.11 optimization/opt_post_process.py
# python3.11 evaluate_FROC.py
python3.11 process_nuclick_masks.py

echo "finished"