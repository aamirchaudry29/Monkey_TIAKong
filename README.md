# Monkey_TIAKong
GitHub Repostory for Monkey TIAKong.  

# Description
Training and evaluation pipeline for mononuclear leukocytes (MNLs) detection.  
Some parts of the pipeline are taken from Kesi's AutoNuclick.

# Run on TIA New Servers (Slurm)
Use the script `submit_job.sbatch`.  
On `Cayenne` server, run `sbatch sumbmit_job.sbatch`.  
You need to change the conda environment to your environment, and the path to the python script.  
Set `data_dir` to `"/mnt/lab-share/Monkey/patches_256/"`.

# Train Cell Detection
An example training code is in `train_detection.py`.  
You need to change the `save_dir`, to where you want the checkpoints and WandB logs to be saved.  

# Train Cell Classification
An example training code is in `train_classification.py`.  
You need to change the `save_dir`, to where you want the checkpoints and WandB logs to be saved.  
