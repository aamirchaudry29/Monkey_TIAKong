# Training code for overall cell classification

import os

import segmentation_models_pytorch as smp
import torch
import wandb
from torch.optim import lr_scheduler

from monkey.config import TrainingIOConfig
from monkey.data.dataset import get_detection_dataloaders
from monkey.model.efficientunetb0.architecture import get_efficientunet_b0_MBConv
from monkey.model.loss_functions import get_loss_function
from monkey.model.utils import get_activation_function
from monkey.train.train_cell_detection import train_det_net

# -----------------------------------------------------------------------
# Specify training config and hyperparameters
run_config = {
    "project_name": "Monkey_Classification",
    "model_name": "efficientunetb0",
    "batch_size": 64,
    "val_fold": 1,  # [1-4]
    "optimizer": "AdamW",
    "learning_rate": 0.03,
    "weight_decay": 0.0004,
    "epochs": 100,
    "loss_function": "BCE",
    "disk_radius": 11,  # Ignored for NuClick masks
    "regression_map": False,
    "do_augmentation": True,
    "activation_function": "sigmoid",
    "module": "detection",
    "use_nuclick_masks": True,  # Whether to use NuClick segmentation masks
}