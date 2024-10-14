import os

import torch
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

from monkey.config import TrainingIOConfig
from monkey.data.data_utils import get_dataloaders
from monkey.data.dataset import InflammatoryDataset
from monkey.model.detection_model.architecture import (
    get_efficientunet_b0_MBConv,
)
from monkey.model.loss_functions import get_loss_function
from monkey.train.train_cell_detection import train_det_net


run_config = {
    "project_name": "Monkey_Cell_Det",
    "model_name": "efficientunetb0",
    "batch_size": 16,
    "val_fold": 1,  # [1-4]
    "loss_function": "Jaccard",
    "optimizer": "AdamW",
    "learning_rate": 0.003,
    "weight_decay": 0.0004,
    "epochs": 10,
    "loss_function": "Dice",
}

IOconfig = TrainingIOConfig(
    dataset_dir="/home/u1910100/Documents/Monkey/patches_256",
    save_dir="/home/u1910100/Documents/Monkey/det_models",
)
os.environ["WANDB_DIR"] = IOconfig.save_dir


train_loader, val_loader = get_dataloaders(
    IOconfig,
    val_fold=run_config["val_fold"],
    task=1,
    batch_size=run_config["batch_size"],
)

model = get_efficientunet_b0_MBConv(out_channels=1)
model.to("cuda")

loss_fn = get_loss_function("Dice")
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=run_config["learning_rate"],
    weight_decay=run_config["weight_decay"],
)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    "min",
    factor=0.5,
)

run = wandb.init(
    project=f"{run_config['project_name']}_{run_config['model_name']}_fold_{run_config['val_fold']}",
    config=run_config,
)


model = train_det_net(
    model=model,
    train_loader=train_loader,
    validation_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    save_dir=IOconfig.save_dir,
    epochs=10,
    wandb_run=run,
)
