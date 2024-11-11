# Training code for overall cell classification

import os

import segmentation_models_pytorch as smp
import torch
import wandb
from torch.optim import lr_scheduler

from monkey.config import TrainingIOConfig
from monkey.data.dataset import get_classification_dataloaders
from monkey.model.classification_model.efficientnet_b0 import (
    EfficientNet_B0,
)
from monkey.model.loss_functions import get_loss_function
from monkey.model.utils import get_activation_function
from monkey.train.train_cell_classification import train_cls_net

# -----------------------------------------------------------------------
# Specify training config and hyperparameters
run_config = {
    "project_name": "Monkey_Classification",
    "model_name": "efficientnetb0",
    "batch_size": 64,
    "val_fold": 4,  # [1-4]
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "epochs": 75,
    "loss_function": "BCE",
    "do_augmentation": True,
    "activation_function": "sigmoid",
    "module": "classification",
    "stack_mask": False,  # Whether to use 4-channel input
}

# Set IOConfig

IOconfig = TrainingIOConfig(
    dataset_dir="/mnt/lab-share/Monkey/classification/",
    save_dir=f"/home/u1910100/cloud_workspace/data/Monkey/cell_cls/{run_config['model_name']}",
)
IOconfig.set_image_dir("/mnt/lab-share/Monkey/classification/patches")
IOconfig.set_mask_dir("/mnt/lab-share/Monkey/classification/patches")
IOconfig.set_checkpoint_save_dir(
    run_name=f"fold_{run_config['val_fold']}"
)
os.environ["WANDB_DIR"] = IOconfig.save_dir

# Create dataloaders

train_loader, val_loader = get_classification_dataloaders(
    IOconfig,
    val_fold=run_config["val_fold"],
    batch_size=run_config["batch_size"],
    do_augmentation=run_config["do_augmentation"],
    stack_mask=run_config["stack_mask"],
)

# Create Classification Model

model = EfficientNet_B0(
    input_channels=3, num_classes=1, pretrained=True
)
model.to("cuda")

# Create loss function, optimizer and scheduler

loss_fn = get_loss_function(run_config["loss_function"])

activation_fn = get_activation_function(
    run_config["activation_function"]
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=run_config["learning_rate"],
    weight_decay=run_config["weight_decay"],
)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    "max",
    factor=0.1,
)

# Create WandB session
run = wandb.init(
    project=f"{run_config['project_name']}_{run_config['model_name']}_bm",
    name=f"fold_{run_config['val_fold']}",
    config=run_config,
)
# run.watch(model, log_freq=1000)
# run = None

# Start training
model = train_cls_net(
    model=model,
    train_loader=train_loader,
    validation_loader=val_loader,
    loss_fn=loss_fn,
    activation=activation_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    save_dir=IOconfig.checkpoint_save_dir,
    epochs=run_config["epochs"],
    wandb_run=run,
)

# Save final checkpoint
final_checkpoint = {
    "epoch": run_config["epochs"],
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
}
checkpoint_name = f"epoch_{run_config['epochs']}.pth"
model_path = os.path.join(
    IOconfig.checkpoint_save_dir, checkpoint_name
)
torch.save(final_checkpoint, model_path)

if run is not None:
    wandb.finish()
