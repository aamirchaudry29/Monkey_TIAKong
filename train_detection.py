# Training code for overall cell detection

import os

import segmentation_models_pytorch as smp
import torch
import wandb
from torch.optim import lr_scheduler

from monkey.config import TrainingIOConfig
from monkey.data.dataset import get_detection_dataloaders
from monkey.model.efficientunetb0.architecture import (
    get_efficientunet_b0_MBConv,
)
from monkey.model.loss_functions import get_loss_function
from monkey.model.utils import get_activation_function
from monkey.train.train_cell_detection import train_det_net

# -----------------------------------------------------------------------
# Specify training config and hyperparameters
run_config = {
    "project_name": "Monkey_Detection",
    "model_name": "unet++_resnext101_32x16d",
    "batch_size": 16,
    "val_fold": 1,  # [1-4]
    "optimizer": "AdamW",
    "learning_rate": 0.03,
    "weight_decay": 0.0004,
    "epochs": 75,
    "loss_function": "BCE_Dice",
    "disk_radius": 11,  # Ignored for NuClick masks
    "regression_map": False,
    "do_augmentation": True,
    "activation_function": "sigmoid",
    "module": "detection",
    "use_nuclick_masks": True,  # Whether to use NuClick segmentation masks
}

# Specify IO config
# ***Change save_dir
IOconfig = TrainingIOConfig(
    dataset_dir="/mnt/lab-share/Monkey/patches_256/",
    save_dir=f"/home/u1910100/cloud_workspace/data/Monkey/cell_det/{run_config['model_name']}",
)
# If use nuclick masks, change mask dir
if run_config["use_nuclick_masks"]:
    IOconfig.set_mask_dir("/mnt/lab-share/Monkey/nuclick_hovernext")


# Create model
# model = get_efficientunet_b0_MBConv(out_channels=1)
model = smp.UnetPlusPlus(
    encoder_name="tu-resnext101_32x16d",
    encoder_weights="imagenet",
    decoder_attention_type="scse",
    in_channels=3,
    classes=1,
)
model.to("cuda")
# torch.compile(
#     model,
#     mode="default"
# )
# -----------------------------------------------------------------------


IOconfig.set_checkpoint_save_dir(
    run_name=f"fold_{run_config['val_fold']}"
)
os.environ["WANDB_DIR"] = IOconfig.save_dir

# Get dataloaders for task
train_loader, val_loader = get_detection_dataloaders(
    IOconfig,
    val_fold=run_config["val_fold"],
    task=1,
    batch_size=run_config["batch_size"],
    disk_radius=run_config["disk_radius"],
    regression_map=run_config["regression_map"],
    do_augmentation=run_config["do_augmentation"],
    use_nuclick_masks=run_config["use_nuclick_masks"],
)


# Create loss function, optimizer and scheduler
loss_fn = get_loss_function(run_config["loss_function"])
activation_fn = get_activation_function(
    run_config["activation_function"]
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=run_config["learning_rate"],
    weight_decay=run_config["weight_decay"],
    # momentum=0.9,
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
model = train_det_net(
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
