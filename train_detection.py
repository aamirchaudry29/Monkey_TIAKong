# Training code for overall cell detection

import os
from pprint import pprint

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
    "project_name": "Monkey_Multiclass_Detection",
    "model_name": "efficientunetb0_seg_3_channel",
    "val_fold": 2,  # [1-5]
    "batch_size": 32,
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "epochs": 10,
    "loss_function": "Weighted_CE_Dice",
    "disk_radius": 11,  # Ignored if using NuClick masks
    "regression_map": False,  # Ignored if using NuClick masks
    "do_augmentation": True,
    "activation_function": "softmax",
    "module": "multiclass_detection",  # 'detection' or 'multiclass_detection'
    "include_background_channel": True,
    "use_nuclick_masks": True,  # Whether to use NuClick segmentation masks
}
pprint(run_config)

# Specify IO config
# ***Change save_dir
IOconfig = TrainingIOConfig(
    dataset_dir="/mnt/lab-share/Monkey/patches_256/",
    save_dir=f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{run_config['model_name']}",
)
# If use nuclick masks, change mask dir
if run_config["use_nuclick_masks"]:
    IOconfig.set_mask_dir("/mnt/lab-share/Monkey/nuclick_hovernext")


# Create model
model = get_efficientunet_b0_MBConv(out_channels=3)
model.to("cuda")
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
    module=run_config["module"],
    include_background_channel=run_config[
        "include_background_channel"
    ],
)


# Create loss function, optimizer and scheduler
loss_fn = get_loss_function(run_config["loss_function"])
if run_config["module"] == "multiclass_detection":
    loss_fn.set_multiclass(True)
    print("using multiclass loss")
    if run_config["include_background_channel"]:
        loss_fn.set_weight(
            torch.tensor([0.2, 0.4, 0.4], device="cuda")
        )
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
    optimizer, "max", factor=0.1, patience=5
)

# Create WandB session
run = wandb.init(
    project=f"{run_config['project_name']}_{run_config['model_name']}",
    name=f"fold_{run_config['val_fold']}_{run_config['module']}",
    config=run_config,
)
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
    run_config=run_config,
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
