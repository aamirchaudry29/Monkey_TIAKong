# Training code for overall cell detection

import os
from pprint import pprint

import torch
import wandb
from torch.optim import lr_scheduler

from monkey.config import TrainingIOConfig
from monkey.data.dataset import get_detection_dataloaders
from monkey.model.efficientunetb0.architecture import (
    get_multihead_efficientunet,
)
from monkey.model.loss_functions import get_loss_function
from monkey.model.utils import get_activation_function
from monkey.train.train_multitask_cell_detection import (
    multitask_train_loop,
)

# -----------------------------------------------------------------------
# Specify training config and hyperparameters
run_config = {
    "project_name": "Monkey_Multiclass_Detection",
    "model_name": "multihead_unet",
    "out_channels": [2, 1, 1],
    "val_fold": 1,  # [1-5]
    "batch_size": 16,
    "optimizer": "AdamW",
    "learning_rate": 0.0004,
    "weight_decay": 0.01,
    "epochs": 50,
    "loss_function": {
        "head_1": "BCE_Dice",
        "head_2": "BCE_Dice",
        "head_3": "BCE_Dice",
    },
    "do_augmentation": True,
    "activation_function": {
        "head_1": "sigmoid",
        "head_2": "sigmoid",
        "head_3": "sigmoid",
    },
    "use_nuclick_masks": True,  # Whether to use NuClick segmentation masks
}
pprint(run_config)

# Specify IO config
# ***Change save_dir
IOconfig = TrainingIOConfig(
    dataset_dir="/mnt/lab-share/Monkey/patches_256/",
    save_dir=f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{run_config['model_name']}",
)

IOconfig.set_mask_dir("/mnt/lab-share/Monkey/nuclick_masks_processed")


# Create model
model = get_multihead_efficientunet(
    out_channels=run_config["out_channels"], pretrained=True
)
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
    dataset_name="multitask",
    batch_size=run_config["batch_size"],
    do_augmentation=run_config["do_augmentation"],
    use_nuclick_masks=run_config["use_nuclick_masks"],
)


# Create loss function, optimizer and scheduler

loss_fn_dict = {
    "head_1": get_loss_function(
        run_config["loss_function"]["head_1"]
    ),
    "head_2": get_loss_function(
        run_config["loss_function"]["head_2"]
    ),
    "head_3": get_loss_function(
        run_config["loss_function"]["head_3"]
    ),
}

loss_fn_dict["head_1"].set_multiclass(True)

activation_fn_dict = {
    "head_1": get_activation_function(
        run_config["activation_function"]["head_1"]
    ),
    "head_2": get_activation_function(
        run_config["activation_function"]["head_2"]
    ),
    "head_3": get_activation_function(
        run_config["activation_function"]["head_3"]
    ),
}


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=run_config["learning_rate"],
    weight_decay=run_config["weight_decay"],
)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, "max", factor=0.1, patience=10
)


# Create WandB session
run = wandb.init(
    project=f"{run_config['project_name']}_{run_config['model_name']}",
    name=f"fold_{run_config['val_fold']}",
    config=run_config,
)
# run = None

# Start training
model = multitask_train_loop(
    model=model,
    num_tasks=len(run_config["out_channels"]),
    train_loader=train_loader,
    validation_loader=val_loader,
    loss_fn_dict=loss_fn_dict,
    activation_dict=activation_fn_dict,
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
}
checkpoint_name = f"epoch_{run_config['epochs']}.pth"
model_path = os.path.join(
    IOconfig.checkpoint_save_dir, checkpoint_name
)
torch.save(final_checkpoint, model_path)

if run is not None:
    wandb.finish()
