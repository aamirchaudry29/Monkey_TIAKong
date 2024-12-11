# Training code for multihead efficientunet

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
    "model_name": "multihead_unet_512",
    "val_fold": 2,  # [1-5]
    "batch_size": 16,
    "optimizer": "AdamW",
    "learning_rate": 0.0004,
    "weight_decay": 0.01,
    "epochs": 30,
    "loss_function": {
        "head_1": "Weighted_BCE_Jaccard",
        "head_2": "Weighted_BCE_Jaccard",
        "head_3": "Weighted_BCE_Jaccard",
    },
    "loss_pos_weight": 1.0,
    "peak_thresholds": [0.5, 0.5, 0.5],  # [inflamm, lymph, mono]
    "do_augmentation": True,
    "activation_function": {
        "head_1": "sigmoid",
        "head_2": "sigmoid",
        "head_3": "sigmoid",
    },
    "use_nuclick_masks": False,  # Whether to use NuClick segmentation masks,
    "include_background_channel": False,
    "disk_radius": 11,
    "regression_map": False,
    "augmentation_prob": 0.85,
    "unfreeze_epoch": 5,
}
pprint(run_config)

# Specify IO config
# ***Change save_dir
IOconfig = TrainingIOConfig(
    dataset_dir="/mnt/lab-share/Monkey/patches_512/",
    save_dir=f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{run_config['model_name']}",
)



# Create model
model = get_multihead_efficientunet(
    out_channels=[1,1,1], pretrained=True
)
model.to("cuda")
device = torch.device('cuda:0')
free, total = torch.cuda.mem_get_info(device)
print(f"GPU memory free: {free/1024**2:.2f} MB")
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
    include_background_channel=run_config[
        "include_background_channel"
    ],
    disk_radius=run_config["disk_radius"],
    augmentation_prob=run_config["augmentation_prob"],
    regression_map=run_config["regression_map"],
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
loss_fn_dict["head_1"].set_weight(run_config["loss_pos_weight"])
loss_fn_dict["head_2"].set_weight(run_config["loss_pos_weight"])
loss_fn_dict["head_3"].set_weight(run_config["loss_pos_weight"])

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
    optimizer, mode="min", factor=0.1, patience=5
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
