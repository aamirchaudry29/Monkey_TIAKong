# Training code for overall cell detection

import os
from pprint import pprint

import segmentation_models_pytorch as smp
import torch
import wandb
from torch.optim import lr_scheduler

from monkey.config import TrainingIOConfig
from monkey.data.dataset import get_detection_dataloaders
from monkey.model.hovernext.model import get_convnext_unet
from monkey.model.loss_functions import get_loss_function
from monkey.model.mapde.model import MapDe
from monkey.model.utils import get_activation_function
from monkey.train.train_cell_detection import train_det_net

# -----------------------------------------------------------------------
# Specify training config and hyperparameters
run_config = {
    "project_name": "Monkey_Detection_Lymph",
    "model_name": "convnext_unet",
    "val_fold": 1,  # [1-5]
    "batch_size": 32,
    "optimizer": "AdamW",
    "learning_rate": 0.0004,
    "weight_decay": 0.01,
    "epochs": 30,
    "loss_function": "BCE_Dice",
    "loss_pos_weight": 1.0,
    "do_augmentation": True,
    "activation_function": "sigmoid",
    "module": "detection",  # 'detection' or 'multiclass_detection'
    "target_cell_type": "lymph",
}
pprint(run_config)

# Specify IO config
# ***Change save_dir
IOconfig = TrainingIOConfig(
    dataset_dir="/mnt/lab-share/Monkey/patches_256/",
    save_dir=f"/home/u1910100/cloud_workspace/data/Monkey/{run_config['project_name']}/{run_config['model_name']}",
)

# Create model
model = get_convnext_unet(
    enc="convnextv2_large.fcmae_ft_in22k_in1k", pretrained=True
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
    dataset_name="detection",
    batch_size=run_config["batch_size"],
    do_augmentation=run_config["do_augmentation"],
    module=run_config["module"],
    target_cell_type=run_config["target_cell_type"],
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
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Create WandB session
run = wandb.init(
    project=f"{run_config['project_name']}_{run_config['model_name']}",
    name=f"fold_{run_config['val_fold']}",
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
}
checkpoint_name = f"epoch_{run_config['epochs']}.pth"
model_path = os.path.join(
    IOconfig.checkpoint_save_dir, checkpoint_name
)
torch.save(final_checkpoint, model_path)

if run is not None:
    wandb.finish()
