# Example training code for overall cell detection

import os

import segmentation_models_pytorch as smp
import torch
import wandb
from torch.optim import lr_scheduler

from monkey.config import TrainingIOConfig
from monkey.data.dataset import get_dataloaders
from monkey.model.efficientunetb0.architecture import get_efficientunet_b0_MBConv
from monkey.model.loss_functions import get_loss_function
from monkey.model.utils import get_activation_function
from monkey.train.train_cell_detection import train_det_net

# -----------------------------------------------------------------------
# Specify training config and hyperparameters
run_config = {
    "project_name": "Monkey_Cell_Det",
    "model_name": "efficientunetb0",
    "batch_size": 32,
    "val_fold": 4,  # [1-4]
    "optimizer": "RMSProp",
    "learning_rate": 0.03,
    "weight_decay": 0.0004,
    "epochs": 100,
    "loss_function": "BCE_Dice",
    "disk_radius": 9,
    "do_augmentation": False,
    "activation_function": "sigmoid",
    "module": "detection",
}

# Specify IO config
# ***Change save_dir
IOconfig = TrainingIOConfig(
    dataset_dir="/mnt/lab-share/Monkey/patches_256/",
    save_dir=f"/home/u1910100/cloud_workspace/data/Monkey/cell_det/runs",
)
# Create model
model = get_efficientunet_b0_MBConv(out_channels=1)
model.to("cuda")
# -----------------------------------------------------------------------


IOconfig.set_checkpoint_save_dir(
    run_name=f"fold_{run_config['val_fold']}"
)
os.environ["WANDB_DIR"] = IOconfig.save_dir

# Get dataloaders for task
train_loader, val_loader = get_dataloaders(
    IOconfig,
    val_fold=run_config["val_fold"],
    task=1,
    batch_size=run_config["batch_size"],
    disk_radius=run_config["disk_radius"],
    do_augmentation=run_config["do_augmentation"],
)


# Create loss function, optimizer and scheduler
loss_fn = get_loss_function(run_config["loss_function"])
activation_fn = get_activation_function(
    run_config["activation_function"]
)
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=run_config["learning_rate"],
    weight_decay=run_config["weight_decay"],
    momentum=0.9,
)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    "max",
    factor=0.1,
)

# Create WandB session
run = wandb.init(
    project=f"{run_config['project_name']}_{run_config['model_name']}",
    name=f"fold_{run_config['val_fold']}",
    config=run_config,
)

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

wandb.finish()
