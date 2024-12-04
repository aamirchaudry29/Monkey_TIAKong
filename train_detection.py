# Training code for overall cell detection

import os
from pprint import pprint

import segmentation_models_pytorch as smp
import torch
import wandb
from torch.optim import lr_scheduler

from monkey.config import TrainingIOConfig
from monkey.data.dataset import get_detection_dataloaders
from monkey.model.hovernext.model import (
    get_convnext_unet,
    get_custom_hovernext,
    load_encoder_weights,
)
from monkey.model.loss_functions import get_loss_function
from monkey.model.utils import get_activation_function
from monkey.train.train_cell_detection import train_det_net

# -----------------------------------------------------------------------
# Specify training config and hyperparameters
run_config = {
    "project_name": "Monkey_Detection_2_channel",
    "model_name": "hovernext_large_lizzard_pretrained_v2",
    "val_fold": 4,  # [1-5]
    "batch_size": 64,
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "epochs": 30,
    "loss_function": "Jaccard_Loss",
    "loss_pos_weight": 1.0,
    "do_augmentation": True,
    "activation_function": "sigmoid",
    "disk_radius": 11,
    "augmentation_prob": 0.5
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
    pretrained=True,
    out_classes=2,
    use_batchnorm=True,
    attention_type="scse",
)
# model = get_custom_hovernext(
#     pretrained=True,
#     num_heads=2,
#     decoders_out_channels=[1, 1],
#     use_batchnorm=True,
#     attention_type="scse",
# )
checkpoint_path = "/home/u1910100/cloud_workspace/data/Monkey/convnextv2_large_lizard"
model.to("cuda")
model = load_encoder_weights(model, checkpoint_path=checkpoint_path)
pprint("Lizzard encoder weights loaded")
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
    use_nuclick_masks=False,
    disk_radius=run_config['disk_radius'],
    augmentation_prob=run_config['augmentation_prob']
)


# Create loss function, optimizer and scheduler
loss_fn = get_loss_function(run_config["loss_function"])
loss_fn.set_multiclass(True)

activation_fn = get_activation_function(
    run_config["activation_function"]
)
optimizer = torch.optim.NAdam(
    model.parameters(),
    lr=run_config["learning_rate"],
    weight_decay=run_config["weight_decay"],
    # momentum=0.9,
)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode="max", factor=0.1, patience=5
)

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
