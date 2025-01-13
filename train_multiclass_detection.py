# Training code for customized hovernext
import os
from pprint import pprint

import torch
import click
import wandb
from torch.optim import lr_scheduler

from monkey.config import TrainingIOConfig
from monkey.data.dataset import get_detection_dataloaders
from monkey.model.cellvit.cellvit import CellVit256_Unet
from monkey.model.hovernext.model import (
    get_custom_hovernext,
    load_encoder_weights,
)
from monkey.model.hovernext.modified_model import (
    get_modified_hovernext,
)
from monkey.model.loss_functions import get_loss_function
from monkey.model.utils import get_activation_function
from monkey.train.train_multitask_cell_detection import (
    multitask_train_loop,
)

@click.command()
@click.option("--fold", default=1)
def train(fold: int = 1):
    # -----------------------------------------------------------------------
    # Specify training config and hyperparameters
    run_config = {
        "project_name": "Monkey_Multiclass_Detection",
        "model_name": "convnextv2_tiny_pannuke_multitask_det_experiment",
        "val_fold": fold,  # [1-5]
        "batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": 0.0004,
        "weight_decay": 0.005,
        "epochs": 75,
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
        "disk_radius": 11,
        "augmentation_prob": 0.95,
        "unfreeze_epoch": 1,
        "strong_augmentation": True,
        "det_version": 2
    }
    pprint(run_config)

    # Specify IO config
    # ***Change save_dir
    IOconfig = TrainingIOConfig(
        dataset_dir="/mnt/lab-share/Monkey/patches_256/",
        save_dir=f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{run_config['model_name']}",
    )
    IOconfig.set_mask_dir(
        mask_dir="/mnt/lab-share/Monkey/nuclick_masks_processed_v2"
    )

    # Create model
    model = get_custom_hovernext(
        enc="convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained=True,
        use_batchnorm=True,
        attention_type="scse",
        decoders_out_channels=[3,3,3]
    )
    # model = get_modified_hovernext(
    #     enc="convnextv2_tiny.fcmae_ft_in22k_in1k",
    #     pretrained=True,
    #     use_batchnorm=True,
    #     attention_type="scse",
    # )
    checkpoint_path = "/home/u1910100/cloud_workspace/data/Monkey/convnextv2_tiny_pannuke"
    model = load_encoder_weights(model, checkpoint_path=checkpoint_path)
    pprint("Encoder weights loaded")
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
        disk_radius=run_config["disk_radius"],
        augmentation_prob=run_config["augmentation_prob"],
        strong_augmentation=run_config["strong_augmentation"],
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
    # optimizer = torch.optim.RAdam(
    #     model.parameters(),
    #     lr=run_config["learning_rate"],
    #     weight_decay=run_config["weight_decay"],
    # )
    # scheduler = None
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=5
    )
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Create WandB session
    # run = None
    run = wandb.init(
        project=f"{run_config['project_name']}_{run_config['model_name']}",
        name=f"fold_{run_config['val_fold']}",
        config=run_config,
        notes="",
    )

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


if __name__ == "__main__":
    train()
