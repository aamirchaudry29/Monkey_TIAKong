# Training code for multihead efficientunet

import os
from pprint import pprint

import torch
import wandb
import click
from torch.optim import lr_scheduler

from monkey.config import TrainingIOConfig
from monkey.data.dataset import get_detection_dataloaders
from monkey.model.efficientunetb0.architecture import (
    get_multihead_efficientunet,
)
from monkey.model.hovernext.model import (
    get_convnext_unet,
    get_custom_hovernext,
    load_encoder_weights,
)
from monkey.model.loss_functions import get_loss_function
from monkey.model.utils import get_activation_function
from monkey.train.train_multitask_segmentation import (
    multitask_train_loop,
)


@click.command()
@click.option("--fold", default=1)
def train(fold: int = 1):
    # -----------------------------------------------------------------------
    # Specify training config and hyperparameters
    run_config = {
        "project_name": "Monkey_Multiclass_Segmentation",
        "model_name": "convnextv2_tiny_pannuke_seg_v3",
        "val_fold": fold,  # [1-5]
        "batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": 0.0004,
        "weight_decay": 0.004,
        "epochs": 50,
        "loss_function": {
            "head_1": "Weighted_BCE_Jaccard",
            "head_2": "Weighted_BCE_Jaccard",
            "head_3": "Weighted_BCE_Jaccard",
        },
        "loss_pos_weight": 1.0,
        "peak_thresholds": [
            0.5,
            0.5,
            0.5,
            0.5,
        ],  # [inflamm, lymph, mono, contour]
        "do_augmentation": True,
        "activation_function": {
            "head_1": "sigmoid",
            "head_2": "sigmoid",
            "head_3": "sigmoid",
        },
        "use_nuclick_masks": True,  # Whether to use NuClick segmentation masks,
        "augmentation_prob": 0.95,
        "strong_augmentation": True,
        "unfreeze_epoch": 1,
        "dataset_version": 2,
        "loss_weights": [1.0, 0.5]
    }
    pprint(run_config)

    # Specify IO config
    # ***Change save_dir
    IOconfig = TrainingIOConfig(
        dataset_dir="/mnt/lab-share/Monkey/patches_256/",
        save_dir=f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{run_config['model_name']}",
    )
    IOconfig.set_mask_dir(
        mask_dir="/mnt/lab-share/Monkey/nuclick_masks_processed"
    )
    if run_config["dataset_version"] == 2:
        IOconfig.set_mask_dir(
            mask_dir="/mnt/lab-share/Monkey/nuclick_masks_processed_v2"
        )


    # Create model
    # model = get_multihead_efficientunet(
    #     out_channels=[2, 1, 1], pretrained=True
    # )
    model = get_custom_hovernext(
        enc="convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained=True,
        num_heads=3,
        decoders_out_channels=[2, 2, 2],
        use_batchnorm=True,
        attention_type="scse",
    )
    checkpoint_path = "/home/u1910100/cloud_workspace/data/Monkey/convnextv2_tiny_pannuke"
    model = load_encoder_weights(model, checkpoint_path=checkpoint_path)
    pprint("Encoder weights loaded")
    model.to("cuda")
    device = torch.device("cuda")
    # -----------------------------------------------------------------------


    IOconfig.set_checkpoint_save_dir(
        run_name=f"fold_{run_config['val_fold']}"
    )
    os.environ["WANDB_DIR"] = IOconfig.save_dir

    # Get dataloaders for task
    train_loader, val_loader = get_detection_dataloaders(
        IOconfig,
        val_fold=run_config["val_fold"],
        dataset_name="segmentation",
        batch_size=run_config["batch_size"],
        do_augmentation=run_config["do_augmentation"],
        use_nuclick_masks=run_config["use_nuclick_masks"],
        strong_augmentation=run_config["strong_augmentation"],
        augmentation_prob=run_config["augmentation_prob"],
        version=run_config["dataset_version"],
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
        optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-6
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


if __name__ == "__main__":
    train()