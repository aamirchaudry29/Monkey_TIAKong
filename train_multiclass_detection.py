# Training code for customized hovernext
import os
from pprint import pprint

import click
import torch
import wandb
from torch.optim import lr_scheduler

from monkey.config import TrainingIOConfig
from monkey.data.dataset import get_detection_dataloaders
from monkey.model.hovernext.model import (
    get_custom_hovernext,
    load_encoder_weights,
)
from monkey.model.loss_functions import (
    MultiTaskLoss,
    get_loss_function,
    AutomatucWeightedLoss
)
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
        "model_name": "efficientnetv2_l_multitask_det_max_aug",
        "center_block": True,
        "val_fold": fold,  # [1-5]
        "batch_size": 64,
        "optimizer": "AdamW",
        "learning_rate": 4e-4,
        "weight_decay": 0.005,
        "epochs": 100,
        "loss_function": {
            "seg_loss": "Weighted_BCE_Dice",
            "contour_loss": "Weighted_BCE_Dice",
            "det_loss": "Jaccard_Dice_Focal_Loss",
        },
        "weight_map_scale": 1.0,
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
        "det_version": 2,
        "train_aux_loss_weights": [1.0, 0.5],  # [seg, contour]
        "val_aux_loss_weights": [1.0, 0.5],  # [seg, contour]
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
        enc="tf_efficientnetv2_l.in21k_ft_in1k",
        # enc="convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained=True,
        use_batchnorm=True,
        attention_type="scse",
        decoders_out_channels=[3, 3, 3],
        center=run_config["center_block"],
    )
    # checkpoint_path = "/home/u1910100/cloud_workspace/data/Monkey/convnextv2_base_lizard"
    # model = load_encoder_weights(model, checkpoint_path=checkpoint_path)
    # pprint("Encoder weights loaded")
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
        weight_map_scale=run_config["weight_map_scale"],
    )

    # Create loss function, optimizer and scheduler

    loss_fn_dict = {
        "seg_loss": get_loss_function(
            run_config["loss_function"]["seg_loss"]
        ),
        "contour_loss": get_loss_function(
            run_config["loss_function"]["contour_loss"]
        ),
        "det_loss": get_loss_function(
            run_config["loss_function"]["det_loss"]
        ),
    }

    # is_regression = torch.tensor([False, False, False], device="cuda")
    # multi_task_loss_instance = MultiTaskLoss(
    #     is_regression=is_regression, reduction="sum"
    # )
    multi_task_loss_instance = AutomatucWeightedLoss(3)
    multi_task_loss_instance.to("cuda")

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

    params = [
                {'params': model.parameters()},
                {'params': multi_task_loss_instance.parameters(), 'weight_decay': 0}
            ]
    optimizer = torch.optim.AdamW(
        params,
        lr=run_config["learning_rate"],
        weight_decay=run_config["weight_decay"],
    )
    # optimizer = torch.optim.RAdam(
    #     model.parameters(),
    #     lr=run_config["learning_rate"],
    #     weight_decay=run_config["weight_decay"],
    # )
    # scheduler = None
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "min", factor=0.5, patience=5
    # )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=0.0
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=20, T_mult=1
    # )

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
        multi_task_loss_instance=multi_task_loss_instance,
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
