from math import inf
import os
from pdb import run
from pprint import pprint
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from monkey.model.hovernext.model import freeze_enc, unfreeze_enc
from monkey.model.loss_functions import Loss_Function
from monkey.model.utils import get_multiclass_patch_F1_score_batch, get_patch_F1_score_batch
from monkey.train.utils import compose_multitask_log_images
from prediction.utils import multihead_seg_post_process, multihead_seg_post_process_v2


def train_one_epoch_v2(
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn_dict: dict[str, Loss_Function],
    run_config: dict,
    activation_dict: dict[str, torch.nn.Module],
):
    epoch_loss = 0.0
    model.train()
    for i, data in enumerate(
        tqdm(training_loader, desc="train", leave=False)
    ):
        images = data["image"].cuda().float()

        inflamm_true_masks = data["inflamm_mask"].cuda().float()
        inflamm_contour_true_masks = data["inflamm_contour_mask"].cuda().float()
        lymph_true_masks = (
            data["lymph_mask"].cuda().float()
        )
        lymph_contour_true_masks = data["lymph_contour_mask"].cuda().float()
        mono_true_masks = (
            data["mono_mask"].cuda().float()
        )
        mono_contour_true_masks = data["mono_contour_mask"].cuda().float()

        optimizer.zero_grad()

        logits_pred = model(images)

        inflamm_logits = logits_pred[:, 0:1, :, :]
        inflamm_contour_logits = logits_pred[:, 1:2, :, :]
        lymph_logits = logits_pred[:, 2:3, :, :]
        lymph_contour_logits = logits_pred[:, 3:4, :, :]
        mono_logits = logits_pred[:, 4:5, :, :]
        mono_contour_logits = logits_pred[:, 5:6, :, :]

        inflamm_probs = activation_dict["head_1"](inflamm_logits)
        inflamm_contour_probs = activation_dict["head_1"](inflamm_contour_logits)
        lymph_probs = activation_dict["head_2"](lymph_logits)
        lymph_contour_probs = activation_dict["head_2"](lymph_contour_logits)
        mono_probs = activation_dict["head_3"](mono_logits)
        mono_contour_probs = activation_dict["head_3"](mono_contour_logits)

        inflamm_loss = loss_fn_dict["head_1"].compute_loss(
            inflamm_probs, inflamm_true_masks
        )
        lymph_loss = loss_fn_dict["head_2"].compute_loss(
            lymph_probs, lymph_true_masks
        )
        mono_loss = loss_fn_dict["head_3"].compute_loss(
            mono_probs, mono_true_masks
        )
        inflamm_contour_loss = loss_fn_dict["head_1"].compute_loss(
            inflamm_contour_probs, inflamm_contour_true_masks
        )
        lymph_contour_loss = loss_fn_dict["head_2"].compute_loss(
            lymph_contour_probs, lymph_contour_true_masks
        )
        mono_contour_loss = loss_fn_dict["head_3"].compute_loss(
            mono_contour_probs, mono_contour_true_masks
        )

        sum_loss = inflamm_loss + lymph_loss + mono_loss + inflamm_contour_loss + lymph_contour_loss + mono_contour_loss
        sum_loss.backward()
        optimizer.step()

        epoch_loss += sum_loss.item() * images.size(0)

    return epoch_loss / len(training_loader.sampler)



def validate_one_epoch_v2(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_fn_dict: dict[str, Loss_Function],
    run_config: dict,
    activation_dict: dict[str, torch.nn.Module],
    wandb_run: Optional[wandb.run] = None,
):
    running_overall_score = 0.0
    running_lymph_score = 0.0
    running_mono_score = 0.0
    running_loss = 0.0

    model.eval()
    for i, data in enumerate(
        tqdm(validation_loader, desc="validation", leave=False)
    ):
        images = data["image"].cuda().float()

        inflamm_true_masks = data["inflamm_mask"].cuda().float()
        inflamm_contour_true_masks = data["inflamm_contour_mask"].cuda().float()
        lymph_true_masks = (
            data["lymph_mask"].cuda().float()
        )
        lymph_contour_true_masks = data["lymph_contour_mask"].cuda().float()
        mono_true_masks = (
            data["mono_mask"].cuda().float()
        )
        mono_contour_true_masks = data["mono_contour_mask"].cuda().float()

        with torch.no_grad():
            logits_pred = model(images)
            inflamm_logits = logits_pred[:, 0:1, :, :]
            inflamm_contour_logits = logits_pred[:, 1:2, :, :]
            lymph_logits = logits_pred[:, 2:3, :, :]
            lymph_contour_logits = logits_pred[:, 3:4, :, :]
            mono_logits = logits_pred[:, 4:5, :, :]
            mono_contour_logits = logits_pred[:, 5:6, :, :]

            inflamm_probs = activation_dict["head_1"](inflamm_logits)
            inflamm_contour_probs = activation_dict["head_1"](inflamm_contour_logits)
            lymph_probs = activation_dict["head_2"](lymph_logits)
            lymph_contour_probs = activation_dict["head_2"](lymph_contour_logits)
            mono_probs = activation_dict["head_3"](mono_logits)
            mono_contour_probs = activation_dict["head_3"](mono_contour_logits)

            inflamm_loss = loss_fn_dict["head_1"].compute_loss(
            inflamm_probs, inflamm_true_masks
            )
            lymph_loss = loss_fn_dict["head_2"].compute_loss(
                lymph_probs, lymph_true_masks
            )
            mono_loss = loss_fn_dict["head_3"].compute_loss(
                mono_probs, mono_true_masks
            )
            inflamm_contour_loss = loss_fn_dict["head_1"].compute_loss(
                inflamm_contour_probs, inflamm_contour_true_masks
            )
            lymph_contour_loss = loss_fn_dict["head_2"].compute_loss(
                lymph_contour_probs, lymph_contour_true_masks
            )
            mono_contour_loss = loss_fn_dict["head_3"].compute_loss(
                mono_contour_probs, mono_contour_true_masks
            )

            sum_loss = inflamm_loss + lymph_loss + mono_loss + inflamm_contour_loss + lymph_contour_loss + mono_contour_loss
            running_loss += sum_loss * images.size(0)

            binary_masks = multihead_seg_post_process_v2(
                inflamm_prob=inflamm_probs,
                lymph_prob=lymph_probs,
                mono_prob=mono_probs,
                inflamm_contour_prob=inflamm_contour_probs,
                lymph_contour_prob=lymph_contour_probs,
                mono_contour_prob=mono_contour_probs,
                thresholds=run_config["peak_thresholds"],
            )
            overall_pred_binary = binary_masks["inflamm_mask"]
            lymph_pred_binary = binary_masks["lymph_mask"]
            mono_pred_binary = binary_masks["mono_mask"]

            # Compute detection F1 score
            overall_metrics = get_multiclass_patch_F1_score_batch(
                overall_pred_binary,
                inflamm_true_masks,
                [5],
                inflamm_probs,
            )
            lymph_metrics = get_multiclass_patch_F1_score_batch(
                lymph_pred_binary,
                lymph_true_masks,
                [4],
                lymph_probs,
            )
            mono_metrics = get_multiclass_patch_F1_score_batch(
                mono_pred_binary,
                mono_true_masks,
                [5],
                mono_probs,
            )

        running_overall_score += (
            overall_metrics["F1"]
        ) * images.size(0)
        running_lymph_score += (lymph_metrics["F1"]) * images.size(0)
        running_mono_score += (mono_metrics["F1"]) * images.size(0)

    # Log an example prediction to WandB
    log_data = compose_multitask_log_images(
        images,
        inflamm_true_masks,
        lymph_true_masks,
        mono_true_masks,
        inflamm_contour_true_masks,
        overall_pred_probs=inflamm_probs,
        lymph_pred_probs=lymph_probs,
        mono_pred_probs=mono_probs,
        contour_pred_probs=inflamm_contour_probs,
    )
    wandb_run.log(log_data)

    return {
        "overall_F1": running_overall_score
        / len(validation_loader.sampler),
        "lymph_F1": running_lymph_score
        / len(validation_loader.sampler),
        "mono_F1": running_mono_score
        / len(validation_loader.sampler),
        "val_loss": running_loss / len(validation_loader.sampler),
    }


def train_one_epoch(
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn_dict: dict[str, Loss_Function],
    run_config: dict,
    activation_dict: dict[str, torch.nn.Module],
):
    epoch_loss = 0.0
    model.train()
    for i, data in enumerate(
        tqdm(training_loader, desc="train", leave=False)
    ):
        images = data["image"].cuda().float()

        binary_true_masks = data["binary_mask"].cuda().float()
        contour_true_masks = data["contour_mask"].cuda().float()
        # head_1_true_masks = torch.concatenate(
        #     (binary_true_masks, contour_masks), dim=1
        # )
        # head_2_true_masks = data["class_mask"].cuda().float()
        lymph_true_masks = (
            data["class_mask"][:, 0:1, :, :].cuda().float()
        )
        mono_true_masks = (
            data["class_mask"][:, 1:2, :, :].cuda().float()
        )

        optimizer.zero_grad()

        logits_pred = model(images)

        inflamm_logits = logits_pred[:, 0:1, :, :]
        contour_logits = logits_pred[:, 1:2, :, :]
        lymph_logits = logits_pred[:, 2:3, :, :]
        mono_logits = logits_pred[:, 3:4, :, :]

        inflamm_probs = activation_dict["head_1"](inflamm_logits)
        contour_probs = activation_dict["head_1"](contour_logits)
        lymph_probs = activation_dict["head_2"](lymph_logits)
        mono_probs = activation_dict["head_3"](mono_logits)

        inflamm_loss = loss_fn_dict["head_1"].compute_loss(
            inflamm_probs, binary_true_masks
        )
        contour_loss = loss_fn_dict["head_1"].compute_loss(
            contour_probs, contour_true_masks
        )
        lymph_loss = loss_fn_dict["head_2"].compute_loss(
            lymph_probs, lymph_true_masks
        )
        mono_loss = loss_fn_dict["head_3"].compute_loss(
            mono_probs, mono_true_masks
        )

        sum_loss = inflamm_loss + contour_loss + lymph_loss + mono_loss
        sum_loss.backward()
        optimizer.step()

        epoch_loss += sum_loss.item() * images.size(0)

    return epoch_loss / len(training_loader.sampler)


def validate_one_epoch(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_fn_dict: dict[str, Loss_Function],
    run_config: dict,
    activation_dict: dict[str, torch.nn.Module],
    wandb_run: Optional[wandb.run] = None,
):
    running_overall_score = 0.0
    running_lymph_score = 0.0
    running_mono_score = 0.0
    running_loss = 0.0

    model.eval()
    for i, data in enumerate(
        tqdm(validation_loader, desc="validation", leave=False)
    ):
        images = data["image"].cuda().float()

        binary_true_masks = data["binary_mask"].cuda().float()
        contour_true_masks = data["contour_mask"].cuda().float()
        lymph_true_masks = (
            data["class_mask"][:, 0:1, :, :].cuda().float()
        )
        mono_true_masks = (
            data["class_mask"][:, 1:2, :, :].cuda().float()
        )

        with torch.no_grad():
            logits_pred = model(images)
            inflamm_logits = logits_pred[:, 0:1, :, :]
            contour_logits = logits_pred[:, 1:2, :, :]
            lymph_logits = logits_pred[:, 2:3, :, :]
            mono_logits = logits_pred[:, 3:4, :, :]

            inflamm_probs = activation_dict["head_1"](inflamm_logits)
            contour_probs = activation_dict["head_1"](contour_logits)
            lymph_probs = activation_dict["head_2"](lymph_logits)
            mono_probs = activation_dict["head_3"](mono_logits)

            inflamm_loss = loss_fn_dict["head_1"].compute_loss(
                inflamm_probs, binary_true_masks
            )
            contour_loss = loss_fn_dict["head_1"].compute_loss(
                contour_probs, contour_true_masks
            )
            lymph_loss = loss_fn_dict["head_2"].compute_loss(
                lymph_probs, lymph_true_masks
            )
            mono_loss = loss_fn_dict["head_3"].compute_loss(
                mono_probs, mono_true_masks
            )

            sum_loss = inflamm_loss + contour_loss + lymph_loss + mono_loss
            running_loss += sum_loss * images.size(0)

            binary_masks = multihead_seg_post_process(
                inflamm_prob=inflamm_probs,
                lymph_prob=lymph_probs,
                mono_prob=mono_probs,
                contour_prob=contour_probs,
                thresholds=run_config["peak_thresholds"],
            )
            overall_pred_binary = binary_masks["inflamm_mask"]
            lymph_pred_binary = binary_masks["lymph_mask"]
            mono_pred_binary = binary_masks["mono_mask"]

            # Compute detection F1 score
            overall_metrics = get_multiclass_patch_F1_score_batch(
                overall_pred_binary,
                binary_true_masks,
                [5],
                inflamm_probs,
            )
            lymph_metrics = get_multiclass_patch_F1_score_batch(
                lymph_pred_binary,
                lymph_true_masks,
                [4],
                lymph_probs,
            )
            mono_metrics = get_multiclass_patch_F1_score_batch(
                mono_pred_binary,
                mono_true_masks,
                [5],
                mono_probs,
            )

        running_overall_score += (
            overall_metrics["F1"]
        ) * images.size(0)
        running_lymph_score += (lymph_metrics["F1"]) * images.size(0)
        running_mono_score += (mono_metrics["F1"]) * images.size(0)

    # Log an example prediction to WandB
    log_data = compose_multitask_log_images(
        images,
        binary_true_masks,
        lymph_true_masks,
        mono_true_masks,
        contour_true_masks,
        overall_pred_probs=inflamm_probs,
        lymph_pred_probs=lymph_probs,
        mono_pred_probs=mono_probs,
        contour_pred_probs=contour_probs,
    )
    wandb_run.log(log_data)

    return {
        "overall_F1": running_overall_score
        / len(validation_loader.sampler),
        "lymph_F1": running_lymph_score
        / len(validation_loader.sampler),
        "mono_F1": running_mono_score
        / len(validation_loader.sampler),
        "val_loss": running_loss / len(validation_loader.sampler),
    }



def multitask_train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn_dict: dict[str, Loss_Function],
    activation_dict: dict[str, torch.nn.Module],
    save_dir: str,
    run_config: dict,
    wandb_run: Optional[wandb.run] = None,
    scheduler: Optional[lr_scheduler.LRScheduler] = None,
) -> torch.nn.Module:
    pprint("Starting training")

    best_val_score = np.inf
    epochs = run_config["epochs"]

    model = freeze_enc(model)
    for epoch in tqdm(
        range(1, epochs + 1), desc="epochs", leave=True
    ):
        pprint(f"EPOCH {epoch}")
        if epoch == run_config["unfreeze_epoch"]:
            model = unfreeze_enc(model)
            pprint("Unfreezing encoder")

        if run_config["dataset_version"] == 2:
            avg_train_loss = train_one_epoch_v2(
                model=model,
                training_loader=train_loader,
                optimizer=optimizer,
                loss_fn_dict=loss_fn_dict,
                run_config=run_config,
                activation_dict=activation_dict,
            )
            avg_scores = validate_one_epoch_v2(
                model=model,
                validation_loader=validation_loader,
                loss_fn_dict=loss_fn_dict,
                run_config=run_config,
                activation_dict=activation_dict,
                wandb_run=wandb_run,
            )
        else:
            avg_train_loss = train_one_epoch(
                model=model,
                training_loader=train_loader,
                optimizer=optimizer,
                loss_fn_dict=loss_fn_dict,
                run_config=run_config,
                activation_dict=activation_dict,
            )
            avg_scores = validate_one_epoch(
                model=model,
                validation_loader=validation_loader,
                loss_fn_dict=loss_fn_dict,
                run_config=run_config,
                activation_dict=activation_dict,
                wandb_run=wandb_run,
            )

        sum_val_score = (
            avg_scores["overall_F1"]
            + avg_scores["lymph_F1"]
            + avg_scores["mono_F1"]
        )

        if scheduler is not None:
            scheduler.step(avg_scores["val_loss"])
            # scheduler.step()

        log_data = {
            "Epoch": epoch,
            "Train loss": avg_train_loss,
            "Val loss": avg_scores["val_loss"],
            "Sum F1 score": sum_val_score,
            "overall F1": avg_scores["overall_F1"],
            "lymph F1": avg_scores["lymph_F1"],
            "mono F1": avg_scores["mono_F1"],
            "Learning rate": optimizer.param_groups[0]["lr"],
        }
        if wandb_run is not None:
            wandb_run.log(log_data)
        pprint(log_data)

        if avg_scores["val_loss"] < best_val_score:
            best_val_score = avg_scores["val_loss"]
            pprint(f"Check Point {epoch}")
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            model_name = f"best.pth"
            model_path = os.path.join(save_dir, model_name)
            torch.save(checkpoint, model_path)

    return model
