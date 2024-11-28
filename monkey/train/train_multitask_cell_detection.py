import os
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
from monkey.model.utils import get_multiclass_patch_F1_score_batch
from monkey.train.utils import compose_multitask_log_images


def hovernext_train_one_epoch(
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
        # contour_masks = data["contour_mask"].cuda().float()
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

        head_1_logits = logits_pred[:, 0:1, :, :]
        head_2_logits = logits_pred[:, 1:2, :, :]
        head_3_logits = logits_pred[:, 2:3, :, :]

        pred_1 = activation_dict["head_1"](head_1_logits)
        pred_2 = activation_dict["head_2"](head_2_logits)
        pred_3 = activation_dict["head_3"](head_3_logits)

        loss_1 = loss_fn_dict["head_1"].compute_loss(
            pred_1, binary_true_masks
        )
        loss_2 = loss_fn_dict["head_2"].compute_loss(
            pred_2, lymph_true_masks
        )
        loss_3 = loss_fn_dict["head_3"].compute_loss(
            pred_3, mono_true_masks
        )

        sum_loss = loss_1 + loss_2 + loss_3
        sum_loss.backward()
        optimizer.step()

        epoch_loss += sum_loss.item() * images.size(0)

    return epoch_loss / len(training_loader.sampler)


def hovernext_validate_one_epoch(
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
        # contour_masks = data["contour_mask"].cuda().float()
        lymph_true_masks = (
            data["class_mask"][:, 0:1, :, :].cuda().float()
        )
        mono_true_masks = (
            data["class_mask"][:, 1:2, :, :].cuda().float()
        )

        with torch.no_grad():
            logits_pred = model(images)
            head_1_logits = logits_pred[:, 0:1, :, :]
            head_2_logits = logits_pred[:, 1:2, :, :]
            head_3_logits = logits_pred[:, 2:3, :, :]

            pred_1 = activation_dict["head_1"](head_1_logits)
            pred_2 = activation_dict["head_2"](head_2_logits)
            pred_3 = activation_dict["head_3"](head_3_logits)

            loss_1 = loss_fn_dict["head_1"].compute_loss(
                pred_1, binary_true_masks
            )
            loss_2 = loss_fn_dict["head_2"].compute_loss(
                pred_2, lymph_true_masks
            )
            loss_3 = loss_fn_dict["head_3"].compute_loss(
                pred_3, mono_true_masks
            )
            sum_loss = loss_1.item() + loss_2.item() + loss_3.item()
            running_loss += sum_loss * images.size(0)

            overall_pred_binary = (pred_1 > 0.5).float()
            lymph_pred_binary = (pred_2 > 0.5).float()
            mono_pred_binary = (pred_3 > 0.5).float()

            # Compute detection F1 score
            overall_metrics = get_multiclass_patch_F1_score_batch(
                overall_pred_binary,
                binary_true_masks,
                pred_1,
            )
            lymph_metrics = get_multiclass_patch_F1_score_batch(
                lymph_pred_binary, lymph_true_masks, pred_2
            )
            mono_metrics = get_multiclass_patch_F1_score_batch(
                mono_pred_binary, mono_true_masks, pred_3
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
        None,
        overall_pred_probs=pred_1,
        lymph_pred_probs=pred_2,
        mono_pred_probs=pred_3,
        contour_pred_probs=None,
    )
    wandb_run.log(log_data)

    # avg_score = running_val_score / len(validation_loader.sampler)

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
        # contour_masks = data["contour_mask"].cuda().float()
        # head_1_true_masks = torch.concatenate(
        #     (binary_true_masks, contour_masks), dim=1
        # )
        lymph_true_masks = (
            data["class_mask"][:, 0:1, :, :].cuda().float()
        )
        mono_true_masks = (
            data["class_mask"][:, 1:2, :, :].cuda().float()
        )

        optimizer.zero_grad()

        logits_pred = model(images)

        head_1_logits = logits_pred["head_1"]
        head_2_logits = logits_pred["head_2"]
        # head_3_logits = logits_pred["head_3"]

        pred_1 = activation_dict["head_1"](head_1_logits)
        pred_2 = activation_dict["head_2"](head_2_logits)
        # pred_3 = activation_dict["head_3"](head_3_logits)

        # loss_1 = loss_fn_dict["head_1"].compute_loss(
        #     pred_1, head_1_true_masks
        # )
        loss_1 = loss_fn_dict["head_1"].compute_loss(
            pred_1, lymph_true_masks
        )
        loss_2 = loss_fn_dict["head_2"].compute_loss(
            pred_2, mono_true_masks
        )
        # loss_3 = loss_fn_dict["head_3"].compute_loss(
        #     pred_3, mono_true_masks
        # )
        # sum_loss = loss_1 + loss_2 + loss_3
        sum_loss = loss_1 + loss_2
        sum_loss.backward()
        optimizer.step()

        epoch_loss += sum_loss.item() * images.size(0)

    return epoch_loss / len(training_loader.sampler)


def validate_one_epoch(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_fn_dict: dict,
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
        # contour_masks = data["contour_mask"].cuda().float()
        lymph_true_masks = (
            data["class_mask"][:, 0:1, :, :].cuda().float()
        )
        mono_true_masks = (
            data["class_mask"][:, 1:2, :, :].cuda().float()
        )

        with torch.no_grad():
            logits_pred = model(images)
            head_1_logits = logits_pred["head_1"]
            head_2_logits = logits_pred["head_2"]
            # head_3_logits = logits_pred["head_3"]

            lymph_probs = activation_dict["head_1"](head_1_logits)
            mono_probs = activation_dict["head_2"](head_2_logits)
            # pred_3 = activation_dict["head_3"](head_3_logits)
            overall_probs = torch.zeros_like(lymph_probs)
            overall_probs += lymph_probs
            overall_probs += mono_probs

            loss_1 = loss_fn_dict["head_1"].compute_loss(
                lymph_probs, lymph_true_masks
            )
            loss_2 = loss_fn_dict["head_2"].compute_loss(
                mono_probs, mono_true_masks
            )
            # loss_3 = loss_fn_dict["head_3"].compute_loss(
            #     pred_3, mono_true_masks
            # )
            # sum_loss = loss_1.item() + loss_2.item() + loss_3.item()
            sum_loss = loss_1.item() + loss_2.item()
            running_loss += sum_loss * images.size(0)

            # overall_pred_binary = (pred_1 > 0.5).float()
            lymph_pred_binary = (lymph_probs > 0.5).float()
            mono_pred_binary = (mono_probs > 0.5).float()
            overall_pred_binary = (overall_probs > 0.5).float()

            # Compute detection F1 score
            overall_metrics = get_multiclass_patch_F1_score_batch(
                overall_pred_binary,
                binary_true_masks,
                overall_probs,
            )
            lymph_metrics = get_multiclass_patch_F1_score_batch(
                lymph_pred_binary, lymph_true_masks, lymph_probs
            )
            mono_metrics = get_multiclass_patch_F1_score_batch(
                mono_pred_binary, mono_true_masks, mono_probs
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
        None,
        overall_pred_probs=overall_probs,
        lymph_pred_probs=lymph_probs,
        mono_pred_probs=mono_probs,
        contour_pred_probs=None,
    )
    if wandb_run is not None:
        wandb_run.log(log_data)

    # avg_score = running_val_score / len(validation_loader.sampler)

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

    best_val_score = -np.inf
    epochs = run_config["epochs"]


    for epoch in tqdm(
        range(1, epochs + 1), desc="epochs", leave=True
    ):
        pprint(f"EPOCH {epoch}")

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
        # avg_train_loss = hovernext_train_one_epoch(
        #     model=model,
        #     training_loader=train_loader,
        #     optimizer=optimizer,
        #     loss_fn_dict=loss_fn_dict,
        #     run_config=run_config,
        #     activation_dict=activation_dict,
        # )
        # avg_scores = hovernext_validate_one_epoch(
        #     model=model,
        #     validation_loader=validation_loader,
        #     loss_fn_dict=loss_fn_dict,
        #     run_config=run_config,
        #     activation_dict=activation_dict,
        #     wandb_run=wandb_run,
        # )

        sum_val_score = (
            avg_scores["overall_F1"]
            + avg_scores["lymph_F1"]
            + avg_scores["mono_F1"]
        )

        if scheduler is not None:
            # scheduler.step(sum_val_score)
            scheduler.step()

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

        if sum_val_score > best_val_score:
            best_val_score = sum_val_score
            pprint(f"Check Point {epoch}")
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            model_name = f"epoch_{epoch}.pth"
            model_path = os.path.join(save_dir, model_name)
            torch.save(checkpoint, model_path)

    return model
