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

from monkey.model.loss_functions import (
    Loss_Function,
    inter_class_exclusion_loss,
    jaccard_loss,
)
from monkey.model.mapde import model
from monkey.model.utils import get_multiclass_patch_F1_score_batch
from monkey.train.utils import (
    compose_log_images,
    compose_multitask_log_images,
)
from prediction.utils import post_process_batch


def train_one_epoch_mapde(
    model: model.MapDe,
    training_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Loss_Function,
    run_config: dict,
):
    epoch_loss = 0.0
    model.train()
    for i, data in enumerate(
        tqdm(training_loader, desc="train", leave=False)
    ):
        images, true_labels = (
            data["image"].cuda().float(),
            data["mask"].cuda().float(),
        )
        images = model.reshape_transform(images)
        true_labels = model.reshape_transform(true_labels)
        true_labels = model.blur_cell_points(true_labels)
        optimizer.zero_grad()

        probs_pred = model(images)

        loss = loss_fn.compute_loss(probs_pred, true_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(training_loader.sampler)


def validate_one_epoch_mapde(
    model: model.MapDe,
    validation_loader: DataLoader,
    run_config: dict,
    loss_fn: Loss_Function,
    wandb_run: Optional[wandb.run] = None,
):
    running_val_score = 0.0
    module = run_config["module"]

    model.eval()
    for i, data in enumerate(
        tqdm(validation_loader, desc="validation", leave=False)
    ):
        images, true_masks = (
            data["image"].cuda().float(),
            data["mask"].cuda().float(),
        )
        images = model.reshape_transform(images)
        true_masks = model.reshape_transform(true_masks)
        true_masks = model.blur_cell_points(true_masks)
        with torch.no_grad():
            logits_pred = model(images)
            probs_pred = model.logits_to_probs(logits_pred)
            pred_cell_masks = model.postproc(logits_pred)
            pred_cell_masks = pred_cell_masks[:, np.newaxis, :, :]
            pred_cell_masks = torch.tensor(
                pred_cell_masks, device="cuda", dtype=float
            )
            pred_cell_masks = model.blur_cell_points(pred_cell_masks)

            # Compute val loss
            val_loss = loss_fn.compute_loss(
                logits_pred, true_masks
            ).item()

        running_val_score += val_loss * images.size(0)

    # Log an example prediction to WandB
    log_data = compose_log_images(
        images=images,
        true_masks=true_masks,
        pred_masks=pred_cell_masks,
        pred_probs=probs_pred,
        module=module,
        has_background_channel=False,
    )
    wandb_run.log(log_data)

    avg_score = running_val_score / len(validation_loader.sampler)
    return avg_score


def train_one_epoch(
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Loss_Function,
    run_config: dict,
    activation: torch.nn = torch.nn.Sigmoid,
):
    epoch_loss = 0.0
    model.train()
    for i, data in enumerate(
        tqdm(training_loader, desc="train", leave=False)
    ):
        images, true_labels = (
            data["image"].cuda().float(),
            data["class_mask"].cuda().float(),
        )
        binary_true_masks = data["binary_mask"].cuda().float()
        optimizer.zero_grad()

        logits_pred = model(images)
        pred_probs = activation(logits_pred)
        class_loss = loss_fn.compute_loss(pred_probs, true_labels)
        inter_class_loss = inter_class_exclusion_loss(
            pred_probs[:, 0, :, :], pred_probs[:, 1, :, :]
        )
        overall_probs = (
            pred_probs[:, 0:1, :, :] + pred_probs[:, 1:2, :, :]
        )
        overall_loss = jaccard_loss(
            overall_probs, binary_true_masks, multiclass=False
        )
        final_loss = (
            2 * overall_loss + 2 * class_loss + inter_class_loss
        )
        final_loss.backward()
        optimizer.step()

        epoch_loss += final_loss.item() * images.size(0)

    return epoch_loss / len(training_loader.sampler)


def validate_one_epoch(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_fn: Loss_Function,
    run_config: dict,
    wandb_run: Optional[wandb.run] = None,
    activation: torch.nn = torch.nn.Sigmoid,
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
        class_masks = data["class_mask"].cuda().float()
        lymph_true_masks = (
            data["class_mask"][:, 0:1, :, :].cuda().float()
        )
        mono_true_masks = (
            data["class_mask"][:, 1:2, :, :].cuda().float()
        )
        with torch.no_grad():
            logits_pred = model(images)
            pred_probs = activation(logits_pred)
            class_loss = loss_fn.compute_loss(
                pred_probs, class_masks
            ).item()
            inter_class_loss = inter_class_exclusion_loss(
                pred_probs[:, 0, :, :], pred_probs[:, 1, :, :]
            ).item()
            overall_probs = (
                pred_probs[:, 0:1, :, :] + pred_probs[:, 1:2, :, :]
            )
            overall_loss = jaccard_loss(
                overall_probs, binary_true_masks, multiclass=False
            ).item()
            final_loss = (
                2 * overall_loss + 2 * class_loss + inter_class_loss
            )

        lymph_pred_binary = post_process_batch(
            pred_probs[:, 0:1, :, :],
            threshold=0.5,
            min_distance=9,
        )
        mono_pred_binary = post_process_batch(
            pred_probs[:, 1:2, :, :],
            threshold=0.5,
            min_distance=13,
        )
        inflamm_pred_probs = (
            pred_probs[:, 0:1, :, :] + pred_probs[:, 1:2, :, :]
        )
        inflamm_pred_binary = lymph_pred_binary + mono_pred_binary

        # Compute detection F1 score
        lymph_metrics = get_multiclass_patch_F1_score_batch(
            lymph_pred_binary[:, np.newaxis, :, :],
            lymph_true_masks,
            [4],
            pred_probs,
        )
        mono_metrics = get_multiclass_patch_F1_score_batch(
            mono_pred_binary[:, np.newaxis, :, :],
            mono_true_masks,
            [10],
            pred_probs,
        )
        inflamm_metrics = get_multiclass_patch_F1_score_batch(
            inflamm_pred_binary[:, np.newaxis, :, :],
            binary_true_masks,
            [7.5],
            inflamm_pred_probs,
        )

        running_overall_score += (
            inflamm_metrics["F1"]
        ) * images.size(0)
        running_lymph_score += (lymph_metrics["F1"]) * images.size(0)
        running_mono_score += (mono_metrics["F1"]) * images.size(0)
        running_loss += final_loss * images.size(0)

    # Log an example prediction to WandB
    log_data = compose_multitask_log_images(
        images,
        binary_true_masks,
        lymph_true_masks,
        mono_true_masks,
        None,
        overall_pred_probs=inflamm_pred_probs,
        lymph_pred_probs=pred_probs[:, 0:1, :, :],
        mono_pred_probs=pred_probs[:, 1:2, :, :],
        contour_pred_probs=None,
    )
    wandb_run.log(log_data)

    avg_loss = running_loss / len(validation_loader.sampler)
    return {
        "overall_F1": running_overall_score
        / len(validation_loader.sampler),
        "lymph_F1": running_lymph_score
        / len(validation_loader.sampler),
        "mono_F1": running_mono_score
        / len(validation_loader.sampler),
        "val_loss": avg_loss,
    }


def train_det_net(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Loss_Function,
    activation: torch.nn,
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
            model,
            train_loader,
            optimizer,
            loss_fn,
            run_config,
            activation,
        )
        avg_score = validate_one_epoch(
            model,
            validation_loader,
            loss_fn,
            run_config,
            wandb_run,
            activation,
        )
        # avg_train_loss = train_one_epoch_mapde(
        #     model,
        #     train_loader,
        #     optimizer,
        #     loss_fn,
        #     run_config,
        # )
        # avg_score = validate_one_epoch_mapde(
        #     model,
        #     validation_loader,
        #     run_config,
        #     loss_fn,
        #     wandb_run,
        # )
        sum_val_score = (
            avg_score["overall_F1"]
            + avg_score["lymph_F1"]
            + avg_score["mono_F1"]
        )

        if scheduler is not None:
            scheduler.step(sum_val_score)

        log_data = {
            "Epoch": epoch,
            "Train loss": avg_train_loss,
            "Val loss": avg_score["val_loss"],
            "Sum F1 score": sum_val_score,
            "overall F1": avg_score["overall_F1"],
            "lymph F1": avg_score["lymph_F1"],
            "mono F1": avg_score["mono_F1"],
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
