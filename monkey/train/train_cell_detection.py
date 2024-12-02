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

from monkey.model.loss_functions import Loss_Function, MapDe_Loss
from monkey.model.mapde import model
from monkey.model.utils import get_multiclass_patch_F1_score_batch
from monkey.train.utils import compose_log_images


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
            data["mask"].cuda().float(),
        )
        optimizer.zero_grad()

        logits_pred = model(images)
        pred = activation(logits_pred)
        loss = loss_fn.compute_loss(pred, true_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(training_loader.sampler)


def validate_one_epoch(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_fn: Loss_Function,
    run_config: dict,
    wandb_run: Optional[wandb.run] = None,
    activation: torch.nn = torch.nn.Sigmoid,
):
    running_val_score = 0.0
    running_loss = 0.0
    module = run_config["module"]
    if run_config["target_cell_type"] == "inflamm":
        margin = 7.5
    elif run_config["target_cell_type"] == "lymph":
        margin = 4
    else:
        margin = 10

    model.eval()
    for i, data in enumerate(
        tqdm(validation_loader, desc="validation", leave=False)
    ):
        images, true_masks = (
            data["image"].cuda().float(),
            data["mask"].cuda().float(),
        )
        with torch.no_grad():
            logits_pred = model(images)
            pred_probs = activation(logits_pred)
            loss = loss_fn.compute_loss(pred_probs, true_masks).item()

            mask_pred_binary = (pred_probs > 0.5).float()

            # Compute detection F1 score
            metrics = get_multiclass_patch_F1_score_batch(
                mask_pred_binary, true_masks, [margin], pred_probs
            )

        running_val_score += metrics["F1"] * images.size(0)
        running_loss += loss * images.size(0)

    # Log an example prediction to WandB
    log_data = compose_log_images(
        images=images,
        true_masks=true_masks,
        pred_masks=mask_pred_binary,
        pred_probs=pred_probs,
        module=module,
        has_background_channel=False,
    )
    wandb_run.log(log_data)

    avg_score = running_val_score / len(validation_loader.sampler)
    avg_loss = running_loss / len(validation_loader.sampler)
    return {"F1": avg_score, "Val_loss": avg_loss}


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

        if scheduler is not None:
            scheduler.step()

        log_data = {
            "Epoch": epoch,
            "Train loss": avg_train_loss,
            "Val loss": avg_score["Val_loss"],
            "F1": avg_score["F1"],
            "Learning rate": optimizer.param_groups[0]["lr"],
        }
        if wandb_run is not None:
            wandb_run.log(log_data)
        pprint(log_data)

        if avg_score["F1"] > best_val_score:
            best_val_score = avg_score["F1"]
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
