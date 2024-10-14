import logging
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

from monkey.model.loss_functions import compute_dice_coefficient


def train_one_epoch(
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn,
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
        logits_pred = torch.relu(logits_pred)

        loss = loss_fn.compute_loss(logits_pred, true_labels)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(training_loader.sampler)


def validate_one_epoch(
    model: nn.Module,
    validation_loader: DataLoader,
):
    running_dice = 0.0
    model.eval()
    for i, data in enumerate(
        tqdm(validation_loader, desc="validation", leave=False)
    ):
        images, true_masks = (
            data["image"].cuda().float(),
            data["mask"].cuda().int(),
        )
        with torch.no_grad():
            logits_pred = model(images)
            mask_pred = torch.sigmoid(logits_pred)

            threshold = 0.5
            mask_pred_binary = (mask_pred > threshold).float()

            dice_score = compute_dice_coefficient(
                mask_pred_binary, true_masks, multiclass=False
            )

        running_dice += dice_score * images.size(0)

    avg_dice = running_dice / len(validation_loader.sampler)
    return avg_dice


def train_det_net(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn,
    save_dir: str,
    epochs: int,
    wandb_run: wandb.run,
    scheduler: Optional[lr_scheduler.LRScheduler] = None,
) -> torch.nn.Module:
    logging.info("Starting training")

    best_val_score = -np.inf

    for epoch in tqdm(range(epochs), desc="epochs", leave=True):
        logging.info(f"EPOCH {epoch}")

        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn
        )
        avg_score = validate_one_epoch(model, validation_loader)

        if scheduler is not None:
            scheduler.step(avg_score)
        log_data = {
            "Epoch": epoch,
            "Train loss": avg_train_loss,
            "Val score": avg_score,
            "Learning rate": optimizer.param_groups[0]["lr"],
        }
        wandb_run.log(log_data)
        logging.info(log_data)

        if avg_score > best_val_score:
            best_val_score = avg_score
            pprint(f"Check Point {epoch}")
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            model_name = f"epoch_{epoch}.pth"
            model_path = os.path.join(save_dir, model_name)
            torch.save(checkpoint, model_path)

    return model
