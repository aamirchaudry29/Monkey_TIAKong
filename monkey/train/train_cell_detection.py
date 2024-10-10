import math
import os
from pprint import pprint
from typing import Optional

import numpy as np
import segmentation_models_pytorch as smp
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
    for i, data in training_loader:
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
    activation_function,
):
    running_dice = 0.0
    model.eval()
    for i, data in validation_loader:
        images, true_masks = (
            data["image"].cuda().float(),
            data["mask"].cuda().int(),
        )
        with torch.no_grad():
            logits_pred = model(images)
            mask_pred = activation_function(logits_pred)

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
    save_checkpoint: bool = True,
    scheduler: Optional[lr_scheduler.LRScheduler] = None,
):
    pass
