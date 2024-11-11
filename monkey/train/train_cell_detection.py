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

from monkey.model.loss_functions import Loss_Function
from monkey.model.utils import get_multiclass_patch_F1_score_batch


def train_one_epoch(
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Loss_Function,
    module: str,
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
    module: str,
    wandb_run: Optional[wandb.run] = None,
    activation: torch.nn = torch.nn.Sigmoid,
):
    running_val_score = 0.0
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
            mask_pred = activation(logits_pred)

            mask_pred_binary = (mask_pred > 0.5).float()

            # Compute detection F1 score
            metrics = get_multiclass_patch_F1_score_batch(
                mask_pred_binary, true_masks, logits_pred
            )

        running_val_score += metrics["F1"] * images.size(0)

    # Log an example prediction to WandB
    if wandb_run is not None:
        if module == "detection":
            log_data = {
                "images": wandb.Image(images[0, :3, :, :].cpu()),
                "masks": {
                    "true": wandb.Image(
                        true_masks[0].float().cpu(), mode="L"
                    ),
                    "pred_probs": wandb.Image(
                        logits_pred[0, 0, :, :].float().cpu()
                    ),
                    "Final_pred": wandb.Image(
                        mask_pred_binary[0, 0, :, :].float().cpu(),
                        mode="L",
                    ),
                },
            }
        elif module == "multiclass_detection":
            log_data = {
                "images": wandb.Image(images[0, :3, :, :].cpu()),
                "masks": {
                    "true_lymph": wandb.Image(
                        true_masks[0, 0, :, :].float().cpu(), mode="L"
                    ),
                    "true_mono": wandb.Image(
                        true_masks[0, 1, :, :].float().cpu(), mode="L"
                    ),
                    "pred_lymph_probs": wandb.Image(
                        logits_pred[0, 0, :, :].float().cpu()
                    ),
                    "pred_mono_probs": wandb.Image(
                        logits_pred[0, 1, :, :].float().cpu()
                    ),
                },
            }
        else:
            log_data = {}
        wandb_run.log(log_data)

    avg_score = running_val_score / len(validation_loader.sampler)
    return avg_score


def train_det_net(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Loss_Function,
    activation: torch.nn,
    save_dir: str,
    epochs: int,
    module: str,
    wandb_run: Optional[wandb.run] = None,
    scheduler: Optional[lr_scheduler.LRScheduler] = None,
) -> torch.nn.Module:
    pprint("Starting training")

    best_val_score = -np.inf

    for epoch in tqdm(
        range(1, epochs + 1), desc="epochs", leave=True
    ):
        pprint(f"EPOCH {epoch}")

        avg_train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            module,
            activation,
        )
        avg_score = validate_one_epoch(
            model, validation_loader, module, wandb_run, activation
        )

        if scheduler is not None:
            scheduler.step(avg_score)

        log_data = {
            "Epoch": epoch,
            "Train loss": avg_train_loss,
            "Val score": avg_score,
            "Learning rate": optimizer.param_groups[0]["lr"],
        }
        if wandb_run is not None:
            wandb_run.log(log_data)
        pprint(log_data)

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
