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

from monkey.model.utils import get_classification_metrics


def train_one_epoch(
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn,
    activation: torch.nn = torch.nn.Softmax,
):
    epoch_loss = 0.0
    model.train()
    for i, data in enumerate(
        tqdm(training_loader, desc="train", leave=False)
    ):
        images, true_labels = (
            data["image"].cuda().float(),
            data["label"].cuda().long(),
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
    wandb_run: Optional[wandb.run] = None,
    activation: torch.nn = torch.nn.Softmax,
):
    running_val_score = 0.0
    model.eval()
    for i, data in enumerate(
        tqdm(validation_loader, desc="validation", leave=False)
    ):
        images, true_labels = (
            data["image"].cuda().float(),
            data["label"].cuda().long(),
        )

        pred_labels_list = []
        true_labels_list = true_labels.cpu().tolist()

        with torch.no_grad():
            logits_pred = model(images)
            pred_probs = activation(logits_pred)

            pred_labels = (
                torch.argmax(pred_probs, dim=1).cpu().tolist()
            )
            pred_labels_list.extend(pred_labels)

            metrics = get_classification_metrics(
                true_labels_list, pred_labels_list
            )

        running_val_score += metrics["F1"] * images.size(0)

    avg_f1 = running_val_score / len(validation_loader.sampler)
    return avg_f1


def train_cls_net(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn,
    activation: torch.nn,
    save_dir: str,
    epochs: int,
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
            model, train_loader, optimizer, loss_fn, activation
        )
        avg_score = validate_one_epoch(
            model, validation_loader, wandb_run, activation
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
