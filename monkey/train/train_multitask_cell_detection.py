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
from monkey.model.loss_functions import Loss_Function, MultiTaskLoss
from monkey.model.utils import (
    EarlyStopper,
    get_multiclass_patch_F1_score_batch,
)
from monkey.train.utils import compose_multitask_log_images
from prediction.utils import multihead_det_post_process_batch


def det_v2_train_one_epoch(
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn_dict: dict[str, Loss_Function],
    run_config: dict,
    activation_dict: dict[str, torch.nn.Module],
    multi_task_loss_instance: Optional[MultiTaskLoss] = None,
):
    seg_loss_weight = run_config["train_aux_loss_weights"][0]
    contour_loss_weight = run_config["train_aux_loss_weights"][1]
    hv_loss_weight = run_config["train_aux_loss_weights"][2]
    epoch_loss = 0.0
    model.train()
    multi_task_loss_instance.train()
    for i, data in enumerate(
        tqdm(training_loader, desc="train", leave=False)
    ):
        images = data["image"].cuda().float()

        inflamm_seg_masks = data["inflamm_mask"].cuda().float()
        lymph_seg_masks = data["lymph_mask"].cuda().float()
        mono_seg_masks = data["mono_mask"].cuda().float()
        inflamm_centroid_masks = (
            data["inflamm_centroid_mask"].cuda().float()
        )
        lymph_centroid_masks = (
            data["lymph_centroid_mask"].cuda().float()
        )
        mono_centroid_masks = (
            data["mono_centroid_mask"].cuda().float()
        )
        inflamm_contour_masks = (
            data["inflamm_contour_mask"].cuda().float()
        )
        lymph_contour_masks = (
            data["lymph_contour_mask"].cuda().float()
        )
        mono_contour_masks = data["mono_contour_mask"].cuda().float()
        inflamm_hv_maps = data["inflamm_hv_map"].cuda().float()
        lymph_hv_maps = data["lymph_hv_map"].cuda().float()
        mono_hv_maps = data["mono_hv_map"].cuda().float()
        # inflamm_weight_masks = data['inflamm_weight_mask'].cuda().float()
        # lymph_weight_masks = data['lymph_weight_mask'].cuda().float()
        # mono_weight_masks = data['mono_weight_mask'].cuda().float()

        optimizer.zero_grad()

        logits_pred = model(images)

        inflamm_seg_logits = logits_pred[:, 0:1, :, :]
        inflamm_contour_logits = logits_pred[:, 1:2, :, :]
        inflamm_centroid_logits = logits_pred[:, 2:3, :, :]
        inflamm_hv_logits = logits_pred[:, 3:5, :, :]

        lymph_seg_logits = logits_pred[:, 3:4, :, :]
        lymph_contour_logits = logits_pred[:, 4:5, :, :]
        lymph_centroid_logits = logits_pred[:, 5:6, :, :]
        lymph_hv_logits = logits_pred[:, 6:8, :, :]

        mono_seg_logits = logits_pred[:, 6:7, :, :]
        mono_contour_logits = logits_pred[:, 7:8, :, :]
        mono_centroid_logits = logits_pred[:, 8:9, :, :]
        mono_hv_logits = logits_pred[:, 9:11, :, :]

        inflamm_seg_pred = activation_dict["head_1"](
            inflamm_seg_logits
        )
        inflamm_contour_pred = activation_dict["head_1"](
            inflamm_contour_logits
        )
        inflamm_centroid_pred = activation_dict["head_1"](
            inflamm_centroid_logits
        )
        inflamm_hv_pred = activation_dict["hv"](inflamm_hv_logits)

        lymph_seg_pred = activation_dict["head_2"](lymph_seg_logits)
        lymph_contour_pred = activation_dict["head_2"](
            lymph_contour_logits
        )
        lymph_centroid_pred = activation_dict["head_2"](
            lymph_centroid_logits
        )
        lymph_hv_pred = activation_dict["hv"](lymph_hv_logits)

        mono_seg_pred = activation_dict["head_3"](mono_seg_logits)
        mono_contour_pred = activation_dict["head_3"](
            mono_contour_logits
        )
        mono_centroid_pred = activation_dict["head_3"](
            mono_centroid_logits
        )
        mono_hv_pred = activation_dict["hv"](mono_hv_logits)

        inflamm_seg_loss = loss_fn_dict["seg_loss"].compute_loss(
            inflamm_seg_pred, inflamm_seg_masks
        )
        inflamm_contour_loss = loss_fn_dict[
            "contour_loss"
        ].compute_loss(inflamm_contour_pred, inflamm_contour_masks)
        # inflamm_centroid_loss = loss_fn_dict["det_loss"].compute_loss(
        #     inflamm_centroid_pred, inflamm_centroid_masks, inflamm_weight_masks
        # )
        inflamm_centroid_loss = loss_fn_dict["det_loss"].compute_loss(
            inflamm_centroid_pred, inflamm_centroid_masks
        )
        inflamm_hv_loss = loss_fn_dict["hv_loss"].compute_loss(
            inflamm_hv_pred, inflamm_hv_maps, inflamm_seg_masks
        )
        lymph_seg_loss = loss_fn_dict["seg_loss"].compute_loss(
            lymph_seg_pred, lymph_seg_masks
        )
        lymph_contour_loss = loss_fn_dict[
            "contour_loss"
        ].compute_loss(lymph_contour_pred, lymph_contour_masks)
        # lymph_centroid_loss = loss_fn_dict["det_loss"].compute_loss(
        #     lymph_centroid_pred, lymph_centroid_masks, lymph_weight_masks
        # )
        lymph_centroid_loss = loss_fn_dict["det_loss"].compute_loss(
            lymph_centroid_pred, lymph_centroid_masks
        )
        lymph_hv_loss = loss_fn_dict["hv_loss"].compute_loss(
            lymph_hv_pred, lymph_hv_maps, lymph_seg_masks
        )

        mono_seg_loss = loss_fn_dict["seg_loss"].compute_loss(
            mono_seg_pred, mono_seg_masks
        )
        mono_contour_loss = loss_fn_dict["contour_loss"].compute_loss(
            mono_contour_pred, mono_contour_masks
        )
        # mono_centroid_loss = loss_fn_dict["det_loss"].compute_loss(
        #     mono_centroid_pred, mono_centroid_masks, mono_weight_masks
        # )
        mono_centroid_loss = loss_fn_dict["det_loss"].compute_loss(
            mono_centroid_pred, mono_centroid_masks
        )
        mono_hv_loss = loss_fn_dict["hv_loss"].compute_loss(
            mono_hv_pred, mono_hv_maps, mono_seg_masks
        )

        loss_1 = (
            seg_loss_weight * inflamm_seg_loss
            + contour_loss_weight * inflamm_contour_loss
            + hv_loss_weight * inflamm_hv_loss
            + inflamm_centroid_loss
        )
        loss_2 = (
            seg_loss_weight * lymph_seg_loss
            + contour_loss_weight * lymph_contour_loss
            + hv_loss_weight * lymph_hv_loss
            + lymph_centroid_loss
        )
        loss_3 = (
            seg_loss_weight * mono_seg_loss
            + contour_loss_weight * mono_contour_loss
            + hv_loss_weight * mono_hv_loss
            + mono_centroid_loss
        )

        # sum_loss = loss_1 + loss_2 + loss_3
        # sum_loss.backward()
        # epoch_loss += sum_loss.item() * images.size(0)
        stack_loss = torch.stack((loss_1, loss_2, loss_3))
        multi_task_loss = multi_task_loss_instance(stack_loss)
        multi_task_loss.backward()
        epoch_loss += multi_task_loss.item() * images.size(0)

        optimizer.step()

    return epoch_loss / len(training_loader.sampler)


def det_v2_validate_one_epoch(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_fn_dict: dict[str, Loss_Function],
    run_config: dict,
    activation_dict: dict[str, torch.nn.Module],
    wandb_run: Optional[wandb.run] = None,
    multi_task_loss_instance: Optional[MultiTaskLoss] = None,
):
    running_overall_score = 0.0
    running_lymph_score = 0.0
    running_mono_score = 0.0
    running_loss = 0.0

    seg_loss_weight = run_config["val_aux_loss_weights"][0]
    contour_loss_weight = run_config["val_aux_loss_weights"][1]
    hv_loss_weight = run_config["val_aux_loss_weights"][2]
    model.eval()
    multi_task_loss_instance.eval()
    for i, data in enumerate(
        tqdm(validation_loader, desc="validation", leave=False)
    ):
        images = data["image"].cuda().float()

        inflamm_seg_masks = data["inflamm_mask"].cuda().float()
        lymph_seg_masks = data["lymph_mask"].cuda().float()
        mono_seg_masks = data["mono_mask"].cuda().float()
        inflamm_centroid_masks = (
            data["inflamm_centroid_mask"].cuda().float()
        )
        lymph_centroid_masks = (
            data["lymph_centroid_mask"].cuda().float()
        )
        mono_centroid_masks = (
            data["mono_centroid_mask"].cuda().float()
        )
        inflamm_contour_masks = (
            data["inflamm_contour_mask"].cuda().float()
        )
        lymph_contour_masks = (
            data["lymph_contour_mask"].cuda().float()
        )
        mono_contour_masks = data["mono_contour_mask"].cuda().float()
        inflamm_hv_maps = data["inflamm_hv_map"].cuda().float()
        lymph_hv_maps = data["lymph_hv_map"].cuda().float()
        mono_hv_maps = data["mono_hv_map"].cuda().float()
        # inflamm_weight_masks = data['inflamm_weight_mask'].cuda().float()
        # lymph_weight_masks = data['lymph_weight_mask'].cuda().float()
        # mono_weight_masks = data['mono_weight_mask'].cuda().float()

        with torch.no_grad():
            logits_pred = model(images)

            inflamm_seg_logits = logits_pred[:, 0:1, :, :]
            inflamm_contour_logits = logits_pred[:, 1:2, :, :]
            inflamm_centroid_logits = logits_pred[:, 2:3, :, :]
            inflamm_hv_logits = logits_pred[:, 3:5, :, :]

            lymph_seg_logits = logits_pred[:, 3:4, :, :]
            lymph_contour_logits = logits_pred[:, 4:5, :, :]
            lymph_centroid_logits = logits_pred[:, 5:6, :, :]
            lymph_hv_logits = logits_pred[:, 6:8, :, :]

            mono_seg_logits = logits_pred[:, 6:7, :, :]
            mono_contour_logits = logits_pred[:, 7:8, :, :]
            mono_centroid_logits = logits_pred[:, 8:9, :, :]
            mono_hv_logits = logits_pred[:, 9:11, :, :]

            inflamm_seg_pred = activation_dict["head_1"](
                inflamm_seg_logits
            )
            inflamm_contour_pred = activation_dict["head_1"](
                inflamm_contour_logits
            )
            inflamm_centroid_pred = activation_dict["head_1"](
                inflamm_centroid_logits
            )
            inflamm_hv_pred = activation_dict["hv"](inflamm_hv_logits)

            lymph_seg_pred = activation_dict["head_2"](
                lymph_seg_logits
            )
            lymph_contour_pred = activation_dict["head_2"](
                lymph_contour_logits
            )
            lymph_centroid_pred = activation_dict["head_2"](
                lymph_centroid_logits
            )
            lymph_hv_pred = activation_dict["hv"](lymph_hv_logits)

            mono_seg_pred = activation_dict["head_3"](mono_seg_logits)
            mono_contour_pred = activation_dict["head_3"](
                mono_contour_logits
            )
            mono_centroid_pred = activation_dict["head_3"](
                mono_centroid_logits
            )
            mono_hv_pred = activation_dict["hv"](mono_hv_logits)

            inflamm_seg_loss = loss_fn_dict["seg_loss"].compute_loss(
                inflamm_seg_pred, inflamm_seg_masks
            )
            inflamm_contour_loss = loss_fn_dict[
                "contour_loss"
            ].compute_loss(
                inflamm_contour_pred, inflamm_contour_masks
            )
            # inflamm_centroid_loss = loss_fn_dict["det_loss"].compute_loss(
            #     inflamm_centroid_pred, inflamm_centroid_masks, inflamm_weight_masks
            # )
            inflamm_centroid_loss = loss_fn_dict[
                "det_loss"
            ].compute_loss(
                inflamm_centroid_pred, inflamm_centroid_masks
            )
            inflamm_hv_loss = loss_fn_dict["hv_loss"].compute_loss(
                inflamm_hv_pred, inflamm_hv_maps, inflamm_seg_masks
            )

            lymph_seg_loss = loss_fn_dict["seg_loss"].compute_loss(
                lymph_seg_pred, lymph_seg_masks
            )
            lymph_contour_loss = loss_fn_dict[
                "contour_loss"
            ].compute_loss(lymph_contour_pred, lymph_contour_masks)
            # lymph_centroid_loss = loss_fn_dict["det_loss"].compute_loss(
            #     lymph_centroid_pred, lymph_centroid_masks, lymph_weight_masks
            # )
            lymph_centroid_loss = loss_fn_dict[
                "det_loss"
            ].compute_loss(lymph_centroid_pred, lymph_centroid_masks)
            lymph_hv_loss = loss_fn_dict["hv_loss"].compute_loss(
                lymph_hv_pred, lymph_hv_maps, lymph_seg_masks
            )

            mono_seg_loss = loss_fn_dict["seg_loss"].compute_loss(
                mono_seg_pred, mono_seg_masks
            )
            mono_contour_loss = loss_fn_dict[
                "contour_loss"
            ].compute_loss(mono_contour_pred, mono_contour_masks)
            # mono_centroid_loss = loss_fn_dict["det_loss"].compute_loss(
            #     mono_centroid_pred, mono_centroid_masks, mono_weight_masks
            # )
            mono_centroid_loss = loss_fn_dict[
                "det_loss"
            ].compute_loss(mono_centroid_pred, mono_centroid_masks)
            mono_hv_loss = loss_fn_dict["hv_loss"].compute_loss(
                mono_hv_pred, mono_hv_maps, mono_seg_masks
            )

            loss_1 = (
                seg_loss_weight * inflamm_seg_loss
                + contour_loss_weight * inflamm_contour_loss
                + hv_loss_weight * inflamm_hv_loss
                + inflamm_centroid_loss
            )
            loss_2 = (
                seg_loss_weight * lymph_seg_loss
                + contour_loss_weight * lymph_contour_loss
                + hv_loss_weight * lymph_hv_loss
                + lymph_centroid_loss
            )
            loss_3 = (
                seg_loss_weight * mono_seg_loss
                + contour_loss_weight * mono_contour_loss
                + hv_loss_weight * mono_hv_loss
                + mono_centroid_loss
            )

            # sum_loss = loss_1 + loss_2 + loss_3
            # running_loss += sum_loss.item() * images.size(0)

            stack_loss = torch.stack((loss_1, loss_2, loss_3))
            multi_task_loss = multi_task_loss_instance(stack_loss)
            running_loss += multi_task_loss.item() * images.size(0)

            binary_masks = multihead_det_post_process_batch(
                inflamm_prob=inflamm_centroid_pred,
                lymph_prob=lymph_centroid_pred,
                mono_prob=mono_centroid_pred,
                thresholds=run_config["peak_thresholds"],
                min_distances=[11, 11, 11],
            )
            overall_pred_binary = binary_masks["inflamm_mask"]
            lymph_pred_binary = binary_masks["lymph_mask"]
            mono_pred_binary = binary_masks["mono_mask"]

            # Compute detection F1 score
            overall_metrics = get_multiclass_patch_F1_score_batch(
                overall_pred_binary[:, np.newaxis, :, :],
                inflamm_centroid_masks,
                [5],
                inflamm_centroid_pred,
            )
            lymph_metrics = get_multiclass_patch_F1_score_batch(
                lymph_pred_binary[:, np.newaxis, :, :],
                lymph_centroid_masks,
                [4],
                lymph_centroid_pred,
            )
            mono_metrics = get_multiclass_patch_F1_score_batch(
                mono_pred_binary[:, np.newaxis, :, :],
                mono_centroid_masks,
                [5],
                mono_centroid_pred,
            )

        running_overall_score += (
            overall_metrics["F1"]
        ) * images.size(0)
        running_lymph_score += (lymph_metrics["F1"]) * images.size(0)
        running_mono_score += (mono_metrics["F1"]) * images.size(0)

    # Log an example prediction to WandB
    log_data = compose_multitask_log_images(
        images,
        inflamm_centroid_masks,
        lymph_centroid_masks,
        mono_centroid_masks,
        None,
        overall_pred_probs=inflamm_centroid_pred,
        lymph_pred_probs=lymph_centroid_pred,
        mono_pred_probs=mono_centroid_pred,
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

            binary_masks = multihead_det_post_process_batch(
                inflamm_prob=pred_1,
                lymph_prob=pred_2,
                mono_prob=pred_3,
                thresholds=run_config["peak_thresholds"],
                min_distances=[20, 16, 20],
            )
            overall_pred_binary = binary_masks["inflamm_mask"]
            lymph_pred_binary = binary_masks["lymph_mask"]
            mono_pred_binary = binary_masks["mono_mask"]

            # Compute detection F1 score
            overall_metrics = get_multiclass_patch_F1_score_batch(
                overall_pred_binary[:, np.newaxis, :, :],
                binary_true_masks,
                [5],
                pred_1,
            )
            lymph_metrics = get_multiclass_patch_F1_score_batch(
                lymph_pred_binary[:, np.newaxis, :, :],
                lymph_true_masks,
                [4],
                pred_2,
            )
            mono_metrics = get_multiclass_patch_F1_score_batch(
                mono_pred_binary[:, np.newaxis, :, :],
                mono_true_masks,
                [5],
                pred_3,
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

        head_1_logits = logits_pred[:, 0:1, :, :]
        head_2_logits = logits_pred[:, 1:2, :, :]
        head_3_logits = logits_pred[:, 2:3, :, :]

        lymph_probs = activation_dict["head_2"](head_2_logits)
        mono_probs = activation_dict["head_3"](head_3_logits)
        inflamm_probs = activation_dict["head_1"](head_1_logits)

        loss_1 = loss_fn_dict["head_1"].compute_loss(
            lymph_probs, lymph_true_masks
        )
        loss_2 = loss_fn_dict["head_2"].compute_loss(
            mono_probs, mono_true_masks
        )
        loss_3 = loss_fn_dict["head_1"].compute_loss(
            inflamm_probs, binary_true_masks
        )

        sum_loss = loss_1 + loss_2 + loss_3
        # sum_loss = loss_1 + loss_2
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
            head_1_logits = logits_pred[:, 0:1, :, :]
            head_2_logits = logits_pred[:, 1:2, :, :]
            head_3_logits = logits_pred[:, 2:3, :, :]

            lymph_probs = activation_dict["head_1"](head_1_logits)
            mono_probs = activation_dict["head_2"](head_2_logits)
            inflamm_probs = activation_dict["head_1"](head_1_logits)

            loss_1 = loss_fn_dict["head_1"].compute_loss(
                lymph_probs, lymph_true_masks
            )
            loss_2 = loss_fn_dict["head_2"].compute_loss(
                mono_probs, mono_true_masks
            )
            loss_3 = loss_fn_dict["head_1"].compute_loss(
                inflamm_probs, binary_true_masks
            )

            sum_loss = loss_1.item() + loss_2.item() + loss_3.item()
            # sum_loss = loss_1.item() + loss_2.item()
            running_loss += sum_loss * images.size(0)

            # overall_pred_binary = (pred_1 > 0.5).float()
            lymph_pred_binary = (lymph_probs > 0.5).float()
            mono_pred_binary = (mono_probs > 0.5).float()
            inflamm_pred_binary = (inflamm_probs > 0.5).float()

            # Compute detection F1 score
            overall_metrics = get_multiclass_patch_F1_score_batch(
                inflamm_pred_binary,
                binary_true_masks,
                [5],
                inflamm_probs,
            )
            lymph_metrics = get_multiclass_patch_F1_score_batch(
                lymph_pred_binary, lymph_true_masks, [4], lymph_probs
            )
            mono_metrics = get_multiclass_patch_F1_score_batch(
                mono_pred_binary, mono_true_masks, [5], mono_probs
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
        overall_pred_probs=inflamm_probs,
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
    multi_task_loss_instance: Optional[MultiTaskLoss] = None,
) -> torch.nn.Module:
    pprint("Starting training")

    best_val_score = np.inf
    best_f1_score = 0.0
    epochs = run_config["epochs"]
    early_stopper = EarlyStopper(patience=20, min_delta=0)

    model = freeze_enc(model)
    for epoch in tqdm(
        range(1, epochs + 1), desc="epochs", leave=True
    ):
        pprint(f"EPOCH {epoch}")
        if epoch == run_config["unfreeze_epoch"]:
            model = unfreeze_enc(model)
            pprint("Unfreezing encoder")

        # avg_train_loss = train_one_epoch(
        #     model=model,
        #     training_loader=train_loader,
        #     optimizer=optimizer,
        #     loss_fn_dict=loss_fn_dict,
        #     run_config=run_config,
        #     activation_dict=activation_dict,
        # )
        # avg_scores = validate_one_epoch(
        #     model=model,
        #     validation_loader=validation_loader,
        #     loss_fn_dict=loss_fn_dict,
        #     run_config=run_config,
        #     activation_dict=activation_dict,
        #     wandb_run=wandb_run,
        # )

        if run_config["det_version"] == 2:
            avg_train_loss = det_v2_train_one_epoch(
                model=model,
                training_loader=train_loader,
                optimizer=optimizer,
                loss_fn_dict=loss_fn_dict,
                run_config=run_config,
                activation_dict=activation_dict,
                multi_task_loss_instance=multi_task_loss_instance,
            )
            avg_scores = det_v2_validate_one_epoch(
                model=model,
                validation_loader=validation_loader,
                loss_fn_dict=loss_fn_dict,
                run_config=run_config,
                activation_dict=activation_dict,
                wandb_run=wandb_run,
                multi_task_loss_instance=multi_task_loss_instance,
            )
        else:
            avg_train_loss = hovernext_train_one_epoch(
                model=model,
                training_loader=train_loader,
                optimizer=optimizer,
                loss_fn_dict=loss_fn_dict,
                run_config=run_config,
                activation_dict=activation_dict,
            )
            avg_scores = hovernext_validate_one_epoch(
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
            # scheduler.step(avg_scores["val_loss"])
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

        if sum_val_score > best_f1_score:
            best_f1_score = sum_val_score
            pprint(f"Best F1 Score Check Point {epoch}")
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            model_name = f"best_f1.pth"
            model_path = os.path.join(save_dir, model_name)
            torch.save(checkpoint, model_path)

        if avg_scores["val_loss"] < best_val_score:
            best_val_score = avg_scores["val_loss"]
            pprint(f"Best Val Score Check Point {epoch}")
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            model_name = f"best_val.pth"
            model_path = os.path.join(save_dir, model_name)
            torch.save(checkpoint, model_path)

        if early_stopper.early_stop(avg_scores["val_loss"]):
            pprint("Early stopping")
            break

    return model
