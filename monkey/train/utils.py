import numpy as np
import torch
import wandb
from torch import Tensor


def compose_log_images(
    images: Tensor | np.ndarray,
    true_masks: Tensor | np.ndarray,
    pred_probs: Tensor | np.ndarray,
    pred_masks: Tensor | np.ndarray,
    module: str,
    has_background_channel: bool,
) -> dict:

    if torch.is_tensor(images):
        images = images.numpy(force=True)
        images = np.moveaxis(images, 1, 3)
    if torch.is_tensor(true_masks):
        true_masks = true_masks.numpy(force=True)
    if torch.is_tensor(pred_probs):
        pred_probs = pred_probs.numpy(force=True)
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.numpy(force=True)

    if module == "detection":
        log_data = {}
        log_data["images"] = wandb.Image(images[0, :, :, 0:3])

        if has_background_channel:
            log_data["masks"] = {
                "true": wandb.Image(true_masks[1], mode="L"),
                "pred_probs": wandb.Image(pred_probs[0, 1, :, :]),
                "Final_pred": wandb.Image(
                    pred_masks[0, 1, :, :],
                    mode="L",
                ),
            }
        else:
            log_data["masks"] = {
                "true": wandb.Image(true_masks[0], mode="L"),
                "pred_probs": wandb.Image(pred_probs[0, 0, :, :]),
                "Final_pred": wandb.Image(
                    pred_masks[0, 0, :, :],
                    mode="L",
                ),
            }
    elif module == "multiclass_detection":
        log_data = {}
        log_data["images"] = wandb.Image(images[0, :, :, 0:3])

        if has_background_channel:
            log_data["masks"] = {
                "other": wandb.Image(
                    true_masks[0, 0, :, :], mode="L"
                ),
                "true_lymph": wandb.Image(
                    true_masks[0, 1, :, :], mode="L"
                ),
                "true_mono": wandb.Image(
                    true_masks[0, 2, :, :], mode="L"
                ),
                "pred_lymph_probs": wandb.Image(
                    pred_probs[0, 1, :, :]
                ),
                "pred_mono_probs": wandb.Image(
                    pred_probs[0, 2, :, :]
                ),
                "pred_other_probs": wandb.Image(
                    true_masks[0, 0, :, :]
                ),
            }
        else:
            log_data["masks"] = {
                "true_lymph": wandb.Image(
                    true_masks[0, 0, :, :], mode="L"
                ),
                "true_mono": wandb.Image(
                    true_masks[0, 1, :, :], mode="L"
                ),
                "pred_lymph_probs": wandb.Image(
                    pred_probs[0, 0, :, :]
                ),
                "pred_mono_probs": wandb.Image(
                    pred_probs[0, 1, :, :]
                ),
            }

    else:
        log_data = {}

    return log_data


def compose_multitask_log_images(
    images: Tensor,
    overall_true_masks: Tensor,
    lymph_true_masks: Tensor,
    mono_true_masks: Tensor,
    contour_true_masks: Tensor | None,
    overall_pred_probs: Tensor,
    lymph_pred_probs: Tensor,
    mono_pred_probs: Tensor,
    contour_pred_probs: Tensor | None,
) -> dict:

    log_data = {}
    log_data["images"] = wandb.Image(images[0, :3, :, :].cpu())
    log_data["masks"] = {
        "true_overall": wandb.Image(
            overall_true_masks[0, 0, :, :].float().cpu(), mode="L"
        ),
        "true_lymph": wandb.Image(
            lymph_true_masks[0, 0, :, :].float().cpu(), mode="L"
        ),
        "true_mono": wandb.Image(
            mono_true_masks[0, 0, :, :].float().cpu(), mode="L"
        ),
        "pred_lymph_probs": wandb.Image(
            lymph_pred_probs[0, 0, :, :].float().cpu()
        ),
        "pred_mono_probs": wandb.Image(
            mono_pred_probs[0, 0, :, :].float().cpu()
        ),
        "pred_overall_probs": wandb.Image(
            overall_pred_probs[0, 0, :, :].float().cpu()
        ),
    }
    if contour_true_masks is not None:
        log_data["masks"]["true_contour"] = wandb.Image(
            contour_true_masks[0, 0, :, :].float().cpu(), mode="L"
        )
    if contour_pred_probs is not None:
        log_data["masks"]["pred_contour_probs"] = wandb.Image(
            contour_pred_probs[0, 0, :, :].float().cpu(), mode="L"
        )

    return log_data
