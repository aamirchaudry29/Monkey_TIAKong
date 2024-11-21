import numpy as np
import wandb
from torch import Tensor


def compose_log_images(
    images: Tensor,
    true_masks: Tensor,
    pred_probs: Tensor,
    pred_masks: Tensor,
    module: str,
    has_background_channel: bool,
) -> dict:
    if module == "detection":
        log_data = {}
        log_data["images"] = wandb.Image(images[0, :3, :, :].cpu())

        if has_background_channel:
            log_data["masks"] = {
                "true": wandb.Image(
                    true_masks[1].float().cpu(), mode="L"
                ),
                "pred_probs": wandb.Image(
                    pred_probs[0, 1, :, :].float().cpu()
                ),
                "Final_pred": wandb.Image(
                    pred_masks[0, 1, :, :].float().cpu(),
                    mode="L",
                ),
            }
        else:
            log_data["masks"] = {
                "true": wandb.Image(
                    true_masks[0].float().cpu(), mode="L"
                ),
                "pred_probs": wandb.Image(
                    pred_probs[0, 0, :, :].float().cpu()
                ),
                "Final_pred": wandb.Image(
                    pred_masks[0, 0, :, :].float().cpu(),
                    mode="L",
                ),
            }
    elif module == "multiclass_detection":
        log_data = {}
        log_data["images"] = wandb.Image(images[0, :3, :, :].cpu())

        if has_background_channel:
            log_data["masks"] = {
                "other": wandb.Image(
                    true_masks[0, 0, :, :].float().cpu(), mode="L"
                ),
                "true_lymph": wandb.Image(
                    true_masks[0, 1, :, :].float().cpu(), mode="L"
                ),
                "true_mono": wandb.Image(
                    true_masks[0, 2, :, :].float().cpu(), mode="L"
                ),
                "pred_lymph_probs": wandb.Image(
                    pred_probs[0, 1, :, :].float().cpu()
                ),
                "pred_mono_probs": wandb.Image(
                    pred_probs[0, 2, :, :].float().cpu()
                ),
                "pred_other_probs": wandb.Image(
                    true_masks[0, 0, :, :].float().cpu()
                ),
            }
        else:
            log_data["masks"] = {
                "true_lymph": wandb.Image(
                    true_masks[0, 0, :, :].float().cpu(), mode="L"
                ),
                "true_mono": wandb.Image(
                    true_masks[0, 1, :, :].float().cpu(), mode="L"
                ),
                "pred_lymph_probs": wandb.Image(
                    pred_probs[0, 0, :, :].float().cpu()
                ),
                "pred_mono_probs": wandb.Image(
                    pred_probs[0, 1, :, :].float().cpu()
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
    contour_true_masks: Tensor,
    overall_pred_probs: Tensor,
    lymph_pred_probs: Tensor,
    mono_pred_probs: Tensor,
    contour_pred_probs: Tensor,
) -> dict:

    log_data = {}
    log_data["images"] = wandb.Image(images[0, :3, :, :].cpu())
    log_data["masks"] = {
        "true_overall": wandb.Image(
            overall_true_masks[0, 0, :, :].float().cpu(), mode="L"
        ),
        "true_contour": wandb.Image(
            contour_true_masks[0, 0, :, :].float().cpu(), mode="L"
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
        "pred_contour_probs": wandb.Image(
            contour_pred_probs[0, 0, :, :].float().cpu()
        ),
    }

    return log_data
