import numpy as np
import wandb


def compose_log_images(
    images: np.ndarray,
    true_masks: np.ndarray,
    pred_probs: np.ndarray,
    pred_masks: np.ndarray,
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
