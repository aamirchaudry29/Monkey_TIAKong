from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from monkey.config import TrainingIOConfig
from monkey.data.data_utils import (
    erode_mask,
    morphological_post_processing,
)
from monkey.data.dataset import get_detection_dataloaders
from monkey.model.efficientunetb0.architecture import (
    get_efficientunet_b0_MBConv,
)
from monkey.model.utils import (
    get_multiclass_patch_F1_score_batch,
    get_patch_F1_score_batch,
)


def evaluate_multiclass_detection(fold_number: int = 1):
    model = get_efficientunet_b0_MBConv(
        pretrained=False, out_channels=3
    )
    val_fold = fold_number

    checkpoint_path = f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/efficientunetb0_seg_3_channel/fold_{val_fold}/epoch_75.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to("cuda")

    IOconfig = TrainingIOConfig(
        dataset_dir="/mnt/lab-share/Monkey/patches_256/",
        save_dir=f"./",
    )

    # Get dataloaders for task
    _, val_loader = get_detection_dataloaders(
        IOconfig,
        val_fold=val_fold,
        batch_size=32,
        disk_radius=11,
        do_augmentation=False,
        module="multiclass_detection",
        include_background_channel=True,
    )
    # thresholds = [0.3, 0.5, 0.7]
    # thresholds = [0.3]
    # best_thresh_lymph = thresholds[0]
    # best_thresh_mono = thresholds[0]
    # best_F1_lymph = 0.0
    # best_F1_mono = 0.0

    sum_F1_lymph = []
    sum_precision_lymph = []
    sum_recall_lymph = []
    sum_F1_mono = []
    sum_precision_mono = []
    sum_recall_mono = []
    for data in tqdm(val_loader):
        images = data["image"].cuda().float()
        gt_masks = data["mask"].cuda().float()

        with torch.no_grad():
            out = model(images)
            out = torch.softmax(out, dim=1)

            mask_pred = torch.argmax(out, dim=1)
            # to_one_hot
            mask_pred_binary = torch.zeros_like(out).scatter_(
                1, mask_pred.unsqueeze(1), 1.0
            )

        mask_pred_binary = mask_pred_binary.numpy(force=True).astype(
            np.uint8
        )
        mask_pred_binary = morphological_post_processing(
            mask_pred_binary
        )
        lymphocyte_pred = mask_pred_binary[:, 1, :, :]
        monocyte_pred = mask_pred_binary[:, 2, :, :]

        lymph_metrics = get_patch_F1_score_batch(
            lymphocyte_pred, gt_masks[:, 1, :, :], out[:, 1, :, :]
        )
        sum_F1_lymph.append(lymph_metrics["F1"])
        sum_precision_lymph.append(lymph_metrics["Precision"])
        sum_recall_lymph.append(lymph_metrics["Recall"])

        mono_metrics = get_patch_F1_score_batch(
            monocyte_pred, gt_masks[:, 2, :, :], out[:, 2, :, :]
        )
        sum_F1_mono.append(mono_metrics["F1"])
        sum_precision_mono.append(mono_metrics["Precision"])
        sum_recall_mono.append(mono_metrics["Recall"])

    sum_F1_lymph = [x for x in sum_F1_lymph if x is not None]
    sum_precision_lymph = [
        x for x in sum_precision_lymph if x is not None
    ]
    sum_recall_lymph = [x for x in sum_recall_lymph if x is not None]

    sum_F1_mono = [x for x in sum_F1_mono if x is not None]
    sum_precision_mono = [
        x for x in sum_precision_mono if x is not None
    ]
    sum_recall_mono = [x for x in sum_recall_mono if x is not None]

    pprint(f"Lymph F1 {np.mean(sum_F1_lymph)}")
    pprint(f"Lymph Precision {np.mean(sum_precision_lymph)}")
    pprint(f"Lymph Recall {np.mean(sum_recall_lymph)}")

    pprint(f"Mono F1 {np.mean(sum_F1_mono)}")
    pprint(f"Mono Precision {np.mean(sum_precision_mono)}")
    pprint(f"Mono Recall {np.mean(sum_recall_mono)}")


if __name__ == "__main__":
    for i in range(3, 6):
        pprint(f"Evaluating fold {i}")
        evaluate_multiclass_detection(i)
