import json
import os

import albumentations as alb
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from monkey.config import TrainingIOConfig
from monkey.data.augmentation import get_augmentation
from monkey.data.data_utils import (
    centre_cross_validation_split,
    dilate_mask,
    generate_regression_map,
    get_file_names,
    imagenet_normalise,
    load_image,
    load_mask,
)


def class_mask_to_binary(class_mask: np.ndarray) -> np.ndarray:
    """Converts 2D cell class mask to binary mask
    Example:
        [1,0,0
         0,0,2
         0,0,1]
         ->
        [1,0,0
         0,0,1
         0,0,1]
    """
    binary_mask = np.zeros_like(class_mask)
    binary_mask[class_mask != 0] = 1
    return binary_mask


class InflammatoryDataset(Dataset):
    """Dataset for overall cell detection
    Detecting Lymphocytes and Monocytes
    Data: RGB image and binary cell mask
    """

    def __init__(
        self,
        IOConfig: TrainingIOConfig,
        file_ids: list,
        phase: str = "train",
        do_augment: bool = True,
        disk_radius: int = 11,
        module: str = "detection",
    ):
        self.IOConfig = IOConfig
        self.file_ids = file_ids
        self.phase = phase
        self.do_augment = do_augment
        self.disk_radius = disk_radius
        self.module = module

        if self.do_augment:
            self.augmentation = get_augmentation(
                module=self.module, gt_type="mask", aug_prob=0.7
            )

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> dict:
        # Load image and mask
        file_id = self.file_ids[idx]
        image = load_image(file_id, self.IOConfig)
        image = image / 255
        cell_mask = load_mask(file_id, self.IOConfig)

        # Convert cell class mask to binary mask
        # for overall detection
        cell_binary_mask = class_mask_to_binary(cell_mask)
        # augmentation
        if self.do_augment:
            augmented_data = self.augmentation(
                image=image, mask=cell_binary_mask
            )
            image, cell_binary_mask = (
                augmented_data["image"],
                augmented_data["mask"],
            )

        # Dilate cell centroids
        cell_binary_mask = dilate_mask(
            cell_binary_mask, disk_radius=self.disk_radius
        )
        # Generate regression map
        cell_map = generate_regression_map(
            binary_mask=cell_binary_mask, d_thresh=7, alpha=5, scale=3
        )

        # HxW -> 1xHxW
        cell_map = cell_map[np.newaxis, :, :]
        # HxWx3 -> 3xHxW
        image = imagenet_normalise(image)
        image = np.moveaxis(image, -1, 0)

        data = {
            "id": file_id,
            "image": image,
            "mask": cell_map,
        }

        return data


def get_dataloaders(
    IOConfig: TrainingIOConfig,
    val_fold=1,
    task=1,
    batch_size=4,
    disk_radius=11,
    module: str = "segmentation",
    do_augmentation: bool = False,
):
    """Get training and validation dataloaders
    Task 1: Overall Inflammation cell (MNL) detection
    Task 2: Detect and distinguish monocytes and lymphocytes
    """

    if task not in [1, 2]:
        raise ValueError(f"Task {task} is in invalid")

    if module not in ["detection", "classification", "segmentation"]:
        raise ValueError(f"Module {module} is in invalid")

    file_ids = get_file_names(IOConfig)
    split = centre_cross_validation_split(
        file_ids=file_ids, val_fold=val_fold
    )

    train_dataset = InflammatoryDataset(
        IOConfig=IOConfig,
        file_ids=split["train_file_ids"],
        phase="Train",
        do_augment=do_augmentation,
        disk_radius=disk_radius,
        module=module,
    )
    val_dataset = InflammatoryDataset(
        IOConfig=IOConfig,
        file_ids=split["val_file_ids"],
        phase="test",
        do_augment=False,
        disk_radius=disk_radius,
        module=module,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader, val_loader
