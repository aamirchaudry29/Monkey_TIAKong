import json
import os

import albumentations as alb
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from monkey.config import TrainingIOConfig


def load_image(
    file_id: str, IOConfig: TrainingIOConfig
) -> np.ndarray:
    image_name = f"{file_id}.npy"
    image_path = os.path.join(IOConfig.image_dir, image_name)
    image = np.load(image_path)
    return image


def load_mask(file_id: str, IOConfig: TrainingIOConfig) -> np.ndarray:
    mask_name = f"{file_id}.npy"
    mask_path = os.path.join(IOConfig.mask_dir, mask_name)
    mask = np.load(mask_path)
    return mask


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


def augmentation(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Example augmentation code"""
    aug = alb.Compose(
        [
            alb.OneOf(
                [
                    alb.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=(-15, 15),
                        val_shift_limit=15,
                        always_apply=False,
                        p=0.5,
                    ),
                    alb.RGBShift(
                        r_shift_limit=15,
                        g_shift_limit=15,
                        b_shift_limit=15,
                        p=0.5,
                    ),
                ],
                p=1.0,
            ),
            alb.OneOf(
                [
                    alb.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    alb.Sharpen(
                        alpha=(0.1, 0.3), lightness=(1.0, 1.0), p=0.5
                    ),
                    alb.ImageCompression(
                        quality_lower=30, quality_upper=80, p=0.5
                    ),
                ],
                p=0.8,
            ),
            alb.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.2, p=0.5
            ),
            alb.ShiftScaleRotate(
                shift_limit=0.01,
                scale_limit=0.2,
                rotate_limit=180,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.8,
            ),
            alb.Flip(p=0.5),
        ],
        p=0.7,
    )
    transformed = aug(image=img, mask=mask)
    img, mask = transformed["image"], transformed["mask"]
    return img, mask


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
    ):
        self.IOConfig = IOConfig
        self.file_ids = file_ids
        self.phase = phase
        self.do_augment = do_augment

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> dict:
        # Load image and mask
        file_id = self.file_ids[idx]
        image = load_image(file_id, self.IOConfig)
        cell_mask = load_mask(file_id, self.IOConfig)

        # Convert cell class mask to binary mask
        # for overall detection
        cell_binary_mask = class_mask_to_binary(cell_mask)

        # augmentation
        if self.do_augment:
            image, cell_binary_mask = augmentation(
                image, cell_binary_mask
            )

        # HxW -> 1xHxW
        cell_binary_mask = cell_binary_mask[np.newaxis, :, :]
        # HxWx3 -> 3xHxW
        image = np.moveaxis(image, -1, 0)

        data = {
            "id": file_id,
            "image": image,
            "mask": cell_binary_mask,
        }

        return data
