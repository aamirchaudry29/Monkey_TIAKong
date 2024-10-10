import os
import re
from typing import Tuple

import numpy as np

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


def extract_id(file_name: str):
    """
    Give a file name such as 'A_P000001_PAS_CPG.tif',
    Extract the ID: 'A_P000001'
    """
    match = re.match(r"([A-Z]_P\d+)_", file_name, re.IGNORECASE)

    if match:
        return match.group(1)
    else:
        return None


def get_file_names(IOConfig: TrainingIOConfig) -> list[str]:
    """Return all file name from the entire image dataset
    (without extension)
    """
    results = []
    file_names = os.listdir(IOConfig.image_dir)
    for fn in file_names:
        name = os.path.splitext(fn)[0]
        results.append(name)
    return results


def centre_cross_validation_split(
    file_ids: list[str], test_centre: str = "A"
) -> dict:
    """
    Split files for cross validation based on centres.
    Centres = ["A", "B", "C", "D"]
    Example output:
        {
            "train_file_ids": [
                "A_1000", "B_1000", ...
            ],
            "val_file_ids": [
                "D_1000", "D_1001", ...
            ]
        }
    """
    test_centre = test_centre.upper()
    if test_centre not in ["A", "B", "C", "D"]:
        raise ValueError(f"Invalid test centre {test_centre}")

    train_file_ids = []
    val_file_ids = []

    for id in file_ids:
        if id[0] != test_centre:
            train_file_ids.append(id)
        else:
            val_file_ids.append(id)

    split = {
        "train_file_ids": train_file_ids,
        "val_file_ids": val_file_ids,
    }
    return split


def imagenet_denormalise(img: np.ndarray) -> np.ndarray:
    """Normalize RGB image to ImageNet mean and std"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std) + mean
    return img


def imagenet_normalise(img: np.ndarray) -> np.ndarray:
    """Revert ImageNet normalized RGB"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img - mean
    img = img / std
    return img
