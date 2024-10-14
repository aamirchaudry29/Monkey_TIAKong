import os
import re
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader

from monkey.config import TrainingIOConfig
from monkey.data.dataset import InflammatoryDataset


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
    file_ids: list[str], val_fold: int = 1
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
    centres = ["A", "B", "C", "D"]
    if val_fold < 0 or val_fold > 4:
        raise ValueError(f"Invalid test centre {val_fold}")

    test_centre = centres[val_fold]

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


def get_dataloaders(
    IOConfig: TrainingIOConfig, val_fold=1, task=1, batch_size=4
):
    """Get training and validation dataloaders
    Task 1: Overall Inflammation cell (MNL) detection
    Task 2: Detect and distinguish monocytes and lymphocytes
    """

    if task not in [1, 2]:
        raise ValueError(f"Task {task} is in invalid")

    file_ids = get_file_names(IOConfig)
    split = centre_cross_validation_split(
        file_ids=file_ids, val_fold=val_fold
    )

    train_dataset = InflammatoryDataset(
        IOConfig=IOConfig,
        file_ids=split["train_file_ids"],
        phase="Train",
        do_augment=True,
    )
    val_dataset = InflammatoryDataset(
        IOConfig=IOConfig,
        file_ids=split["val_file_ids"],
        phase="test",
        do_augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
