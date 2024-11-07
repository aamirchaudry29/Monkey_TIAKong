import json
import os

import numpy as np
import torchvision.transforms.v2 as transforms
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from monkey.config import TrainingIOConfig
from monkey.data.augmentation import get_augmentation
from monkey.data.data_utils import (
    dilate_mask,
    generate_regression_map,
    get_label_from_class_id,
    get_split_from_json,
    imagenet_normalise,
    load_classification_data_example,
    load_image,
    load_mask,
    load_nuclick_annotation,
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


class DetectionDataset(Dataset):
    """Dataset for overall cell detection
    Detecting Lymphocytes and Monocytes
    Data: RGB image and binary cell mask
    """

    def __init__(
        self,
        IOConfig: TrainingIOConfig,
        file_ids: list,
        phase: str = "train",
        do_augment: bool = False,
        disk_radius: int = 9,
        regression_map: bool = False,
        module: str = "detection",
        use_nuclick_masks: bool = False,
    ):
        self.IOConfig = IOConfig
        self.file_ids = file_ids
        self.phase = phase
        self.do_augment = do_augment
        self.disk_radius = disk_radius
        self.module = module
        self.regression_map = regression_map
        self.use_nuclick_masks = use_nuclick_masks

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

        if self.use_nuclick_masks:
            cell_mask = load_nuclick_annotation(
                file_id, self.IOConfig
            )
        else:
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

        if self.use_nuclick_masks:
            cell_map = cell_binary_mask
        else:
            # Dilate cell centroids
            cell_map = dilate_mask(
                cell_binary_mask, disk_radius=self.disk_radius
            )
        # Generate regression map
        if self.regression_map:
            cell_map = generate_regression_map(
                binary_mask=cell_binary_mask,
                d_thresh=7,
                alpha=5,
                scale=1,
            )

        # HxW -> 1xHxW
        cell_map = cell_map[np.newaxis, :, :]
        # HxWx3 -> 3xHxW
        image = image / 255
        image = imagenet_normalise(image)
        image = np.moveaxis(image, -1, 0)

        data = {
            "id": file_id,
            "image": image,
            "mask": cell_map,
        }

        return data


class ClassificationDataset(Dataset):
    """Dataset for cell classification
    Lymphocytes vs Monocytes
    Data: RGB image,binary cell mask, class label
    1 -> lymphocyte, 2 -> monocyte
    """

    def __init__(
        self,
        IOConfig: TrainingIOConfig,
        file_ids: list,
        phase: str = "train",
        do_augment: bool = False,
        module: str = "classification",
        stack_mask: bool = False,
    ):
        self.IOConfig = IOConfig
        self.file_ids = file_ids
        self.phase = phase
        self.do_augment = do_augment
        self.module = module
        self.patch_size = 32
        self.stack_mask = stack_mask

        if self.do_augment:
            self.augmentation = get_augmentation(
                module=self.module, gt_type="mask", aug_prob=0.7
            )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(
                    size=(self.patch_size, self.patch_size)
                ),
            ]
        )

    def crop_transform(self, image):
        out = self.transform(image)
        out = np.array(out)
        return out

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> dict:
        # Load image and mask
        file_id = self.file_ids[idx]
        data = load_classification_data_example(
            file_id, self.IOConfig
        )

        image = data["image"]
        mask = data["mask"]
        label = data["label"]

        image = self.crop_transform(image)
        mask = self.crop_transform(mask)

        # augmentation
        if self.do_augment:
            augmented_data = self.augmentation(image=image, mask=mask)
            image, mask = (
                augmented_data["image"],
                augmented_data["mask"],
            )

        # HxW -> 1xHxW
        mask = mask[np.newaxis, :, :]
        # HxWx3 -> 3xHxW
        image = image / 255
        image = imagenet_normalise(image)
        image = np.moveaxis(image, -1, 0)

        if self.stack_mask:
            image = np.concatenate((image, mask), axis=0)

        data = {
            "id": file_id,
            "image": image,
            "mask": mask,
            "label": label,
        }

        return data


def get_classification_dataloaders(
    IOConfig: TrainingIOConfig,
    val_fold=1,
    batch_size=4,
    do_augmentation: bool = False,
    stack_mask: bool = False,
):
    split = get_split_from_json(IOConfig, val_fold)
    train_file_ids = split["train_file_ids"]
    test_file_ids = split["test_file_ids"]

    train_sampler = get_classification_sampler(
        file_ids=train_file_ids
    )

    print(f"train patches: {len(train_file_ids)}")
    print(f"test patches: {len(test_file_ids)}")

    train_dataset = ClassificationDataset(
        IOConfig=IOConfig,
        file_ids=train_file_ids,
        phase="Train",
        do_augment=do_augmentation,
        stack_mask=stack_mask,
    )
    val_dataset = ClassificationDataset(
        IOConfig=IOConfig,
        file_ids=test_file_ids,
        phase="Test",
        do_augment=False,
        stack_mask=stack_mask,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    return train_loader, val_loader


def get_classification_sampler(file_ids):
    """
    Get Weighted Sampler.
    To balance lymphocyte and monocyte patches.
    """
    class_instances = []
    class_counts = [0, 0]  # [lymphocytes, monocytes]

    for id in file_ids:
        label = get_label_from_class_id(id)
        if label == 0:
            class_instances.append(0)
            class_counts[0] += 1
        else:
            class_instances.append(1)
            class_counts[1] += 1

    print(class_counts)

    sample_weights = []
    for i in class_instances:
        sample_weights.append(1 / class_counts[i])

    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(file_ids),
        replacement=True,
    )

    return weighted_sampler


def get_detection_dataloaders(
    IOConfig: TrainingIOConfig,
    val_fold=1,
    task=1,
    batch_size=4,
    disk_radius=11,
    regression_map: bool = False,
    module: str = "detection",
    do_augmentation: bool = False,
    use_nuclick_masks: bool = False,
):
    """Get training and validation dataloaders
    Task 1: Overall Inflammation cell (MNL) detection
    Task 2: Detect and distinguish monocytes and lymphocytes
    """

    if task not in [1, 2]:
        raise ValueError(f"Task {task} is in invalid")

    if module not in ["detection", "segmentation"]:
        raise ValueError(f"Module {module} is in invalid")

    split = get_split_from_json(IOConfig, val_fold)
    train_file_ids = split["train_file_ids"]
    test_file_ids = split["test_file_ids"]

    train_sampler = get_detection_sampler(
        file_ids=train_file_ids, IOConfig=IOConfig
    )

    print(f"train patches: {len(train_file_ids)}")
    print(f"test patches: {len(test_file_ids)}")

    train_dataset = DetectionDataset(
        IOConfig=IOConfig,
        file_ids=train_file_ids,
        phase="Train",
        do_augment=do_augmentation,
        disk_radius=disk_radius,
        regression_map=regression_map,
        module=module,
        use_nuclick_masks=use_nuclick_masks,
    )
    val_dataset = DetectionDataset(
        IOConfig=IOConfig,
        file_ids=test_file_ids,
        phase="Test",
        do_augment=False,
        disk_radius=disk_radius,
        regression_map=regression_map,
        module=module,
        use_nuclick_masks=use_nuclick_masks,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    return train_loader, val_loader


def get_detection_sampler(file_ids, IOConfig):
    """
    Get Weighted Sampler.
    To balance positive and negative patches.
    """
    patch_stats_path = os.path.join(
        IOConfig.dataset_dir, "patch_stats.json"
    )
    with open(patch_stats_path, "r") as file:
        patch_stats = json.load(file)

    class_instances = []
    class_counts = [0, 0]  # [negatives, positives]

    for id in file_ids:
        stats = patch_stats[id]
        total_cells = stats["lymph_count"] + stats["mono_count"]
        if total_cells == 0:
            class_instances.append(0)
            class_counts[0] += 1
        else:
            class_instances.append(1)
            class_counts[1] += 1

    print(class_counts)

    sample_weights = []
    for i in class_instances:
        sample_weights.append(1 / class_counts[i])

    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(file_ids),
        replacement=True,
    )

    return weighted_sampler
