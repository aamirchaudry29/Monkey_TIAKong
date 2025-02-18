import json
import os

import numpy as np
import torchvision.transforms as T
from strong_augment import StrongAugment
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from monkey.config import TrainingIOConfig
from monkey.data.augmentation import get_augmentation
from monkey.data.data_utils import (
    dilate_mask,
    generate_distance_map,
    get_split_from_json,
    imagenet_normalise,
    load_image,
    load_mask,
    load_nuclick_annotation_v2,
)

# Strong augmentation
AUGMENT_SPACE = {
    "red": (0.0, 2.0),
    "green": (0.0, 2.0),
    "blue": (0.0, 2.0),
    "hue": (-0.5, 0.5),
    "saturation": (0.0, 2.0),
    "brightness": (0.1, 2.0),
    "contrast": (0.1, 2.0),
    "gamma": (0.1, 2.0),
    "solarize": (0, 255),
    "posterize": (1, 8),
    "sharpen": (0.0, 1.0),
    "emboss": (0.0, 1.0),
    "blur": (0.0, 3.0),
    "noise": (0.0, 0.2),
    "jpeg": (0, 100),
    "tone": (0.0, 1.0),
    "autocontrast": (True, True),
    "equalize": (True, True),
    "grayscale": (True, True),
}


def class_mask_to_binary(class_mask: np.ndarray) -> np.ndarray:
    """Converts cell class mask to binary mask
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


def class_mask_to_multichannel_mask(
    class_mask: np.ndarray,
) -> np.ndarray:
    """Converts cell class mask to multi-channel masks
    Example:
        [1,0,0
         0,0,2
         0,0,1]
         ->
        [[1,0,0
         0,0,0
         0,0,1],
         [0,0,0
         0,0,1
         0,0,0]]
    """
    num_classes = 2

    mask = np.zeros(
        shape=(num_classes, class_mask.shape[0], class_mask.shape[1]),
        dtype=np.uint8,
    )
    for idx in range(num_classes):
        label = idx + 1
        mask[idx, :, :] = np.where(class_mask == label, 1, 0)
    return mask


class Multitask_Dataset(Dataset):
    """
    Dataset for multihead model
    """

    def __init__(
        self,
        IOConfig: TrainingIOConfig,
        file_ids: list,
        phase: str = "train",
        do_augment: bool = True,
        disk_radius: int = 11,
        augmentation_prob: float = 0.9,
        strong_augmentation: bool = False,
        weight_map_scale: int = 1,
    ):
        self.IOConfig = IOConfig
        self.file_ids = file_ids
        self.phase = phase
        self.do_augment = do_augment
        self.use_nuclick_masks = False
        self.module = "multiclass_detection"
        self.include_background_channel = False
        self.disk_radius = disk_radius
        self.strong_augmentation = strong_augmentation
        self.weight_map_scale = weight_map_scale

        if self.do_augment:
            self.augmentation = get_augmentation(
                module=self.module,
                gt_type="mask",
                aug_prob=augmentation_prob,
            )
            if self.strong_augmentation:
                self.trnsf = T.Compose(
                    [
                        StrongAugment(
                            operations=[2, 3, 4],
                            probabilites=[0.5, 0.3, 0.2],
                            augment_space=AUGMENT_SPACE,
                        )
                    ]
                )

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> dict:
        # Load image and nuclick masks
        file_id = self.file_ids[idx]
        image = load_image(file_id, self.IOConfig)
        annotation_mask = load_nuclick_annotation_v2(
            file_id, self.IOConfig
        )

        inflamm_mask = annotation_mask["inflamm_mask"]
        inflamm_contour_mask = annotation_mask["inflamm_contour_mask"]
        lymph_mask = annotation_mask["lymph_mask"]
        lymph_contour_mask = annotation_mask["lymph_contour_mask"]
        mono_mask = annotation_mask["mono_mask"]
        mono_contour_mask = annotation_mask["mono_contour_mask"]

        # Load cell centroid masks
        cell_centroid_masks = load_mask(file_id, self.IOConfig)

        # augmentation
        if self.do_augment:
            augmented_data = self.augmentation(
                image=image,
                masks=[
                    inflamm_mask,
                    lymph_mask,
                    mono_mask,
                    inflamm_contour_mask,
                    lymph_contour_mask,
                    mono_contour_mask,
                    cell_centroid_masks,
                ],
            )
            image, masks = (
                augmented_data["image"],
                augmented_data["masks"],
            )
            inflamm_mask = masks[0]
            lymph_mask = masks[1]
            mono_mask = masks[2]
            inflamm_contour_mask = masks[3]
            lymph_contour_mask = masks[4]
            mono_contour_mask = masks[5]
            cell_centroid_masks = masks[6]
            if self.strong_augmentation:
                image = self.trnsf(image)

        lymph_mono_centroid_masks = class_mask_to_multichannel_mask(
            cell_centroid_masks
        )
        for i in range(lymph_mono_centroid_masks.shape[0]):
            lymph_mono_centroid_masks[i] = dilate_mask(
                lymph_mono_centroid_masks[i],
                disk_radius=self.disk_radius,
            )

        inflamm_centroid_masks = class_mask_to_binary(
            cell_centroid_masks
        )
        inflamm_centroid_masks = dilate_mask(
            inflamm_centroid_masks, disk_radius=self.disk_radius
        )

        # HxW -> 1xHxW
        inflamm_centroid_masks = inflamm_centroid_masks[
            np.newaxis, :, :
        ]
        inflamm_mask = inflamm_mask[np.newaxis, :, :]
        lymph_mask = lymph_mask[np.newaxis, :, :]
        mono_mask = mono_mask[np.newaxis, :, :]
        inflamm_contour_mask = inflamm_contour_mask[np.newaxis, :, :]
        lymph_contour_mask = lymph_contour_mask[np.newaxis, :, :]
        mono_contour_mask = mono_contour_mask[np.newaxis, :, :]

        # HxWx3 -> 3xHxW
        image = image / 255
        image = imagenet_normalise(image)
        image = np.moveaxis(image, -1, 0)

        data = {
            "id": file_id,
            "image": image,
            "inflamm_mask": inflamm_mask,
            "lymph_mask": lymph_mask,
            "mono_mask": mono_mask,
            "inflamm_contour_mask": inflamm_contour_mask,
            "lymph_contour_mask": lymph_contour_mask,
            "mono_contour_mask": mono_contour_mask,
            "inflamm_centroid_mask": inflamm_centroid_masks,
            "lymph_centroid_mask": lymph_mono_centroid_masks[
                0:1, :, :
            ],
            "mono_centroid_mask": lymph_mono_centroid_masks[
                1:2, :, :
            ],
        }
        return data


def get_detection_dataloaders(
    IOConfig: TrainingIOConfig,
    val_fold=1,
    dataset_name="detection",
    batch_size=4,
    disk_radius=11,
    module: str = "detection",
    do_augmentation: bool = False,
    train_full_dataset: bool = False,
    augmentation_prob: float = 0.8,
    strong_augmentation: bool = False,
    weight_map_scale: int = 1,
):
    """
    Get training and validation dataloaders
    """

    if dataset_name not in ["multitask"]:
        raise ValueError(f"Dataset Name {dataset_name} is in invalid")

    if module not in ["detection", "multiclass_detection"]:
        raise ValueError(f"Module {module} is in invalid")

    if val_fold not in [1, 2, 3, 4, 5]:
        raise ValueError(f"val_fold {val_fold} is in invalid")

    split = get_split_from_json(IOConfig, val_fold)
    train_file_ids = split["train_file_ids"]
    test_file_ids = split["test_file_ids"]

    if train_full_dataset:
        # Train using entire dataset
        train_file_ids.extend(test_file_ids)

    # if target_cell_type is None:
    train_sampler = get_detection_sampler_v2(
        file_ids=train_file_ids,
        IOConfig=IOConfig,
        cell_radius=disk_radius,
    )

    print(f"train patches: {len(train_file_ids)}")
    print(f"test patches: {len(test_file_ids)}")

    if dataset_name == "multitask":
        train_dataset = Multitask_Dataset(
            IOConfig=IOConfig,
            file_ids=train_file_ids,
            phase="Train",
            do_augment=do_augmentation,
            disk_radius=disk_radius,
            augmentation_prob=augmentation_prob,
            strong_augmentation=strong_augmentation,
            weight_map_scale=weight_map_scale,
        )
        val_dataset = Multitask_Dataset(
            IOConfig=IOConfig,
            file_ids=test_file_ids,
            phase="Test",
            do_augment=False,
            disk_radius=disk_radius,
            weight_map_scale=weight_map_scale,
        )
    else:
        raise ValueError("Invalid dataset name")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return train_loader, val_loader


def get_detection_sampler_v2(file_ids, IOConfig, cell_radius=11):
    """
    Get Weighted Sampler.
    To balance positive and negative patches at pixel level.
    """
    patch_stats_path = os.path.join(
        IOConfig.dataset_dir, "patch_stats.json"
    )
    with open(patch_stats_path, "r") as file:
        patch_stats = json.load(file)

    class_instances = []
    total_class_pixels = np.array(
        [0, 0, 0]
    )  # [negatives, lymph, mono]

    patch_area = 256 * 256
    lymph_size = 16 * 16
    mono_size = 20 * 20

    # Calculate total pixel per class
    for id in file_ids:
        stats = patch_stats[id]
        lymph_count = stats["lymph_count"]
        lymph_area = lymph_count * lymph_size
        mono_count = stats["mono_count"]
        mono_area = mono_count * mono_size
        background_area = patch_area - lymph_area - mono_area

        total_class_pixels += [background_area, lymph_area, mono_area]
        # class_instances.append(
        #     [background_area, lymph_area, mono_area]
        # )

    # Calculate class weights
    total_pixels = np.sum(total_class_pixels)
    class_weights = np.log(total_pixels / total_class_pixels)

    print(f"negative pixels: {total_class_pixels[0]}")
    print(f"lymph pixels: {total_class_pixels[1]}")
    print(f"mono pixels: {total_class_pixels[2]}")

    # Calculate patch weights
    patch_weights = []
    for id in file_ids:
        stats = patch_stats[id]
        lymph_count = stats["lymph_count"]
        mono_count = stats["mono_count"]
        lymph_area = lymph_count * lymph_size
        mono_area = mono_count * mono_size
        background_area = patch_area - lymph_area - mono_area

        # Normalize patch class areas
        patch_class_areas = np.array(
            [background_area, lymph_area, mono_area]
        )
        patch_class_ratios = patch_class_areas / np.sum(
            patch_class_areas
        )
        # Weighted sum of patch class contributions
        patch_weight = np.sum(patch_class_ratios * class_weights)
        patch_weights.append(patch_weight)

    # print(patch_weights)
    weighted_sampler = WeightedRandomSampler(
        weights=patch_weights,
        num_samples=len(file_ids),
        replacement=True,
    )

    return weighted_sampler
