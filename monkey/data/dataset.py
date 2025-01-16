import json
import os

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.v2 as transforms
from strong_augment import StrongAugment
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from monkey.config import TrainingIOConfig
from monkey.data.augmentation import get_augmentation
from monkey.data.data_utils import (
    add_background_channel,
    dilate_mask,
    generate_regression_map,
    get_label_from_class_id,
    get_split_from_json,
    imagenet_normalise,
    load_classification_data_example,
    load_image,
    load_json_annotation,
    load_mask,
    load_nuclick_annotation,
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
    # "solarize": (0, 255),
    # "posterize": (1, 8),
    "sharpen": (0.0, 1.0),
    # "emboss": (0.0, 1.0),
    "blur": (0.0, 3.0),
    "noise": (0.0, 0.2),
    "jpeg": (0, 100),
    "tone": (0.0, 1.0),
    "autocontrast": (True, True),
    "equalize": (True, True),
    # "grayscale": (True, True),
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


class BoundingBoxDataset(Dataset):
    """
    Dataset for training YOLO, RCNN,...
    """

    def __init__(
        self,
        IOConfig: TrainingIOConfig,
        file_ids: list,
        phase: str = "train",
        do_augment: bool = False,
        box_radius: int = 9,
    ):
        self.IOConfig = IOConfig
        self.file_ids = file_ids
        self.phase = phase
        self.do_augment = do_augment
        self.box_radius = box_radius

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> dict:
        # Load image and mask
        file_id = self.file_ids[idx]
        image = load_image(file_id, self.IOConfig)
        annotation_json = load_json_annotation(file_id, self.IOConfig)


class DetectionDataset(Dataset):
    """Dataset for binary cell detection
    Detecting Lymphocytes and Monocytes
    Data: RGB image and cell mask
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
        include_background_channel: bool = False,
        target_cell_type: str | None = None,
        augmentation_prob: float = 0.9,
    ):
        self.IOConfig = IOConfig
        self.file_ids = file_ids
        self.phase = phase
        self.do_augment = do_augment
        self.disk_radius = disk_radius
        self.module = "detection"
        self.regression_map = False
        self.use_nuclick_masks = use_nuclick_masks
        self.include_background_channel = False
        self.target_cell_type = target_cell_type

        if self.target_cell_type is not None:
            self.include_background_channel = False
            if self.target_cell_type == "inflamm":
                self.disk_radius = 11
            if self.target_cell_type == "lymph":
                self.disk_radius = 9
            if self.target_cell_type == "mono":
                self.disk_radius = 13

        if self.do_augment:
            self.augmentation = get_augmentation(
                module=self.module,
                gt_type="mask",
                aug_prob=augmentation_prob,
            )

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> dict:
        # Load image and mask
        file_id = self.file_ids[idx]
        image = load_image(file_id, self.IOConfig)

        if self.use_nuclick_masks:
            nuclick_annotation = load_nuclick_annotation(
                file_id, self.IOConfig
            )
            cell_mask = nuclick_annotation["class_mask"]
        else:
            cell_mask = load_mask(file_id, self.IOConfig)

        # augmentation
        if self.do_augment:
            augmented_data = self.augmentation(
                image=image, mask=cell_mask
            )
            image, cell_mask = (
                augmented_data["image"],
                augmented_data["mask"],
            )

        if self.target_cell_type == "inflamm":
            cell_mask = class_mask_to_binary(cell_mask)
        if self.target_cell_type == "lymph":
            cell_mask = class_mask_to_multichannel_mask(cell_mask)[0]
        if self.target_cell_type == "mono":
            cell_mask = class_mask_to_multichannel_mask(cell_mask)[1]

        # Dilate cell centroids
        if not self.use_nuclick_masks:
            if len(cell_mask.shape) == 2:
                cell_mask = dilate_mask(
                    cell_mask, disk_radius=self.disk_radius
                )
            else:
                for i in range(cell_mask.shape[0]):
                    cell_mask[i] = dilate_mask(
                        cell_mask[i], disk_radius=self.disk_radius
                    )
        # Generate regression map
        if self.regression_map:
            if len(cell_mask.shape) == 2:
                cell_mask = generate_regression_map(
                    binary_mask=cell_mask,
                    d_thresh=3,
                    alpha=5,
                    scale=1,
                )
            else:
                for i in range(cell_mask.shape[0]):
                    cell_mask[i] = generate_regression_map(
                        binary_mask=cell_mask[i],
                        d_thresh=7,
                        alpha=5,
                        scale=1,
                    )

        if len(cell_mask.shape) == 2:
            # HxW -> 1xHxW
            cell_mask = cell_mask[np.newaxis, :, :]
        # if self.include_background_channel:
        #     cell_mask = add_background_channel(cell_mask)

        # HxWx3 -> 3xHxW
        image = image / 255
        image = imagenet_normalise(image)
        image = np.moveaxis(image, -1, 0)

        data = {
            "id": file_id,
            "image": image,
            "mask": cell_mask,
        }

        return data


class Multitask_Dataset(Dataset):
    """
    Dataset for multihead unet
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
                            operations=[1, 2, 3],
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
                    cell_centroid_masks
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

        lymph_mono_centroid_masks = class_mask_to_multichannel_mask(cell_centroid_masks)
        lymph_weight_mask = generate_regression_map(lymph_mono_centroid_masks[0], d_thresh=self.disk_radius, alpha=1, scale=1)
        mono_weight_mask = generate_regression_map(lymph_mono_centroid_masks[1], d_thresh=self.disk_radius, alpha=1, scale=1)
        for i in range(lymph_mono_centroid_masks.shape[0]):
            lymph_mono_centroid_masks[i] = dilate_mask(
                lymph_mono_centroid_masks[i], disk_radius=self.disk_radius
            )
        inflamm_centroid_masks = class_mask_to_binary(cell_centroid_masks)
        inflamm_weight_mask = generate_regression_map(inflamm_centroid_masks, d_thresh=self.disk_radius, alpha=1, scale=1)
        inflamm_centroid_masks = dilate_mask(
            inflamm_centroid_masks, disk_radius=self.disk_radius
        )
        inflamm_centroid_masks = inflamm_centroid_masks[np.newaxis, :, :]
        # HxW -> 1xHxW
        inflamm_mask = inflamm_mask[np.newaxis, :, :]
        lymph_mask = lymph_mask[np.newaxis, :, :]
        mono_mask = mono_mask[np.newaxis, :, :]
        inflamm_contour_mask = inflamm_contour_mask[np.newaxis, :, :]
        lymph_contour_mask = lymph_contour_mask[np.newaxis, :, :]
        mono_contour_mask = mono_contour_mask[np.newaxis, :, :]
        lymph_weight_mask = lymph_weight_mask[np.newaxis, :, :]
        mono_weight_mask = mono_weight_mask[np.newaxis, :, :]
        inflamm_weight_mask = inflamm_weight_mask[np.newaxis, :, :]

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
            "lymph_centroid_mask": lymph_mono_centroid_masks[0:1, :, :],
            "mono_centroid_mask": lymph_mono_centroid_masks[1:2, :, :],
            "inflamm_weight_mask": inflamm_weight_mask,
            "lymph_weight_mask": lymph_weight_mask,
            "mono_weight_mask": mono_weight_mask,
        }
        return data


class Segmentation_Dataset(Dataset):
    """
    Dataset for multihead unet
    NuClick Segmentation masks only
    """

    def __init__(
        self,
        IOConfig: TrainingIOConfig,
        file_ids: list,
        phase: str = "train",
        do_augment: bool = True,
        augmentation_prob: float = 0.9,
        strong_augmentation: bool = False,
        version: int = 1,
    ):
        if version not in [1, 2]:
            raise ValueError("Invalid version")
        else:
            self.version = version

        self.IOConfig = IOConfig
        self.file_ids = file_ids
        self.phase = phase
        self.do_augment = do_augment

        self.module = "multiclass_detection"
        self.strong_augmentation = strong_augmentation

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
                            operations=[1, 2, 3],
                            probabilites=[0.5, 0.3, 0.2],
                            augment_space=AUGMENT_SPACE,
                        )
                    ]
                )

    def __len__(self) -> int:
        return len(self.file_ids)

    def get_item_v1(self, idx: int) -> dict:
        # Load image and mask
        file_id = self.file_ids[idx]
        image = load_image(file_id, self.IOConfig)
        annotation_mask = load_nuclick_annotation(
            file_id, self.IOConfig
        )

        binary_mask = annotation_mask["binary_mask"]
        class_mask = annotation_mask["class_mask"]
        class_mask = class_mask_to_multichannel_mask(class_mask)
        contour_mask = annotation_mask["contour_mask"]

        # augmentation
        if self.do_augment:
            augmented_data = self.augmentation(
                image=image,
                masks=[
                    binary_mask,
                    class_mask[0],
                    class_mask[1],
                    contour_mask,
                ],
            )
            image, masks = (
                augmented_data["image"],
                augmented_data["masks"],
            )
            binary_mask = masks[0]
            class_mask[0] = masks[1]
            class_mask[1] = masks[2]
            contour_mask = masks[3]
            if self.strong_augmentation:
                image = self.trnsf(image)

        # HxW -> 1xHxW
        binary_mask = binary_mask[np.newaxis, :, :]
        contour_mask = contour_mask[np.newaxis, :, :]

        # HxWx3 -> 3xHxW
        image = image / 255
        image = imagenet_normalise(image)
        image = np.moveaxis(image, -1, 0)

        data = {
            "id": file_id,
            "image": image,
            "binary_mask": binary_mask,
            "class_mask": class_mask,
            "contour_mask": contour_mask,
        }
        return data

    def get_item_v2(self, idx: int) -> dict:
        # Load image and mask
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
            if self.strong_augmentation:
                image = self.trnsf(image)

        # HxW -> 1xHxW
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
        }
        return data

    def __getitem__(self, idx: int) -> dict:
        if self.version == 1:
            return self.get_item_v1(idx)
        else:
            return self.get_item_v2(idx)


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
                transforms.Resize((224, 224)),
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
    dataset_name="detection",
    batch_size=4,
    disk_radius=11,
    regression_map: bool = False,
    module: str = "detection",
    do_augmentation: bool = False,
    use_nuclick_masks: bool = False,
    include_background_channel: bool = False,
    train_full_dataset: bool = False,
    target_cell_type: str | None = None,
    augmentation_prob: float = 0.8,
    strong_augmentation: bool = False,
    version: int = 1,
):
    """
    Get training and validation dataloaders
    """

    if dataset_name not in ["detection", "multitask", "segmentation"]:
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
    # else:
    #     train_sampler = get_detection_sampler_v2_binary(
    #         file_ids=train_file_ids,
    #         IOConfig=IOConfig,
    #         cell_type=target_cell_type,
    #     )
    # train_sampler = get_detection_sampler(
    #     file_ids=train_file_ids, IOConfig=IOConfig
    # )

    print(f"train patches: {len(train_file_ids)}")
    print(f"test patches: {len(test_file_ids)}")

    if dataset_name == "detection":
        train_dataset = DetectionDataset(
            IOConfig=IOConfig,
            file_ids=train_file_ids,
            phase="Train",
            do_augment=do_augmentation,
            disk_radius=disk_radius,
            regression_map=regression_map,
            module=module,
            use_nuclick_masks=use_nuclick_masks,
            include_background_channel=include_background_channel,
            target_cell_type=target_cell_type,
            augmentation_prob=augmentation_prob,
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
            include_background_channel=include_background_channel,
            target_cell_type=target_cell_type,
            augmentation_prob=augmentation_prob,
        )
    elif dataset_name == "multitask":
        train_dataset = Multitask_Dataset(
            IOConfig=IOConfig,
            file_ids=train_file_ids,
            phase="Train",
            do_augment=do_augmentation,
            disk_radius=disk_radius,
            augmentation_prob=augmentation_prob,
            strong_augmentation=strong_augmentation,
        )
        val_dataset = Multitask_Dataset(
            IOConfig=IOConfig,
            file_ids=test_file_ids,
            phase="Test",
            do_augment=False,
            disk_radius=disk_radius,
        )
    elif dataset_name == "segmentation":
        train_dataset = Segmentation_Dataset(
            IOConfig=IOConfig,
            file_ids=train_file_ids,
            phase="Train",
            do_augment=do_augmentation,
            augmentation_prob=augmentation_prob,
            strong_augmentation=strong_augmentation,
            version=version,
        )
        val_dataset = Segmentation_Dataset(
            IOConfig=IOConfig,
            file_ids=test_file_ids,
            phase="Test",
            do_augment=False,
            augmentation_prob=augmentation_prob,
            strong_augmentation=strong_augmentation,
            version=version,
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


# def get_detection_sampler(file_ids, IOConfig):
#     """
#     Get Weighted Sampler.
#     To balance positive and negative patches.
#     """
#     patch_stats_path = os.path.join(
#         IOConfig.dataset_dir, "patch_stats.json"
#     )
#     with open(patch_stats_path, "r") as file:
#         patch_stats = json.load(file)

#     class_instances = []
#     class_counts = [
#         0,
#         0,
#         0,
#         0,
#     ]  # [negatives, lymph only, mono only, both type]

#     total_cell_counts = [0, 0]

#     for id in file_ids:
#         stats = patch_stats[id]
#         lymph_count = stats["lymph_count"]
#         total_cell_counts[0] += lymph_count
#         mono_count = stats["mono_count"]
#         total_cell_counts[1] += mono_count
#         total_cells = lymph_count + mono_count
#         if total_cells == 0:
#             class_instances.append(0)
#             class_counts[0] += 1
#         else:
#             if lymph_count > 0 and mono_count == 0:
#                 class_instances.append(1)
#                 class_counts[1] += 1
#             elif lymph_count == 0 and mono_count > 0:
#                 class_instances.append(2)
#                 class_counts[2] += 1
#             else:
#                 class_instances.append(3)
#                 class_counts[3] += 1

#     print(f"negative patches: {class_counts[0]}")
#     print(f"lymph only patches: {class_counts[1]}")
#     print(f"mono only patches: {class_counts[2]}")
#     print(f"inflamm patches: {class_counts[3]}")
#     print(f"Total lymph cells {total_cell_counts[0]}")
#     print(f"Total mono cells {total_cell_counts[1]}")

#     sample_weights = []
#     for i in class_instances:
#         sample_weights.append(1 / class_counts[i])

#     weighted_sampler = WeightedRandomSampler(
#         weights=sample_weights,
#         num_samples=len(file_ids),
#         replacement=True,
#     )

#     return weighted_sampler


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

    # class_instances = np.array(class_instances)
    # pixel_class_sum = np.sum(class_instances, axis=0)

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

        # weight_vector = class_instances[i] / pixel_class_sum
        # patch_weights.append(np.sum(weight_vector))

    # print(patch_weights)
    weighted_sampler = WeightedRandomSampler(
        weights=patch_weights,
        num_samples=len(file_ids),
        replacement=True,
    )

    return weighted_sampler


def get_detection_sampler_v2_binary(
    file_ids, IOConfig, cell_type: str = "inflamm"
):
    """
    Get Weighted Sampler.
    To balance positive and negative patches at pixel level.
    """

    if cell_type not in ["inflamm", "lymph", "mono"]:
        raise ValueError("Invalid cell type")

    patch_stats_path = os.path.join(
        IOConfig.dataset_dir, "patch_stats.json"
    )
    with open(patch_stats_path, "r") as file:
        patch_stats = json.load(file)

    class_instances = []
    class_areas_total = [
        0,
        0,
    ]  # [others, cell_type of interest]

    patch_area = 512 * 512
    cell_area = 7 * 7

    for id in file_ids:
        stats = patch_stats[id]

        lymph_count = stats["lymph_count"]
        lymph_area = lymph_count * cell_area
        mono_count = stats["mono_count"]
        mono_area = mono_count * cell_area

        target_cell_area = 0.0
        if cell_type == "inflamm":
            target_cell_area = lymph_area + mono_area
        if cell_type == "lymph":
            target_cell_area = lymph_area
        if cell_type == "mono":
            target_cell_area = mono_area

        background_area = patch_area - target_cell_area
        class_instances.append([background_area, target_cell_area])

    class_instances = np.array(class_instances)
    pixel_class_sum = np.sum(class_instances, axis=0)

    print(f"others pixels: {pixel_class_sum[0]}")
    print(f"target cell pixels: {pixel_class_sum[1]}")

    sample_weights = []
    for i in range(class_instances.shape[0]):
        weight_vector = class_instances[i] / pixel_class_sum
        sample_weights.append(np.sum(weight_vector))

    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(file_ids),
        replacement=True,
    )

    return weighted_sampler
