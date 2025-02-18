import os
import pickle
from typing import Tuple

import numpy as np
import torch
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    collate_fn,
    imagenet_normalise_torch,
)
from monkey.model.utils import get_activation_function
from torch.amp import autocast


def detection_in_tile(
    image_tile: np.ndarray,
    models: list[torch.nn.Module],
    config: PredictionIOConfig,
) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Detection in tile image [2048x2048]

    Args:
        image_tile: input tile image
        model: model to be used
        config: PredictionIOConfig object
    Returns:
        (predictions, coordinates):
            prediction: a list of patch probs.
            coordinates: a list of bounding boxes corresponding to
                each patch prediction
    """
    patch_size = config.patch_size
    stride = config.stride

    # Create patch extractor
    tile_reader = VirtualWSIReader.open(image_tile)

    patch_extractor = get_patch_extractor(
        input_img=tile_reader,
        method_name="slidingwindow",
        patch_size=(patch_size, patch_size),
        stride=(stride, stride),
        resolution=0,
        units="level",
    )

    predictions = {
        "inflamm_prob": [],
        "lymph_prob": [],
        "mono_prob": [],
    }


    batch_size = 16
    dataloader = DataLoader(
        patch_extractor,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    inflamm_prob_np = np.zeros((len(patch_extractor), patch_size, patch_size), dtype=np.float16)
    lymph_prob_np = np.zeros_like(inflamm_prob_np)
    mono_prob_np = np.zeros_like(inflamm_prob_np)

    activation_dict = {
        "head_1": get_activation_function("sigmoid"),
        "head_2": get_activation_function("sigmoid"),
        "head_3": get_activation_function("sigmoid"),
    }

    start_idx = 0
    for imgs in dataloader:
        batch_size_actual = imgs.shape[0]  # In case last batch is smaller
        end_idx = start_idx + batch_size_actual
        imgs = torch.permute(imgs, (0, 3, 1, 2))
        imgs = imgs / 255
        imgs = imagenet_normalise_torch(imgs)
        imgs = imgs.to("cuda").float()

        inflamm_prob = torch.zeros(
            size=(imgs.shape[0], patch_size, patch_size), device="cuda"
        )
        lymph_prob = torch.zeros(
            size=(imgs.shape[0], patch_size, patch_size), device="cuda"
        )
        mono_prob = torch.zeros(
            size=(imgs.shape[0], patch_size, patch_size), device="cuda"
        )

        with torch.no_grad():
            for model in models:
                with autocast(device_type='cuda'):
                    logits_pred = model(imgs)
                _inflamm_prob = activation_dict["head_1"](logits_pred[:, 2, :, :])
                _lymph_prob = activation_dict["head_2"](logits_pred[:, 5, :, :])
                _mono_prob = activation_dict["head_3"](logits_pred[:, 8, :, :])

                _inflamm_seg_prob = activation_dict["head_1"](logits_pred[:, 0, :, :])
                _lymph_seg_prob = activation_dict["head_2"](logits_pred[:, 3, :, :])
                _mono_seg_prob = activation_dict["head_3"](logits_pred[:, 6, :, :])

                _inflamm_seg_prob *= (_inflamm_prob >= config.thresholds[0]).to(dtype=torch.float16)
                _lymph_seg_prob *= (_lymph_prob >= config.thresholds[1]).to(dtype=torch.float16)
                _mono_seg_prob *= (_mono_prob >= config.thresholds[2]).to(dtype=torch.float16)

                _inflamm_prob = (
                    _inflamm_seg_prob * 0.4 + _inflamm_prob * 0.6
                )
                _lymph_prob = (
                    _lymph_seg_prob * 0.4 + _lymph_prob * 0.6
                )
                _mono_prob = _mono_seg_prob * 0.4 + _mono_prob * 0.6

                inflamm_prob += _inflamm_prob
                lymph_prob += _lymph_prob
                mono_prob += _mono_prob

        inflamm_prob = inflamm_prob / len(models)
        lymph_prob = lymph_prob / len(models)
        mono_prob = mono_prob / len(models)

        inflamm_prob_np[start_idx:end_idx] = inflamm_prob.cpu().numpy()
        lymph_prob_np[start_idx:end_idx] = lymph_prob.cpu().numpy()
        mono_prob_np[start_idx:end_idx] = mono_prob.cpu().numpy()

        start_idx = end_idx  # Update index for next batch

    predictions["inflamm_prob"] = list(inflamm_prob_np)
    predictions["lymph_prob"] = list(lymph_prob_np)
    predictions["mono_prob"] = list(mono_prob_np)

    return predictions, patch_extractor.coordinate_list


def wsi_raw_prediction(
    wsi_name: str,
    mask_name: str,
    config: PredictionIOConfig,
    models: list[torch.nn.Module],
) -> None:
    """
    Perform detection on a WSI
    Args:
        wsi_name: name of the wsi with file extension
        mask_name: name of the mask with file extension (multi-res)
        config: PredictionIOConfig object
    Returns:
        detected_points: list[dict]
            A list of detection records: [{'x', 'y', 'type', 'prob'}]
    """
    wsi_dir = config.wsi_dir
    mask_dir = config.mask_dir
    raw_prediction_dir = os.path.join(
        config.output_dir, "raw_prob_maps"
    )

    wsi_without_ext = os.path.splitext(wsi_name)[0]

    wsi_path = os.path.join(wsi_dir, wsi_name)
    mask_path = os.path.join(mask_dir, mask_name)

    wsi_reader = WSIReader.open(wsi_path)
    mask_reader = WSIReader.open(mask_path)

    # Get baseline resolution in mpp
    base_mpp = wsi_reader.convert_resolution_units(
        input_res=0, input_unit="level", output_unit="mpp"
    )[0]
    print(f"baseline mpp = {base_mpp}")
    # Get ROI mask
    mask_thumbnail = mask_reader.slide_thumbnail(
        resolution=2.0, units="mpp"
    )
    binary_mask = np.where(mask_thumbnail > 0, 1, 0).astype(np.uint8)
    # Create tile extractor
    resolution = config.resolution
    units = config.units
    tile_extractor = get_patch_extractor(
        input_img=wsi_reader,
        input_mask=binary_mask,
        method_name="slidingwindow",
        patch_size=(2048, 2048),
        resolution=resolution,
        units=units,
    )

    tile_predictions = []
    tile_coordinates = []
    bounding_boxes = []
    tile_masks = []

    for model in models:
        model.eval()

    for i, tile in enumerate(
        tqdm(
            tile_extractor,
            leave=False,
            desc=f"{wsi_without_ext} detection progress",
        )
    ):
        bounding_box = tile_extractor.coordinate_list[
            i
        ]  # (x_start, y_start, x_end, y_end)
        predictions, coordinates = detection_in_tile(
            tile, models, config
        )
        mask_tile = mask_reader.read_rect(
            location=(bounding_box[0], bounding_box[1]),
            size=(2048, 2048),
            resolution=0,
            units="level",
        )[:, :, 0].astype(np.uint8)
        mask_tile[mask_tile > 0] = 1

        tile_masks.append(mask_tile)
        tile_predictions.append(predictions)
        tile_coordinates.append(coordinates)
        bounding_boxes.append(bounding_box)

    # Save raw predictions
    data = {
        "tile_predictions": tile_predictions,
        "tile_coordinates": tile_coordinates,
        "bounding_boxes": bounding_boxes,
        "tile_masks": tile_masks,
    }
    data_path = os.path.join(
        raw_prediction_dir, f"{wsi_without_ext}.pkl"
    )
    os.makedirs(raw_prediction_dir, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Raw predictions saved at {raw_prediction_dir}")
