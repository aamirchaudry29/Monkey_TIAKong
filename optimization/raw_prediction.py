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
        # "contour_prob": [],
    }
    batch_size = 16
    dataloader = DataLoader(
        patch_extractor,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    activation_dict = {
        "head_1": get_activation_function("sigmoid"),
        "head_2": get_activation_function("sigmoid"),
        "head_3": get_activation_function("sigmoid"),
    }

    for i, imgs in enumerate(dataloader):
        imgs = torch.permute(imgs, (0, 3, 1, 2))
        imgs = imgs / 255
        imgs = imagenet_normalise_torch(imgs)
        imgs = imgs.to("cuda").float()

        inflamm_prob = np.zeros(
            shape=(imgs.shape[0], patch_size, patch_size)
        )
        lymph_prob = np.zeros(
            shape=(imgs.shape[0], patch_size, patch_size)
        )
        mono_prob = np.zeros(
            shape=(imgs.shape[0], patch_size, patch_size)
        )
        with torch.no_grad():
            for model in models:
                model.eval()
                logits_pred = model(imgs)
                head_1_logits = logits_pred[:, 0, :, :]
                head_2_logits = logits_pred[:, 1, :, :]
                head_3_logits = logits_pred[:, 2, :, :]

                _inflamm_prob = activation_dict["head_1"](
                    head_1_logits
                ).numpy(force=True)
                _lymph_prob = activation_dict["head_2"](
                    head_2_logits
                ).numpy(force=True)
                _mono_prob = activation_dict["head_3"](
                    head_3_logits
                ).numpy(force=True)

                inflamm_prob += _inflamm_prob
                lymph_prob += _lymph_prob
                mono_prob += _mono_prob

        inflamm_prob = inflamm_prob / len(models)
        lymph_prob = lymph_prob / len(models)
        mono_prob = mono_prob / len(models)

        predictions["inflamm_prob"].extend(list(inflamm_prob))
        predictions["lymph_prob"].extend(list(lymph_prob))
        predictions["mono_prob"].extend(list(mono_prob))

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
        resolution=8.0, units="mpp"
    )
    binary_mask = mask_thumbnail[:, :, 0]
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
        tile_predictions.append(predictions)
        tile_coordinates.append(coordinates)
        bounding_boxes.append(bounding_box)

    # Save raw predictions
    data = {
        "tile_predictions": tile_predictions,
        "tile_coordinates": tile_coordinates,
        "bounding_boxes": bounding_boxes,
    }
    data_path = os.path.join(
        raw_prediction_dir, f"{wsi_without_ext}.pkl"
    )
    os.makedirs(raw_prediction_dir, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Raw predictions saved at {raw_prediction_dir}")
