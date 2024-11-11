import os
from typing import Tuple

import numpy as np
import skimage.measure
import skimage.morphology
import torch
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
)
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader
from torch.utils.data import DataLoader
from tqdm import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    collate_fn,
    erode_mask,
    filter_detection_with_mask,
    imagenet_normalise_torch,
    slide_nms,
)


def detection_in_tile(
    image_tile: np.ndarray,
    models: list[torch.nn.Module],
    IOConfig: PredictionIOConfig,
) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Detection in tile image [1024x1024]

    Args:
        image_tile: input tile image
        model: model to be used
        IOConfig: PredictionIOConfig object
    Returns:
        (predictions, coordinates):
            prediction: a list of raw patch predictions.
            coordinates: a list of bounding boxes corresponding to
                each patch prediction
    """
    patch_size = IOConfig.patch_size
    stride = IOConfig.stride

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

    predictions = []
    batch_size = 16
    dataloader = DataLoader(
        patch_extractor,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    for i, imgs in enumerate(dataloader):
        imgs = torch.permute(imgs, (0, 3, 1, 2))
        imgs = imgs / 255
        imgs = imagenet_normalise_torch(imgs)
        imgs = imgs.to("cuda").float()

        final_out = np.zeros(
            shape=(imgs.shape[0], patch_size, patch_size)
        )
        with torch.no_grad():
            for model in models:
                model.eval()
                out = model(imgs)
                out = torch.sigmoid(out)
                out = out.cpu().detach().numpy()[:, 0, :, :]
                final_out += out

        final_out = final_out / len(models)

        predictions.extend(list(final_out))

    return predictions, patch_extractor.coordinate_list


def process_tile_detection_masks(
    pred_masks: list[np.ndarray],
    coordinate_list: list,
    x_start: int,
    y_start: int,
    threshold: float = 0.5,
    min_size: int = 7,
):
    """
    Process cell detection of tile image
    x_start and y_start are used to convert detected cells to WSI coordinates

    Args:
        pred_masks: list of raw predictions [1024x1024]
        coordinate_list: list of coordinates from patch extractor
        x_start: starting x coordinate of this tile
        y_start: starting y coordinate of this tile
        threshold: threshold for raw prediction
    Returns:
        detected_points: list[dict]
            A list of detection records: [{'x', 'y', 'type', 'probability'}]

    """
    if len(pred_masks) == 0:
        tile_prediction = np.zeros(shape=(1024, 1024), dtype=np.uint8)
    else:
        tile_prediction = SemanticSegmentor.merge_prediction(
            (1024, 1024), pred_masks, coordinate_list
        )

    tile_prediction_binary = (tile_prediction > threshold).astype(
        np.uint8
    )
    # tile_prediction_binary = erode_mask(tile_prediction_binary)
    tile_prediction_binary = skimage.morphology.remove_small_objects(
        ar=tile_prediction_binary.astype(bool), min_size=min_size
    )

    mask_labels = skimage.measure.label(tile_prediction_binary)
    stats = skimage.measure.regionprops(
        mask_labels, intensity_image=tile_prediction
    )

    points = []
    for region in stats:
        centroid = region["centroid"]

        c, r, confidence = (
            centroid[1],
            centroid[0],
            region["mean_intensity"],
        )

        c1 = c + x_start
        r1 = r + y_start

        prediction_record = {
            "x": np.round(c1),
            "y": np.round(r1),
            "type": "inflammatory",
            "prob": float(confidence),
        }

        points.append(prediction_record)

    return points


def wsi_detection_in_mask(
    wsi_name: str,
    mask_name: str,
    IOConfig: PredictionIOConfig,
    models: list[torch.nn.Module],
) -> list[dict]:
    """
    Cell Detection in WSI

    Args:
        wsi_name: name of the wsi with file extension
        mask_name: name of the mask with file extension (multi-res)
        IOConfig: PredictionIOConfig object
    Returns:
        detected_points: list[dict]
            A list of detection records: [{'x', 'y', 'type', 'prob'}]
    """
    wsi_dir = IOConfig.wsi_dir
    mask_dir = IOConfig.mask_dir

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
    resolution = IOConfig.resolution
    units = IOConfig.units
    tile_extractor = get_patch_extractor(
        input_img=wsi_reader,
        input_mask=binary_mask,
        method_name="slidingwindow",
        patch_size=(1024, 1024),
        resolution=resolution,
        units=units,
    )

    # Detection in tile
    detected_points: list[dict] = []
    for i, tile in enumerate(
        tqdm(
            tile_extractor,
            leave=False,
            desc=f"{wsi_without_ext} progress",
        )
    ):
        bounding_box = tile_extractor.coordinate_list[
            i
        ]  # (x_start, y_start, x_end, y_end)
        predictions, coordinates = detection_in_tile(
            tile, models, IOConfig
        )
        output_points_tile = process_tile_detection_masks(
            predictions,
            coordinates,
            bounding_box[0],
            bounding_box[1],
            threshold=IOConfig.threshold,
            min_size=IOConfig.min_size,
        )
        detected_points.extend(output_points_tile)

    print(f"Before filtering: {len(detected_points)}")
    # Filter detected_points using ROI mask
    final_detected_records = filter_detection_with_mask(
        detected_points, binary_mask, points_mpp=base_mpp, mask_mpp=8
    )
    print(f"Before nms: {len(final_detected_records)}")
    # nms
    nms_records = slide_nms(
        wsi_reader=wsi_reader,
        binary_mask=binary_mask,
        detection_record=final_detected_records,
        tile_size=2048,
        box_size=30,
        overlap_thresh=0.5,
    )

    return nms_records
