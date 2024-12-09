import os
from typing import Tuple
import pickle

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
from tqdm.auto import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    collate_fn,
    filter_detection_with_mask,
    imagenet_normalise_torch,
    slide_nms,
)
from monkey.model.efficientunetb0.architecture import (
    EfficientUnet_MBConv_Multihead,
)
from monkey.model.utils import get_activation_function
from prediction.utils import multihead_det_post_process


def process_tile_detection_masks(
    pred_results: list,
    coordinate_list: list,
    config: PredictionIOConfig,
    x_start: int,
    y_start: int,
    min_size: int = 3,
    tile_size: int = 2048,
) -> dict:
    """
    Process cell detection of tile image
    x_start and y_start are used to convert detected cells to WSI coordinates

    Args:
        pred_masks: list of predicted probs[HxWx3]
        coordinate_list: list of coordinates from patch extractor
        config: PredictionIOConfig
        x_start: starting x coordinate of this tile
        y_start: starting y coordinate of this tile
        threshold: threshold for raw prediction
    Returns:
        detected_points: list[dict]
            A list of detection records: [{'x', 'y', 'type', 'probability'}]

    """
    # inflamm_prediction = np.zeros(
    #     shape=(tile_size, tile_size), dtype=np.uint8
    # )
    # contour_prediction = np.zeros(
    #     shape=(tile_size, tile_size), dtype=np.uint8
    # )
    # lymph_prediction = np.zeros(
    #     shape=(tile_size, tile_size), dtype=np.uint8
    # )
    # mono_prediction = np.zeros(
    #     shape=(tile_size, tile_size), dtype=np.uint8
    # )

    inflamm_probs_map = np.zeros(shape=(tile_size, tile_size))
    lymph_probs_map = np.zeros(shape=(tile_size, tile_size))
    mono_probs_map = np.zeros(shape=(tile_size, tile_size))

    if len(pred_results["lymph_prob"]) != 0:
        lymph_probs_map = SemanticSegmentor.merge_prediction(
            (tile_size, tile_size),
            pred_results["lymph_prob"],
            coordinate_list,
        )[:, :, 0]
        # lymph_prediction[lymph_probs_map > 0.5] = 1

    if len(pred_results["mono_prob"]) != 0:
        mono_probs_map = SemanticSegmentor.merge_prediction(
            (tile_size, tile_size),
            pred_results["mono_prob"],
            coordinate_list,
        )[:, :, 0]
        # mono_prediction[mono_probs_map > 0.5] = 1

    if len(pred_results["inflamm_prob"]) != 0:
        inflamm_probs_map = SemanticSegmentor.merge_prediction(
            (tile_size, tile_size),
            pred_results["inflamm_prob"],
            coordinate_list,
        )[:, :, 0]
        # inflamm_prediction[inflamm_probs_map > 0.5] = 1


    processed_masks = multihead_det_post_process(
        inflamm_probs_map,
        lymph_probs_map,
        mono_probs_map,
        thresholds=config.thresholds,
        min_distances=config.min_distances,
    )

    inflamm_labels = skimage.measure.label(
        processed_masks["inflamm_mask"]
    )
    inflamm_stats = skimage.measure.regionprops(
        inflamm_labels, intensity_image=inflamm_probs_map
    )

    lymph_labels = skimage.measure.label(
        processed_masks["lymph_mask"]
    )
    lymph_stats = skimage.measure.regionprops(
        lymph_labels, intensity_image=lymph_probs_map
    )

    mono_labels = skimage.measure.label(processed_masks["mono_mask"])
    mono_stats = skimage.measure.regionprops(
        mono_labels, intensity_image=mono_probs_map
    )

    inflamm_points = []
    lymph_points = []
    mono_points = []
    baseline_mpp = 0.24199951445730394
    scale_factor = 1.0

    for region in inflamm_stats:
        centroid = region["centroid"]

        c, r, confidence = (
            centroid[1],
            centroid[0],
            region["mean_intensity"],
        )
        c1 = c + x_start
        r1 = r + y_start

        prediction_record = {
            "x": int(np.round(c1 * scale_factor)),
            "y": int(np.round(r1 * scale_factor)),
            "type": "inflammatory",
            "prob": float(confidence),
        }

        inflamm_points.append(prediction_record)

    for region in lymph_stats:
        centroid = region["centroid"]

        c, r, confidence = (
            centroid[1],
            centroid[0],
            region["mean_intensity"],
        )
        c1 = c + x_start
        r1 = r + y_start

        prediction_record = {
            "x": int(np.round(c1 * scale_factor)),
            "y": int(np.round(r1 * scale_factor)),
            "type": "lymphocyte",
            "prob": float(confidence),
        }

        lymph_points.append(prediction_record)

    for region in mono_stats:
        centroid = region["centroid"]

        c, r, confidence = (
            centroid[1],
            centroid[0],
            region["mean_intensity"],
        )
        c1 = c + x_start
        r1 = r + y_start

        prediction_record = {
            "x": int(np.round(c1 * scale_factor)),
            "y": int(np.round(r1 * scale_factor)),
            "type": "monocyte",
            "prob": float(confidence),
        }

        mono_points.append(prediction_record)

    return {
        "inflamm_points": inflamm_points,
        "lymph_points": lymph_points,
        "mono_points": mono_points,
    }


def post_process_detection(
    wsi_name: str,
    mask_name: str,
    config: PredictionIOConfig,
    models: list[torch.nn.Module],
) -> dict:
    """
    Post process detection tile probs

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

    # Detection in tile
    detected_inflamm_points: list[dict] = []
    detected_lymph_points = []
    detected_mono_points = []

    raw_prediction_dir = os.path.join(config.output_dir, "raw_prob_maps")
    data_path = os.path.join(raw_prediction_dir, f"{wsi_without_ext}.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    tile_predictions = data["tile_predictions"]
    tile_coordinates = data["tile_coordinates"]
    bounding_boxes = data["bounding_boxes"]

    for i in range(len(tile_predictions)):
        predictions = tile_predictions[i]
        coordinates = tile_coordinates[i]
        bounding_box = bounding_boxes[i]

        output_points_tile = process_tile_detection_masks(
            predictions,
            coordinates,
            config,
            bounding_box[0],
            bounding_box[1],
            min_size=config.min_size,
        )
        detected_inflamm_points.extend(
            output_points_tile["inflamm_points"]
        )
        detected_lymph_points.extend(
            output_points_tile["lymph_points"]
        )
        detected_mono_points.extend(output_points_tile["mono_points"])

    print(f"Inflamm before filtering: {len(detected_inflamm_points)}")
    print(f"Lymph before filtering: {len(detected_lymph_points)}")
    print(f"Mono before filtering: {len(detected_mono_points)}")
    # Filter detected_points using ROI mask
    filtered_inflamm_records = filter_detection_with_mask(
        detected_inflamm_points,
        binary_mask,
        points_mpp=base_mpp,
        mask_mpp=8,
    )
    filtered_lymph_records = filter_detection_with_mask(
        detected_lymph_points,
        binary_mask,
        points_mpp=base_mpp,
        mask_mpp=8,
    )
    filtered_mono_records = filter_detection_with_mask(
        detected_mono_points,
        binary_mask,
        points_mpp=base_mpp,
        mask_mpp=8,
    )

    print(f"Inflamm before nms: {len(filtered_inflamm_records)}")
    print(f"Lymph before nms: {len(filtered_lymph_records)}")
    print(f"Mono before nms: {len(filtered_mono_records)}")

    # nms
    final_inflamm_records = slide_nms(
        wsi_reader=wsi_reader,
        binary_mask=binary_mask,
        detection_record=filtered_inflamm_records,
        tile_size=4096,
        box_size=config.nms_boxes[0],
        overlap_thresh=config.nms_overlap_thresh,
    )

    final_lymph_records = slide_nms(
        wsi_reader=wsi_reader,
        binary_mask=binary_mask,
        detection_record=filtered_lymph_records,
        tile_size=4096,
        box_size=config.nms_boxes[1],
        overlap_thresh=config.nms_overlap_thresh,
    )

    final_mono_records = slide_nms(
        wsi_reader=wsi_reader,
        binary_mask=binary_mask,
        detection_record=filtered_mono_records,
        tile_size=4096,
        box_size=config.nms_boxes[2],
        overlap_thresh=config.nms_overlap_thresh,
    )

    return {
        "inflamm_records": final_inflamm_records,
        "lymph_records": final_lymph_records,
        "mono_records": final_mono_records,
    }
