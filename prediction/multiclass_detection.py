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
from tqdm.auto import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    collate_fn,
    filter_detection_with_mask,
    imagenet_normalise_torch,
    slide_nms,
    check_image_mask_shape
)
from monkey.model.efficientunetb0.architecture import (
    EfficientUnet_MBConv_Multihead,
)
from monkey.model.utils import get_activation_function
from prediction.utils import multihead_det_post_process, multihead_det_post_process_batch_v2


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
        "inflamm_seg_prob": [],
        "lymph_seg_prob": [],
        "mono_seg_prob": [],
    }
    batch_size = 8
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

        # inflamm_seg_prob = np.zeros(
        #     shape=(imgs.shape[0], patch_size, patch_size)
        # )
        # lymph_seg_prob = np.zeros(
        #     shape=(imgs.shape[0], patch_size, patch_size)
        # )
        # mono_seg_prob = np.zeros(
        #     shape=(imgs.shape[0], patch_size, patch_size)
        # )

        with torch.no_grad():
            for model in models:
                model.eval()
                logits_pred = model(imgs)
                # head_1_logits = logits_pred[:, 0, :, :]
                # head_2_logits = logits_pred[:, 1, :, :]
                # head_3_logits = logits_pred[:, 2, :, :]
                head_1_logits = logits_pred[:, 2, :, :]
                head_2_logits = logits_pred[:, 5, :, :]
                head_3_logits = logits_pred[:, 8, :, :]

                inflamm_seg_logits = logits_pred[:, 0, :, :]
                lymph_seg_logits = logits_pred[:, 3, :, :]
                mono_seg_logits = logits_pred[:, 6, :, :]
                _inflamm_seg_prob = activation_dict["head_1"](inflamm_seg_logits).numpy(force=True)
                _lymph_seg_prob = activation_dict["head_2"](lymph_seg_logits).numpy(force=True)
                _mono_seg_prob = activation_dict["head_3"](mono_seg_logits).numpy(force=True)

                _inflamm_prob = activation_dict["head_1"](
                    head_1_logits
                ).numpy(force=True)
                _lymph_prob = activation_dict["head_2"](
                    head_2_logits
                ).numpy(force=True)
                _mono_prob = activation_dict["head_3"](
                    head_3_logits
                ).numpy(force=True)

                _inflamm_seg_prob[_inflamm_prob < config.thresholds[0]] = 0
                _lymph_seg_prob[_lymph_prob < config.thresholds[1]] = 0
                _mono_seg_prob[_mono_prob < config.thresholds[2]] = 0

                _inflamm_prob = _inflamm_seg_prob * 0.4 + _inflamm_prob * 0.6
                _lymph_prob = _lymph_seg_prob * 0.4 + _lymph_prob * 0.6
                _mono_prob = _mono_seg_prob * 0.4 + _mono_prob * 0.6

                inflamm_prob += _inflamm_prob
                lymph_prob += _lymph_prob
                mono_prob += _mono_prob
                # inflamm_seg_prob += _inflamm_seg_prob
                # lymph_seg_prob += _lymph_seg_prob
                # mono_seg_prob += _mono_seg_prob

        inflamm_prob = inflamm_prob / len(models)
        lymph_prob = lymph_prob / len(models)
        mono_prob = mono_prob / len(models)
        # inflamm_seg_prob /= len(models)
        # lymph_seg_prob /= len(models)
        # mono_seg_prob /= len(models)

        predictions["inflamm_prob"].extend(list(inflamm_prob))
        predictions["lymph_prob"].extend(list(lymph_prob))
        predictions["mono_prob"].extend(list(mono_prob))
        # predictions["inflamm_seg_prob"].extend(list(inflamm_seg_prob))
        # predictions["lymph_seg_prob"].extend(list(lymph_seg_prob))
        # predictions["mono_seg_prob"].extend(list(mono_seg_prob))

    return predictions, patch_extractor.coordinate_list


def process_tile_detection_masks(
    pred_results: list,
    coordinate_list: list,
    config: PredictionIOConfig,
    x_start: int,
    y_start: int,
    mask_tile: np.ndarray,
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
    inflamm_probs_map = np.zeros(shape=(tile_size, tile_size))
    lymph_probs_map = np.zeros(shape=(tile_size, tile_size))
    mono_probs_map = np.zeros(shape=(tile_size, tile_size))
    # inflamm_seg_probs_map = np.zeros(shape=(tile_size, tile_size))
    # lymph_seg_probs_map = np.zeros(shape=(tile_size, tile_size))
    # mono_seg_probs_map = np.zeros(shape=(tile_size, tile_size))

    if len(pred_results["lymph_prob"]) != 0:
        lymph_probs_map = SemanticSegmentor.merge_prediction(
            (tile_size, tile_size),
            pred_results["lymph_prob"],
            coordinate_list,
        )[:, :, 0]
        # lymph_seg_probs_map = SemanticSegmentor.merge_prediction(
        #     (tile_size, tile_size),
        #     pred_results["lymph_seg_prob"],
        #     coordinate_list,
        # )[:, :, 0]

    if len(pred_results["mono_prob"]) != 0:
        mono_probs_map = SemanticSegmentor.merge_prediction(
            (tile_size, tile_size),
            pred_results["mono_prob"],
            coordinate_list,
        )[:, :, 0]
        # mono_seg_probs_map = SemanticSegmentor.merge_prediction(
        #     (tile_size, tile_size),
        #     pred_results["mono_seg_prob"],
        #     coordinate_list,
        # )[:, :, 0]

    if len(pred_results["inflamm_prob"]) != 0:
        inflamm_probs_map = SemanticSegmentor.merge_prediction(
            (tile_size, tile_size),
            pred_results["inflamm_prob"],
            coordinate_list,
        )[:, :, 0]
        # inflamm_seg_probs_map = SemanticSegmentor.merge_prediction(
        #     (tile_size, tile_size),
        #     pred_results["inflamm_seg_prob"],
        #     coordinate_list,
        # )[:, :, 0]

    inflamm_probs_map = inflamm_probs_map * mask_tile
    lymph_probs_map = lymph_probs_map * mask_tile
    mono_probs_map = mono_probs_map * mask_tile
    # inflamm_seg_probs_map = inflamm_seg_probs_map * mask_tile
    # lymph_seg_probs_map = lymph_seg_probs_map * mask_tile
    # mono_seg_probs_map = mono_seg_probs_map * mask_tile

    processed_masks = multihead_det_post_process(
        inflamm_probs_map,
        lymph_probs_map,
        mono_probs_map,
        thresholds=config.thresholds,
        min_distances=config.min_distances,
    )
    # processed_masks = multihead_det_post_process_v2(
    #     inflamm_probs_map,
    #     lymph_probs_map,
    #     mono_probs_map,
    #     thresholds=config.thresholds,
    #     min_distances=config.min_distances,
    # )

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
            "x": c1,
            "y": r1,
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
            "x": c1,
            "y": r1,
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
            "x": c1,
            "y": r1,
            "type": "monocyte",
            "prob": float(confidence),
        }

        mono_points.append(prediction_record)

    return {
        "inflamm_points": inflamm_points,
        "lymph_points": lymph_points,
        "mono_points": mono_points,
    }


def wsi_detection_in_mask_v2(
    wsi_name: str,
    mask_name: str,
    config: PredictionIOConfig,
    models: list[torch.nn.Module],
) -> dict:
    """
    Cell Detection and classification in WSI

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

    # Raise exception if wsi shape != mask shape
    check_image_mask_shape(wsi_path, mask_path)

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
    )[:, :, 0].astype(np.uint8)
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

    # Detection in tile
    detected_inflamm_points: list[dict] = []
    detected_lymph_points = []
    detected_mono_points = []

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
        output_points_tile = process_tile_detection_masks(
            predictions,
            coordinates,
            config,
            bounding_box[0],
            bounding_box[1],
            mask_tile,
            min_size=config.min_size,
        )
        detected_inflamm_points.extend(
            output_points_tile["inflamm_points"]
        )
        detected_lymph_points.extend(
            output_points_tile["lymph_points"]
        )
        detected_mono_points.extend(output_points_tile["mono_points"])

    print(f"Inflamm before nms: {len(detected_inflamm_points)}")
    print(f"Lymph before nms: {len(detected_lymph_points)}")
    print(f"Mono before nms: {len(detected_mono_points)}")

    # nms
    final_inflamm_records = slide_nms(
        wsi_reader=wsi_reader,
        binary_mask=binary_mask,
        detection_record=detected_inflamm_points,
        tile_size=4096,
        box_size=config.nms_boxes[0],
        overlap_thresh=config.nms_overlap_thresh,
    )

    final_lymph_records = slide_nms(
        wsi_reader=wsi_reader,
        binary_mask=binary_mask,
        detection_record=detected_lymph_points,
        tile_size=4096,
        box_size=config.nms_boxes[1],
        overlap_thresh=config.nms_overlap_thresh,
    )

    final_mono_records = slide_nms(
        wsi_reader=wsi_reader,
        binary_mask=binary_mask,
        detection_record=detected_mono_points,
        tile_size=4096,
        box_size=config.nms_boxes[2],
        overlap_thresh=config.nms_overlap_thresh,
    )

    return {
        "inflamm_records": final_inflamm_records,
        "lymph_records": final_lymph_records,
        "mono_records": final_mono_records,
    }
