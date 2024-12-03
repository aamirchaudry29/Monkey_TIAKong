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
)
from monkey.model.efficientunetb0.architecture import (
    EfficientUnet_MBConv_Multihead,
)
from monkey.model.utils import get_activation_function
from prediction.utils import multihead_det_post_process


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
        # contour_prob = np.zeros(
        #     shape=(imgs.shape[0], patch_size, patch_size)
        # )

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
                # processed_outputs = EfficientUnet_MBConv_Multihead.multihead_unet_post_process(
                #     logits,
                #     activation_dict,
                #     thresholds=[0.5, 0.5, 0.5, 0.5],
                # )
                inflamm_prob += _inflamm_prob
                lymph_prob += _lymph_prob
                mono_prob += _mono_prob
                # contour_prob += processed_outputs["contour_prob"]

        inflamm_prob = inflamm_prob / len(models)
        lymph_prob = lymph_prob / len(models)
        mono_prob = mono_prob / len(models)
        # contour_prob = contour_prob / len(models)

        predictions["inflamm_prob"].extend(list(inflamm_prob))
        predictions["lymph_prob"].extend(list(lymph_prob))
        predictions["mono_prob"].extend(list(mono_prob))
        # predictions["contour_prob"].extend(list(contour_prob))

    return predictions, patch_extractor.coordinate_list


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
        pred_masks: list of predicted probs[256x256x3]
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

    # if len(pred_results["contour_prob"]) != 0:
    #     contour_probs_map = SemanticSegmentor.merge_prediction(
    #         (tile_size, tile_size),
    #         pred_results["contour_prob"],
    #         coordinate_list,
    #     )[:, :, 0]
    # contour_prediction[contour_probs_map > 0.5] = 1
    # inflamm_prediction[contour_prediction == 1] = 0
    # lymph_prediction[contour_prediction == 1] = 0
    # mono_prediction[contour_prediction == 1] = 0

    # inflamm_probs_map = inflamm_probs_map + lymph_probs_map + mono_probs_map
    # min = np.min(np.ravel(inflamm_probs_map))
    # max = np.max(np.ravel(inflamm_probs_map))
    # inflamm_probs_map = (inflamm_probs_map - min) / (max-min)

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

    mono_labels = skimage.measure.label(
        processed_masks["mono_mask"]
    )
    mono_stats = skimage.measure.regionprops(
        mono_labels, intensity_image=mono_probs_map
    )

    inflamm_points = []
    lymph_points = []
    mono_points = []
    baseline_mpp = 0.24199951445730394
    scale_factor = config.resolution / baseline_mpp

    for region in inflamm_stats:
        centroid = region["centroid"]

        c, r, confidence = (
            centroid[1],
            centroid[0],
            region["mean_intensity"],
        )
        # if confidence < 0.3:
        #     continue
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
        # if confidence < 0.3:
        #     continue
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
        # if confidence < 0.3:
        #     continue
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


def wsi_detection_in_mask(
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
        box_size=30,
        overlap_thresh=0.5,
    )

    final_lymph_records = slide_nms(
        wsi_reader=wsi_reader,
        binary_mask=binary_mask,
        detection_record=filtered_lymph_records,
        tile_size=4096,
        box_size=16,
        overlap_thresh=0.5,
    )

    final_mono_records = slide_nms(
        wsi_reader=wsi_reader,
        binary_mask=binary_mask,
        detection_record=filtered_mono_records,
        tile_size=4096,
        box_size=40,
        overlap_thresh=0.5,
    )

    return {
        "inflamm_records": final_inflamm_records,
        "lymph_records": final_lymph_records,
        "mono_records": final_mono_records,
    }
