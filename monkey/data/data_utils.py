import json
import os
import re

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from scipy import ndimage
from shapely import Point, Polygon
from skimage.feature import peak_local_max
from skimage.measure import label
from tiatoolbox.annotation.storage import (
    Annotation,
    AnnotationStore,
    SQLiteStore,
)
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import WSIReader

from monkey.config import PredictionIOConfig, TrainingIOConfig


def load_image(
    file_id: str, IOConfig: TrainingIOConfig
) -> np.ndarray:
    """
    Load a single RGB image for cell detection
    """
    image_name = f"{file_id}.npy"
    image_path = os.path.join(IOConfig.image_dir, image_name)
    image = np.load(image_path)
    return image


def load_mask(file_id: str, IOConfig: TrainingIOConfig) -> np.ndarray:
    """
    Load a single mask for cell detection
    """
    mask_name = f"{file_id}.npy"
    mask_path = os.path.join(
        IOConfig.cell_centroid_mask_dir, mask_name
    )
    mask = np.load(mask_path)
    return mask


def load_nuclick_annotation(file_id: str, IOConfig: TrainingIOConfig):
    """
    Load a single NuClick annotation mask
    Nuclick file format: 6 channel .np file
    channel 1-3: RGB image
    channel 4: binary segmentation mask
    channel 5: class mask (1:lymphocytes,2:monocytes)
    channel 6: binary contour mask

    Returns:
        annotation: {'binary_mask', 'class_mask', 'contour_mask'}
    """
    file_name = f"{file_id}.npy"
    file_path = os.path.join(IOConfig.mask_dir, file_name)
    data = np.load(file_path)
    data = data.astype(np.uint8)
    binary_mask = data[:, :, 3]
    class_mask = data[:, :, 4]
    contour = data[:, :, 5]

    annotation = {
        "binary_mask": binary_mask,
        "class_mask": class_mask,
        "contour_mask": contour,
    }

    return annotation


def load_nuclick_annotation_v2(
    file_id: str, IOConfig: TrainingIOConfig
):
    """
    Load a single NuClick annotation mask
    Nuclick file format: 8 channel .np file
    channel 1-3: RGB image
    channel 4: inflamm segmentation mask
    channel 5: inflamm contour mask
    channel 6: lymph segmentation mask
    channel 7: lymph contour mask
    channel 8: mono segmentation mask
    channel 9: mono contour mask

    Returns:
        annotation: {'inflamm_mask', 'inflamm_contour_mask', 'lymph_mask', 'lymph_contour_mask', 'mono_mask', 'mono_contour_mask'}
    """
    file_name = f"{file_id}.npy"
    file_path = os.path.join(IOConfig.mask_dir, file_name)
    data = np.load(file_path)
    data = data.astype(np.uint8)

    annotation = {
        "inflamm_mask": data[:, :, 3],
        "inflamm_contour_mask": data[:, :, 4],
        "lymph_mask": data[:, :, 5],
        "lymph_contour_mask": data[:, :, 6],
        "mono_mask": data[:, :, 7],
        "mono_contour_mask": data[:, :, 8],
    }

    return annotation


def get_label_from_class_id(file_id: str):
    """
    Get cell type from classification file_id
    Example file_id "D_P000019_9856_52192_10112_52448_lymph_3"
    Returns 0 for lymphocyte, 1 for monocyte
    """
    parts = file_id.split("_")
    name = parts[-2]
    if name == "lymph":
        return 0
    elif name == "mono":
        return 1
    else:
        raise ValueError(f"Unknown name {name}")


def load_classification_data_example(
    file_id: str, IOConfig: TrainingIOConfig
):
    """
    Load a single classification image patch.
    channel 1-3: RGB image
    channel 4: binary segmentation mask

    Returns:
        {"image","mask","label"}
    """
    file_name = f"{file_id}.npy"
    class_label = get_label_from_class_id(file_id)
    data_path = os.path.join(IOConfig.mask_dir, file_name)
    data = np.load(data_path)
    data = data.astype(np.uint8)
    image = data[:, :, 0:3]
    mask = data[:, :, 3]
    return {"image": image, "mask": mask, "label": class_label}


def load_json_annotation(
    file_id: str, IOConfig: TrainingIOConfig
) -> np.ndarray:
    """Load patch-level cell coordinates"""
    json_name = f"{file_id}.json"
    json_path = os.path.join(IOConfig.json_dir, json_name)
    return open_json_file(json_path)


def open_json_file(json_path: str):
    """Extract annotations from json file"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


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
    if val_fold < 1 or val_fold > 4:
        raise ValueError(f"Invalid test centre {val_fold}")

    test_centre = centres[val_fold - 1]

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


def get_split_from_json(
    IOConfig: TrainingIOConfig, val_fold: int = 1
):
    """
    Retrieve train and validation patch ids from pre-processed json file
    """
    split_info_json_path = os.path.join(
        IOConfig.dataset_dir, "patch_level_split.json"
    )

    split_info = open_json_file(split_info_json_path)
    fold_info = split_info[f"Fold_{val_fold}"]
    split = {
        "train_file_ids": fold_info["train_files"],
        "test_file_ids": fold_info["test_files"],
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


def imagenet_normalise_torch(img: torch.tensor) -> torch.tensor:
    """Normalises input image to ImageNet mean and std
    Input torch tensor (B,3,H,W)
    """

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    for i in range(3):
        img[:, i, :, :] = (img[:, i, :, :] - mean[i]) / std[i]
    return img


def extract_cetroids_from_mask(
    mask: np.ndarray, area_threshold: int = 3
):
    """Extract cell centroids from mask"""
    # dilating the instance to connect separated instances
    inst_map = mask
    inst_map_np = np.asarray(inst_map)
    obj_total = np.unique(inst_map_np)
    obj_ids = obj_total
    obj_ids = obj_ids[1:]  # first id is the background, so remove it
    num_objs = len(obj_ids)
    centroids = []
    for i in range(
        num_objs
    ):  ##num_objs is how many bounding boxes in each tile.
        this_mask = inst_map_np == obj_ids[i]
        this_mask_int = this_mask.astype(
            "uint8"
        )  # because of mirror operation in patch sampling process,
        # the instance with same index may appear more than onece, e.g. two areas is 27 for the mirrored nucleus.
        # find the centroids for each unique connected area, although those may have same index number
        contours, _ = cv2.findContours(
            this_mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            if cv2.contourArea(c) < area_threshold:
                continue
            # calculate moments for each contour
            M = cv2.moments(c)
            # calculate x,y coordinate of cente
            centroids.append(
                (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            )
    return centroids


def draw_disks(
    canvas_size: tuple, centroids: list, disk_radius: np.uint8
):
    """Draw disks based on centroid points on a canvas with predefined size.

    canvas_size: size of the canvas to draw disks on, usually the same size as the input image
        but only with 1 channel. In (height, width) format.
    x_list: list of x coordinates of centroids
    y_list: list of Y coordinates of centroids
    disk_radius: the radius to draw disks on canvas
    """
    gt_circles = np.zeros(canvas_size, dtype=np.uint8)

    if disk_radius == 0:  # put a point if the radius is zero
        for cX, cY in centroids:
            gt_circles[cY, cX] = 1
    else:  # draw a circle otherwise
        for cX, cY in centroids:
            cv2.circle(
                gt_circles, (cX, cY), disk_radius, (255, 255, 255), -1
            )
    gt_circles = np.float32(gt_circles > 0)
    return gt_circles


def dilate_mask(mask: np.ndarray, disk_radius: int):
    """
    Draw dilate mask
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (disk_radius, disk_radius)
    )
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask


def generate_regression_map(
    binary_mask: np.ndarray,
    d_thresh: int = 5,
    alpha: int = 3,
    scale: int = 3,
):
    dist = ndi.distance_transform_edt(binary_mask == 0)
    M = (np.exp(alpha * (1 - dist / d_thresh)) - 1) / (
        np.exp(alpha) - 1
    )
    M[M < 0] = 0
    M *= scale
    return M


def px_to_mm(px: int, mpp: float = 0.24199951445730394):
    """
    Convert pixel coordinate to millimeters
    """
    return px * mpp / 1000


def write_json_file(location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def extract_dotmaps(
    original_prediction: np.ndarray,
    distance_threshold_local_max: int,
    prediction_dots_threshold: float | None = None,
    method: str = "local_max",
):
    if method == "local_max":
        coordinate = peak_local_max(
            original_prediction,
            min_distance=distance_threshold_local_max,
            threshold_abs=prediction_dots_threshold,
        )
        coordinate_change_xy = coordinate[:, [1, 0]]
        centroids_list = coordinate_change_xy.tolist()
    elif (
        method == "threshold" and distance_threshold_local_max == None
    ):
        binary_map = np.where(
            original_prediction > prediction_dots_threshold, 1, 0
        ).astype(np.uint8)
        connectivity = 4  # or whatever you prefer
        output = cv2.connectedComponentsWithStats(
            binary_map, connectivity, cv2.CV_32S
        )
        # Get the results
        # num_labels = output[0] - 1  # The first cell is the number of labels
        # labels = output[1][1:]  # The second cell is the label matrix
        # stats = output[2][1:]  # The third cell is the stat matrix
        centroids = output[3][
            1:
        ]  # The fourth cell is the centroid matrix
        # np.savetxt('/home/kesix/mnt/predict_centroids_MBConv.csv', centroids, delimiter=',', fmt='%d')
        centroids_int = np.rint(centroids).astype(int)
        centroids_list = centroids_int.tolist()
    else:
        raise ValueError(f"Unknown postprocessing method: {method}")
    return centroids_list


def collate_fn(batch):
    # Apply the make_writable function to each element in the batch
    batch = np.asarray(batch)
    writable_batch = batch.copy()
    # Convert each element to a tensor
    return torch.as_tensor(writable_batch, dtype=torch.float)


def check_coord_in_mask(x, y, mask, coord_res, mask_res):
    """Checks if a given coordinate is inside the tissue mask
    Coordinate (x, y)
    Binary tissue mask default at 1.25x
    """
    if mask is None:
        return True

    try:
        return mask[int(np.round(y)), int(np.round(x))] == 1
    except IndexError:
        return False


def scale_coords(coords: list, scale_factor: float = 1):
    new_coords = []
    for coord in coords:
        x = int(coord[0] * scale_factor)
        y = int(coord[1] * scale_factor)
        new_coords.append([x, y])

    return new_coords


def detection_to_annotation_store(
    detection_records: list[dict],
    scale_factor: float = 1,
    type="polygon",
):
    """
    Convert detection records to annotation store

    Args:
        detection_records: list of {'x','y', 'type', 'probability'}
    """
    annotation_store = SQLiteStore()

    for record in detection_records:
        x = record["x"] * scale_factor
        y = record["y"] * scale_factor

        if type == "polygon":
            entry = Annotation(
                geometry=Polygon.from_bounds(
                    x - 16, y - 16, x + 16, y + 16
                ),
                properties={
                    "type": record["type"],
                    "prob": record["prob"],
                },
            )
        else:
            entry = Annotation(
                geometry=Point(x, y),
                properties={
                    "type": record["type"],
                    "prob": record["prob"],
                },
            )

        annotation_store.append(entry)

    return annotation_store


def save_detection_records_monkey(
    IOConfig: PredictionIOConfig,
    overall_detection_records: list[dict] = [],
    lymph_detection_records: list[dict] = [],
    mono_detection_records: list[dict] = [],
    wsi_id: str | None = None,
    save_mpp: float = 0.24199951445730394,
) -> None:
    """
    Save cell detection records into Monkey challenge format
    """

    output_dir = IOConfig.output_dir

    output_dict_lymphocytes = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    output_dict_monocytes = {
        "name": "monocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    output_dict_inflammatory_cells = {
        "name": "inflammatory-cells",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    for i, record in enumerate(overall_detection_records):
        counter = i + 1
        x = record["x"]
        y = record["y"]
        confidence = record["prob"]
        cell_type = record["type"]
        prediction_record = {
            "name": "Point " + str(counter),
            "point": [
                px_to_mm(x, save_mpp),
                px_to_mm(y, save_mpp),
                0.24199951445730394,
            ],
            "probability": confidence,
        }
        output_dict_inflammatory_cells["points"].append(
            prediction_record
        )

    for i, record in enumerate(lymph_detection_records):
        counter = i + 1
        x = record["x"]
        y = record["y"]
        confidence = record["prob"]
        cell_type = record["type"]
        prediction_record = {
            "name": "Point " + str(counter),
            "point": [
                px_to_mm(x, save_mpp),
                px_to_mm(y, save_mpp),
                0.24199951445730394,
            ],
            "probability": confidence,
        }

        output_dict_lymphocytes["points"].append(prediction_record)

    for i, record in enumerate(mono_detection_records):
        counter = i + 1
        x = record["x"]
        y = record["y"]
        confidence = record["prob"]
        cell_type = record["type"]
        prediction_record = {
            "name": "Point " + str(counter),
            "point": [
                px_to_mm(x, save_mpp),
                px_to_mm(y, save_mpp),
                0.24199951445730394,
            ],
            "probability": confidence,
        }
        output_dict_monocytes["points"].append(prediction_record)

    if wsi_id is not None:
        json_filename_lymphocytes = (
            f"{wsi_id}_detected-lymphocytes.json"
        )
        json_filename_monocytes = f"{wsi_id}_detected-monocytes.json"
        json_filename_inflammatory_cells = (
            f"{wsi_id}_detected-inflammatory-cells.json"
        )
    else:
        json_filename_lymphocytes = "detected-lymphocytes.json"
        json_filename_monocytes = "detected-monocytes.json"
        json_filename_inflammatory_cells = (
            "detected-inflammatory-cells.json"
        )

    output_path_json = os.path.join(
        output_dir, json_filename_lymphocytes
    )
    write_json_file(
        location=output_path_json, content=output_dict_lymphocytes
    )

    output_path_json = os.path.join(
        output_dir, json_filename_monocytes
    )
    write_json_file(
        location=output_path_json, content=output_dict_monocytes
    )

    output_path_json = os.path.join(
        output_dir, json_filename_inflammatory_cells
    )
    write_json_file(
        location=output_path_json,
        content=output_dict_inflammatory_cells,
    )


def save_detection_records_monkey_v2(
    IOConfig: PredictionIOConfig,
    lymph_detection_records: list[dict] = [],
    mono_detection_records: list[dict] = [],
    wsi_id: str | None = None,
    save_mpp: float = 0.24199951445730394,
) -> None:
    """
    Save cell detection records into Monkey challenge format
    """

    output_dir = IOConfig.output_dir

    output_dict_lymphocytes = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    output_dict_monocytes = {
        "name": "monocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    output_dict_inflammatory_cells = {
        "name": "inflammatory-cells",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    for i, record in enumerate(lymph_detection_records):
        counter = i + 1
        x = record["x"]
        y = record["y"]
        confidence = record["prob"]
        cell_type = record["type"]
        prediction_record = {
            "name": "Point " + str(counter),
            "point": [
                px_to_mm(x, save_mpp),
                px_to_mm(y, save_mpp),
                0.24199951445730394,
            ],
            "probability": confidence,
        }

        output_dict_lymphocytes["points"].append(prediction_record)
        output_dict_inflammatory_cells["points"].append(
            prediction_record
        )

    for i, record in enumerate(mono_detection_records):
        counter = i + 1
        x = record["x"]
        y = record["y"]
        confidence = record["prob"]
        cell_type = record["type"]
        prediction_record = {
            "name": "Point " + str(counter),
            "point": [
                px_to_mm(x, save_mpp),
                px_to_mm(y, save_mpp),
                0.24199951445730394,
            ],
            "probability": confidence,
        }
        output_dict_monocytes["points"].append(prediction_record)
        output_dict_inflammatory_cells["points"].append(
            prediction_record
        )

    if wsi_id is not None:
        json_filename_lymphocytes = (
            f"{wsi_id}_detected-lymphocytes.json"
        )
        json_filename_monocytes = f"{wsi_id}_detected-monocytes.json"
        json_filename_inflammatory_cells = (
            f"{wsi_id}_detected-inflammatory-cells.json"
        )
    else:
        json_filename_lymphocytes = "detected-lymphocytes.json"
        json_filename_monocytes = "detected-monocytes.json"
        json_filename_inflammatory_cells = (
            "detected-inflammatory-cells.json"
        )

    output_path_json = os.path.join(
        output_dir, json_filename_lymphocytes
    )
    write_json_file(
        location=output_path_json, content=output_dict_lymphocytes
    )

    output_path_json = os.path.join(
        output_dir, json_filename_monocytes
    )
    write_json_file(
        location=output_path_json, content=output_dict_monocytes
    )

    output_path_json = os.path.join(
        output_dir, json_filename_inflammatory_cells
    )
    write_json_file(
        location=output_path_json,
        content=output_dict_inflammatory_cells,
    )


def filter_detection_with_mask(
    detection_records: list[dict],
    mask: np.ndarray,
    points_mpp: float = 0.24199951445730394,
    mask_mpp: float = 8.0,
    margin: int = 1,
) -> list[dict]:
    """
    Filter detected points: [{'x','y','type','prob'}]
    Using binary mask.
    Points outside the binary mask are removed

    Args:
        detection_records: [{'x','y','type','prob'}]
        mask: binary mask to for filtering
        points_mpp: resolution of the detected points in mpp
        mask_mpp: resolution of the binary mask in mpp
        margin: margin in pixels to add around the mask
    Returns:
        fitlered_records: [{'x','y','type','prob'}]
    """
    scale_factor = mask_mpp / points_mpp

    filtered_records: list[dict] = []
    for record in detection_records:
        x = record["x"]
        y = record["y"]

        x_in_mask = int(np.round(x / scale_factor))
        y_in_mask = int(np.round(y / scale_factor))
        top_left = (x_in_mask - margin, y_in_mask - margin)
        top_right = (x_in_mask + margin, y_in_mask - margin)
        bottom_left = (x_in_mask - margin, y_in_mask + margin)
        bottom_right = (x_in_mask + margin, y_in_mask + margin)
        indices = [
            (int(round(xi)), int(round(yi)))
            for xi, yi in [
                top_left,
                top_right,
                bottom_left,
                bottom_right,
            ]
        ]
        valid_indices = [
            (xi, yi)
            for xi, yi in indices
            if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]
        ]
        ones_count = sum(mask[yi, xi] for xi, yi in valid_indices)
        if len(valid_indices) == 0:
            continue
        else:
            if ones_count / len(valid_indices) >= 0.5:
                filtered_records.append(record)
            else:
                continue
        # try:
        #     ones_count = sum(mask[])
        #     if mask[y_in_mask, x_in_mask] != 0:
        #         filtered_records.append(record)
        #     else:
        #         continue
        # except IndexError:
        #     continue

    return filtered_records


def normalize_detection_probs(
    detection_records: list[dict],
    min_prob: float = 0.5,
) -> list[dict]:
    new_records = []

    detected_probs = []
    for record in detection_records:
        prob = record["prob"]
        detected_probs.append(prob)

    max_detected_prob = max(detected_probs)
    min_detected_prob = min(detected_probs)

    for record in detection_records:
        prob = record["prob"]
        normalized_prob = (prob - min_detected_prob) / (
            max_detected_prob - min_detected_prob
        )
        final_prob = normalized_prob * 0.5 + min_prob
        record["prob"] = final_prob
        new_records.append(record)
    return new_records


def non_max_suppression_fast(boxes, overlapThresh):
    """Very efficient NMS function taken from pyimagesearch"""

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs,
            np.concatenate(
                ([last], np.where(overlap > overlapThresh)[0])
            ),
        )
    # return only the bounding boxes that were picked using the
    # integer data type
    # return boxes[pick].astype("int")
    return pick


def nms(boxes: np.ndarray, overlapThresh: float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        overlapThresh: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # we extract coordinates for every
    # prediction box present in P
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # we extract the confidence scores as well
    scores = boxes[:, 4]

    # calculate area of every block in P
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort the prediction boxes in P
    # according to their confidence scores
    idxs = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    pick = []

    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs,
            np.concatenate(
                ([last], np.where(overlap > overlapThresh)[0])
            ),
        )
    # return only the bounding boxes that were picked using the
    # integer data type
    # return boxes[pick].astype("int")

    return pick


def get_centerpoints(box, dist):
    """Returns centerpoints of box"""
    return (box[0] + dist, box[1] + dist)


def get_points_within_box(
    annotation_store: AnnotationStore, box
) -> list:
    query_poly = Polygon.from_bounds(box[0], box[1], box[2], box[3])
    anns = annotation_store.query(geometry=query_poly)
    results = []
    for point in anns.items():
        entry = {
            "x": point[1].coords[0][0],
            "y": point[1].coords[0][1],
            "type": point[1].properties["type"],
            "prob": point[1].properties["prob"],
        }
        results.append(entry)
    return results


def point_to_box(x, y, size, prob=None):
    """
    Convert centerpoint to bounding box of fixed size
    Args:
        x: x coordinate
        y: y coordinate
        size: radius of the box
        prob: probability of the point
    Returns:
        box: np.ndarray[4], if prob is not None [5]
    """
    if prob == None:
        return np.array([x - size, y - size, x + size, y + size])
    else:
        return np.array(
            [x - size, y - size, x + size, y + size, prob]
        )


def slide_nms(
    wsi_reader: WSIReader,
    binary_mask: np.ndarray,
    detection_record: list[dict],
    tile_size: int = 2048,
    box_size: int = 5,
    overlap_thresh: float = 0.5,
):
    """
    Iterate over detection records and perform NMS.
    For this to properly work, tiles need to be larger than
    model inference patches.
    """
    # Open WSI and detection points file

    tile_extractor = get_patch_extractor(
        input_img=wsi_reader,
        input_mask=binary_mask,
        method_name="slidingwindow",
        patch_size=(tile_size, tile_size),
        resolution=0,
        units="level",
    )

    annotation_store = detection_to_annotation_store(
        detection_record, scale_factor=1, type="Point"
    )

    center_nms_points = []
    # get 2048x2048 patch coordinates without overlap
    for bb in tile_extractor.coordinate_list:
        x_pos = bb[0]
        y_pos = bb[1]
        # Select annotations within 2048x2048 box
        box = [x_pos, y_pos, x_pos + tile_size, y_pos + tile_size]
        patch_points = get_points_within_box(annotation_store, box)

        if len(patch_points) < 2:
            continue

        # Convert each point to a box
        boxes = np.array(
            [
                point_to_box(
                    entry["x"], entry["y"], box_size, entry["prob"]
                )
                for entry in patch_points
            ]
        )
        indices = nms(boxes, overlap_thresh)
        for i in indices:
            center_nms_points.append(patch_points[i])
        # for box in nms_boxes:
        #     center_nms_points.append(get_centerpoints(box, box_size))

    return center_nms_points


def erode_mask(mask, size=3, iterations=1):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (size, size)
    )
    if mask.ndim == 4:
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i, j, :, :] = cv2.erode(
                    mask[i, j, :, :], kernel, iterations=iterations
                )
    else:
        mask = cv2.erode(mask, kernel, iterations=iterations)

    return mask


def morphological_post_processing(mask, size=3):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (size, size)
    )
    if mask.ndim == 4:
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i, j, :, :] = cv2.morphologyEx(
                    mask[i, j, :, :], cv2.MORPH_OPEN, kernel
                )
                mask[i, j, :, :] = cv2.morphologyEx(
                    mask[i, j, :, :], cv2.MORPH_CLOSE, kernel
                )
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def add_background_channel(input_mask: np.ndarray):
    """
    Add background mask to channel 0

    Args:
        input_mask: [CxHxW]
    Returns
        output_mask: [(C+1)xHxW]
    """

    output_mask = np.zeros(
        shape=(
            input_mask.shape[0] + 1,
            input_mask.shape[1],
            input_mask.shape[2],
        ),
        dtype=np.uint8,
    )

    mask_union = np.zeros(
        shape=(input_mask.shape[1], input_mask.shape[2]),
        dtype=np.uint8,
    )
    for i in range(input_mask.shape[0]):
        output_mask[i + 1, :, :] = input_mask[i, :, :]

        mask_union = np.logical_or(mask_union, input_mask[i, :, :])

    output_mask[0, :, :] = np.logical_not(mask_union)

    return output_mask


def check_image_mask_shape(wsi_path: str, mask_path: str) -> None:
    """
    Check if the image and mask have the same shape and mpp
    """
    wsi_reader = WSIReader.open(wsi_path)
    wsi_shape = wsi_reader.slide_dimensions(
        resolution=0, units="level"
    )
    mask_reader = WSIReader.open(mask_path)
    mask_shape = mask_reader.slide_dimensions(
        resolution=0, units="level"
    )

    if (wsi_shape[0] != mask_shape[0]) or (
        wsi_shape[1] != mask_shape[1]
    ):
        message = f"Image and mask have different shapes: {wsi_shape} vs {mask_shape}"
        raise ValueError(message)

    wsi_info = wsi_reader.info.as_dict()
    mask_info = mask_reader.info.as_dict()
    wsi_mpp = wsi_info["mpp"]
    mask_mpp = mask_info["mpp"]
    if (round(wsi_mpp[0],3) != round(mask_mpp[0],3)) or (round(wsi_mpp[1],3) != round(mask_mpp[1],3)):
        message = f"Image and mask have different mpp: {wsi_mpp} vs {mask_mpp}"
        raise ValueError(message)


def get_gradient_maps(image, binary_mask):

    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask), np.zeros_like(binary_mask)

    # convert image to gray scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0

    sx = ndimage.sobel(image, axis=0)
    sy = ndimage.sobel(image, axis=1)
    # normalize to [-1, 1]
    sx = sx / np.max(np.abs(sx))
    sy = sy / np.max(np.abs(sy))
    sx[binary_mask == 0] = 0
    sy[binary_mask == 0] = 0

    # Apply Gaussian filter
    sx = cv2.GaussianBlur(sx, (3, 3), 0)
    sy = cv2.GaussianBlur(sy, (3, 3), 0)
    return sx, sy


def compute_hv_map(mask):
    """
    Preprocessing step for HoVer-Net architecture.
    Compute center of mass for each nucleus, then compute distance of each nuclear pixel to its corresponding center
    of mass.
    Nuclear pixel distances are normalized to (-1, 1). Background pixels are left as 0.
    Operates on a single mask.
    Can be used in Dataset object to make Dataloader compatible with HoVer-Net.

    Based on https://github.com/vqdang/hover_net/blob/195ed9b6cc67b12f908285492796fb5c6c15a000/src/loader/augs.py#L192

    Args:
        mask (np.ndarray): Mask indicating individual nuclei. Array of shape (H, W)
    Returns:
        np.ndarray: array of hv maps of shape (2, H, W). First channel corresponds to horizontal and second vertical.
    """
    assert (
        mask.ndim == 2
    ), f"Input mask has shape {mask.shape}. Expecting a mask with 2 dimensions (H, W)"

    out = np.zeros((2, mask.shape[0], mask.shape[1]))

    mask = label(mask)
    # each individual nucleus is indexed with a different number
    inst_list = list(np.unique(mask))

    try:
        inst_list.remove(0)  # 0 is background
    # TODO: change to specific exception
    except Exception:
        return out

    for inst_id in inst_list:
        # get the mask for the nucleus
        inst_map = mask == inst_id
        inst_map = inst_map.astype(np.uint8)
        contours, _ = cv2.findContours(
            inst_map, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
        )

        # get center of mass coords
        mom = cv2.moments(contours[0])
        com_x = mom["m10"] / (mom["m00"] + 1e-6)
        com_y = mom["m01"] / (mom["m00"] + 1e-6)
        inst_com = (int(com_y), int(com_x))

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        # add to output mask
        # this works assuming background is 0, and each pixel is assigned to only one nucleus.
        out[0, :, :] += inst_x
        out[1, :, :] += inst_y
    return out
