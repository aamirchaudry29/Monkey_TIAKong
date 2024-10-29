import json
import os
import re

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from shapely import Polygon
from skimage.feature import peak_local_max
from tiatoolbox.annotation.storage import Annotation, SQLiteStore

from monkey.config import PredictionIOConfig, TrainingIOConfig


def load_image(
    file_id: str, IOConfig: TrainingIOConfig
) -> np.ndarray:
    image_name = f"{file_id}.npy"
    image_path = os.path.join(IOConfig.image_dir, image_name)
    image = np.load(image_path)
    return image


def load_mask(file_id: str, IOConfig: TrainingIOConfig) -> np.ndarray:
    mask_name = f"{file_id}.npy"
    mask_path = os.path.join(IOConfig.mask_dir, mask_name)
    mask = np.load(mask_path)
    return mask


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
    IOConfig: TrainingIOConfig,
    val_fold: int = 1
):
    """
    Retrieve train and validation patch ids from pre-processed json file
    """
    split_info_json_path = os.path.join(
        IOConfig.dataset_dir,
        'patch_level_split.json'
    )

    split_info = open_json_file(split_info_json_path)
    fold_info = split_info[f'Fold_{val_fold}']
    split = {
        "train_file_ids": fold_info['train_files'],
        "test_file_ids": fold_info['test_files']
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
    detection_records: list[dict], scale_factor: float = 1
):
    """
    Convert detection records to annotation store

    Args:
        detection_records: list of {'x','y', 'type', 'probability'}
    """
    annotation_store = SQLiteStore()

    for record in detection_records:
        x = int(record["x"] * scale_factor)
        y = int(record["y"] * scale_factor)
        annotation_store.append(
            Annotation(
                geometry=Polygon.from_bounds(
                    x - 16, y - 16, x + 16, y + 16
                ),
                properties={
                    "type": record["type"],
                    "prob": record["prob"],
                },
            )
        )

    return annotation_store


def save_detection_records_monkey(
    detection_records: list[dict], IOConfig: PredictionIOConfig
):
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

    for i, record in enumerate(detection_records):
        counter = i + 1
        x = record["x"]
        y = record["y"]
        confidence = record["prob"]
        cell_type = record["type"]
        prediction_record = {
            "name": "Point " + str(counter),
            "point": [
                px_to_mm(x, 0.24199951445730394),
                px_to_mm(y, 0.24199951445730394),
                0.24199951445730394,
            ],
            "probability": confidence,
        }
        if cell_type == "lymphocyte":
            output_dict_lymphocytes["points"].append(
                prediction_record
            )
        if cell_type == "monocytes":
            output_dict_monocytes["points"].append(prediction_record)
        output_dict_inflammatory_cells["points"].append(
            prediction_record
        )

    json_filename_lymphocytes = "detected-lymphocytes.json"
    output_path_json = os.path.join(
        output_dir, json_filename_lymphocytes
    )
    write_json_file(
        location=output_path_json, content=output_dict_lymphocytes
    )

    json_filename_monocytes = "detected-monocytes.json"
    output_path_json = os.path.join(
        output_dir, json_filename_monocytes
    )
    write_json_file(
        location=output_path_json, content=output_dict_monocytes
    )

    json_filename_inflammatory_cells = (
        "detected-inflammatory-cells.json"
    )
    # it should be replaced with correct json files
    output_path_json = os.path.join(
        output_dir, json_filename_inflammatory_cells
    )
    write_json_file(
        location=output_path_json,
        content=output_dict_inflammatory_cells,
    )
