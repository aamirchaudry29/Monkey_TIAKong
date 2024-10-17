import json
import os
import re
from typing import Tuple

import cv2
import numpy as np
import scipy.ndimage as ndi

from monkey.config import TrainingIOConfig


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
    if val_fold < 0 or val_fold > 4:
        raise ValueError(f"Invalid test centre {val_fold}")

    test_centre = centres[val_fold]

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
