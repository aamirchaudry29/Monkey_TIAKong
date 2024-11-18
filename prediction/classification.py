import os
from typing import Tuple

import numpy as np
import torch
from tiatoolbox.wsicore.wsireader import WSIReader
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import imagenet_normalise_torch


def classify_patch(
    patch: np.ndarray, models: list[torch.nn.Module], thresh=0.7
):
    """
    Classify a single patch image into lymphocyte or monocyte

    Args:
        patch: RGB image: (HxWx3)
        models
    Returns:
        result: {'type'}
    """
    patch = np.moveaxis(patch, 2, 0)
    patch = torch.tensor(patch, dtype=torch.float)
    patch = resize(patch, (224, 224))
    patch = torch.unsqueeze(patch, dim=0)

    patch = patch / 255
    patch = imagenet_normalise_torch(patch)
    patch = patch.to("cuda").float()

    with torch.no_grad():
        monocyte_prob = 0.0
        for model in models:
            logits = model(patch)
            pred_prob = torch.sigmoid(logits)
            pred_prob = pred_prob.cpu().detach().numpy()
            monocyte_prob += pred_prob[0][0]

    monocyte_prob = monocyte_prob / len(models)

    if monocyte_prob > thresh:
        return {"type": "monocyte"}
    else:
        return {"type": "lymphocyte"}


def detected_cell_classification(
    detected_cells: list[dict],
    wsi_name: str,
    IOConfig: PredictionIOConfig,
    models: list[torch.nn.Module],
    thresh: float = 0.7,
):
    """
    Classify inflammatory cells into lymphocytes and monocytes

    Args:
        detected_cells: [{'x', 'y', 'type', 'prob'}]
        wsi_name
        IOConfig
        models
    Returns:
        detected_cells_classified: [{'x', 'y', 'type', 'prob'}]
    """

    wsi_dir = IOConfig.wsi_dir
    wsi_without_ext = os.path.splitext(wsi_name)[0]
    wsi_path = os.path.join(wsi_dir, wsi_name)

    wsi_reader = WSIReader.open(wsi_path)

    detection_classification_records = []

    for detection_record in tqdm(
        detected_cells,
        leave=False,
        desc=f"{wsi_without_ext} classification progress",
    ):
        top_left_x = int(detection_record["x"] - 16)
        top_left_y = int(detection_record["y"] - 16)

        image_patch = wsi_reader.read_rect(
            (top_left_x, top_left_y), (32, 32)
        )

        classification_result = classify_patch(
            image_patch, models, thresh
        )

        record = {
            "x": detection_record["x"],
            "y": detection_record["y"],
            "type": classification_result["type"],
            "prob": detection_record["prob"],
        }
        detection_classification_records.append(record)

    return detection_classification_records
