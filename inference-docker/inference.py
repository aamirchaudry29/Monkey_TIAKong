import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely import Point, Polygon, box
from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.models import MapDe
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader

from monkey.data.data_utils import px_to_mm, write_json_file

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MODEL_DIR = Path("/opt/ml/model")


def load_mapde(weights_path) -> MapDe:
    """Loads MapDe model with specified weights."""
    model = MapDe(min_distance=4, threshold_abs=250)
    map_location = select_device(on_gpu=True)
    print(weights_path)
    pretrained = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(pretrained)

    return model


def scale_coords(coords: list, scale_factor: float = 1):
    new_coords = []
    for coord in coords:
        x = int(coord[0] * scale_factor)
        y = int(coord[1] * scale_factor)
        new_coords.append([x, y])

    return new_coords


def detect():
    print("Starting detection")
    image_paths = glob(
        os.path.join(
            INPUT_PATH,
            "images/kidney-transplant-biopsy-wsi-pas/*.tif",
        )
    )
    mask_paths = glob(
        os.path.join(INPUT_PATH, "images/tissue-mask/*.tif")
    )

    wsi_path = image_paths[0]
    mask_path = mask_paths[0]

    wsi_reader = WSIReader.open(wsi_path)
    mask_reader = WSIReader.open(mask_path)

    base_mpp = wsi_reader.convert_resolution_units(
        input_res=0, input_unit="level", output_unit="mpp"
    )[0]

    mask_thumbnail = mask_reader.slide_thumbnail()
    binary_mask = mask_thumbnail[:, :, 0]

    patch_extractor = get_patch_extractor(
        input_img=wsi_reader,
        input_mask=binary_mask,
        method_name="slidingwindow",
        patch_size=(252, 252),
        stride=(252, 252),
        resolution=0.5,
        units="mpp",
    )

    print(len(patch_extractor))
    mapde_result = []

    weight_root = os.path.join(MODEL_DIR, "mapde-conic.pth")
    model = load_mapde(weight_root)
    model.to("cuda")

    for i, patch in enumerate(patch_extractor):
        bb = patch_extractor.coordinate_list[i]

        patch_input = torch.from_numpy(patch)[None]
        patch_input.to("cuda").float()

        output = model.infer_batch(model, patch_input, on_gpu=True)
        output = model.postproc(output[0])
        for coord in output:
            mapde_result.append([coord[0] + bb[0], coord[1] + bb[1]])

    scale_factor = 0.5 / 0.24199951445730394
    mapde_result = scale_coords(mapde_result, scale_factor)

    output_path = OUTPUT_PATH

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

    for i, coord in enumerate(mapde_result):
        counter = i + 1
        x = coord[0]
        y = coord[1]
        confidence = 0.9
        prediction_record = {
            "name": "Point " + str(counter),
            "point": [
                px_to_mm(x, 0.24199951445730394),
                px_to_mm(y, 0.24199951445730394),
                0.24199951445730394,
            ],
            "probability": confidence,
        }
        output_dict_lymphocytes["points"].append(prediction_record)
        output_dict_monocytes["points"].append(
            prediction_record
        )  # should be replaced with detected monocytes
        output_dict_inflammatory_cells["points"].append(
            prediction_record
        )

    json_filename = "detected-lymphocytes.json"
    output_path = OUTPUT_PATH
    output_path_json = os.path.join(output_path, json_filename)
    write_json_file(
        location=output_path_json, content=output_dict_lymphocytes
    )

    json_filename_monocytes = "detected-monocytes.json"
    # it should be replaced with correct json files
    output_path_json = os.path.join(
        output_path, json_filename_monocytes
    )
    write_json_file(
        location=output_path_json, content=output_dict_monocytes
    )

    json_filename_inflammatory_cells = (
        "detected-inflammatory-cells.json"
    )
    # it should be replaced with correct json files
    output_path_json = os.path.join(
        output_path, json_filename_inflammatory_cells
    )
    write_json_file(
        location=output_path_json,
        content=output_dict_inflammatory_cells,
    )

    print("finished")


if __name__ == "__main__":
    detect()
