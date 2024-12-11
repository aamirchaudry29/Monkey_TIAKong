# Singular pipeline for lymphocyte and monocyte detection
# Detect and classify cells using a single model
import sys

sys.path.insert(
    0, "/home/u1910100/cloud_workspace/GitHub/Monkey_TIAKong/"
)

import os
from multiprocessing import Pool
from pprint import pprint

import torch
from tiatoolbox.wsicore.wsireader import WSIReader
from tqdm.auto import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    detection_to_annotation_store,
    extract_id,
    normalize_detection_probs,
    open_json_file,
    save_detection_records_monkey,
)
from monkey.model.cellvit.cellvit import CellVit256_Unet
from monkey.model.efficientunetb0.architecture import (
    get_multihead_efficientunet,
)
from monkey.model.hovernext.model import (
    get_convnext_unet,
    get_custom_hovernext,
)
from optimization.post_process import post_process_detection


def cross_validation(fold_number: int = 1):
    detector_model_name = "convnext_base_lizard_512"
    fold = fold_number
    pprint(
        f"Post-processing raw prediction from {detector_model_name}"
    )
    model_res = 0
    units = "level"
    pprint(f"Detect at {model_res} {units}")

    config = PredictionIOConfig(
        wsi_dir="/mnt/lab-share/Monkey/Dataset/images/pas-cpg",
        mask_dir="/mnt/lab-share/Monkey/Dataset/images/tissue-masks",
        output_dir=f"/home/u1910100/cloud_workspace/data/Monkey/local_output/{detector_model_name}/Fold_{fold}",
        patch_size=512,
        resolution=model_res,
        units=units,
        stride=480,
        thresholds=[0.3, 0.3, 0.3],
        min_distances=[11, 11, 11],
        nms_boxes=[20, 11, 20],
        nms_overlap_thresh=0.5,
    )
    print(f"thresholds: {config.thresholds}")
    print(f"min_distances: {config.min_distances}")
    print(f"nms_boxes: {config.nms_boxes}")
    # config = PredictionIOConfig(
    #     wsi_dir="/home/u1910100/Downloads/Monkey/images/pas-cpg",
    #     mask_dir="/home/u1910100/Downloads/Monkey/images/tissue-masks",
    #     output_dir=f"/home/u1910100/Documents/Monkey/local_output/{detector_model_name}/Fold_{fold}",
    #     patch_size=256,
    #     resolution=0,
    #     units="level",
    #     stride=216,
    # )

    split_info = open_json_file(
        "/mnt/lab-share/Monkey/patches_256/wsi_level_split.json"
    )
    # split_info = open_json_file(
    #     "/home/u1910100/Documents/Monkey/patches_256/wsi_level_split.json"
    # )

    val_wsi_files = split_info[f"Fold_{fold}"]["test_files"]

    print(val_wsi_files)

    detectors = []

    with Pool(20) as p:
        p.starmap(
            process_one_wsi,
            [
                (config, detectors, wsi_name)
                for wsi_name in val_wsi_files
            ],
        )


def process_one_wsi(config, detectors, wsi_name):
    wsi_id = extract_id(wsi_name)
    mask_name = f"{wsi_id}_mask.tif"
    wsi_dir = config.wsi_dir
    wsi_path = os.path.join(wsi_dir, wsi_name)
    wsi_reader = WSIReader.open(wsi_path)
    base_mpp = wsi_reader.convert_resolution_units(
        input_res=0, input_unit="level", output_unit="mpp"
    )[0]

    detection_records = post_process_detection(
        wsi_name, mask_name, config, detectors
    )

    inflamm_records = detection_records["inflamm_records"]
    lymph_records = detection_records["lymph_records"]
    mono_records = detection_records["mono_records"]
    print(f"{len(inflamm_records)} final detected inflamm")
    print(f"{len(lymph_records)} final detected lymph")
    print(f"{len(mono_records)} final detected mono")

    save_detection_records_monkey(
        config,
        inflamm_records,
        lymph_records,
        mono_records,
        wsi_id=wsi_id,
        save_mpp=base_mpp,
    )

    print("finished")


if __name__ == "__main__":
    folds = [1, 2, 3, 4, 5]

    for i in folds:
        pprint(f"Fold {i}")
        cross_validation(i)
