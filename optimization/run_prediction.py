# Singular pipeline for lymphocyte and monocyte detection
# Detect and classify cells using a single model
import sys

sys.path.insert(
    0, "/home/u1910100/cloud_workspace/GitHub/Monkey_TIAKong/"
)

import os
from pprint import pprint

import torch
from tqdm.auto import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    detection_to_annotation_store,
    extract_id,
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
from optimization.raw_prediction import wsi_raw_prediction


def cross_validation(fold_number: int = 1):
    detector_model_name = "convnext_base_lizard_512"
    fold = fold_number
    pprint(f"Making raw prediction using {detector_model_name} fold {fold}")
    model_res = 0
    units='level'

    pprint(f"Detect at {model_res} {units}")

    config = PredictionIOConfig(
        wsi_dir="/mnt/lab-share/Monkey/Dataset/images/pas-cpg",
        mask_dir="/mnt/lab-share/Monkey/Dataset/images/tissue-masks",
        output_dir=f"/home/u1910100/cloud_workspace/data/Monkey/local_output/{detector_model_name}/Fold_{fold}",
        patch_size=512,
        resolution=model_res,
        units=units,
        stride=480,
        thresholds=[0.5, 0.5, 0.5],
        min_distances=[11, 11, 11],
        nms_boxes=[11, 11, 11],
        nms_overlap_thresh=0.7,
    )
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

    # Load models
    detector_weight_paths = [
        f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_{fold}/best.pth",
        # f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_2/best.pth",
        # f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_4/best.pth",
    ]
    detectors = []
    for weight_path in detector_weight_paths:
        # model = get_multihead_efficientunet(
        #     pretrained=False, out_channels=[1, 1, 1]
        # )
        # model = get_custom_hovernext(pretrained=False)
        model = get_custom_hovernext(
            enc="convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=False,
            use_batchnorm=True,
            attention_type="scse",
        )
        checkpoint = torch.load(weight_path)
        print(f"epoch: {checkpoint['epoch']}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model.to("cuda")
        detectors.append(model)

    for wsi_name in tqdm(val_wsi_files):
        wsi_id = extract_id(wsi_name)
        mask_name = f"{wsi_id}_mask.tif"

        wsi_raw_prediction(wsi_name, mask_name, config, detectors)

        print("finished")


if __name__ == "__main__":
    for i in range(3, 6):
        pprint(f"Fold {i}")
        cross_validation(i)
