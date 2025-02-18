# Singular pipeline for lymphocyte and monocyte detection
# Detect and classify cells using a single model
import sys

sys.path.insert(
    0, "/home/u1910100/cloud_workspace/GitHub/Monkey_TIAKong/"
)

import os
from pprint import pprint
import ttach as tta
import torch
from tqdm.auto import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    extract_id,
    open_json_file,
)

from monkey.model.hovernext.model import (
    get_custom_hovernext,
)
from optimization.raw_prediction import wsi_raw_prediction
import click

@click.command()
@click.option("--fold", default=1)
def cross_validation(fold: int = 1):
    detector_model_name = "efficientnetv2_l_multitask_det_decoder_v4"
    pprint(
        f"Making raw prediction using {detector_model_name} fold {fold}"
    )
    model_res = 0
    units = "level"

    pprint(f"Detect at {model_res} {units}")

    config = PredictionIOConfig(
        wsi_dir="/mnt/lab-share/Monkey/Dataset/images/pas-cpg",
        mask_dir="/mnt/lab-share/Monkey/Dataset/images/tissue-masks",
        output_dir=f"/home/u1910100/cloud_workspace/data/Monkey/local_output/{detector_model_name}/Fold_{fold}",
        patch_size=256,
        resolution=model_res,
        units=units,
        stride=224,
        thresholds=[0.5, 0.5, 0.5],
        min_distances=[11, 11, 11],
        nms_boxes=[11, 11, 11],
        nms_overlap_thresh=0.5,
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
        f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_{fold}/best_val.pth",
        # f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_2/best.pth",
        # f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_4/best.pth",
    ]
    detectors = []
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )
    for weight_path in detector_weight_paths:

        model = get_custom_hovernext(
            enc="tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=False,
            use_batchnorm=True,
            attention_type="scse",
            decoders_out_channels=[3, 3, 3],
            center=True,
        )
        checkpoint = torch.load(weight_path)
        print(f"epoch: {checkpoint['epoch']}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model.to("cuda")
        model = tta.SegmentationTTAWrapper(model, transforms)
        detectors.append(model)

    for wsi_name in tqdm(val_wsi_files):
        wsi_id = extract_id(wsi_name)
        mask_name = f"{wsi_id}_mask.tif"

        wsi_raw_prediction(wsi_name, mask_name, config, detectors)

        print("finished")


if __name__ == "__main__":
    cross_validation()
