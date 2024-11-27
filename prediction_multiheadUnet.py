# Singular pipeline for lymphocyte and monocyte detection
# Detect and classify cells using a single model

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
from monkey.model.efficientunetb0.architecture import (
    get_multihead_efficientunet,
)
from prediction.multihead_unet_prediction import wsi_detection_in_mask


def cross_validation(fold_number: int = 1):
    detector_model_name = "multihead_unet_experiment"
    fold = fold_number
    pprint(f"Multiclass detection using {detector_model_name}")

    config = PredictionIOConfig(
        wsi_dir="/mnt/lab-share/Monkey/Dataset/images/pas-cpg",
        mask_dir="/mnt/lab-share/Monkey/Dataset/images/tissue-masks",
        output_dir=f"/home/u1910100/cloud_workspace/data/Monkey/local_output/{detector_model_name}/Fold_{fold}",
        patch_size=256,
        resolution=0,
        units="level",
        stride=224,
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
    # detector_weight_paths = [
    #     f"/home/u1910100/Documents/Monkey/runs/cell_multiclass_det/{detector_model_name}/fold_{fold}/epoch_100.pth",
    # ]
    detector_weight_paths = [
        f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_{fold}/epoch_100.pth",
    ]
    detectors = []
    for weight_path in detector_weight_paths:
        model = get_multihead_efficientunet(
            pretrained=False, out_channels=[2, 1, 1]
        )
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model.to("cuda")
        detectors.append(model)

    for wsi_name in tqdm(val_wsi_files):
        wsi_name_without_ext = os.path.splitext(wsi_name)[0]
        wsi_id = extract_id(wsi_name)
        mask_name = f"{wsi_id}_mask.tif"

        detection_records = wsi_detection_in_mask(
            wsi_name, mask_name, config, detectors
        )

        inflamm_records = detection_records["inflamm_records"]
        lymph_records = detection_records["lymph_records"]
        mono_records = detection_records["mono_records"]
        print(f"{len(inflamm_records)} final detected inflamm")
        print(f"{len(lymph_records)} final detected lymph")
        print(f"{len(mono_records)} final detected mono")

        # Save to AnnotationStore for visualization
        # If model if not running at baseline res:
        # scale_factor = model_res / 0.24199951445730394
        # annoation_store = detection_to_annotation_store(
        #     detection_records, scale_factor=1.0
        # )
        # store_save_path = os.path.join(
        #     config.output_dir, f"{wsi_name_without_ext}.db"
        # )
        # annoation_store.dump(store_save_path)

        # Save result in Monkey Challenge format
        # L2 lymphocyte vs monocyte detection
        save_detection_records_monkey(
            config,
            inflamm_records,
            lymph_records,
            mono_records,
            wsi_id=wsi_id,
        )
        print("finished")


if __name__ == "__main__":
    for i in range(1, 3):
        pprint(f"Fold {i}")
        cross_validation(i)
