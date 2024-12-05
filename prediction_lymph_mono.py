import os
from pprint import pprint

import click
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
from prediction.lymph_mono_det_prediction import wsi_detection_in_mask


@click.command()
@click.option("--fold", default=1)
def cross_validation(fold: int = 1):
    detector_model_name = "hovernext_large_lizzard_pretrained_v2"
    pprint(f"Multiclass detection using {detector_model_name}")
    pprint(f"Fold {fold}")
    model_mpp = 0.24199951445730394
    baseline_mpp = 0.24199951445730394
    pprint(f"Detect at {model_mpp} mpp")

    config = PredictionIOConfig(
        wsi_dir="/mnt/lab-share/Monkey/Dataset/images/pas-cpg",
        mask_dir="/mnt/lab-share/Monkey/Dataset/images/tissue-masks",
        output_dir=f"/home/u1910100/cloud_workspace/data/Monkey/local_output/{detector_model_name}/Fold_{fold}",
        patch_size=256,
        resolution=model_mpp,
        units="mpp",
        stride=224,
        thresholds=[0.5, 0.5, 0.5],
        min_distances=[7, 7, 7],
        nms_boxes=[30, 16, 40],
        nms_overlap_thresh=0.5,
    )

    split_info = open_json_file(
        "/mnt/lab-share/Monkey/patches_256/wsi_level_split.json"
    )
    val_wsi_files = split_info[f"Fold_{fold}"]["test_files"]

    print(val_wsi_files)

    # Load models
    detector_weight_paths = [
        f"/home/u1910100/cloud_workspace/data/Monkey/Monkey_Detection_2_channel/{detector_model_name}/fold_{fold}/best.pth",
    ]
    detectors = []
    for weight_path in detector_weight_paths:
        # model = get_multihead_efficientunet(
        #     pretrained=False, out_channels=[1, 1, 1]
        # )
        # model = get_custom_hovernext(
        #     pretrained=False,
        #     num_heads=2,
        #     decoders_out_channels=[1,1],
        #     use_batchnorm=True,
        #     attention_type='scse'
        # )
        model = get_convnext_unet(
            enc="convnextv2_large.fcmae_ft_in22k_in1k",
            pretrained=False,
            out_classes=2,
            use_batchnorm=True,
            attention_type="scse",
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
        #     lymph_records, scale_factor=1.0
        # )
        # store_save_path = os.path.join(
        #     config.output_dir, f"{wsi_name_without_ext}_lymph.db"
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
    cross_validation()
