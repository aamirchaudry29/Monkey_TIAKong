# Singular pipeline for lymphocyte and monocyte detection
# Detect and classify cells using a single model

import os
from pprint import pprint

import click
import torch
import ttach as tta
from tiatoolbox.wsicore.wsireader import WSIReader
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
from prediction.multiclass_segmentation import (
    wsi_segmentation_in_mask,
)


@click.command()
@click.option("--fold", default=1)
def cross_validation(fold: int = 1):
    detector_model_name = "convnextv2_tiny_pannuke_seg_v2"
    pprint(f"Multiclass segmentation using {detector_model_name}")
    pprint(f"Fold {fold}")
    model_res = 0
    units = "level"
    pprint(f"Model running at {model_res} {units}")

    config = PredictionIOConfig(
        wsi_dir="/mnt/lab-share/Monkey/Dataset/images/pas-cpg",
        mask_dir="/mnt/lab-share/Monkey/Dataset/images/tissue-masks",
        output_dir=f"/home/u1910100/cloud_workspace/data/Monkey/local_output/{detector_model_name}/Fold_{fold}",
        patch_size=256,
        resolution=model_res,
        units=units,
        stride=216,
        thresholds=[0.5, 0.5, 0.5, 0.3],
        nms_boxes=[11, 11, 11],
        nms_overlap_thresh=0.5,
        seg_model_version=2
    )
    # config = PredictionIOConfig(
    #     wsi_dir="/home/u1910100/Downloads/Monkey/images/pas-cpg",
    #     mask_dir="/home/u1910100/Downloads/Monkey/images/tissue-masks",
    #     output_dir=f"/home/u1910100/Documents/Monkey/local_output/{detector_model_name}/Fold_{fold}",
    #     patch_size=256,
    #     resolution=model_res,
    #     units=units,
    #     stride=224,
    #     thresholds=[0.3, 0.3, 0.3],
    #     min_distances=[11, 11, 11],
    #     nms_boxes=[11, 11, 11],
    #     nms_overlap_thresh=0.3,
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
        f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_{fold}/epoch_75.pth",
        # f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_2/epoch_75.pth",
        # f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_4/epoch_75.pth",
    ]
    detectors = []
    transforms = tta.Compose(
        [
            # tta.HorizontalFlip(),
            # tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 180, 90, 270]),
        ]
    )
    for weight_path in detector_weight_paths:
        # model = get_multihead_efficientunet(
        #     pretrained=False, out_channels=[2, 1, 1]
        # )
        # model = get_custom_hovernext(pretrained=False)
        model = get_custom_hovernext(
            enc="convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=False,
            decoders_out_channels=[2, 2, 2],
            use_batchnorm=True,
            attention_type="scse",
        )
        checkpoint = torch.load(weight_path)
        print(f"Epoch: {checkpoint['epoch']}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model.to("cuda")
        # model = tta.SegmentationTTAWrapper(model, transforms)
        detectors.append(model)

    for wsi_name in tqdm(val_wsi_files):
        wsi_name_without_ext = os.path.splitext(wsi_name)[0]
        wsi_id = extract_id(wsi_name)
        mask_name = f"{wsi_id}_mask.tif"

        detection_records = wsi_segmentation_in_mask(
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

        wsi_dir = config.wsi_dir
        wsi_path = os.path.join(wsi_dir, wsi_name)
        wsi_reader = WSIReader.open(wsi_path)
        base_mpp = wsi_reader.convert_resolution_units(
            input_res=0, input_unit="level", output_unit="mpp"
        )[0]
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
    cross_validation()
