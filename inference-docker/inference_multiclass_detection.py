import os
from glob import glob
from pathlib import Path

import torch
import ttach as tta
from tiatoolbox.wsicore.wsireader import WSIReader

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import save_detection_records_monkey
from monkey.model.efficientunetb0.architecture import (
    get_multihead_efficientunet,
)
from monkey.model.hovernext.model import get_custom_hovernext
from prediction.multiclass_detection import wsi_detection_in_mask_v2

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MODEL_DIR = Path("/opt/ml/model")


def load_detectors() -> list[torch.nn.Module]:
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 180, 90, 270]),
        ]
    )
    detectors = []
    detector_weight_paths = [
        os.path.join(MODEL_DIR, "1.pth"),
        os.path.join(MODEL_DIR, "2.pth"),
        os.path.join(MODEL_DIR, "4.pth"),
    ]
    for weight_path in detector_weight_paths:
        detector = get_custom_hovernext(
            enc="tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=False,
            use_batchnorm=True,
            attention_type="scse",
            decoders_out_channels=[3, 3, 3],
            center=True
        )
        checkpoint = torch.load(weight_path)
        detector.load_state_dict(checkpoint["model"])
        detector.eval()
        detector.to("cuda")
        detector = tta.SegmentationTTAWrapper(detector, transforms)
        detectors.append(detector)
    return detectors


def detect():
    print("Starting detection")

    wsi_dir = os.path.join(
        INPUT_PATH,
        "images/kidney-transplant-biopsy-wsi-pas",
    )

    mask_dir = os.path.join(INPUT_PATH, "images/tissue-mask")

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
    print(f"wsi_path={wsi_path}")
    mask_path = mask_paths[0]
    print(f"mask_path={mask_path}")

    wsi_name = os.path.basename(wsi_path)
    mask_name = os.path.basename(mask_path)

    print(f"wsi_name={wsi_name}")
    print(f"mask_name={mask_name}")

    model_res = 0
    units = "level"

    print(f"Detect at {model_res} {units}")
    config = PredictionIOConfig(
        wsi_dir=wsi_dir,
        mask_dir=mask_dir,
        output_dir=OUTPUT_PATH,
        patch_size=256,
        resolution=model_res,
        units=units,
        stride=224,
        thresholds=[0.5, 0.5, 0.5],
        min_distances=[11, 11, 11],
        nms_boxes=[11, 11, 11],
        nms_overlap_thresh=0.5,
    )

    detectors = load_detectors()

    print("start detection")
    detection_records = wsi_detection_in_mask_v2(
        wsi_name, mask_name, config, detectors
    )

    inflamm_records = detection_records["inflamm_records"]
    lymph_records = detection_records["lymph_records"]
    mono_records = detection_records["mono_records"]
    print(f"{len(inflamm_records)} final detected inflamm")
    print(f"{len(lymph_records)} final detected lymph")
    print(f"{len(mono_records)} final detected mono")

    # Save result in Monkey Challenge format
    wsi_reader = WSIReader.open(wsi_path)
    base_mpp = wsi_reader.convert_resolution_units(
        input_res=0, input_unit="level", output_unit="mpp"
    )[0]
    print(f"Base mpp {base_mpp}")
    save_detection_records_monkey(
        config,
        inflamm_records,
        lymph_records,
        mono_records,
        wsi_id=None,
        save_mpp=base_mpp,
    )
    print("finished")


if __name__ == "__main__":
    detect()
