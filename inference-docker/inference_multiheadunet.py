import os
from glob import glob
from pathlib import Path

import torch

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import save_detection_records_monkey
from monkey.model.efficientunetb0.architecture import (
    get_multihead_efficientunet,
)
from prediction.multihead_unet_prediction import wsi_detection_in_mask

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MODEL_DIR = Path("/opt/ml/model")


def load_detectors() -> list[torch.nn.Module]:
    detectors = []
    detector_weight_paths = [
        os.path.join(MODEL_DIR, "1.pth"),
        os.path.join(MODEL_DIR, "2.pth"),
        os.path.join(MODEL_DIR, "3.pth"),
    ]
    for weight_path in detector_weight_paths:
        detector = get_multihead_efficientunet(
            [
                2,
                1,
                1,
            ],
            pretrained=False,
        )
        checkpoint = torch.load(weight_path)
        detector.load_state_dict(checkpoint["model"])
        detector.eval()
        detector.to("cuda")
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

    config = PredictionIOConfig(
        wsi_dir=wsi_dir,
        mask_dir=mask_dir,
        output_dir=OUTPUT_PATH,
        patch_size=256,
        resolution=0,
        units="level",
        stride=216,
    )

    detectors = load_detectors()

    print("start detection")
    detection_records = wsi_detection_in_mask(
        wsi_name, mask_name, config, detectors
    )

    inflamm_records = detection_records["inflamm_records"]
    lymph_records = detection_records["lymph_records"]
    mono_records = detection_records["mono_records"]
    print(f"{len(inflamm_records)} final detected inflamm")
    print(f"{len(lymph_records)} final detected lymph")
    print(f"{len(mono_records)} final detected mono")

    # Save result in Monkey Challenge format
    save_detection_records_monkey(
        config,
        inflamm_records,
        lymph_records,
        mono_records,
        wsi_id=None,
    )
    print("finished")


if __name__ == "__main__":
    detect()
