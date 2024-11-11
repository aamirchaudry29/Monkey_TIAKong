import os
from glob import glob
from pathlib import Path

import torch

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import save_detection_records_monkey
from monkey.model.efficientunetb0.architecture import (
    get_efficientunet_b0_MBConv,
)
from prediction.detection import wsi_detection_in_mask

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MODEL_DIR = Path("/opt/ml/model")


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
        stride=224,
        threshold=0.3,
        min_size=7,
    )

    models = []
    model_path = os.path.join(MODEL_DIR, "efficientunetb0_seg.pth")
    model = get_efficientunet_b0_MBConv(pretrained=False)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"])
    model.to("cuda")
    model.eval()
    models.append(model)
    detection_records = wsi_detection_in_mask(
        wsi_name, mask_name, config, models
    )
    print(f"{len(detection_records)} final detected cells")
    # Save result in Monkey Challenge format
    save_detection_records_monkey(
        detection_records, config, wsi_id=None
    )
    print("finished")


if __name__ == "__main__":
    detect()
