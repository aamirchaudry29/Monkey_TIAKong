import os
from glob import glob
from pathlib import Path

import torch

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import save_detection_records_monkey
from monkey.model.classification_model.efficientnet_b0 import (
    EfficientNet_B0,
)
from monkey.model.efficientunetb0.architecture import (
    get_efficientunet_b0_MBConv,
)
from prediction.classification import detected_cell_classification
from Github.Monkey_TIAKong.prediction.overall_detection import wsi_detection_in_mask

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MODEL_DIR = Path("/opt/ml/model")


def load_detectors() -> list[torch.nn.Module]:
    detectors = []
    detector_weight_paths = [
        os.path.join(MODEL_DIR, "efficientunetb0_seg_1.pth"),
        os.path.join(MODEL_DIR, "efficientunetb0_seg_2.pth"),
        os.path.join(MODEL_DIR, "efficientunetb0_seg_3.pth"),
    ]
    for weight_path in detector_weight_paths:
        detector = get_efficientunet_b0_MBConv(pretrained=False)
        checkpoint = torch.load(weight_path)
        detector.load_state_dict(checkpoint["model"])
        detector.eval()
        detector.to("cuda")
        detectors.append(detector)
    return detectors


def load_classifiers() -> list[torch.nn.Module]:
    classifiers = []
    classifier_weight_paths = [
        os.path.join(MODEL_DIR, "efficientnetb0_cls_1.pth"),
        os.path.join(MODEL_DIR, "efficientnetb0_cls_2.pth"),
        os.path.join(MODEL_DIR, "efficientnetb0_cls_3.pth"),
    ]
    for weight_path in classifier_weight_paths:
        classifier = EfficientNet_B0(
            input_channels=3, num_classes=1, pretrained=False
        )
        checkpoint = torch.load(weight_path)
        classifier.load_state_dict(checkpoint["model"])
        classifier.eval()
        classifier.to("cuda")
        classifiers.append(classifier)
    return classifiers


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
        threshold=0.5,
        min_size=3,
        threshold=0.5,
        min_size=3,
    )

    detectors = load_detectors()
    classifiers = load_classifiers()

    print("start overall detection")
    overall_detection_records = wsi_detection_in_mask(
        wsi_name, mask_name, config, detectors
    )
    print(f"{len(overall_detection_records)} overall detected cells")
    # Save result in Monkey Challenge format
    save_detection_records_monkey(
        overall_detection_records,
        config,
        task="overall_detection",
        wsi_id=None,
    )

    print("start classification")

    classification_detection_records = detected_cell_classification(
        overall_detection_records,
        wsi_name,
        config,
        classifiers,
        thresh=0.43,
    )

    save_detection_records_monkey(
        classification_detection_records,
        config,
        wsi_id=None,
        task="detection_classification",
    )

    print("finished")


if __name__ == "__main__":
    detect()
