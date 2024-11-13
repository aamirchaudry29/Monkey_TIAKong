import os

import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    detection_to_annotation_store,
    extract_id,
    open_json_file,
    save_detection_records_monkey,
)
from monkey.model.classification_model.efficientnet_b0 import (
    EfficientNet_B0,
)
from monkey.model.efficientunetb0.architecture import (
    get_efficientunet_b0_MBConv,
)
from prediction.classification import detected_cell_classification
from prediction.detection import wsi_detection_in_mask

if __name__ == "__main__":
    for fold in range(1, 5):
        # fold = 1
        detector_model_name = "efficientunetb0_seg_bm"

        thresh = 0.3

        config = PredictionIOConfig(
            wsi_dir="/home/u1910100/Downloads/Monkey/images/pas-cpg",
            mask_dir="/home/u1910100/Downloads/Monkey/images/tissue-masks",
            output_dir=f"/home/u1910100/Documents/Monkey/local_output/{detector_model_name}/Fold_{fold}",
            patch_size=256,
            resolution=0,
            units="level",
            stride=224,
            threshold=thresh,
            min_size=7,
        )

        split_info = open_json_file(
            "/home/u1910100/Documents/Monkey/patches_256/wsi_level_split.json"
        )

        val_wsi_files = split_info[f"Fold_{fold}"]["test_files"]

        print(val_wsi_files)

        # Load models
        detector_weight_paths = [
            f"/home/u1910100/Documents/Monkey/runs/detection/{detector_model_name}/fold_1/epoch_100.pth",
            # f"/home/u1910100/Documents/Monkey/runs/detection/{detector_model_name}/fold_2/epoch_100.pth",
            # f"/home/u1910100/Documents/Monkey/runs/detection/{detector_model_name}/fold_4/epoch_100.pth",
        ]
        detectors = []
        for weight_path in detector_weight_paths:
            detector = get_efficientunet_b0_MBConv(pretrained=False)
            checkpoint = torch.load(weight_path)
            detector.load_state_dict(checkpoint["model"])
            detector.eval()
            detector.to("cuda")
            detectors.append(detector)

        classifiers = []
        classifier_weight_paths = [
            "/home/u1910100/Documents/Monkey/runs/cls/efficientnetb0/fold_1/epoch_4.pth",
            "/home/u1910100/Documents/Monkey/runs/cls/efficientnetb0/fold_2/epoch_2.pth",
            "/home/u1910100/Documents/Monkey/runs/cls/efficientnetb0/fold_3/epoch_6.pth",
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

        for wsi_name in tqdm(val_wsi_files):
            wsi_name_without_ext = os.path.splitext(wsi_name)[0]
            wsi_id = extract_id(wsi_name)
            mask_name = f"{wsi_id}_mask.tif"

            overall_detection_records = wsi_detection_in_mask(
                wsi_name, mask_name, config, detectors
            )

            print(
                f"{len(overall_detection_records)} final detected cells"
            )

            # Classify detected inflammatory cells
            classification_detection_records = (
                detected_cell_classification(
                    overall_detection_records,
                    wsi_name,
                    config,
                    classifiers,
                )
            )
            # detection_classification_records = detection_records

            # Save to AnnotationStore for visualization
            # If model if not running at baseline res:
            # scale_factor = 0.25 / 0.24199951445730394
            annoation_store = detection_to_annotation_store(
                classification_detection_records, scale_factor=1.0
            )
            store_save_path = os.path.join(
                config.output_dir, f"{wsi_name_without_ext}.db"
            )
            annoation_store.dump(store_save_path)

            # Save result in Monkey Challenge format
            # Must save results separately

            # L1 overall inflamm detection
            save_detection_records_monkey(
                overall_detection_records,
                config,
                wsi_id=wsi_id,
                task="overall_detection",
            )

            # L2 lymphocyte vs monocyte detection
            save_detection_records_monkey(
                classification_detection_records,
                config,
                wsi_id=wsi_id,
                task="detection_classification",
            )
            print("finished")
