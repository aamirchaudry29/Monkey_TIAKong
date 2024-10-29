import os

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    detection_to_annotation_store,
    save_detection_records_monkey,
)
from prediction.detection import wsi_detection_in_mask

if __name__ == "__main__":
    config = PredictionIOConfig(
        wsi_dir="/home/u1910100/Downloads/Monkey/images/pas-cpg",
        mask_dir="/home/u1910100/Downloads/Monkey/images/tissue-masks",
        output_dir="/home/u1910100/Documents/Monkey/local_output",
        model_name="MiTb0Unet",
        model_path="/home/u1910100/Documents/Monkey/runs/MiTb0Unet/fold_4/epoch_100.pth",
        patch_size=256,
        resolution=0,
        units="level",
        stride=224,
        threshold=0.5,
    )

    wsi_name = "A_P000001_PAS_CPG.tif"
    wsi_name_without_ext = os.path.splitext(wsi_name)[0]
    mask_name = "A_P000001_mask.tif"

    detection_records = wsi_detection_in_mask(
        wsi_name, mask_name, config
    )

    print(f"{len(detection_records)} detected cells")

    # Save to AnnotationStore for visualization
    # scale_factor = 0.25 / 0.24199951445730394
    annoation_store = detection_to_annotation_store(
        detection_records, scale_factor=1.0
    )
    store_save_path = os.path.join(
        config.output_dir, f"{wsi_name_without_ext}.db"
    )
    annoation_store.dump(store_save_path)

    # Save result in Monkey Challenge format
    save_detection_records_monkey(detection_records, config)
    print("finished")
