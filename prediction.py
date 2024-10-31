import os

from tqdm import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    detection_to_annotation_store,
    extract_id,
    open_json_file,
    save_detection_records_monkey,
)
from prediction.detection import wsi_detection_in_mask

if __name__ == "__main__":
    fold = 2

    config = PredictionIOConfig(
        wsi_dir="/home/u1910100/Downloads/Monkey/images/pas-cpg",
        mask_dir="/home/u1910100/Downloads/Monkey/images/tissue-masks",
        output_dir=f"/home/u1910100/Documents/Monkey/local_output/Fold_{fold}",
        model_name="efficientunetb0",
        model_path=f"/home/u1910100/Documents/Monkey/runs/efficientunetb0/fold_{fold}/epoch_100.pth",
        patch_size=256,
        resolution=0,
        units="level",
        stride=224,
        threshold=0.9,
        min_size=32,
    )

    split_info = open_json_file(
        "/home/u1910100/Documents/Monkey/patches_256/wsi_level_split.json"
    )

    val_wsi_files = split_info[f"Fold_{fold}"]["test_files"]

    print(val_wsi_files)

    for wsi_name in tqdm(val_wsi_files):
        wsi_name_without_ext = os.path.splitext(wsi_name)[0]
        wsi_id = extract_id(wsi_name)
        mask_name = f"{wsi_id}_mask.tif"

        detection_records = wsi_detection_in_mask(
            wsi_name, mask_name, config
        )

        # wsi_name = "A_P000001_PAS_CPG.tif"
        # wsi_name_without_ext = os.path.splitext(wsi_name)[0]
        # mask_name = "A_P000001_mask.tif"

        # detection_records = wsi_detection_in_mask(
        #     wsi_name, mask_name, config
        # )

        # print(f"{len(detection_records)} detected cells")

        # # Save to AnnotationStore for visualization
        # # scale_factor = 0.25 / 0.24199951445730394
        annoation_store = detection_to_annotation_store(
            detection_records, scale_factor=1.0
        )
        store_save_path = os.path.join(
            config.output_dir, f"{wsi_name_without_ext}.db"
        )
        annoation_store.dump(store_save_path)

        # Save result in Monkey Challenge format
        save_detection_records_monkey(
            detection_records, config, wsi_id=wsi_id
        )
        print("finished")
