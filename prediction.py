from monkey.config import PredictionIOConfig
from tiatoolbox.wsicore.wsireader import WSIReader
import os

from prediction.detection import wsi_detection_in_mask
from monkey.data.data_utils import scale_coords, points_to_annotation_store



if __name__ == "__main__":
    config = PredictionIOConfig(
        wsi_dir="/home/u1910100/Downloads/Monkey/images/pas-cpg",
        mask_dir="/home/u1910100/Downloads/Monkey/images/tissue-masks",
        output_dir="/home/u1910100/Documents/Monkey/local_output",
        model_name="MiTb0Unet",
        model_path="/home/u1910100/Documents/Monkey/runs/MiTb0Unet/fold_4/epoch_100.pth",
        patch_size=256,
        resolution=0,
        units='level',
        stride=224,
        threshold=0.5 
    )

    wsi_name = "A_P000001_PAS_CPG.tif"
    mask_name = "A_P000001_mask.tif"

    detected_points = wsi_detection_in_mask(wsi_name, mask_name, config)

    points = []
    for record in detected_points:
        points.append([
            record['x'],
            record['y']
        ])

    print(f"{len(points)} detected cells")


    # Convert to AnnotationStore for visualization
    # scale_factor = 0.25 / 0.24199951445730394
    # scaled_result = scale_coords(points, scale_factor)
    annoation_store = points_to_annotation_store(points)
    annoation_store.dump(
        "/home/u1910100/Documents/Monkey/test/input/images/overlays/A_P000001_PAS_CPG.db"
    )