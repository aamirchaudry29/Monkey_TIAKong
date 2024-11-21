import os
from multiprocessing import Pool
from pprint import pprint

from evaluation.evaluate import (
    convert_mm_to_pixel,
    get_F1_scores,
    get_froc_vals,
)
from monkey.data.data_utils import extract_id, open_json_file


def compute_FROC(fold: int = 1):
    GROUND_TRUTH_DIRECTORY = (
        "/mnt/lab-share/Monkey/Dataset/annotations/json"
    )
    FOLD = fold
    model_name = "efficientunetb0_seg_3_channel"
    PREDICT_DIR = f"/home/u1910100/cloud_workspace/data/Monkey/local_output/{model_name}/Fold_{FOLD}"
    SPACING_LEVEL0 = 0.24199951445730394

    split_info = open_json_file(
        "/mnt/lab-share/Monkey/patches_256/wsi_level_split.json"
    )

    val_wsi_files = split_info[f"Fold_{FOLD}"]["test_files"]

    inflamm_sum_score = 0.0
    inflamm_sum_f1 = 0.0
    lymph_sum_score = 0.0
    lymph_sum_f1 = 0.0
    mono_sum_score = 0.0
    mono_sum_f1 = 0.0

    for wsi_name in val_wsi_files:
        wsi_id = extract_id(wsi_name)

        gt_inf_cells = open_json_file(
            json_path=os.path.join(
                GROUND_TRUTH_DIRECTORY,
                f"{wsi_id}_inflammatory-cells.json",
            )
        )
        gt_lymphocytes = open_json_file(
            json_path=os.path.join(
                GROUND_TRUTH_DIRECTORY, f"{wsi_id}_lymphocytes.json"
            )
        )
        gt_monocytes = open_json_file(
            json_path=os.path.join(
                GROUND_TRUTH_DIRECTORY, f"{wsi_id}_monocytes.json"
            )
        )

        location_detected_inflamm = os.path.join(
            PREDICT_DIR, f"{wsi_id}_detected-inflammatory-cells.json"
        )
        result_detected_inflamm = open_json_file(
            json_path=location_detected_inflamm,
        )
        result_detected_inflamm = convert_mm_to_pixel(
            result_detected_inflamm
        )

        location_detected_lymphocytes = os.path.join(
            PREDICT_DIR, f"{wsi_id}_detected-lymphocytes.json"
        )
        result_detected_lymphocytes = open_json_file(
            json_path=location_detected_lymphocytes,
        )
        result_detected_lymphocytes = convert_mm_to_pixel(
            result_detected_lymphocytes
        )

        location_detected_monocytes = os.path.join(
            PREDICT_DIR, f"{wsi_id}_detected-monocytes.json"
        )
        result_detected_monocytes = open_json_file(
            json_path=location_detected_monocytes,
        )
        result_detected_monocytes = convert_mm_to_pixel(
            result_detected_monocytes
        )

        inflamm_froc = get_froc_vals(
            gt_inf_cells,
            result_detected_inflamm,
            radius=int(7.5 / SPACING_LEVEL0),
        )

        inflamm_f1 = get_F1_scores(
            gt_inf_cells,
            result_detected_inflamm,
            radius=int(7.5 / SPACING_LEVEL0),
        )

        lymph_froc = get_froc_vals(
            gt_lymphocytes,
            result_detected_lymphocytes,
            radius=int(7.5 / SPACING_LEVEL0),
        )

        lymph_f1 = get_F1_scores(
            gt_lymphocytes,
            result_detected_lymphocytes,
            radius=int(7.5 / SPACING_LEVEL0),
        )

        mono_froc = get_froc_vals(
            gt_monocytes,
            result_detected_monocytes,
            radius=int(7.5 / SPACING_LEVEL0),
        )

        mono_f1 = get_F1_scores(
            gt_monocytes,
            result_detected_monocytes,
            radius=int(7.5 / SPACING_LEVEL0),
        )

        inflamm_sum_score += inflamm_froc["froc_score_slide"]
        inflamm_sum_f1 += inflamm_f1["F1"]

        lymph_sum_score += lymph_froc["froc_score_slide"]
        lymph_sum_f1 += lymph_f1["F1"]

        mono_sum_score += mono_froc["froc_score_slide"]
        mono_sum_f1 += mono_f1["F1"]

    # pprint(f"Model {model_name} fold {fold}")
    # pprint(
    #     f"Average Inflamm FROC = {inflamm_sum_score / len(val_wsi_files)}"
    # )
    # pprint(
    #     f"Average Inflamm F1 = {inflamm_sum_f1 / len(val_wsi_files)}"
    # )
    # pprint(
    #     f"Average Lymphocytes FROC = {lymph_sum_score / len(val_wsi_files)}"
    # )
    # pprint(
    #     f"Average Lymphocytes F1 = {lymph_sum_f1 / len(val_wsi_files)}"
    # )
    # pprint(
    #     f"Average Monocytes FROC = {mono_sum_score / len(val_wsi_files)}"
    # )
    # pprint(
    #     f"Average Monocytes F1 = {mono_sum_f1 / len(val_wsi_files)}"
    # )

    results = {
        "model_name": model_name,
        "fold": fold,
        "Inflamm FROC": inflamm_sum_score / len(val_wsi_files),
        "Inflamm F1": inflamm_sum_f1 / len(val_wsi_files),
        "Lymphocytes FROC": lymph_sum_score / len(val_wsi_files),
        "Lymphocytes F1": lymph_sum_f1 / len(val_wsi_files),
        "Monocytes FROC": mono_sum_score / len(val_wsi_files),
        "Monocytes F1": mono_sum_f1 / len(val_wsi_files),
    }
    return results


if __name__ == "__main__":
    with Pool(5) as p:
        results = p.map(compute_FROC, [1, 2, 3, 4, 5])

    for result in results:
        pprint(result)
