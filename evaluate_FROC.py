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
    # GROUND_TRUTH_DIRECTORY = (
    #     "/home/u1910100/Downloads/Monkey/annotations/json"
    # )

    FOLD = fold
    model_name = "efficientnetv2_l_multitask_det_max_aug"
    print(f"Model name: {model_name}")
    PREDICT_DIR = f"/home/u1910100/cloud_workspace/data/Monkey/local_output/{model_name}/Fold_{FOLD}"
    print(os.listdir(PREDICT_DIR))
    # PREDICT_DIR = f"/home/u1910100/Documents/Monkey/local_output/{model_name}/Fold_{FOLD}"
    SPACING_LEVEL0 = 0.24199951445730394

    split_info = open_json_file(
        "/mnt/lab-share/Monkey/patches_256/wsi_level_split.json"
    )
    # split_info = open_json_file(
    #     "/home/u1910100/Documents/Monkey/patches_256/wsi_level_split.json"
    # )

    val_wsi_files = split_info[f"Fold_{FOLD}"]["test_files"]

    inflamm_sum_score = 0.0
    inflamm_sum_f1 = 0.0
    inflamm_sum_precision = 0.0
    inflamm_sum_recall = 0.0
    lymph_sum_score = 0.0
    lymph_sum_f1 = 0.0
    lymph_sum_precision = 0.0
    lymph_sum_recall = 0.0
    mono_sum_score = 0.0
    mono_sum_f1 = 0.0
    mono_sum_precision = 0.0
    mono_sum_recall = 0.0
    total_inflamm = 0
    total_lymph = 0
    total_mono = 0

    for wsi_name in val_wsi_files:
        wsi_id = extract_id(wsi_name)

        gt_inf_cells = open_json_file(
            json_path=os.path.join(
                GROUND_TRUTH_DIRECTORY,
                f"{wsi_id}_inflammatory-cells.json",
            )
        )
        total_inflamm += len(gt_inf_cells["points"])

        gt_lymphocytes = open_json_file(
            json_path=os.path.join(
                GROUND_TRUTH_DIRECTORY, f"{wsi_id}_lymphocytes.json"
            )
        )
        total_lymph += len(gt_lymphocytes["points"])

        gt_monocytes = open_json_file(
            json_path=os.path.join(
                GROUND_TRUTH_DIRECTORY, f"{wsi_id}_monocytes.json"
            )
        )
        total_mono += len(gt_monocytes["points"])

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
            radius=int(5 / SPACING_LEVEL0),
        )

        inflamm_f1 = get_F1_scores(
            gt_inf_cells,
            result_detected_inflamm,
            radius=int(5 / SPACING_LEVEL0),
        )

        lymph_froc = get_froc_vals(
            gt_lymphocytes,
            result_detected_lymphocytes,
            radius=int(4 / SPACING_LEVEL0),
        )

        lymph_f1 = get_F1_scores(
            gt_lymphocytes,
            result_detected_lymphocytes,
            radius=int(4 / SPACING_LEVEL0),
        )

        mono_froc = get_froc_vals(
            gt_monocytes,
            result_detected_monocytes,
            radius=int(5 / SPACING_LEVEL0),
        )

        mono_f1 = get_F1_scores(
            gt_monocytes,
            result_detected_monocytes,
            radius=int(5 / SPACING_LEVEL0),
        )

        inflamm_sum_score += inflamm_froc["froc_score_slide"]
        inflamm_sum_f1 += inflamm_f1["F1"]
        inflamm_sum_precision += inflamm_f1["Precision"]
        inflamm_sum_recall += inflamm_f1["Recall"]

        lymph_sum_score += lymph_froc["froc_score_slide"]
        lymph_sum_f1 += lymph_f1["F1"]
        lymph_sum_precision += lymph_f1["Precision"]
        lymph_sum_recall += lymph_f1["Recall"]

        mono_sum_score += mono_froc["froc_score_slide"]
        mono_sum_f1 += mono_f1["F1"]
        mono_sum_precision += mono_f1["Precision"]
        mono_sum_recall += mono_f1["Recall"]

    results = {
        "model_name": model_name,
        "fold": fold,
        "Inflamm FROC": inflamm_sum_score / len(val_wsi_files),
        "Lymphocytes FROC": lymph_sum_score / len(val_wsi_files),
        "Monocytes FROC": mono_sum_score / len(val_wsi_files),
        "Inflamm F1": inflamm_sum_f1 / len(val_wsi_files),
        "Lymphocytes F1": lymph_sum_f1 / len(val_wsi_files),
        "Monocytes F1": mono_sum_f1 / len(val_wsi_files),
        "Inflamm Precision": inflamm_sum_precision
        / len(val_wsi_files),
        "Lymph Precision": lymph_sum_precision / len(val_wsi_files),
        "Mono Precision": mono_sum_precision / len(val_wsi_files),
        "Inflamm Recall": inflamm_sum_recall / len(val_wsi_files),
        "Lymph Recall": lymph_sum_recall / len(val_wsi_files),
        "Mono Recall": mono_sum_recall / len(val_wsi_files),
        "Total Inflamm cells": total_inflamm,
        "Total Lymphocytes": total_lymph,
        "Total Monocytes": total_mono,
    }
    return results


if __name__ == "__main__":
    folds = [1, 2, 3, 4, 5]
    # folds = [1]
    with Pool(5) as p:
        results = p.map(compute_FROC, folds)

    # Sum across folds
    inflamm_froc_sum = 0.0
    lymph_froc_sum = 0.0
    mono_froc_sum = 0.0
    # Sum F1 scores across folds
    inflamm_f1_sum = 0.0
    lymph_f1_sum = 0.0
    mono_f1_sum = 0.0
    for result in results:
        pprint(result)
        inflamm_froc_sum += result["Inflamm FROC"]
        lymph_froc_sum += result["Lymphocytes FROC"]
        mono_froc_sum += result["Monocytes FROC"]
        inflamm_f1_sum += result["Inflamm F1"]
        lymph_f1_sum += result["Lymphocytes F1"]
        mono_f1_sum += result["Monocytes F1"]

    pprint(f"Avg inflamm FROC {inflamm_froc_sum /  len(folds)}")
    pprint(f"Avg lymph FROC {lymph_froc_sum /  len(folds)}")
    pprint(f"Avg mono FROC {mono_froc_sum /  len(folds)}")
    pprint(f"Avg inflamm F1 {inflamm_f1_sum /  len(folds)}")
    pprint(f"Avg lymph F1 {lymph_f1_sum /  len(folds)}")
    pprint(f"Avg mono F1 {mono_f1_sum /  len(folds)}")
