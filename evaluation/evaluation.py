import numpy as np
from skimage.measure import label
from skimage.morphology import reconstruction


def evaluate_detection_F1_score(
    pred_npy, GT_npy, save_dir_pred, save
):
    #####This is to evaluate the F1 score of the detection model##
    """if the detection point inside the mask, it is considered as TP. If the detection point is outside the mask,
    it is considered as FP,if more than one detection point is inside the mask,
    it is considered as TP, see the mask_based_evaluate function for more details
    Input pred_npy is a batch
    input GT_npy is a batch
    """
    total_F1_score = 0
    total_recall = 0
    total_precision = 0
    total_TP = 0
    total_FP = 0
    total_FN = 0
    processed_dots_map = extract_batch_dotmaps(
        pred_npy,
        save_dir_pred,
        save,
        distance_threshold_local_max=1,
        prediction_dots_threshold=0.3,
    )
    # Compute the evaluation metrics
    if save == True:
        print(
            "extracting dotmaps saved in",
            os.path.join(save_dir_pred, "extract_dotmaps_batch.npy"),
        )
    (
        TP,
        FP,
        FN,
        recall,
        precision,
        f1score,
        num_masks,
    ) = mask_based_evaluate_batch(GT_npy, processed_dots_map)
    total_F1_score += f1score
    total_recall += recall
    total_precision += precision
    total_TP += TP
    total_FP += FP
    total_FN += FN
    return {
        "F1_score": total_F1_score,
        "recall": total_recall,
        "precision": total_precision,
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "num_masks": num_masks,
    }


def mask_based_evaluate(gt_mask, pred_mask):
    TP = 0
    FP = 0
    FN = 0

    gt_label = gt_mask
    unique_labels = np.unique(gt_label)

    for i in unique_labels[1:]:  # range(1, np.max(gt_label)+1):
        this_obj = gt_label == i
        detection = this_obj * pred_mask
        if np.sum(detection) == 1:  # object is found
            TP += 1
        elif np.sum(detection) > 1:
            TP += 1
            FP += np.sum(detection) - 1
        else:
            FN += 1

    # finding the FP
    residuals = (pred_mask > 0) - reconstruction(
        np.uint8(pred_mask) * np.uint8(gt_mask > 0),
        np.uint8(pred_mask),
    )
    residuals_label = label(residuals > 0)
    FP = np.max(residuals_label)

    return TP, FP, FN
