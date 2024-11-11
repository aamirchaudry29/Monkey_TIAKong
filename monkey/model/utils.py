import numpy as np
import torch.nn
from skimage.measure import label, regionprops
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor

from evaluation.evaluate import (
    calculate_f1_metrics,
    match_coordinates,
)


def get_activation_function(name: str):
    """
    Return torch.nn Activation function
    matching input name
    """

    functions = {
        "relu": torch.nn.ReLU,
        "sigmoid": torch.nn.Sigmoid,
        "softmax": torch.nn.Softmax,
        "tanh": torch.nn.Tanh,
    }  # add more as needed

    name = name.lower()
    if name in functions:
        if name == "softmax":
            return functions[name](dim=1)
        else:
            return functions[name]()
    else:
        raise ValueError(f"Undefined loss function: {name}")


def get_cell_centers(
    cell_mask: np.ndarray, intensity_image: np.ndarray | None = None
) -> list[list]:
    """
    Get cell centroids from binary mask

    Args:
        cell_mask: binary mask
        intensity_images: for calculating probs
    Returns:
        dict:{"centers", "probs"}
    """
    mask_label = label(cell_mask)
    stats = regionprops(mask_label, intensity_image=intensity_image)
    centers = []
    probs = []
    for region in stats:
        centroid = region["centroid"]
        centers.append(centroid)
        if intensity_image is not None:
            probs.append(region["mean_intensity"])

    return {"centers": centers, "probs": probs}


def get_patch_F1_score_batch(
    batch_pred_patch: np.ndarray | Tensor,
    batch_target_patch: np.ndarray | Tensor,
    batch_intensity_image: np.ndarray | Tensor | None,
) -> dict:
    """
    Calculate detection F1 score from binary masks
    Average over batches

    Args:
        pred_patch: Prediction mask [BxHxW]
        target_parch: ground truth mask [BxHxW]
    Returns:
        metrics: {"F1", "Precision", "Recall"}
    """

    if torch.is_tensor(batch_pred_patch):
        batch_pred_patch = batch_pred_patch.numpy(force=True)
    if torch.is_tensor(batch_target_patch):
        batch_target_patch = batch_target_patch.numpy(force=True)
    if torch.is_tensor(batch_intensity_image):
        batch_intensity_image = batch_intensity_image.numpy(
            force=True
        )

    sum_f1 = 0.0
    sum_precision = 0.0
    sum_recall = 0.0

    batch_count = batch_pred_patch.shape[0]
    for i in range(batch_count):
        pred_patch = batch_pred_patch[i, :, :]
        target_patch = batch_target_patch[i, :, :]
        intensity_image = batch_intensity_image[i, :, :]
        metrics = get_patch_F1_score(
            pred_patch, target_patch, intensity_image
        )
        sum_f1 += metrics["F1"]
        sum_precision += metrics["Precision"]
        sum_recall += metrics["Recall"]

    return {
        "F1": sum_f1 / batch_count,
        "Precision": sum_precision / batch_count,
        "Recall": sum_recall / batch_count,
    }


def get_patch_F1_score(
    pred_patch: np.ndarray,
    target_patch: np.ndarray,
    intensity_image: np.ndarray | None,
) -> dict:
    """
    Calculate detection F1 score from binary masks

    Args:
        pred_patch: Prediction mask [HxW]
        target_parch: ground truth mask [HxW]
        intensity_image: [HxW]
    Returns:
        metrics: {"F1", "Precision", "Recall"}
    """

    pred_stats = get_cell_centers(pred_patch, intensity_image)
    pred_centers = pred_stats["centers"]
    pred_probs = pred_stats["probs"]
    true_centers = get_cell_centers(target_patch)["centers"]
    metrics = evaluate_cell_predictions(
        true_centers, pred_centers, pred_probs
    )

    return metrics


def evaluate_cell_predictions(
    gt_centers, pred_centers, probs, mpp=0.24199951445730394
) -> dict:
    """
    Calculate detection F1 score from binary masks

    Args:
        gt_centers: Prediction mask [HxW]
        pred_centers: ground truth mask [HxW]
        mpp: baseline resolution, default=0.24
    Returns:
        metrics: {"F1", "Precision", "Recall"}
    """
    if len(probs) != len(pred_centers):
        probs = []
        probs = [1.0 for i in range(len(pred_centers))]

    (
        tp,
        fn,
        fp,
        _,
        _,
    ) = match_coordinates(
        gt_centers,
        pred_centers,
        probs,
        int(7.5 / mpp),
    )

    # print(f"tp:{tp}, fn:{fn}, fp:{fp}")

    return calculate_f1_metrics(tp, fn, fp)


def get_classification_metrics(
    gt_labels: np.ndarray | list, pred_labels: np.ndarray | list
):
    """
    Calculate:
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score

    Args:
        gt: list [N] of true labels
        pred: list [N] of pred labels
    Returns:
        metrics: {"Balanced_Accuracy", "Precision", "Recall", "F1"}
    """

    metrics = {
        "Balanced_Accuracy": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
    }

    accuracy = balanced_accuracy_score(gt_labels, pred_labels)
    precision = precision_score(
        gt_labels, pred_labels, average="binary", zero_division=0.0
    )
    recall = recall_score(
        gt_labels, pred_labels, average="binary", zero_division=0.0
    )
    f1 = f1_score(
        gt_labels, pred_labels, average="binary", zero_division=0.0
    )

    metrics = {
        "Balanced_Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    return metrics
