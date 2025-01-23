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


def get_multiclass_patch_F1_score_batch(
    batch_pred_patch: np.ndarray | Tensor,
    batch_target_patch: np.ndarray | Tensor,
    margins: list,
    batch_intensity_image: np.ndarray | Tensor | None,
) -> dict:
    """
    Calculate detection F1 score from binary masks
    Average over batches and channels

    Args:
        batch_pred_patch: Prediction mask [BxCxHxW]
        batch_target_parch: ground truth mask [BxCxHxW]
        batch_intensity_image: [BxCxHxW]
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

    class_count = batch_pred_patch.shape[1]
    for i in range(class_count):
        class_pred = batch_pred_patch[:, i, :, :]
        class_target = batch_target_patch[:, i, :, :]
        class_intensity = batch_intensity_image[:, i, :, :]
        metrics = get_patch_F1_score_batch(
            class_pred, class_target, margins[i], class_intensity
        )
        sum_f1 += metrics["F1"]
        sum_precision += metrics["Precision"]
        sum_recall += metrics["Recall"]

    return {
        "F1": sum_f1 / class_count,
        "Precision": sum_precision / class_count,
        "Recall": sum_recall / class_count,
    }


def get_patch_F1_score_batch(
    batch_pred_patch: np.ndarray | Tensor,
    batch_target_patch: np.ndarray | Tensor,
    margin: float,
    batch_intensity_image: np.ndarray | Tensor | None,
) -> dict:
    """
    Calculate detection F1 score from binary masks
    Average over batches

    Args:
        batch_pred_patch: Prediction mask [BxHxW]
        batch_target_parch: ground truth mask [BxHxW]
        batch_intensity_image: [BxHxW]
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
            pred_patch, target_patch, margin, intensity_image
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
    pred_patch: np.ndarray | Tensor,
    target_patch: np.ndarray | Tensor,
    margin: float,
    intensity_image: np.ndarray | Tensor | None,
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

    if torch.is_tensor(pred_patch):
        pred_patch = pred_patch.numpy(force=True)
    if torch.is_tensor(target_patch):
        target_patch = target_patch.numpy(force=True)
    if torch.is_tensor(intensity_image):
        intensity_image = intensity_image.numpy(force=True)

    pred_stats = get_cell_centers(pred_patch, intensity_image)
    pred_centers = pred_stats["centers"]
    pred_probs = pred_stats["probs"]
    true_centers = get_cell_centers(target_patch)["centers"]
    metrics = evaluate_cell_predictions(
        true_centers, pred_centers, pred_probs, margin
    )

    return metrics


def evaluate_cell_predictions(
    gt_centers,
    pred_centers,
    probs,
    margin: float,
    mpp=0.24199951445730394,
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
        int(margin / mpp),
    )

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


class EarlyStopper:
    """
    A utility class to perform early stopping during model training.
    Attributes:
        patience (int): The number of epochs to wait after the last time the validation loss improved.
        min_delta (float): The minimum change in the monitored quantity to qualify as an improvement.
        counter (int): The number of epochs since the last improvement in validation loss.
        min_validation_loss (float): The minimum validation loss observed so far.
    Methods:
        early_stop(validation_loss):
            Checks if training should be stopped early based on the validation loss.
            Args:
                validation_loss (float): The current epoch's validation loss.
            Returns:
                bool: True if training should be stopped early, False otherwise.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
    
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False