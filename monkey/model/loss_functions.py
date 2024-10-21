from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import ConfusionMatrixMetric, DiceMetric
from torch import Tensor


# Returns a Loss Function instance depending on the loss type
def get_loss_function(loss_type):
    loss_functions = {
        "Jaccard_Loss": Jaccard_Loss,
        "Dice": Dice_Loss,
        "BCE": BCE_Loss,
        "Weighted_BCE": Weighted_BCE_Loss,
        "BCE_Dice": BCE_Dice_Loss,
        "Weighted_BCE_Dice": Weighted_BCE_Dice_Loss,
        "MSE": MSE_loss,
        "Combined_Dice_Focal": CombinedLoss,
        # To add a new loss function, first create a subclass of Loss_Function
        # Then add a new entry here:
        # "<loss_type>": <class name>
    }

    if loss_type in loss_functions:
        return loss_functions[loss_type]()
    else:
        raise ValueError(f"Undefined loss function: {loss_type}")


# Abstract class for loss functions
# All loss functions need to be a subclass of this class
class Loss_Function(ABC):
    def __init__(self, name, use_weights) -> None:
        self.name = name
        self.use_weight = use_weights

    @abstractmethod
    def compute_loss(self):
        pass


# -------------------------------------Classes implementing loss functions---------------------------------
# MSE loss
class MSE_loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("MSE", False)

    def compute_loss(self, input: Tensor, target: Tensor):
        assert (
            input.size() == target.size()
        )  # "Input size {} must be the same as target size {}".format(input.size(), target.size())
        return nn.MSELoss()(input, target)


# Jaccard loss
class Jaccard_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Jaccard Loss", False)
        self.jaccard_loss = DiceLoss(jaccard=True, sigmoid=False)

    def compute_loss(self, input: Tensor, target: Tensor):
        return self.jaccard_loss(input, target)


# Dice loss
class Dice_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Dice Loss", False)
        self.dice_loss = DiceLoss(sigmoid=False)

    def compute_loss(self, input: Tensor, target: Tensor):
        return self.dice_loss(input, target)


# Binary cross entropy loss
class BCE_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("BCE Loss", False)

    def compute_loss(self, input: Tensor, target: Tensor):
        return nn.BCELoss()(input, target.float())


# Weighted binary cross entropy loss
class Weighted_BCE_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Weighted BCE Loss", True)

    def compute_loss(
        self, input: Tensor, target: Tensor, weight: Tensor
    ):
        return nn.BCELoss(weight=weight)(input, target.float())


# Binary cross entropy + Dice loss
class BCE_Dice_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("BCE + Dice Loss", False)
        self.dice_ce_loss = DiceCELoss(sigmoid=False)

    def compute_loss(self, input: Tensor, target: Tensor):
        return self.dice_ce_loss(input, target)


# Weighted binary cross entropy + Dice loss
class Weighted_BCE_Dice_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Weighted BCE Loss + Dice Loss", True)
        self.dice_ce_loss = DiceCELoss(sigmoid=False)

    def compute_loss(
        self, input: Tensor, target: Tensor, weight: Tensor
    ):
        return self.dice_ce_loss(input, target, weight)


# Combined Dice and Focal loss
class CombinedLoss(Loss_Function):
    def __init__(
        self, dice_weight=0.5, focal_weight=0.5, alpha=0.25, gamma=2
    ):
        super().__init__("Combined Dice + Focal Loss", False)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma
        self.dice_loss = Dice_Loss()

    def compute_loss(self, input: Tensor, target: Tensor):
        # Dice loss
        dice_loss = self.dice_loss.compute_loss(input, target)

        # Focal loss
        bce_loss = F.binary_cross_entropy_with_logits(
            input, target.float(), reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        # Combined loss
        combined_loss = (
            self.dice_weight * dice_loss
            + self.focal_weight * focal_loss
        )
        return combined_loss


# -------------------------------------Metric functions---------------------------------
def compute_dice_coefficient(
    input: Tensor, target: Tensor, multiclass: bool = False
):
    """
    Compute the Dice coefficient for either multiclass or single-class segmentation.

    Args:
        input (Tensor): The predicted segmentation mask.
        target (Tensor): The ground truth segmentation mask.
        multiclass (bool): Flag indicating whether it is a multiclass or single-class segmentation.
                           If True, the input and target are expected to have shape (batch_size, num_classes, ...).
                           If False, the input and target are expected to have shape (batch_size, ...) or (...,).

    Returns:
        float: The computed Dice coefficient.
    """
    if multiclass:
        # Multiclass Dice coefficient
        dice_metric = DiceMetric(
            include_background=True,
            reduction="mean",
            get_not_nans=False,
        )
    else:
        # Single-class Dice coefficient
        dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )

    dice_metric.reset()  # Reset the metric before computing

    if input.dim() == target.dim():
        # No batch size, compute Dice coefficient directly
        dice_metric(y_pred=input.unsqueeze(0), y=target.unsqueeze(0))
    else:
        # Batch size exists, compute Dice coefficient for each batch
        for i in range(input.shape[0]):
            dice_metric(
                y_pred=input[i].unsqueeze(0), y=target[i].unsqueeze(0)
            )

    # Aggregate the results and return the mean Dice coefficient
    return dice_metric.aggregate().item()


def compute_f1_score(input: Tensor, target: Tensor):
    """
    Compute the F1 score for segmentation.

    Args:
        input (Tensor): The predicted segmentation mask.
        target (Tensor): The ground truth segmentation mask.

    Returns:
        float: The computed F1 score.
    """
    confusion_matrix = ConfusionMatrixMetric(
        include_background=True,
        metric_name="f1_score",
        reduction="mean",
    )
    confusion_matrix.reset()  # Reset the metric before computing

    if input.dim() == target.dim():
        # No batch size, compute F1 score directly
        confusion_matrix(
            y_pred=input.unsqueeze(0), y=target.unsqueeze(0)
        )
    else:
        # Batch size exists, compute F1 score for each batch
        for i in range(input.shape[0]):
            confusion_matrix(
                y_pred=input[i].unsqueeze(0), y=target[i].unsqueeze(0)
            )

    # Aggregate the results and return the mean F1 score
    return confusion_matrix.aggregate()[0].item()
