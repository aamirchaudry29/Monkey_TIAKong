from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss
from torch import Tensor


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, n_tasks: int):
        super(AutomaticWeightedLoss, self).__init__()
        self.n_tasks = n_tasks
        params = torch.ones(n_tasks, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (
                self.params[i] ** 2
            ) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class MultiTaskLoss(torch.nn.Module):
    def __init__(
        self,
        is_regression: torch.Tensor = torch.tensor(
            [False, False, False]
        ),
        reduction="none",
    ):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, losses: torch.Tensor):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars) ** 0.5).to(dtype).to(device)
        self.is_regression = self.is_regression.to(device).to(dtype)
        coeffs = 1 / ((self.is_regression + 1) * stds**2)
        multi_task_losses = coeffs * losses + self.log_vars

        if self.reduction == "sum":
            return multi_task_losses.sum()
        if self.reduction == "mean":
            return multi_task_losses.mean()
        return multi_task_losses


# Abstract class for loss functions
# All loss functions need to a subclass of this class
class Loss_Function(ABC):
    def __init__(self, name, use_weights) -> None:
        self.name = name
        self.use_weight = use_weights

    @abstractmethod
    def compute_loss(self):
        """
        Args:
            input: Tensor: Model output
            target: Tensor: Ground truth
        """
        pass

    @abstractmethod
    def set_multiclass(self):
        pass


# Returns a Loss Function instance depending on the loss type
def get_loss_function(loss_type: str) -> Loss_Function:
    """
    Returns an initialized loss function object.

    """
    loss_functions = {
        "Jaccard_Loss": Jaccard_Loss,
        "Dice": Dice_Loss,
        "BCE": BCE_Loss,
        "Jaccard_Focal_Loss": Jaccard_Focal_Loss,
        "Focal_Loss": Focal_Loss,
        "Jaccard_Dice_Focal_Loss": Jaccard_Dice_Focal_Loss,
        # To add a new loss function, first create a subclass of Loss_Function
        # Then add a new entry here:
        # "<loss_type>": <class name>
    }

    if loss_type in loss_functions:
        return loss_functions[loss_type]()
    else:
        raise ValueError(f"Undefined loss function: {loss_type}")


# -------------------------------------Classes implementing loss functions--------------------------------


class Focal_Loss(Loss_Function):
    def __init__(self, use_weights=False):
        super().__init__("name", use_weights)
        self.loss_fn = FocalLoss(
            include_background=True,
            gamma=2.0,
            alpha=0.25,
            reduction="mean",
        )

    def compute_loss(self, input: Tensor, target: Tensor):
        return self.loss_fn(input, target)

    def set_multiclass(self):
        return

    def set_gamma(self, gamma):
        self.loss_fn.gamma = gamma
        return

    def set_alpha(self, alpha):
        self.loss_fn.alpha = alpha
        return


class Weighted_Focal_Loss(Loss_Function):
    def __init__(self, use_weights=False):
        super().__init__("name", use_weights)
        self.loss_fn = FocalLoss(
            include_background=True,
            gamma=2.0,
            alpha=0.25,
            reduction="none",
        )

    def compute_loss(
        self, input: Tensor, target: Tensor, weight_map: Tensor
    ):
        focal_loss = self.loss_fn(input, target)
        return (focal_loss * weight_map).mean()

    def set_multiclass(self):
        return

    def set_gamma(self, gamma):
        self.loss_fn.gamma = gamma
        return

    def set_alpha(self, alpha):
        self.loss_fn.alpha = alpha
        return


class Jaccard_Dice_Focal_Loss(Loss_Function):
    def __init__(self, use_weights=False):
        super().__init__("name", use_weights)
        self.focal_loss = Focal_Loss()
        self.jaccard_loss = Jaccard_Loss()
        self.dice_loss = Dice_Loss()
        self.multiclass = False

    def set_weight(self, pos_weight):
        return

    def compute_loss(self, input: Tensor, target: Tensor):
        loss_1 = self.focal_loss.compute_loss(input, target)
        loss_2 = self.jaccard_loss.compute_loss(input, target)
        loss_3 = self.dice_loss.compute_loss(input, target)
        return loss_1 + loss_2 + loss_3

    def set_multiclass(self, multiclass: bool):
        self.multiclass = multiclass
        self.jaccard_loss.set_multiclass(multiclass)
        self.dice_loss.set_multiclass(multiclass)
        return


class Jaccard_Focal_Loss(Loss_Function):
    def __init__(self, use_weights=False):
        super().__init__("name", use_weights)
        self.focal_loss = Focal_Loss()
        self.jaccard_loss = Jaccard_Loss()
        self.multiclass = False

    def set_weight(self, pos_weight):
        return

    def compute_loss(self, input: Tensor, target: Tensor):
        loss_1 = self.focal_loss.compute_loss(input, target)
        loss_2 = self.jaccard_loss.compute_loss(input, target)
        return loss_1 + loss_2

    def set_multiclass(self, multiclass: bool):
        self.multiclass = multiclass
        self.jaccard_loss.set_multiclass(multiclass)
        return


# Jaccard loss
class Jaccard_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Jaccard Loss", False)
        self.multiclass = False

    def set_multiclass(self, multiclass: bool):
        self.multiclass = multiclass

    def set_weight(self, pos_weight: float):
        return

    def compute_loss(self, input: Tensor, target: Tensor):
        return jaccard_loss(
            input.float(), target.float(), multiclass=self.multiclass
        )


# Dice loss
class Dice_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Dice Loss", False)
        self.multiclass = False

    def set_multiclass(self, multiclass: bool):
        self.multiclass = multiclass

    def compute_loss(self, input: Tensor, target: Tensor):
        input = input.float()
        target = target.float()
        return dice_loss(input, target, multiclass=self.multiclass)


# Binary cross entropy loss
class BCE_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("BCE Loss", False)
        self.multiclass = False

    def compute_loss(self, input: Tensor, target: Tensor):
        return nn.BCELoss()(input, target.float())

    def set_multiclass(self, _):
        return False


# ------------------------------------------Dice loss functions--------------------------------------
def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon=1e-6,
):
    # Average of Dice coefficient for all batches, or for a single mask

    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f"Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})"
        )

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon=1e-6,
):

    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(
            input[:, channel, ...],
            target[:, channel, ...],
            reduce_batch_first,
            epsilon,
        )

    return dice / input.shape[1]


def dice_loss(
    input: Tensor, target: Tensor, multiclass: bool = False
):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


# -------------------------------------------------------Jaccard loss function --------------------------------
def jaccard_coef(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon=1e-6,
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f"Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})"
        )

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(torch.pow(input, 2)) + torch.sum(
            torch.pow(target, 2)
        )
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (inter + epsilon) / (sets_sum - inter + epsilon)
    else:
        # compute and average metric for each batch element
        jaccard = 0
        for i in range(input.shape[0]):
            jaccard += jaccard_coef(input[i, ...], target[i, ...])
        return jaccard / input.shape[0]


def multiclass_jaccard_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon=1e-6,
):
    # Average of jaccard coefficient for all classes
    assert input.size() == target.size()
    jaccard = 0
    for channel in range(input.shape[1]):
        jaccard += jaccard_coef(
            input[:, channel, ...],
            target[:, channel, ...],
            reduce_batch_first,
            epsilon,
        )
    return jaccard / input.shape[1]


def jaccard_loss(
    input: Tensor, target: Tensor, multiclass: bool = False
):
    # Jaccard loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_jaccard_coeff if multiclass else jaccard_coef
    return 1 - fn(input, target, reduce_batch_first=True)
