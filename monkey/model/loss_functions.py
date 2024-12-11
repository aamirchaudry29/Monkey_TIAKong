from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss
from torch import Tensor


# Abstract class for loss functions
# All loss functions need to a subclass of this class
class Loss_Function(ABC):
    def __init__(self, name, use_weights) -> None:
        self.name = name
        self.use_weight = use_weights

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def set_multiclass(self):
        pass


# Returns a Loss Function instance depending on the loss type
def get_loss_function(loss_type: str) -> Loss_Function:
    """
    Returns an initialized loss function object.

    Options:
        {"Jaccard_Loss", "Dice", "BCE", "Weighted_BCE", "BCE_Dice",
        "Weighted_BCE_Dice","MSE", "Weighted_CrossEntropy", "Dice_Focal_Loss"
        "Weighted_CE_Dice"}
    """
    loss_functions = {
        "Jaccard_Loss": Jaccard_Loss,
        "Dice": Dice_Loss,
        "BCE": BCE_Loss,
        "Weighted_BCE": Weighted_BCE_Loss,
        "BCE_Dice": BCE_Dice_Loss,
        "Weighted_BCE_Dice": Weighted_BCE_Dice_Loss,
        "MSE": MSE_Loss,
        "Weighted_CrossEntropy": CrossEntropy_Loss,
        "Dice_Focal_Loss": Dice_Focal_Loss,
        "Weighted_CE_Dice": Weighted_CE_Dice_Loss,
        "MapDe_Loss": MapDe_Loss,
        "Weighted_BCE_Jaccard": Weighted_BCE_Jaccard_Loss,
        "Weighted_MSE": Weighted_MSE_Loss,
        # To add a new loss function, first create a subclass of Loss_Function
        # Then add a new entry here:
        # "<loss_type>": <class name>
    }

    if loss_type in loss_functions:
        return loss_functions[loss_type]()
    else:
        raise ValueError(f"Undefined loss function: {loss_type}")


# -------------------------------------Classes implementing loss functions---------------------------------
class MapDe_Loss(Loss_Function):
    def __init__(self, use_weights=True):
        super().__init__("MapDe_Loss", use_weights)
        self.multiclass = False
        self.pos_weight = 1000.0

    def set_multiclass(self, multiclass):
        self.multiclass = multiclass

    def set_weight(self, pos_weight):
        self.pos_weight = pos_weight

    def ce_loss(self, input: Tensor, target: Tensor):
        epsilon = 1e-7
        log_weight = 1 + (self.pos_weight - 1) * target
        clipped_logits = torch.clamp(
            input, min=epsilon, max=1.0 - epsilon
        )
        clipped_logits = torch.sigmoid(clipped_logits)
        loss = (
            target * -clipped_logits.log() * log_weight
            + (1 - target) * -(1.0 - clipped_logits).log()
        )
        return torch.mean(torch.flatten(loss))

    def compute_loss(self, input: Tensor, target: Tensor):
        if self.multiclass:
            bce_loss = 0.0
            for channel in range(input.shape[1]):
                bce_loss += self.ce_loss(
                    input[:, channel, ...],
                    target[:, channel, ...].float(),
                )
        else:
            bce_loss = self.ce_loss(input, target.float())
        return bce_loss


class Focal_Loss(Loss_Function):
    def __init__(self, use_weights=False):
        super().__init__("name", use_weights)
        self.loss_fn = FocalLoss(
            include_background=True, gamma=2.0, alpha=0.25
        )

    def compute_loss(self, input: Tensor, target: Tensor):
        return self.loss_fn(input, target)

    def set_multiclass(self):
        return


class Dice_Focal_Loss(Loss_Function):
    def __init__(self, use_weights=False):
        super().__init__("name", use_weights)
        self.focal_loss = Focal_Loss()
        self.dice_loss = Dice_Loss()
        self.multiclass = False

    def compute_loss(self, input: Tensor, target: Tensor):
        loss_1 = self.focal_loss.compute_loss(input, target)
        loss_2 = self.dice_loss.compute_loss(input, target)
        return loss_1 + loss_2

    def set_multiclass(self, multiclass: bool):
        self.multiclass = multiclass
        self.dice_loss.set_multiclass(multiclass)
        return


class CrossEntropy_Loss(Loss_Function):
    def __init__(self, use_weights=True):
        super().__init__("name", use_weights)
        self.class_weight = torch.tensor([0.5, 0.5], device="cuda")

    def set_weight(self, class_weight):
        self.class_weight = class_weight

    def compute_loss(
        self,
        input: Tensor,
        target: Tensor,
    ):
        loss = nn.CrossEntropyLoss(weight=self.class_weight)
        return loss(input, target)

    def set_multiclass(self):
        return


# MSE loss
class MSE_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("MSE", False)
        self.multiclass = False
        self.pos_weight = 1.0
        self.mse_ln = nn.MSELoss()

    def set_multiclass(self, multiclass):
        self.multiclass = multiclass

    def set_weight(self, pos_weight):
        self.pos_weight = pos_weight

    def compute_loss(
        self,
        input: Tensor,
        target: Tensor,
    ):
        assert (
            input.size() == target.size()
        )  # "Input size {} must be the same as target size {}".format(input.size(), target.size())
        target = target * self.pos_weight
        input = torch.clamp_max(input, self.pos_weight)
        return self.mse_ln(input, target)


class Weighted_MSE_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Weighted_MSE", False)
        self.multiclass = False
        self.non_zero_weight = 1.0

    def set_multiclass(self, multiclass):
        self.multiclass = multiclass

    def set_weight(self, weight):
        self.non_zero_weight = weight

    def compute_loss(
        self,
        input: Tensor,
        target: Tensor,
    ):
        assert (
            input.size() == target.size()
        )  # "Input size {} must be the same as target size {}".format(input.size(), target.size())
        weights = torch.where((target > 0), self.non_zero_weight, 1.0)
        squared_diff = (input - target) ** 2
        weighted_loss = squared_diff * weights
        return weighted_loss.mean()


# Jaccard loss
class Jaccard_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Jaccard Loss", False)
        self.multiclass = False

    def set_multiclass(self, multiclass: bool):
        self.multiclass = multiclass

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


# Weighted binary cross entropy loss
class Weighted_BCE_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Weighted BCE Loss", True)
        self.multiclass = False
        self.pos_weight = 1.0

    def set_multiclass(self, multiclass):
        self.multiclass = multiclass

    def set_weight(self, pos_weight: float):
        self.pos_weight = pos_weight

    def compute_loss(self, input: Tensor, target: Tensor):
        weight_map = 1 + (self.pos_weight - 1) * target
        loss = nn.functional.binary_cross_entropy(
            input.float(),
            target.float(),
            weight=weight_map,
            reduction="mean",
        )
        return loss


# Binary cross entropy + Dice loss
class BCE_Dice_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("BCE + Dice Loss", False)
        self.multiclass = False
        self.bce_loss_fn = nn.BCELoss()

    def set_multiclass(self, multiclass: bool):
        self.multiclass = multiclass

    def set_weight(self, weight):
        return

    def compute_loss(self, input: Tensor, target: Tensor):
        if self.multiclass:
            bce_loss = 0.0
            for channel in range(input.shape[1]):
                bce_loss += self.bce_loss_fn(
                    input[:, channel, ...],
                    target[:, channel, ...].float(),
                )
        else:
            bce_loss = self.bce_loss_fn(input, target.float())

        return bce_loss + dice_loss(
            input.float(), target.float(), multiclass=self.multiclass
        )


# Weighted binary cross entropy + Dice loss
class Weighted_BCE_Dice_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Weighted BCE Loss + Dice Loss", True)
        self.multiclass = False
        self.pos_weight = 1000.0
        self.bce_loss = Weighted_BCE_Loss()
        self.bce_loss.set_weight(self.pos_weight)

    def set_multiclass(self, multiclass):
        self.multiclass = multiclass

    def set_weight(self, pos_weight: float):
        self.pos_weight = pos_weight
        self.bce_loss.set_weight(self.pos_weight)

    def compute_loss(self, input: Tensor, target: Tensor):
        bce = self.bce_loss.compute_loss(input, target)
        dice = dice_loss(
            input.float(), target.float(), multiclass=self.multiclass
        )
        return bce + dice


class Weighted_BCE_Jaccard_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Weighted BCE Loss + Jaccard Loss", True)
        self.multiclass = False
        self.pos_weight = 1.0
        self.bce_loss = Weighted_BCE_Loss()
        self.bce_loss.set_weight(self.pos_weight)

    def set_multiclass(self, multiclass):
        self.multiclass = multiclass

    def set_weight(self, pos_weight: float):
        self.pos_weight = pos_weight
        self.bce_loss.set_weight(self.pos_weight)

    def compute_loss(self, input: Tensor, target: Tensor):
        bce = self.bce_loss.compute_loss(input, target)
        dice = jaccard_loss(
            input.float(), target.float(), multiclass=self.multiclass
        )
        return bce + dice


class Weighted_CE_Dice_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Weighted CE Loss + Dice Loss", True)
        self.multiclass = True
        self.weights = torch.tensor([0.4, 0.6], device="cuda")

    def set_multiclass(self, multiclass: bool):
        self.multiclass = multiclass

    def set_weight(self, weights: Tensor):
        self.weights = weights
        self.weights.to("cuda")

    def compute_loss(self, input: Tensor, target: Tensor):

        # convert target to one-hot
        # BxCxHxW -> BxHxW
        target_indices = torch.argmax(target, dim=1)

        return nn.CrossEntropyLoss(
            weight=self.weights, reduction="sum"
        )(input, target_indices) + dice_loss(
            input.float(), target.float(), multiclass=self.multiclass
        )


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


# ------------------------ inter and intra class loss ----------------------
def inter_class_exclusion_loss(dist_pred_pos, dist_pred_neg):
    """
    Penalize overlapping peaks between positive and negative maps.
    dist_pred_pos: Predicted distance map for positive class [batch_size, H, W]
    dist_pred_neg: Predicted distance map for negative class [batch_size, H, W]
    """

    # Compute overlap per sample
    overlap = dist_pred_pos * dist_pred_neg  # [batch_size, H, W]
    batch_size = dist_pred_pos.size(0)
    overlap_flat = overlap.view(batch_size, -1)
    loss_per_sample = overlap_flat.sum(dim=1) / overlap_flat.size(1)
    loss = loss_per_sample.mean()
    return loss


# def intra_class_repulsion_loss_single(dist_pred):
#     """
#     Penalize peaks that are too close to each other within the same map.
#     dist_pred: Predicted distance map for one class [batch_size, H, W]
#     """
#     # Apply ReLU activation
#     batch_size = dist_pred.size(0)
#     loss = torch.tensor(0.0, device=dist_pred.device)

#     # Find peaks using max pooling
#     max_pooled = F.max_pool2d(dist_pred, kernel_size=3, stride=1, padding=1)
#     peaks = (dist_pred == max_pooled) & (dist_pred > self.threshold)

#     # Convert peaks to coordinates
#     for i in range(batch_size):
#         peak_indices = peaks[i].nonzero(as_tuple=False).float()  # [num_peaks, 2]
#         if peak_indices.size(0) > 1:
#             # Compute pairwise distances
#             pdist = torch.cdist(peak_indices, peak_indices, p=2)
#             # Remove self-distances
#             pdist = pdist + torch.eye(pdist.size(0), device=pdist.device) * 1e6
#             # Penalize close peaks
#             close_peaks = (pdist < self.min_distance).float()
#             loss += close_peaks.sum() / (peak_indices.size(0) ** 2)

#     return loss / batch_size
