from abc import ABC, abstractmethod

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# import torch.nn.functional as F
# from monai.losses import DiceCELoss, DiceLoss
# from monai.metrics import ConfusionMatrixMetric, DiceMetric
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
        "Weighted_BCE_Dice","MSE", "Weighted_CrossEntropy"}
    """
    loss_functions = {
        "Jaccard_Loss": Jaccard_Loss,
        "Dice": Dice_Loss,
        "BCE": BCE_Loss,
        "Weighted_BCE": Weighted_BCE_Loss,
        "BCE_Dice": BCE_Dice_Loss,
        "Weighted_BCE_Dice": Weighted_BCE_Dice_Loss,
        "MSE": MSE_loss,
        "Weighted_CrossEntropy": CrossEntropy_loss,
        # To add a new loss function, first create a subclass of Loss_Function
        # Then add a new entry here:
        # "<loss_type>": <class name>
    }

    if loss_type in loss_functions:
        return loss_functions[loss_type]()
    else:
        raise ValueError(f"Undefined loss function: {loss_type}")


# -------------------------------------Classes implementing loss functions---------------------------------
class CrossEntropy_loss(Loss_Function):
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


# MSE loss
class MSE_loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("MSE", False)

    def compute_loss(
        self,
        input: Tensor,
        target: Tensor,
        pos_Weight: float = 1000.0,
    ):
        assert (
            input.size() == target.size()
        )  # "Input size {} must be the same as target size {}".format(input.size(), target.size())
        target = target * pos_Weight
        return nn.MSELoss()(input, target)

        ####


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
        if self.multiclass:
            loss = dice_loss(
                input, target, multiclass=self.multiclass
            )
            # Add loss for channel-wise exclusive predictions
            channel_similarity = dice_coeff(
                target[:, 0, :, :],
                target[:, 1, :, :],
                reduce_batch_first=True,
            )
            return loss + channel_similarity
        else:
            return dice_loss(
                input, target, multiclass=self.multiclass
            )


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
        self.multiclass = False

    def set_multiclass(self, multiclass: bool):
        self.multiclass = multiclass

    def compute_loss(self, input: Tensor, target: Tensor):
        return nn.BCELoss()(input, target.float()) + dice_loss(
            input.float(), target.float(), multiclass=self.multiclass
        )


# Weighted binary cross entropy + Dice loss
class Weighted_BCE_Dice_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__("Weighted BCE Loss + Dice Loss", True)
        self.multiclass = False

    def set_multiclass(self, multiclass: bool):
        self.multiclass = multiclass

    def compute_loss(
        self, input: Tensor, target: Tensor, weight: Tensor
    ):
        return nn.BCELoss(weight=weight)(
            input, target.float()
        ) + dice_loss(
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
    # Average of Dice coefficient for all classes
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
