from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as functional
from scipy.signal import convolve2d
from skimage.feature import peak_local_max
from torchvision.transforms import v2


def weights_init(m):
    classname = m.__class__.__name__
    # ! Fixed the type checking
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(
            m.weight, gain=nn.init.calculate_gain("tanh")
        )
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)

    if "norm" in classname.lower():
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    if "linear" in classname.lower():
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    return


def group1_forward_branch(layer, in_tensor, resized_feat):
    """Defines group 1 connections.

    Args:
        layer (torch.nn.Module):
            Network layer.
        in_tensor (torch.Tensor):
            Input tensor.
        resized_feat (torch.Tensor):
            Resized input.

    Returns:
        torch.Tensor:
            Output of group 1 layer.

    """
    a = layer["conv1"](in_tensor)
    a = layer["conv2"](a)
    a = layer["pool"](a)
    b = layer["conv3"](resized_feat)
    b = layer["conv4"](b)
    return torch.cat(tensors=(a, b), dim=1)


def group2_forward_branch(layer, in_tensor):
    """Defines group 1 connections.

    Args:
        layer (torch.nn.Module):
            Network layer.
        in_tensor (torch.Tensor):
            Input tensor.

    Returns:
        torch.Tensor:
            Output of group 1 layer.

    """
    a = layer["conv1"](in_tensor)
    return layer["conv2"](a)


def group3_forward_branch(layer, main_feat, skip):
    """Defines group 1 connections.

    Args:
        layer (torch.nn.Module):
            Network layer.
        main_feat (torch.Tensor):
            Input tensor.
        skip (torch.Tensor):
            Skip connection.

    Returns:
        torch.Tensor: Output of group 1 layer.

    """
    a = layer["up1"](main_feat)
    a = layer["conv1"](a)
    a = layer["conv2"](a)

    b1 = layer["up2"](a)
    b2 = layer["up3"](skip)
    b = torch.cat(tensors=(b1, b2), dim=1)
    return layer["conv3"](b)


def group4_forward_branch(layer, in_tensor):
    """Defines group 1 connections.

    Args:
        layer (torch.nn.Module):
            Network layer.
        in_tensor (torch.Tensor):
            Input tensor.

    Returns:
        torch.Tensor: Output of group 1 layer.

    """
    a = layer["up1"](in_tensor)
    return layer["conv1"](a)


def group1_arch_branch(in_ch: int, resized_in_ch: int, out_ch: int):
    """Group1 branch for MicroNet.

    Args:
        in_ch (int):
            Number of input channels.
        resized_in_ch (int):
            Number of input channels from resized input.
        out_ch (int):
            Number of output channels.

    Returns:
        torch.nn.ModuleDict:
            An output of type :class:`torch.nn.ModuleDict`

    """
    module_dict = OrderedDict()
    module_dict["conv1"] = nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        nn.Tanh(),
        nn.BatchNorm2d(out_ch),
    )
    module_dict["conv2"] = nn.Sequential(
        nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        nn.Tanh(),
    )
    module_dict["pool"] = nn.MaxPool2d(2, padding=0)  # check padding

    module_dict["conv3"] = nn.Sequential(
        nn.Conv2d(
            resized_in_ch,
            out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        nn.Tanh(),
        nn.BatchNorm2d(out_ch),
    )
    module_dict["conv4"] = nn.Sequential(
        nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        nn.Tanh(),
    )
    return nn.ModuleDict(module_dict)


def group2_arch_branch(in_ch, out_ch):
    """Group2 branch for MicroNet.

    Args:
        in_ch (int):
            Number of input channels.
        out_ch (int):
            Number of output channels.

    Returns:
        torch.nn.ModuleDict:
            An output of type :class:`torch.nn.ModuleDict`

    """
    module_dict = OrderedDict()
    module_dict["conv1"] = nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        nn.Tanh(),
    )
    module_dict["conv2"] = nn.Sequential(
        nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        nn.Tanh(),
    )
    return nn.ModuleDict(module_dict)


def group3_arch_branch(in_ch, skip, out_ch):
    """Group3 branch for MicroNet.

    Args:
        in_ch (int):
            Number of input channels.
        skip (int):
            Number of channels for the skip connection.
        out_ch (int):
            Number of output channels.

    Returns:
        torch.nn.ModuleDict:
            An output of type :class:`torch.nn.ModuleDict`

    """
    module_dict = OrderedDict()
    module_dict["up1"] = nn.ConvTranspose2d(
        in_ch, out_ch, kernel_size=(2, 2), stride=(2, 2)
    )
    module_dict["conv1"] = nn.Sequential(
        nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        nn.Tanh(),
    )
    module_dict["conv2"] = nn.Sequential(
        nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        nn.Tanh(),
    )
    module_dict["up2"] = nn.ConvTranspose2d(
        out_ch, out_ch, kernel_size=(5, 5), stride=(1, 1)
    )

    module_dict["up3"] = nn.ConvTranspose2d(
        skip, out_ch, kernel_size=(5, 5), stride=(1, 1)
    )

    module_dict["conv3"] = nn.Sequential(
        nn.Conv2d(
            2 * out_ch,
            out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        nn.Tanh(),
    )
    return nn.ModuleDict(module_dict)


def group4_arch_branch(
    in_ch,
    out_ch,
    up_kernel=(2, 2),
    up_strides=(2, 2),
    activation="tanh",
):
    """Group4 branch for MicroNet.

    Args:
        in_ch (int):
            Number of input channels.
        out_ch (int):
            Number of output channels.
        up_kernel (tuple of int):
            Kernel size for
            :class:`torch.nn.ConvTranspose2d`.
        up_strides (tuple of int):
            Stride size for
            :class:`torch.nn.ConvTranspose2d`.

    Returns:
        torch.nn.ModuleDict:
            An output of type :class:`torch.nn.ModuleDict`

    """
    if activation == "relu":
        activation = nn.ReLU()
    else:
        activation = nn.Tanh()

    module_dict = OrderedDict()
    module_dict["up1"] = nn.ConvTranspose2d(
        in_ch, out_ch, kernel_size=up_kernel, stride=up_strides
    )
    module_dict["conv1"] = nn.Sequential(
        nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        activation,
    )
    return nn.ModuleDict(module_dict)


def out_arch_branch(in_ch, num_class=2, activation="softmax"):
    """Group5 branch for MicroNet.

    Args:
        in_ch (int):
            Number of input channels.
        num_class (int):
            Number of output channels. default=2.

    Returns:
        torch.nn.Sequential:
            An output of type :class:`torch.nn.Sequential`

    """
    if activation == "relu":
        activation = nn.ReLU()
    else:
        activation = nn.Softmax(dim=1)
    return nn.Sequential(
        nn.Dropout2d(p=0.5),
        nn.Conv2d(
            in_ch,
            num_class,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            bias=True,
        ),
        activation,
    )


class MicroNet(nn.Module):
    def __init__(
        self,
        num_input_channels=3,
        num_class=2,
        out_activation="softmax",
    ):
        super().__init__()
        if num_class < 2:
            raise ValueError("Number of classes should be >=2.")
        self.__num_class = num_class
        self.in_ch = num_input_channels

        module_dict = OrderedDict()
        module_dict["b1"] = group1_arch_branch(
            num_input_channels, num_input_channels, 64
        )
        module_dict["b2"] = group1_arch_branch(
            128, num_input_channels, 128
        )
        module_dict["b3"] = group1_arch_branch(
            256, num_input_channels, 256
        )
        module_dict["b4"] = group1_arch_branch(
            512, num_input_channels, 512
        )

        module_dict["b5"] = group2_arch_branch(1024, 2048)

        module_dict["b6"] = group3_arch_branch(2048, 1024, 1024)
        module_dict["b7"] = group3_arch_branch(1024, 512, 512)
        module_dict["b8"] = group3_arch_branch(512, 256, 256)
        module_dict["b9"] = group3_arch_branch(256, 128, 128)

        module_dict["fm1"] = group4_arch_branch(
            128, 64, (2, 2), (2, 2), activation=out_activation
        )
        module_dict["fm2"] = group4_arch_branch(
            256, 128, (4, 4), (4, 4), activation=out_activation
        )
        module_dict["fm3"] = group4_arch_branch(
            512, 256, (8, 8), (8, 8), activation=out_activation
        )

        module_dict["aux_out1"] = out_arch_branch(
            64, num_class=self.__num_class
        )
        module_dict["aux_out2"] = out_arch_branch(
            128, num_class=self.__num_class
        )
        module_dict["aux_out3"] = out_arch_branch(
            256, num_class=self.__num_class
        )

        module_dict["out"] = out_arch_branch(
            64 + 128 + 256,
            num_class=self.__num_class,
            activation=out_activation,
        )

        self.layer = nn.ModuleDict(module_dict)

        self.apply(weights_init)

    def forward(
        self, input_tensor: torch.Tensor
    ):  # skipcq: PYL-W0221
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            input_tensor (torch.Tensor):
                Input images, the tensor is in the shape of NHCW.

        Returns:
            list:
                A list of main and auxiliary outputs. The expected
                format is `[main_output, aux1, aux2, aux3]`.

        """
        b1 = group1_forward_branch(
            self.layer["b1"],
            input_tensor,
            functional.interpolate(
                input_tensor, size=(128, 128), mode="bicubic"
            ),
        )
        b2 = group1_forward_branch(
            self.layer["b2"],
            b1,
            functional.interpolate(
                input_tensor, size=(64, 64), mode="bicubic"
            ),
        )
        b3 = group1_forward_branch(
            self.layer["b3"],
            b2,
            functional.interpolate(
                input_tensor, size=(32, 32), mode="bicubic"
            ),
        )
        b4 = group1_forward_branch(
            self.layer["b4"],
            b3,
            functional.interpolate(
                input_tensor, size=(16, 16), mode="bicubic"
            ),
        )
        b5 = group2_forward_branch(self.layer["b5"], b4)
        b6 = group3_forward_branch(self.layer["b6"], b5, b4)
        b7 = group3_forward_branch(self.layer["b7"], b6, b3)
        b8 = group3_forward_branch(self.layer["b8"], b7, b2)
        b9 = group3_forward_branch(self.layer["b9"], b8, b1)
        fm1 = group4_forward_branch(self.layer["fm1"], b9)
        fm2 = group4_forward_branch(self.layer["fm2"], b8)
        fm3 = group4_forward_branch(self.layer["fm3"], b7)

        aux1 = self.layer["aux_out1"](fm1)
        aux2 = self.layer["aux_out2"](fm2)
        aux3 = self.layer["aux_out3"](fm3)

        out = torch.cat(tensors=(fm1, fm2, fm3), dim=1)
        out = self.layer["out"](out)

        return [out, aux1, aux2, aux3]


def gauss_2d_filter(shape=(11, 11)):
    sigma = int((shape[0] - 1) / 6)
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    h = h / (h[int(m), int(n)])
    return h


class MapDe(MicroNet):
    def __init__(
        self,
        num_input_channels: int = 3,
        min_distance: int = 4,
        threshold_abs: float = 250,
        num_classes: int = 1,
        filter_size: int = 31,
    ) -> None:
        """Initialize :class:`MapDe`."""
        super().__init__(
            num_class=2,
            num_input_channels=num_input_channels,
            out_activation="relu",
        )

        self.filter_size = filter_size

        dist_filter = gauss_2d_filter(
            shape=(filter_size, filter_size)
        )
        dist_filter = np.expand_dims(dist_filter, axis=(0, 1))  # NCHW

        self.register_buffer(
            "dist_filter_2d",
            torch.from_numpy(dist_filter.astype(np.float32)),
        )
        self.dist_filter_2d.requires_grad = False

        dist_filter = np.repeat(
            dist_filter, repeats=num_classes * 2, axis=1
        )
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.register_buffer(
            "dist_filter",
            torch.from_numpy(dist_filter.astype(np.float32)),
        )
        self.dist_filter.requires_grad = False

        self.preprocess_reshape = v2.Compose([v2.Resize((252, 252))])

    def reshape_transform(self, input_tensor: torch.Tensor):
        out = self.preprocess_reshape(input_tensor)
        return out

    def logits_to_probs(self, logits: torch.Tensor):
        probs = torch.zeros_like(logits)
        for i in range(logits.shape[0]):
            probs[i] = logits[i] / torch.max(logits[i])
        return probs

    def blur_cell_points(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Blur input cell masks (points) using dist_filter
        """
        if torch.is_tensor(input_tensor):
            out = F.conv2d(
                input_tensor.float(),
                self.dist_filter_2d,
                padding="same",
            )
            return out
        else:
            input_tensor = torch.tensor(input_tensor)
            out = F.conv2d(
                input_tensor.float(),
                self.dist_filter_2d,
                padding="same",
            )
            return out.numpy(force=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            input_tensor (torch.Tensor):
                Input images, the tensor is in the shape of NCHW.

        Returns:
            torch.Tensor:
                Output map for cell detection. Peak detection should be applied
                to this output for cell detection.

        """
        logits, _, _, _ = super().forward(input_tensor)
        out = F.conv2d(logits, self.dist_filter, padding="same")
        return F.relu(out)

    def postproc(self, prediction_map: torch.Tensor) -> np.ndarray:
        """Post-processing script for MicroNet.

        Performs peak detection and extracts coordinates in x, y format.

        Args:
            prediction_map (Tensor):
                Logits after forward pass (Bx1x252x252).

        Returns:
            :class:`numpy.ndarray`:
                Pixel-wise nuclear instance segmentation
                prediction.

        """
        prediction_map_numpy = prediction_map.numpy(force=True)
        prediction_map_numpy = np.squeeze(
            prediction_map_numpy, axis=1
        )
        batches = prediction_map_numpy.shape[0]
        output_mask = np.zeros(shape=(batches, 252, 252))

        for i in range(0, batches):
            coordinates = peak_local_max(
                prediction_map_numpy[i],
                min_distance=self.min_distance,
                threshold_abs=self.threshold_abs,
                exclude_border=False,
            )
            output_mask[i][coordinates[:, 0], coordinates[:, 1]] = 1

        return output_mask
