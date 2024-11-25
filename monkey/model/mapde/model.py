import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import peak_local_max
from tiatoolbox.models.architecture.micronet import MicroNet
from torchvision.transforms import v2


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
            num_output_channels=num_classes * 2,
            num_input_channels=num_input_channels,
            out_activation="relu",
        )

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

    def blur_cell_points(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Blur input cell masks (points) using dist_filter
        """
        out = F.conv2d(
            input_tensor.float(), self.dist_filter_2d, padding="same"
        )
        return out

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
            prediction_map_numpy = prediction_map_numpy / np.max(prediction_map_numpy)
            coordinates = peak_local_max(
                prediction_map_numpy[i],
                min_distance=self.min_distance,
                threshold_abs=self.threshold_abs,
                exclude_border=False,
            )
            output_mask[i][coordinates[:, 0], coordinates[:, 1]] = 1

        return output_mask
