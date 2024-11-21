import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import peak_local_max
from tiatoolbox.models.architecture.micronet import MicroNet


def gauss_2d_filter(shape=(11, 11), sigma=1):
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
    ) -> None:
        """Initialize :class:`MapDe`."""
        super().__init__(
            num_output_channels=num_classes * 2,
            num_input_channels=num_input_channels,
            out_activation="relu",
        )

        dist_filter = gauss_2d_filter()

        dist_filter = np.expand_dims(dist_filter, axis=(0, 1))  # NCHW
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
        print(logits.size())
        print(self.dist_filter.size())
        out = F.conv2d(logits, self.dist_filter, padding="same")
        return F.relu(out)

    def postproc(self, prediction_map: np.ndarray) -> np.ndarray:
        """Post-processing script for MicroNet.

        Performs peak detection and extracts coordinates in x, y format.

        Args:
            prediction_map (ndarray):
                Input image of type numpy array.

        Returns:
            :class:`numpy.ndarray`:
                Pixel-wise nuclear instance segmentation
                prediction.

        """
        coordinates = peak_local_max(
            np.squeeze(prediction_map[0], axis=2),
            min_distance=self.min_distance,
            threshold_abs=self.threshold_abs,
            exclude_border=False,
        )
        return np.fliplr(coordinates)
