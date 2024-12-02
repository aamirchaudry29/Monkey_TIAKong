import numpy as np
import torch
from skimage.feature import peak_local_max

from monkey.data.data_utils import morphological_post_processing


def multihead_det_post_process(
    inflamm_prob: torch.Tensor,
    lymph_prob: torch.Tensor,
    mono_prob: torch.Tensor,
    thresholds: list = [0.5, 0.5, 0.5],
    min_distances: list = [5, 5, 5],
):
    if torch.is_tensor(inflamm_prob):
        inflamm_prob = inflamm_prob.numpy(force=True)
    if torch.is_tensor(lymph_prob):
        lymph_prob = lymph_prob.numpy(force=True)
    if torch.is_tensor(mono_prob):
        mono_prob = mono_prob.numpy(force=True)

    inflamm_output_mask = np.zeros(
        shape=inflamm_prob.shape, dtype=np.uint8
    )
    lymph_output_mask = np.zeros(
        shape=lymph_prob.shape, dtype=np.uint8
    )
    mono_output_mask = np.zeros(shape=mono_prob.shape, dtype=np.uint8)

    inflamm_coordinates = peak_local_max(
        inflamm_prob,
        min_distance=min_distances[0],
        threshold_abs=thresholds[0],
        exclude_border=False,
    )
    inflamm_output_mask[
        inflamm_coordinates[:, 0], inflamm_coordinates[:, 1]
    ] = 1
    # inflamm_output_mask = (inflamm_prob > thresholds[0]).astype(np.uint8)
    # inflamm_output_mask = morphological_post_processing(inflamm_output_mask)

    lymph_coordinates = peak_local_max(
        lymph_prob,
        min_distance=min_distances[1],
        threshold_abs=thresholds[1],
        exclude_border=False,
    )
    lymph_output_mask[
        lymph_coordinates[:, 0], lymph_coordinates[:, 1]
    ] = 1
    # lymph_output_mask = (lymph_prob > thresholds[1]).astype(np.uint8)
    # lymph_output_mask = morphological_post_processing(lymph_output_mask)

    mono_coordinates = peak_local_max(
        mono_prob,
        min_distance=min_distances[2],
        threshold_abs=thresholds[2],
        exclude_border=False,
    )
    mono_output_mask[
        mono_coordinates[:, 0], mono_coordinates[:, 1]
    ] = 1
    # mono_output_mask = (mono_prob > thresholds[2]).astype(np.uint8)
    # mono_output_mask = morphological_post_processing(mono_output_mask)

    return {
        "inflamm_mask": inflamm_output_mask,
        "lymph_mask": lymph_output_mask,
        "mono_mask": mono_output_mask,
    }


def multihead_det_post_process_batch(
    inflamm_prob: torch.Tensor,
    lymph_prob: torch.Tensor,
    mono_prob: torch.Tensor,
    thresholds: list = [0.5, 0.5, 0.5],
    min_distances: list = [5, 5, 5],
):

    if torch.is_tensor(inflamm_prob):
        inflamm_prob = inflamm_prob.numpy(force=True)
    if torch.is_tensor(lymph_prob):
        lymph_prob = lymph_prob.numpy(force=True)
    if torch.is_tensor(mono_prob):
        mono_prob = mono_prob.numpy(force=True)

    inflamm_prob = np.squeeze(inflamm_prob, axis=1)
    lymph_prob = np.squeeze(lymph_prob, axis=1)
    mono_prob = np.squeeze(mono_prob, axis=1)

    batches = inflamm_prob.shape[0]
    inflamm_output_mask = np.zeros(
        shape=(batches, inflamm_prob.shape[1], inflamm_prob.shape[2]),
        dtype=np.uint8,
    )
    lymph_output_mask = np.zeros(
        shape=(batches, lymph_prob.shape[1], lymph_prob.shape[2]),
        dtype=np.uint8,
    )
    mono_output_mask = np.zeros(
        shape=(batches, mono_prob.shape[1], mono_prob.shape[2]),
        dtype=np.uint8,
    )

    for i in range(0, batches):
        inflamm_coordinates = peak_local_max(
            inflamm_prob[i],
            min_distance=min_distances[0],
            threshold_abs=thresholds[0],
            exclude_border=False,
        )
        inflamm_output_mask[i][
            inflamm_coordinates[:, 0], inflamm_coordinates[:, 1]
        ] = 1

        lymph_coordinates = peak_local_max(
            lymph_prob[i],
            min_distance=min_distances[1],
            threshold_abs=thresholds[1],
            exclude_border=False,
        )
        lymph_output_mask[i][
            lymph_coordinates[:, 0], lymph_coordinates[:, 1]
        ] = 1

        mono_coordinates = peak_local_max(
            mono_prob[i],
            min_distance=min_distances[2],
            threshold_abs=thresholds[2],
            exclude_border=False,
        )
        mono_output_mask[i][
            mono_coordinates[:, 0], mono_coordinates[:, 1]
        ] = 1

    return {
        "inflamm_mask": inflamm_output_mask,
        "lymph_mask": lymph_output_mask,
        "mono_mask": mono_output_mask,
    }
