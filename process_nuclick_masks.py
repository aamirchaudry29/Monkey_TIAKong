"""
Process data from nuclick_hovernext folder.
Main purpose is to separate touching nuclei in binary segmentation mask
And generate contours
"""

import os
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

NUCLICK_DIR = "/home/u1910100/Documents/Monkey/patches_256/annotations/nuclick_hovernext"

SAVE_DIR = "/home/u1910100/Documents/Monkey/patches_256/annotations/nuclick_masks_processed"


def process_instance_and_class_map(instance_map, class_map):
    # get initial binary mask from instance map
    binary_mask = np.zeros(shape=(instance_map.shape), dtype=np.uint8)
    binary_mask = np.where(instance_map > 0, 1, 0).astype(np.uint8)

    # get gradient map
    sx = ndimage.sobel(instance_map, axis=0)
    sy = ndimage.sobel(instance_map, axis=1)
    gradient = np.hypot(sx, sy)
    gradient = (gradient > 0).astype(np.uint8)

    # Erode binary mask by gradient map
    binary_mask[gradient == 1] = 0
    # Erode class_map by gradient map
    class_map[gradient == 1] = 0
    class_map = class_map.astype(np.uint8)

    return {
        "binary_mask": binary_mask,
        "class_mask": class_map,
        "contour_mask": gradient,
    }


def process_nuclick_data_file(file_name):
    data_path = os.path.join(NUCLICK_DIR, file_name)
    data = np.load(data_path)

    image = data[:, :, 0:3]

    image = image.astype(np.uint8)

    instance_map = data[:, :, 3]
    class_map = data[:, :, 4]

    processed_masks = process_instance_and_class_map(
        instance_map, class_map
    )

    new_data = np.zeros(
        shape=(data.shape[0], data.shape[1], 6), dtype=np.uint8
    )
    new_data[:, :, 0:3] = image
    new_data[:, :, 3] = processed_masks["binary_mask"]
    new_data[:, :, 4] = processed_masks["class_mask"]
    new_data[:, :, 5] = processed_masks["contour_mask"]

    save_path = os.path.join(SAVE_DIR, file_name)
    np.save(save_path, new_data)


if __name__ == "__main__":

    files = os.listdir(NUCLICK_DIR)
    with Pool(10) as p:
        p.map(process_nuclick_data_file, files)
