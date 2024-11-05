from monkey.data.data_utils import open_json_file, write_json_file
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader
from tiatoolbox.tools.patchextraction import get_patch_extractor
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
import cv2
import json
from tqdm.auto import tqdm

def extract_patch_and_mask(image, mask, coords, size=36, label=1):
    results = []
    extractor = get_patch_extractor(
            'point',
            input_img = image,
            locations_list = np.array(coords),
            patch_size=size
        )
    mask_reader = VirtualWSIReader.open(mask)
    for i, patch in enumerate(extractor):
        x,y = extractor.locations_df["x"][i], extractor.locations_df["y"][i]
        mask_patch = mask_reader.read_rect(
            (x,y),
            (size,size)
        )[:,:,0]
        mask_patch = np.where(mask_patch==label, 1, 0)

        result_patch = np.zeros(shape=(size,size,4), dtype=np.uint8)
        result_patch[:,:,0:3] = patch
        result_patch[:,:,3] = mask_patch
        results.append(result_patch)
    return results

if __name__ == "__main__":
    # Path to folder containing all the target WSIs
    nuclick_folder = "/home/u1910100/Documents/Monkey/patches_256/annotations/nuclick_hovernext"
    json_annotation_folder = "/home/u1910100/Documents/Monkey/patches_256/annotations/json"

    save_dir = "/home/u1910100/Documents/Monkey/classification/patches"

    data_labels = {}

    files = os.listdir(nuclick_folder)

    for i in tqdm(range(len(files))):
        file_name = files[i]

        file_path = os.path.join(nuclick_folder, file_name)

        file_name_without_ext = os.path.splitext(file_name)[0]

        json_path = os.path.join(json_annotation_folder, f"{file_name_without_ext}.json")
        annotation = open_json_file(json_path)
        lymphocyte_coords = annotation['lymphocytes']
        monocyte_coords = annotation['monocytes']
        

        data = np.load(file_path)
        data = data.astype(np.uint8)
        img = data[:,:,0:3]


        mask = data[:,:,4]
        
        if len(lymphocyte_coords) > 0:
            lymph_results = []
            lymph_results = extract_patch_and_mask(img, mask, lymphocyte_coords, 36)
            for i, data in enumerate(lymph_results):
                save_name = f"{file_name_without_ext}_lymph_{i+1}"
                save_path = os.path.join(save_dir, save_name)
                np.save(save_path, data)
                data_labels[save_name] = 1


        if len(monocyte_coords) > 0:
            mono_results = []
            mono_results = extract_patch_and_mask(img, mask, monocyte_coords, 36, label=2)
            for i, data in enumerate(mono_results):
                save_name = f"{file_name_without_ext}_mono_{i+1}"
                save_path = os.path.join(save_dir, save_name)
                np.save(save_path, data)
                data_labels[save_name] = 2

    data_labels_save_path = os.path.join(
        save_dir, "labels.json"
    )
    write_json_file(data_labels_save_path, data_labels)