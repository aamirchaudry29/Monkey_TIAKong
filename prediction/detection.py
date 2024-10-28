import os



import numpy as np
import torch

from tiatoolbox.tools.patchextraction import get_patch_extractor

from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader
from torch.utils.data import DataLoader
from tqdm import tqdm
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
import skimage.morphology
import skimage.measure
import segmentation_models_pytorch as smp

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    collate_fn,
    imagenet_normalise_torch,
    px_to_mm,
    write_json_file,
    check_coord_in_mask
)


def detection_in_tile(
    image_tile: np.ndarray,
    model: torch.nn.Module,
    IOConfig: PredictionIOConfig,
):
    patch_size = IOConfig.patch_size
    stride = IOConfig.stride

    # Create patch extractor
    tile_reader = VirtualWSIReader.open(image_tile)

    patch_extractor = get_patch_extractor(
        input_img=tile_reader,
        method_name="slidingwindow",
        patch_size=(patch_size, patch_size),
        stride=(stride, stride),
        resolution=0,
        units="level",
    )

    predictions = []
    batch_size = 16
    dataloader = DataLoader(
        patch_extractor,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model.eval()
    for i, imgs in enumerate(dataloader):
        imgs = torch.permute(imgs, (0, 3, 1, 2))
        imgs = imgs / 255
        imgs = imagenet_normalise_torch(imgs)
        imgs = imgs.to("cuda").float()

        with torch.no_grad():
            out = model(imgs)
            out = torch.sigmoid(out)
            out = out.cpu().detach().numpy()[:, 0, :, :]


        # out_mask = skimage.morphology.remove_small_objects(
        #     ar=out_mask, min_size=128
        # )
        predictions.extend(list(out))

    return predictions, patch_extractor.coordinate_list


def process_tile_detection_masks(
    pred_masks: list,
    coordinate_list: list,
    x_start: int,
    y_start: int,
    threshold: float = 0.5,
    mpp: float = 0.24,
    tissue_mask: np.ndarray|None = None # 1.25 objective power
):
    if len(pred_masks) == 0:
        tile_prediction = np.zeros(shape=(1024,1024), dtype=np.uint8)
    else:
        tile_prediction = SemanticSegmentor.merge_prediction(
            (1024,1024),
            pred_masks,
            coordinate_list
        )

    tile_prediction_binary = tile_prediction > threshold
    tile_prediction_binary = skimage.morphology.remove_small_objects(
        ar=tile_prediction_binary, min_size=128
    )

    mask_labels = skimage.measure.label(tile_prediction_binary)
    stats = skimage.measure.regionprops(
        mask_labels, intensity_image=tile_prediction
    )

    points = []
    for region in stats:
        centroid = region["centroid"]

        c, r, confidence = (
            centroid[1],
            centroid[0],
            region["mean_intensity"],
        )

        c1 = c + x_start
        r1 = r + y_start

        prediction_record = {
            "x": np.round(c1),
            "y": np.round(r1),
            "type": "inflammatory",
            "probability": float(confidence),
        }

        points.append(prediction_record)

    return points




def wsi_detection_in_mask(
    wsi_name: str, mask_name: str, IOConfig: PredictionIOConfig
):
    wsi_dir = IOConfig.wsi_dir
    mask_dir = IOConfig.mask_dir
    output_dir = IOConfig.output_dir
    model_path = IOConfig.model_path

    # create model
    model = smp.Unet(   
    encoder_name='mit_b0',
    encoder_weights=None,
    decoder_attention_type='scse',
    in_channels=3,
    classes=1
    )
    model.to("cuda")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    wsi_without_ext = os.path.splitext(wsi_name)[0]

    wsi_path = os.path.join(wsi_dir, wsi_name)
    mask_path = os.path.join(mask_dir, mask_name)

    wsi_reader = WSIReader.open(wsi_path)
    mask_reader = WSIReader.open(mask_path)

    # Get baseline resolution in mpp
    base_mpp = wsi_reader.convert_resolution_units(
        input_res=0, input_unit="level", output_unit="mpp"
    )[0]
    print(f"baseline mpp = {base_mpp}")
    # Get ROI mask
    mask_thumbnail = mask_reader.slide_thumbnail()
    binary_mask = mask_thumbnail[:, :, 0]
    # Create tile extractor
    resolution = IOConfig.resolution
    units = IOConfig.units
    tile_extractor = get_patch_extractor(
        input_img=wsi_reader,
        input_mask=binary_mask,
        method_name="slidingwindow",
        patch_size=(1024, 1024),
        resolution=resolution,
        units=units,
    )

    # Detection in tile
    detected_points: list[dict] = []
    for i, tile in enumerate(
        tqdm(
            tile_extractor,
            leave=False,
            desc=f"{wsi_without_ext} progress",
        )
    ):
        bounding_box = tile_extractor.coordinate_list[
            i
        ]  # (x_start, y_start, x_end, y_end)
        predictions, coordinates = detection_in_tile(tile, model, IOConfig)
        output_points_tile = process_tile_detection_masks(
            predictions,
            coordinates,
            bounding_box[0],
            bounding_box[1],
            threshold=IOConfig.threshold,
            tissue_mask=binary_mask,
        )
        detected_points.extend(output_points_tile)



    return detected_points



