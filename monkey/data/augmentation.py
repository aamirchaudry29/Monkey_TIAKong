import albumentations as alb
import cv2
import numpy as np
from tiatoolbox.tools.stainaugment import StainAugmentor


def get_augmentation(
    module: str, gt_type=None, augment=True, aug_prob=0.7
):
    # Common augmentations
    common_augs = [
        alb.OneOf(
            [
                alb.GaussianBlur(blur_limit=(3, 5), p=0.5),
                alb.Sharpen(
                    alpha=(0.1, 0.3), lightness=(1.0, 1.0), p=0.5
                ),
                alb.ImageCompression(
                    quality_lower=30, quality_upper=80, p=0.5
                ),
            ],
            p=1.0,
        ),
        alb.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        alb.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=180,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5,
        ),
        alb.Flip(p=0.5),
    ]

    if module == "detection":
        stain_matrix = np.array(
            [
                [0.91633014, -0.20408072, -0.34451435],
                [0.17669817, 0.92528011, 0.33561059],
            ]
        )
        augs_list = [
            alb.OneOf(
                [
                    StainAugmentor(
                        "macenko",
                        stain_matrix=stain_matrix,
                        sigma1=0.25,
                        sigma2=0.2,
                        augment_background=False,
                        always_apply=True,
                        p=1.0,
                    ),
                    alb.HueSaturationValue(
                        hue_shift_limit=5,
                        sat_shift_limit=(-40, -30),
                        val_shift_limit=0,
                        always_apply=False,
                        p=0.5,
                    ),
                    alb.HueSaturationValue(
                        hue_shift_limit=5,
                        sat_shift_limit=(10, 30),
                        val_shift_limit=0,
                        always_apply=False,
                        p=0.5,
                    ),
                    alb.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=(-20, 10),
                        val_shift_limit=0,
                        always_apply=False,
                        p=0.5,
                    ),
                    alb.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=(-5, 5),
                        val_shift_limit=0,
                        always_apply=False,
                        p=0.75,
                    ),
                ],
                p=1.0,
            ),
            alb.OneOf(
                [
                    alb.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    alb.Sharpen(
                        alpha=(0.1, 0.3), lightness=(1.0, 1.0), p=0.5
                    ),
                    alb.MotionBlur(blur_limit=(3, 5), p=0.5),
                    alb.GaussNoise(var_limit=(50, 100), p=0.5),
                    alb.JpegCompression(
                        quality_lower=45,
                        quality_upper=80,
                        always_apply=False,
                        p=0.5,
                    ),
                ],
                p=1.0,
            ),
            alb.OneOf(
                [
                    alb.RandomBrightnessContrast(
                        brightness_limit=(0.3, 0.4),
                        contrast_limit=(-0.3, -0.2),
                        p=0.5,
                    ),
                    alb.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=(-0.3, 0.2),
                        p=0.5,
                    ),
                    alb.RandomBrightnessContrast(
                        brightness_limit=(-0.4, -0.3),
                        contrast_limit=(0.3, 0.4),
                        p=0.5,
                    ),
                ],
                p=1.0,
            ),
            *common_augs,
        ]
        if gt_type == "mask" and augment:
            return alb.Compose(augs_list, p=0.4)
        elif gt_type == "point" and augment:
            return alb.Compose(
                augs_list,
                keypoint_params=alb.KeypointParams(format="xy"),
                p=0.75,
            )
        else:
            return None

    elif module == "segmentation":
        augs_list = [
            alb.OneOf(
                [
                    alb.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=(-30, 20),
                        val_shift_limit=0,
                        always_apply=False,
                        p=0.75,
                    ),
                    alb.RGBShift(
                        r_shift_limit=20,
                        g_shift_limit=20,
                        b_shift_limit=20,
                        p=0.75,
                    ),
                ],
                p=1.0,
            ),
            *common_augs,
        ]
        if augment:
            return alb.Compose(
                augs_list,
                additional_targets={"others": "mask"},
                p=0.5,
            )
        else:
            return None

    elif module == "classification":
        stain_matrix = np.array(
            [
                [0.91633014, -0.20408072, -0.34451435],
                [0.17669817, 0.92528011, 0.33561059],
            ]
        )
        augs_list = [
            alb.OneOf(
                [
                    alb.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=0,
                        val_shift_limit=0,
                        always_apply=False,
                        p=0.85,
                    ),  # HED or HSV stain augmentation
                ],
                p=1.0,
            ),
            # Saturation augmentation (Strong Lower, Strong Higher, Mid-range)
            alb.OneOf(
                [
                    alb.HueSaturationValue(
                        hue_shift_limit=0,
                        sat_shift_limit=(-40, -30),
                        val_shift_limit=0,
                        always_apply=False,
                        p=aug_prob,
                    ),
                    alb.HueSaturationValue(
                        hue_shift_limit=0,
                        sat_shift_limit=(10, 30),
                        val_shift_limit=0,
                        always_apply=False,
                        p=aug_prob,
                    ),
                    alb.HueSaturationValue(
                        hue_shift_limit=0,
                        sat_shift_limit=(-20, 10),
                        val_shift_limit=0,
                        always_apply=False,
                        p=aug_prob,
                    ),
                ],
                p=1.0,
            ),
            # Bluring (Gaussian, motion, defocus, or zoom) or Sharpenning
            alb.OneOf(
                [
                    alb.GaussianBlur(blur_limit=(1, 3), p=aug_prob),
                    alb.Sharpen(
                        alpha=(0.1, 0.3),
                        lightness=(1.0, 1.0),
                        p=aug_prob,
                    ),
                ],
                p=1.0,
            ),
            alb.OneOf(
                [
                    alb.GaussNoise(var_limit=(50, 80), p=aug_prob),
                    alb.ISONoise(
                        color_shift=(0.05, 0.1),
                        intensity=(0.7, 0.85),
                        p=aug_prob,
                    ),
                    alb.PixelDropout(
                        dropout_prob=0.1,
                        per_channel=False,
                        drop_value=None,
                        mask_drop_value=None,
                        p=aug_prob,
                    ),
                ],
                p=1.0,
            ),
            alb.OneOf(
                [
                    alb.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=(-0.3, 0.2),
                        p=aug_prob,
                    ),
                ],
                p=1.0,
            ),
            alb.ShiftScaleRotate(
                shift_limit=0,
                scale_limit=0.2,
                rotate_limit=180,
                border_mode=cv2.BORDER_CONSTANT,
                value=(250, 250, 250),
                p=aug_prob,
            ),
            alb.Flip(p=aug_prob),
        ]
        if augment:
            return alb.Compose(augs_list, p=1.0)
        else:
            return None

    else:
        raise ValueError(f"Invalid module: {module}")
