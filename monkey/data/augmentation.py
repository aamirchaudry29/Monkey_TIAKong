import albumentations as alb
import cv2
import numpy as np


def get_augmentation(
    module: str, gt_type=None, augment=True, aug_prob=0.7
):
    aug = alb.Compose(
        [
            alb.OneOf(
                [
                    alb.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=(-40, 40),
                        val_shift_limit=5,
                        always_apply=False,
                        p=0.5,
                    ),  # .8
                    alb.RGBShift(
                        r_shift_limit=30,
                        g_shift_limit=30,
                        b_shift_limit=30,
                        p=0.5,
                    ),  # .7
                ],
                p=1,
            ),
            alb.OneOf(
                [
                    alb.GaussianBlur(blur_limit=(1, 3), p=0.5),
                    alb.Sharpen(
                        alpha=(0.1, 0.3), lightness=(1.0, 1.0), p=0.5
                    ),
                    alb.ImageCompression(
                        quality_lower=30, quality_upper=80, p=0.5
                    ),
                ],
                p=0.5,
            ),
            alb.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.3, p=0.5
            ),
            alb.ShiftScaleRotate(
                shift_limit=0.01,
                scale_limit=0.01,
                rotate_limit=180,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.8,
            ),
            alb.Flip(p=0.5),
        ],
        p=1,
    )
    return aug

    # # Common augmentations
    # common_augs = [
    #     alb.OneOf(
    #         [
    #             alb.GaussianBlur(blur_limit=(3, 5), p=0.5),
    #             alb.Sharpen(
    #                 alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.5
    #             ),
    #         ],
    #         p=1.0,
    #     ),
    #     alb.ShiftScaleRotate(
    #         shift_limit=0.1,
    #         scale_limit=0.2,
    #         rotate_limit=180,
    #         border_mode=cv2.BORDER_CONSTANT,
    #         value=0,
    #         p=0.5,
    #     ),
    #     alb.Flip(p=0.5),
    # ]

    # if module == "detection":
    #     augs_list = [
    #         alb.OneOf(
    #             [
    #                 alb.HueSaturationValue(
    #                     hue_shift_limit=15,
    #                     sat_shift_limit=(-15, 15),
    #                     val_shift_limit=15,
    #                     always_apply=False,
    #                     p=0.5,
    #                 ),  # .8
    #                 alb.RGBShift(
    #                     r_shift_limit=15,
    #                     g_shift_limit=15,
    #                     b_shift_limit=15,
    #                     p=0.5,
    #                 ),  # .7
    #             ],
    #             p=1.0,
    #         ),
    #         alb.OneOf(
    #             [
    #                 alb.GaussianBlur(blur_limit=(3, 5), p=0.5),
    #                 alb.Sharpen(
    #                     alpha=(0.1, 0.3), lightness=(1.0, 1.0), p=0.5
    #                 ),
    #                 alb.ImageCompression(
    #                     quality_lower=30, quality_upper=80, p=0.5
    #                 ),
    #             ],
    #             p=0.8,
    #         ),
    #         alb.RandomBrightnessContrast(
    #             brightness_limit=0.1, contrast_limit=0.2, p=0.5
    #         ),
    #         alb.ShiftScaleRotate(
    #             shift_limit=0.01,
    #             scale_limit=0.2,
    #             rotate_limit=180,
    #             border_mode=cv2.BORDER_CONSTANT,
    #             value=0,
    #             p=0.8,
    #         ),
    #         alb.Flip(p=0.5),

    #     ]
    #     if gt_type == "mask" and augment:
    #         return alb.Compose(augs_list, p=0.4)
    #     elif gt_type == "point" and augment:
    #         return alb.Compose(
    #             augs_list,
    #             keypoint_params=alb.KeypointParams(format="xy"),
    #             p=0.75,
    #         )
    #     else:
    #         return None

    # elif module == "segmentation":
    #     augs_list = [
    #         alb.OneOf(
    #             [
    #                 alb.HueSaturationValue(
    #                     hue_shift_limit=10,
    #                     sat_shift_limit=(-30, 20),
    #                     val_shift_limit=0,
    #                     always_apply=False,
    #                     p=0.75,
    #                 ),
    #                 alb.RGBShift(
    #                     r_shift_limit=20,
    #                     g_shift_limit=20,
    #                     b_shift_limit=20,
    #                     p=0.75,
    #                 ),
    #             ],
    #             p=1.0,
    #         ),
    #         *common_augs,
    #     ]
    #     if augment:
    #         return alb.Compose(
    #             augs_list,
    #             additional_targets={"others": "mask"},
    #             p=0.5,
    #         )
    #     else:
    #         return None

    # elif module == "classification":
    #     augs_list = [
    #         alb.OneOf(
    #             [
    #                 alb.HueSaturationValue(
    #                     hue_shift_limit=15,
    #                     sat_shift_limit=0,
    #                     val_shift_limit=0,
    #                     always_apply=False,
    #                     p=0.85,
    #                 ),  # HED or HSV stain augmentation
    #             ],
    #             p=1.0,
    #         ),
    #         # Saturation augmentation (Strong Lower, Strong Higher, Mid-range)
    #         alb.OneOf(
    #             [
    #                 alb.HueSaturationValue(
    #                     hue_shift_limit=0,
    #                     sat_shift_limit=(-40, -30),
    #                     val_shift_limit=0,
    #                     always_apply=False,
    #                     p=aug_prob,
    #                 ),
    #                 alb.HueSaturationValue(
    #                     hue_shift_limit=0,
    #                     sat_shift_limit=(10, 30),
    #                     val_shift_limit=0,
    #                     always_apply=False,
    #                     p=aug_prob,
    #                 ),
    #                 alb.HueSaturationValue(
    #                     hue_shift_limit=0,
    #                     sat_shift_limit=(-20, 10),
    #                     val_shift_limit=0,
    #                     always_apply=False,
    #                     p=aug_prob,
    #                 ),
    #             ],
    #             p=1.0,
    #         ),
    #         # Bluring (Gaussian, motion, defocus, or zoom) or Sharpenning
    #         alb.OneOf(
    #             [
    #                 alb.GaussianBlur(blur_limit=(1, 3), p=aug_prob),
    #                 alb.Sharpen(
    #                     alpha=(0.1, 0.3),
    #                     lightness=(1.0, 1.0),
    #                     p=aug_prob,
    #                 ),
    #             ],
    #             p=1.0,
    #         ),
    #         alb.OneOf(
    #             [
    #                 alb.GaussNoise(var_limit=(50, 80), p=aug_prob),
    #                 alb.ISONoise(
    #                     color_shift=(0.05, 0.1),
    #                     intensity=(0.7, 0.85),
    #                     p=aug_prob,
    #                 ),
    #                 alb.PixelDropout(
    #                     dropout_prob=0.1,
    #                     per_channel=False,
    #                     drop_value=None,
    #                     mask_drop_value=None,
    #                     p=aug_prob,
    #                 ),
    #             ],
    #             p=1.0,
    #         ),
    #         alb.OneOf(
    #             [
    #                 alb.RandomBrightnessContrast(
    #                     brightness_limit=0.2,
    #                     contrast_limit=(-0.3, 0.2),
    #                     p=aug_prob,
    #                 ),
    #             ],
    #             p=1.0,
    #         ),
    #         alb.ShiftScaleRotate(
    #             shift_limit=0,
    #             scale_limit=0.2,
    #             rotate_limit=180,
    #             border_mode=cv2.BORDER_CONSTANT,
    #             value=(250, 250, 250),
    #             p=aug_prob,
    #         ),
    #         alb.Flip(p=aug_prob),
    #     ]
    #     if augment:
    #         return alb.Compose(augs_list, p=1.0)
    #     else:
    #         return None

    # else:
    #     raise ValueError(f"Invalid module: {module}")
