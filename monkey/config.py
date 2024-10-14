import os
from typing import Optional

# Change this
DEFAULT_DATA_DIR = "/home/u1910100/Documents/Monkey/patches_256"


class TrainingIOConfig:
    def __init__(
        self,
        dataset_dir: str = DEFAULT_DATA_DIR,
        save_dir: str = "./",
    ):
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.image_dir = ""
        self.mask_dir = ""
        self.set_image_dir()
        self.set_mask_dir()
        self.check_dirs_exist()

    def set_image_dir(self):
        image_dir = os.path.join(self.dataset_dir, "images/")
        self.image_dir = image_dir

    def set_mask_dir(self):
        mask_dir = os.path.join(
            self.dataset_dir, "annotations/masks/"
        )
        self.mask_dir = mask_dir

    def check_dirs_exist(self):
        for dir in [self.image_dir, self.mask_dir, self.save_dir]:
            if not os.path.exists(dir):
                print(f"{dir} does not exist!")
                raise ValueError(f"{dir} does not exist!")
