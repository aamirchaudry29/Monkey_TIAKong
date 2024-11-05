import os

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
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.image_dir = ""
        self.mask_dir = ""
        self.json_dir = ""
        self.checkpoint_save_dir = ""

        image_dir = os.path.join(self.dataset_dir, "images/")
        mask_dir = os.path.join(
            self.dataset_dir, "annotations/masks/"
        )
        json_dir = os.path.join(self.dataset_dir, "annotations/json/")

        self.set_image_dir(image_dir)
        self.set_mask_dir(mask_dir)
        self.set_json_dir(json_dir)
        self.check_dirs_exist()

    def set_image_dir(self, image_dir):
        self.image_dir = image_dir

    def set_mask_dir(self, mask_dir):
        self.mask_dir = mask_dir

    def set_json_dir(self, json_dir):
        self.json_dir = json_dir

    def set_checkpoint_save_dir(self, run_name: str):
        dir = os.path.join(self.save_dir, run_name)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        self.checkpoint_save_dir = dir

    def check_dirs_exist(self):
        for dir in [self.image_dir, self.mask_dir, self.save_dir]:
            if not os.path.exists(dir):
                print(f"{dir} does not exist!")
                raise ValueError(f"{dir} does not exist!")


class PredictionIOConfig:
    def __init__(
        self,
        wsi_dir: str,
        mask_dir: str,
        output_dir: str,
        patch_size: int = 256,
        resolution: float = 0,
        units: str = "level",
        stride: int = 256,
        threshold: float = 0.9,
        min_size: int = 96,
    ):
        self.wsi_dir = wsi_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.check_dirs_exist()
        self.patch_size = patch_size
        self.stride = stride
        self.resolution = resolution
        self.units = units
        self.threshold = threshold
        self.min_size = min_size

    def check_dirs_exist(self):
        for dir in [self.wsi_dir, self.mask_dir, self.output_dir]:
            if not os.path.exists(dir):
                print(f"{dir} does not exist!")
                raise ValueError(f"{dir} does not exist!")
