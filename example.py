import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from monkey.config import TrainingIOConfig
from monkey.data.data_utils import centre_cross_validation_split, get_file_names
from monkey.data.dataset import InflammatoryDataset

IOconfig = TrainingIOConfig(
    dataset_dir="/home/u1910100/Documents/Monkey/patches_256"
)
file_ids = get_file_names(IOconfig)

split = centre_cross_validation_split(
    file_ids=file_ids, test_centre="D"
)


train_dataset = InflammatoryDataset(
    IOConfig=IOconfig,
    file_ids=split["train_file_ids"],
    phase="Train",
    do_augment=True,
)
val_dataset = InflammatoryDataset(
    IOConfig=IOconfig,
    file_ids=split["val_file_ids"],
    phase="test",
    do_augment=False,
)
