import cv2
from tqdm import tqdm
import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import shutil

from src.utils.data import three_way_split

__all__ = ['split_in_three', 'train_epoch', 'eval_model']

def split_in_three(conf):
    """
    Perform a stratified split of the X-ray images in three subsets.
    After splitting patient IDs and ground-truths in three subsets,
    corresponding dicoms are placed in the train, val or test folder
    depending on the ground-truth value (normal or pneumonia subfolders).

    Arguments
    ---------
    conf 
        a DynaConf settings object
    """
    # read labels and drop duplicates based on patient ID
    labels = pd.read_csv(conf.LABELS_PATH)
    labels = labels.drop_duplicates(conf.ID_COLUMN)

    # split in train, validation and test sets
    train, val, test = three_way_split(labels, conf.SPLIT_SIZE, conf.TARGET_COLUMN, conf.SEED)

    # dicoms = [x.name for x in Path(conf.DICOMS_DIR).iterdir() if x.is_file()]
    for x in Path(conf.DICOMS_DIR).iterdir():
        if x.is_file() and not x.suffix:
            x.rename(x.with_suffix(".dcm"))

    for subset_name, subset in zip(conf.SPLIT_SIZE.keys(), [train, val, test]):
        # create folder for subset if not exists
        subset_dir = conf[f"{subset_name.upper()}_DIR"]
        Path(subset_dir).mkdir(parents=True, exist_ok=True)
        Path(subset_dir / conf.CONTROL_CLASS).mkdir(parents=True, exist_ok=True)
        Path(subset_dir / conf.CONDITION_CLASS).mkdir(parents=True, exist_ok=True)

        # iterate over tuples of patient Id and ground-truth for current subset
        for pid, gt in subset[[conf.ID_COLUMN, conf.TARGET_COLUMN]].itertuples(index=False):
            curr_dcm = (conf.DICOMS_DIR / str(pid)).with_suffix(".dcm")
            if gt:
                shutil.copy(curr_dcm, (subset_dir / conf.CONTROL_CLASS / str(pid)).with_suffix(".dcm"))
            else:
                shutil.copy(curr_dcm, (subset_dir / conf.CONDITION_CLASS / str(pid)).with_suffix(".dcm"))

def train_step():
    raise NotImplementedError()


def train_epoch():
    raise NotImplementedError()


def eval_step():
    raise NotImplementedError()


def eval_model():
    raise NotImplementedError()
