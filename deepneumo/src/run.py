import cv2
from tqdm import tqdm
import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import shutil

from .utils.data import three_way_split

__all__ = ['split_in_three', 'train_epoch', 'eval_model']

def split_in_three(conf, split_size):
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
    if not conf.get("LABELS_PATH"):
        raise ValueError("A path to a ground-truth file must be specified.")
    
    if not isinstance(conf.get("LABELS_PATH"), Path):
        if isinstance(conf.get("LABELS_PATH"), str):
            conf["LABELS_PATH"] = Path(conf.get("LABELS_PATH"))
        else:
            raise ValueError("Path to ground-truth file isn't either string or PosixPath")
    
    labels = pd.read_csv(conf.get("LABELS_PATH"))
    
    id_col = conf.get("ID_COLUMN")
    if not id_col:
        id_col = labels.keys()[0]
    else:
        if id_col not in labels:
            #id_col = labels.keys()[0]
            raise ValueError(f'{id_col} is not a column in the dataframe.\n')
    
    target_col = conf.get("TARGET_COLUMN")
    if not target_col or target_col not in labels:
        target_col = labels.keys()[-1]

    labels = labels.drop_duplicates(id_col)

    # split in train, validation and test sets
    train, val, test = three_way_split(labels, split_size, conf.get("TARGET_COLUMN"), conf.get("SEED"))

    # dicoms = [x.name for x in Path(conf.get("DICOMS_DIR").iterdir() if x.is_file()]
    for x in Path(conf.get("DICOMS_DIR")).iterdir():
        if x.is_file() and not x.suffix:
            x.rename(x.with_suffix(".dcm"))

    for subset_name, subset in zip(split_size.keys(), [train, val, test]):
        # create folder for subset if not exists
        subset_dir = conf[f"{subset_name.upper()}_DIR"]
        Path(subset_dir).mkdir(parents=True, exist_ok=True)
        Path(subset_dir / conf.get("CONTROL_CLASS")).mkdir(parents=True, exist_ok=True)
        Path(subset_dir / conf.get("CONDITION_CLASS")).mkdir(parents=True, exist_ok=True)

        # iterate over tuples of patient Id and ground-truth for current subset
        for pid, gt in subset[[id_col, target_col]].itertuples(index=False):
            curr_dcm = (conf.get("DICOMS_DIR") / str(pid)).with_suffix(".dcm")
            if gt:
                shutil.copy(curr_dcm, (subset_dir / conf.get("CONTROL_CLASS") / str(pid)).with_suffix(".dcm"))
            else:
                shutil.copy(curr_dcm, (subset_dir / conf.get("CONDITION_CLASS") / str(pid)).with_suffix(".dcm"))

def train_step():
    raise NotImplementedError()


def train_epoch():
    raise NotImplementedError()


def eval_step():
    raise NotImplementedError()


def eval_model():
    raise NotImplementedError()
