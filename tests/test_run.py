import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import logging
import math

from deepneumo.src.run import split_in_three
from deepneumo.src.utils import misc
from conftest import Cases


@pytest.fixture
def conf(tmp_path):
    c = {
        "LABELS_PATH": tmp_path / 'data' / 'ground_truths.csv',
        "ID_COLUMN": None,
        "TARGET_COLUMN": None,
        "SEED": 1234,
        "DICOMS_DIR": tmp_path / 'data' / 'dicoms',
        "TRAIN_DIR": tmp_path / 'data' / 'dicoms' / 'train',
        "VAL_DIR": tmp_path / 'data' / 'dicoms' / 'val',
        "TEST_DIR": tmp_path / 'data' / 'dicoms' / 'test',
        "CONTROL_CLASS": "normal",
        "CONDITION_CLASS": "pneumonia"
    }

    return c


@pytest.mark.parametrize('dim', [(100, 100), (1000, 100), (5000, 100)])
@pytest.mark.parametrize('split_size', Cases.SPLIT_SIZES)
def test_split_in_three(data_array, dim, split_size, tmp_path, conf):
    # write data_array to conf["LABELS_PATH"]
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data_array).to_csv(conf.get("LABELS_PATH"), index=False)

    dicoms_dir = tmp_path / 'data' / 'dicoms'
    dicoms_dir.mkdir(parents=True, exist_ok=True)

    def touch_file(filepath):
        filepath.with_suffix(".dcm").write_text("1")
    
    np.apply_along_axis(
        lambda k: touch_file(dicoms_dir / k.item()),
        axis=1, arr=data_array[:, [0]]
    )

    # n_files = len(list(Path(dicoms_dir).glob("*")))
    n_dicoms = misc.count_files_in_dir(Path(dicoms_dir))
    assert n_dicoms == data_array.shape[0]
    

    split_in_three(conf, split_size)

    for subset_name in split_size.keys():
        n_subset_dicoms = misc.count_files_in_dir(conf.get(f"{subset_name.upper()}_DIR"))
        assert math.floor(dim[0] * split_size[subset_name]) == n_subset_dicoms
    


