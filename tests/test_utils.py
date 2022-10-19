import pytest
import logging

from deepneumo.src.utils import data
from conftest import Cases

@pytest.mark.parametrize('dim', [(100, 100), (1000, 100), (5000, 100)])
@pytest.mark.parametrize('split_size', Cases.SPLIT_SIZES)
def test_three_way_split(data_array, dim, split_size, target, random_state):
    df_train, df_val, df_test = data.three_way_split(data_array, split_size, None, random_state)
    assert df_train.shape[0] == split_size["train"] * dim[0]
    assert df_val.shape[0] == split_size["val"] * dim[0]
    assert df_test.shape[0] == split_size["test"] * dim[0]
    