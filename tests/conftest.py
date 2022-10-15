import pytest
import numpy as np
import random
import string

from deepneumo import settings
from deepneumo.src._typings import Seed

### Define fixtures

class Cases:
    SPLIT_SIZES = [
        ({'train': 0.8, 'val': 0.1, 'test': 0.1}),
        ({'train': 0.6, 'val': 0.2, 'test': 0.2})
    ]

@pytest.fixture
def rng() -> np.random.Generator:
    """Construct a new Random Generator using the seed specified in settings.

    Returns:
        numpy.random.Generator: a Random Generator based on BitGenerator(PCG64)
    """
    return np.random.default_rng(settings.SEED)

@pytest.fixture
def random_state() -> Seed:
    """Construct a new Random Generator using the seed specified in settings.

    Returns:
        numpy.random.Generator: a Random Generator based on BitGenerator(PCG64)
    """
    return np.random.RandomState(seed=settings.SEED)

@pytest.fixture
def alphanumeric_strings(dim, rng):
    keys = rng.choice(list(string.ascii_letters) + list(string.digits), size=(dim[0], 10))
    keys = np.apply_along_axis(
        lambda v: ''.join(v), 
        axis = 1, arr=keys)
    return keys

@pytest.fixture
def data_array(dim, rng, alphanumeric_strings):
    """Generates a data array of this size

    Args:
        dim (Tuple[int]): Dimension of the data array. It is passed by the test function
    """
    # no need to call a fixture. Just specify it as an argument
    # and it will return whatever it should return
    features = alphanumeric_strings
    target = rng.integers(low=0, high=2, size=(dim[0],))
    return np.column_stack((features, target))

@pytest.fixture
def target():
    return settings.TARGET_COLUMN