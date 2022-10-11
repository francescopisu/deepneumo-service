import pytest
import numpy as np

from deepneumo import settings
from deepneumo.src._typings import Seed

### Define fixtures
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
def data_array(dim, rng):
    """Generates a data array of this size

    Args:
        dim (Tuple[int]): Dimension of the data array. It is passed by the test function
    """
    features = np.ones(dim)
    target = rng.integers(low=0, high=2, size=(dim[0],))
    return np.column_stack((features, target))

@pytest.fixture
def target():
    return settings.TARGET_COLUMN