import pytest

from bikes_data import make_rentals
from prepare import split_frame


@pytest.fixture(scope="session")
def frame():
    return make_rentals()


@pytest.fixture(scope="session")
def splits(frame):
    return split_frame(frame)
