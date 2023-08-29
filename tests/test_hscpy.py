import pytest
from hscpy import get_idx_timepoint_from_age


def test_get_idx_timepoint_from_age_simple():
    assert get_idx_timepoint_from_age(41, 81, 4) == (3, 28.0)


def test_get_idx_timepoint_from_age_float():
    assert get_idx_timepoint_from_age(41, 81, 19) == (10, 41.5)


def test_get_idx_timepoint_from_age():
    assert get_idx_timepoint_from_age(48, 81, 19) == (9, 46)
