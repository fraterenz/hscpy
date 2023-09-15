import pytest
from hscpy import get_idx_timepoint_from_age


def test_get_idx_timepoint_from_age_simple_21():
    assert get_idx_timepoint_from_age(40, 80, 21, False) == (11, 40.0)


def test_get_idx_timepoint_from_age_simple():
    assert get_idx_timepoint_from_age(40, 80, 5, False) == (3, 40.0)


def test_get_idx_timepoint_from_age_not_matching_lower_21():
    assert get_idx_timepoint_from_age(41, 80, 21, False) == (11, 40.0)


def test_get_idx_timepoint_from_age_not_matching_21():
    assert get_idx_timepoint_from_age(43, 80, 21, False) == (10, 44.0)
