from futils.snapshot import Histogram
import pytest
from hscpy import burden


def test_mean_variance_ones():
    assert burden.compute_mean_variance(Histogram({1: 10})) == (1.0, 0.0)


def test_mean_variance_twos():
    assert burden.compute_mean_variance(Histogram({2: 10})) == (2.0, 0.0)


def test_mean_variance_symmetric():
    assert burden.compute_mean_variance(Histogram({1: 10, 2: 10})) == (
        1.5,
        0.25,
    )


def test_mean_variance():
    assert burden.compute_mean_variance(Histogram({1: 10, 3: 10})) == (
        2,
        1,
    )
