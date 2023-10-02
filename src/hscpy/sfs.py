from dataclasses import dataclass
import numpy as np
from scipy import stats
from enum import StrEnum, auto
from pathlib import Path
from typing import Dict
from futils import snapshot

from hscpy import Measurement, burden, load_measurement


def load_sfs(
    path2save: Path, runs: int, cells: int, timepoint: int = 1
) -> Dict[str, snapshot.Histogram]:
    return load_measurement(
        path2save, runs, cells, measurement=Measurement.SFS, timepoint=timepoint
    )


def load_sfs_timepoints(
    path2save: Path, nb_timepoints: int, cells: int, runs: int
) -> Dict[int, Dict[str, snapshot.Histogram]]:
    my_sfs = dict()

    for i in range(1, nb_timepoints + 1):
        my_sfs[i] = load_sfs(path2save, runs, cells, timepoint=i)
    return my_sfs


class Correction(StrEnum):
    ONE_OVER_F = auto()
    ONE_OVER_F_SQUARED = auto()


def compute_frequencies(pop_size: int) -> np.ndarray:
    return np.arange(1, pop_size + 1, step=1, dtype=int)


def compute_sampling_correction(n: int, s: int) -> np.ndarray:
    return np.array(
        [
            stats.binom(s, (v + 1) / n).pmf(np.arange(1, s + 1, step=1))
            for v in range(0, n)
        ],
        dtype=float,
    ).T


@dataclass
class CorrectedVariants:
    correction: Correction
    corrected_variants: np.ndarray
    variant2correct: np.ndarray
    frequencies: np.ndarray


def compute_variants(
    correction: Correction, pop_size: int, sample_size: int
) -> CorrectedVariants:
    """
    _f = range(0,1,length=N+1)
    nThGr_f = (1 ./ _f).^2
    nSample_f = Vector{Float64}(undef, S+1)
    for u in 1:S
        nSample_f[1+u] =
            sum(
                [ nTrue_f[1+v] * pdf(Binomial(S, v/N), u) for v=1:N ]
            )
    end
    """
    frequencies = compute_frequencies(pop_size)
    if correction == Correction.ONE_OVER_F:
        variants2correct = 1 / frequencies
    elif correction == Correction.ONE_OVER_F_SQUARED:
        variants2correct = 1 / frequencies**2
    else:
        raise ValueError

    assert variants2correct.shape[0] == pop_size, f"{variants2correct.shape[0]}"
    corrected = (
        compute_sampling_correction(pop_size, sample_size)[:sample_size, :]
        @ variants2correct
    )
    return CorrectedVariants(correction, corrected, variants2correct, frequencies)
