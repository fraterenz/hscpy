import numpy as np
from scipy import stats
from enum import StrEnum, auto
from pathlib import Path
from dataclasses import dataclass
from futils import snapshot
from hscpy.parameters import parameters_from_path
from hscpy import load_histogram


class RealisationSfs:
    def __init__(self, path: Path) -> None:
        assert path.is_file(), f"cannot find SFS file {path}"
        self.parameters = parameters_from_path(path)
        self.sfs = load_histogram(path)


def process_sfs(
    my_sfs: snapshot.Histogram, normalise: bool, log_transform: bool
) -> snapshot.Histogram:
    """This modifies the sfs by removing the entry at 0.
    Normalise means normalise the y axis by dividing all entries by the maximal
    value found on the yaxis.
    """
    my_sfs.pop(0, 0)
    jmuts = list(my_sfs.values())
    if normalise:
        max_ = max(jmuts)
        jmuts = [ele / max_ for ele in jmuts]
    jcells = [np.log(k) for k in my_sfs.keys()]
    if log_transform:
        jmuts = [np.log(val) for val in jmuts]
    return snapshot.Histogram({k: val for k, val in zip(jcells, jmuts)})


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
