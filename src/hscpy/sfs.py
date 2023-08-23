import sys
import csv
import numpy as np
import pandas as pd
from typing import Dict, List, NewType, Tuple
from scipy import stats
from pathlib import Path
from hscpy import Measurement, dir_path_over_timepoint

# Sfs with cell counts, not frequencies
Sfs = NewType("Sfs", List[int])


def load_sfs_entropy(
    path2dir: Path, runs: int, cells: int, timepoint: int = 1
) -> Dict[str, Sfs]:
    return load(path2dir, runs, Measurement.SFS_ENTROPY, cells, timepoint)


def load(
    path2dir: Path, runs: int, measurement: Measurement, cells: int, timepoint: int
) -> Dict[str, Sfs]:
    sfs = dict()
    try:
        timepoint_path = dir_path_over_timepoint(
            measurement=measurement, path2dir=path2dir, cells=cells, timepoint=timepoint
        )
    except AssertionError as e:
        e.add_note(f"cannot load SFS from {path2dir} for timepoint {timepoint}: {e}")
        raise e
    i = 0
    for i, file in enumerate(timepoint_path.iterdir(), 1):
        with open(file, "r") as csvfile:
            sfsreader = csv.reader(csvfile, delimiter=",")
            sfs[file.stem] = list()
            try:
                for row in sfsreader:
                    sfs[file.stem].extend([int(ele) for ele in row if ele])
            except csv.Error as e:
                sys.exit("file {}, line {}: {}".format(file, sfsreader.line_num, e))

    assert i == runs, f"wrong number of runs found: should be {runs} found {i}"
    assert (
        len(sfs) == runs
    ), f"wrong number of runs loaded: should be {runs} found {len(sfs)}"
    return sfs


def load_sfs(
    path2dir: Path, runs: int, cells: int, timepoint: int = 1
) -> Dict[str, Sfs]:
    """load all sfs for a specific timepoint, by default load the burden of the
    last timepoint.

    Remember that rust saves timepoints in decreasing order, hence the last
    timepoint is 1.
    """
    return load(path2dir, runs, Measurement.SFS, cells, timepoint)


def pandafy_sfs_dict(sfs_: Dict[str, Sfs]) -> pd.Series:
    """Transform a dict of sfs (key: run id, values: sfs) into a long-record
    format pandas Series"""
    sfs_all = list()
    for run_id, sfs in sfs_.items():
        sfs_df = (
            pd.Series(sfs, dtype=int)
            .value_counts()
            .reset_index()
            .rename(columns={"index": "# of j cells", "count": "# of muts in j cells"})
        )
        sfs_df["id"] = run_id
        sfs_all.append(sfs_df)
    return pd.concat(sfs_all)


from enum import StrEnum, auto


class Correction(StrEnum):
    ONE_OVER_F = auto()

    ONE_OVER_F_SQUARED = auto()


class SamplingCorrection:
    def __init__(self, pop_size: int, sample_size: int):
        self.pop_size = pop_size
        self.sample_size = sample_size
        self.correction = compute_sampling_correction(n=pop_size, s=sample_size)
        self.frequencies = compute_frequencies(pop_size)


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


def compute_variants(
    s_correction: SamplingCorrection, correction: Correction, sample_size: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    if correction == Correction.ONE_OVER_F:
        variants2correct = 1 / s_correction.frequencies
    elif correction == Correction.ONE_OVER_F_SQUARED:
        variants2correct = 1 / s_correction.frequencies**2
    else:
        raise ValueError

    assert (
        variants2correct.shape[0] == s_correction.pop_size
    ), f"{variants2correct.shape[0]}"
    corrected = s_correction.correction[:sample_size, :] @ variants2correct
    return corrected, variants2correct
