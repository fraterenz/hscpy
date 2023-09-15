from dataclasses import dataclass
import sys
import csv
import numpy as np
import pandas as pd
from enum import StrEnum, auto
from typing import Dict, List, NewType, Tuple
from scipy import stats
from pathlib import Path
from hscpy import Measurement, dir_path_over_timepoint
from futils import snapshot

# Sfs with cell counts, not frequencies
Sfs = NewType("Sfs", List[int])


def load_sfs_entropy(
    path2dir: Path, runs: int, cells: int, timepoint: int = 1
) -> Dict[str, Sfs]:
    return load(path2dir, runs, Measurement.SFS_ENTROPY, cells, timepoint)


def load(
    path2dir: Path,
    runs: int,
    measurement: Measurement,
    cells: int,
    timepoint: int,
    verbosity: bool = False,
) -> Dict[str, Sfs]:
    sfs = dict()
    try:
        timepoint_path = dir_path_over_timepoint(
            measurement=measurement, path2dir=path2dir, cells=cells, timepoint=timepoint
        )
    except AssertionError as e:
        e.add_note(
            f"cannot load {measurement} from {path2dir} for timepoint {timepoint}: {e}"
        )
        raise e
    i = 0
    if verbosity:
        print(f"path: {timepoint_path}")
    for i, file in enumerate(timepoint_path.iterdir(), 1):
        if verbosity:
            print(f"loading from {file}")
        with open(file, "r") as csvfile:
            sfsreader = csv.reader(csvfile, delimiter=",")
            sfs[file.stem] = list()
            try:
                for row in sfsreader:
                    for ele in row:
                        try:
                            intele = int(ele)
                        except ValueError:
                            continue
                        sfs[file.stem].append(intele)
            except csv.Error as e:
                sys.exit("file {}, line {}: {}".format(file, sfsreader.line_num, e))

    assert i == runs, f"wrong number of runs found: should be {runs} found {i}"
    assert (
        len(sfs) == runs
    ), f"wrong number of runs loaded: should be {runs} found {len(sfs)}"
    return sfs


def pooled_sfs(sfs_: Dict[str, Sfs]) -> snapshot.Distribution:
    histograms = [
        snapshot.Histogram(pd.Series(sfs, dtype=int).value_counts().to_dict())
        for sfs in sfs_.values()
    ]
    return snapshot.Uniformise.pooled_distribution(histograms)


def load_sfs(
    path2dir: Path,
    runs: int,
    cells: int,
    timepoint: int = 1,
    verbosity: bool = False,
) -> Dict[str, Sfs]:
    """load all sfs for a specific timepoint, by default load the burden of the
    last timepoint.

    Remember that rust saves timepoints in decreasing order, hence the last
    timepoint is 1.
    """
    return load(path2dir, runs, Measurement.SFS, cells, timepoint, verbosity)


def pandafy_sfs_dict(sfs_: Dict[str, Sfs]) -> pd.DataFrame:
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
