import sys
import csv
from typing import Dict, List, NewType
from hscpy import Measurement, dir_path_over_timepoint
import numpy as np
import pandas as pd
from pathlib import Path

# Sfs with cell counts, not frequencies
Sfs = NewType("Sfs", List[int])


def load_sfs_entropy(path2dir: Path, runs: int, timepoint: int = 1) -> Dict[str, Sfs]:
    return load(path2dir, runs, Measurement.SFS_ENTROPY, timepoint)


def load(
    path2dir: Path, runs: int, measurement: Measurement, timepoint: int = 1
) -> Dict[str, Sfs]:
    sfs = dict()
    try:
        timepoint_path = dir_path_over_timepoint(
            measurement=measurement, path2dir=path2dir, timepoint=timepoint
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


def load_sfs(path2dir: Path, runs: int, timepoint: int = 1) -> Dict[str, Sfs]:
    """load all sfs for a specific timepoint, by default load the burden of the
    last timepoint.

    Remember that rust saves timepoints in decreasing order, hence the last
    timepoint is 1.
    """
    return load(path2dir, runs, Measurement.SFS, timepoint)


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
