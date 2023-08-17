import numpy as np
from pathlib import Path

from hscpy import Measurement, dir_path_over_timepoint


def load_variant_fractions(
    path2dir: Path,
    nb_timepoints: int,
    cells: int,
    timepoint: int,
    runs: int,
    subclones: int,
):
    try:
        timepoint_path = dir_path_over_timepoint(
            measurement=Measurement.VARIANT_FRACTION,
            path2dir=path2dir,
            cells=cells,
            timepoint=timepoint,
        )
    except AssertionError as e:
        e.add_note(
            f"cannot load variant fraction from {path2dir} for timepoint {timepoint}: {e}"
        )
        raise e
    data = []

    i = 0
    for i, file in enumerate(timepoint_path.iterdir(), 1):
        with open(file, "r") as csvfile:
            with open(file, "r") as f:
                # remove wild type clone

                data.extend(
                    [
                        float(ele)
                        for i, ele in enumerate(f.read().split(","))
                        if ele and i > 0
                    ]
                )

    assert i == runs, f"wrong number of runs found: should be {runs} found {i}"
    assert (
        len(data) == runs
    ), f"wrong number of runs loaded: should be {runs} found {len(data)}"

    return np.array(data, dtype=float).reshape(
        nb_timepoints, runs, subclones - 1
    )  # timepoints x RUNS x MAX_SUBCLONES - 1 (rm wildtype)
