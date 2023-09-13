from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple
import numpy as np
from enum import StrEnum, auto


@dataclass
class Donor:
    age: int
    closest_age: int
    cells: int
    name: str
    id_timepoint: int


class Measurement(StrEnum):
    SFS = auto()
    BURDEN = auto()
    SFS_ENTROPY = auto()
    VARIANT_FRACTION = auto()


def get_idx_timepoint_from_age(
    age: int, years: int, nb_timepoints: int, verbosity: bool
) -> Tuple[int, int]:
    """Find the idx of the timepoint associated to `age`.

    Rust saves timepoints in the reverse order, that is idx of 1 corresponds
    to the older timepoint (greater age).
    """
    try:
        age = round(age)
    except TypeError:
        print(f"arg `age` must be int found {type(age)} instead")
        sys.exit(1)

    timepoints = list(np.linspace(1, years + 1, nb_timepoints))[::-1]

    try:
        found = timepoints.index(age) + 1
        closest_age = age
    except ValueError:
        closest_age = min(timepoints, key=lambda x: abs(x - age))
        found = timepoints.index(closest_age) + 1
        if verbosity:
            print(
                f"age {age} cannot be mapped, found mapping of timepoint {found} for the closest age of {closest_age}"
            )
    return found, closest_age


def dir_path_over_timepoint(
    measurement: Measurement, path2dir: Path, cells: int, timepoint: int
) -> Path:
    """Return a Path to the directory storing the `measurement` for a specific
    `timepoint`
    """
    path2burden = path2dir / f"{cells}cells" / str(measurement)
    all_dirs = [x for x in path2burden.iterdir() if x.is_dir()]
    # rust saves the last timepoint as 1
    paths = [x for x in all_dirs if int(x.stem) == timepoint]
    assert len(paths) > 0, "found empty dir"
    return paths[0]
