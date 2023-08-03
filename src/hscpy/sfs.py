from typing import Dict
import numpy as np
import json
import sys
from pathlib import Path
from futils import snapshot


def load_sfs(path2dir: Path, neutral: bool, runs: int, timepoint: int = 1):
    """load all sfs for a specific timepoint, by default load the sfs of the
    last timepoint.

    Remember that rust saves timepoints in decreasing order, hence the last
    timepoint is 1.
    """
    sfs = dict()
    path2sfs = path2dir / "sfs_neutral" if neutral else path2dir / "sfs"
    all_dirs = [x for x in path2sfs.iterdir() if x.is_dir()]
    # assert (
    #     len(all_dirs) == NB_TIMEPOINTS
    # ), f"Wrong nb of timepoints saved: should be {NB_TIMEPOINTS} found {len(all_dirs)}"
    # last_timepoint = min([int(x.stem) for x in all_dirs])
    last_timepoint_path = [x for x in all_dirs if int(x.stem) == timepoint][0]
    i = 0
    for i, file in enumerate(last_timepoint_path.iterdir(), 1):
        try:
            with open(file, "r") as f:
                sfs[file.stem] = {int(x): int(y) for x, y in json.load(f).items()}
        except json.JSONDecodeError as e:
            print(f"Error in opening {file} {e}")
            sys.exit(1)

    assert i == runs, f"wrong number of runs found: should be {runs} found {i}"
    assert (
        len(sfs) == runs
    ), f"wrong number of runs loaded: should be {runs} found {len(sfs)}"
    return sfs


def plot_sfs(sfs_: Dict[int, int], ax, k: str):
    ax.plot(
        list(sfs_.keys()),
        list(sfs_.values()),
        alpha=0.45,
        marker="v",
        linestyle="",
        label=k,
    )
    return ax


def average_sfs(sfs_: Dict[int, Dict[int, int]]):
    # add zeros for values which are not present in all SFS
    sfs_uniformised = snapshot.Uniformise.uniformise_histograms([snapshot.Histogram(sfs) for sfs in sfs_.values()])
    jcells = sfs_uniformised.create_x_array()
    # take the Nj mutations for all simulations
    avg_sfs = sfs_uniformised.y
    # compute the average, not pooling
    return jcells, np.mean(avg_sfs, axis=0)


