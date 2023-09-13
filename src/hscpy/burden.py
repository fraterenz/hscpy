from functools import reduce
from typing import Dict, NewType, Tuple
from hscpy import Measurement, dir_path_over_timepoint
import numpy as np
import json
import sys
from pathlib import Path
from futils import snapshot


def load_burden(
    path2dir: Path, runs: int, cells: int, timepoint: int = 1
) -> Dict[str, snapshot.Histogram]:
    """load all burden for a specific timepoint, by default load the burden of the
    last timepoint.

    Remember that rust saves timepoints in decreasing order, hence the last
    timepoint is 1.
    """
    burden = dict()
    try:
        timepoint_path = dir_path_over_timepoint(
            measurement=Measurement.BURDEN,
            cells=cells,
            path2dir=path2dir,
            timepoint=timepoint,
        )
    except AssertionError as e:
        e.add_note(f"cannot load burden from {path2dir} for timepoint {timepoint}: {e}")
        raise e
    i = 0
    for i, file in enumerate(timepoint_path.iterdir(), 1):
        try:
            with open(file, "r") as f:
                burden[file.stem] = snapshot.Histogram(
                    {int(x): int(y) for x, y in json.load(f).items()}
                )
        except json.JSONDecodeError as e:
            print(f"Error in opening {file} {e}")
            sys.exit(1)

    assert i == runs, f"wrong number of runs found: should be {runs} found {i}"
    assert (
        len(burden) == runs
    ), f"wrong number of runs loaded: should be {runs} found {len(burden)}"
    return burden


def compute_mean_variance(burden: snapshot.Histogram) -> Tuple[float, float]:
    cells = sum(burden.values())
    mean = sum(map(lambda entry: entry[0] * entry[1] / cells, burden.items()))
    variance = sum(
        map(lambda entry: (entry[0] - mean) ** 2 * entry[1] / cells, burden.items())
    )
    return mean, variance


def plot_burden(burden: snapshot.Histogram, ax, **kwargs):
    ax.bar(
        list(burden.keys()),
        list(burden.values()),
        color=kwargs["color"],
        alpha=kwargs["alpha"],
        width=1,
    )
    ymax = max(burden.values())
    mean, var = compute_mean_variance(burden)
    ax.axvline(
        x=mean,
        ymin=0,
        ymax=ymax,
        linestyle="--",
        c=kwargs["color"],
        label=f"{kwargs['label']} years, m={mean:.2f}, var={var:.2f}",
    )
    return ax


def average_burden(burden_dict: Dict[int, snapshot.Histogram]):
    # TODO
    # raise NotImplementedError(
    #    "I think we should pool all the simulations together and then average them, not average them directly"
    # )
    # add zeros for values which are not present in all SFS
    burden_uniformised = snapshot.Uniformise.uniformise_histograms(
        [snapshot.Histogram(burden) for burden in burden_dict.values()]
    )
    jcells = burden_uniformised.create_x_array()
    # take the Nj mutations for all simulations
    avg_burden = burden_uniformised.y
    # compute the average, not pooling
    return jcells, np.mean(avg_burden, axis=0)


def single_cell_mutations_from_burden(burden: snapshot.Histogram) -> np.ndarray:
    """From the SFS, create an array where each entry is the number of
    mutations found in a cell"""
    muts = []
    for jmuts, jcells in burden.items():
        for cell in range(0, jcells):
            muts.append(jmuts)
    return np.array(muts, dtype=int)


def pooled_burden(burden_: Dict[str, snapshot.Histogram]) -> snapshot.Distribution:
    histograms = [b for b in burden_.values()]
    return snapshot.Uniformise.pooled_distribution(histograms)
