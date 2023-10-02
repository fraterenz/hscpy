from functools import reduce
from typing import Dict, NewType, Tuple
from hscpy import Measurement, dir_path_over_timepoint, load_measurement
import numpy as np
import json
import sys
from pathlib import Path
from futils import snapshot


def load_burden(
    path2save: Path, runs: int, cells: int, timepoint: int = 1
) -> Dict[str, snapshot.Histogram]:
    return load_measurement(
        path2save, runs, cells, measurement=Measurement.BURDEN, timepoint=timepoint
    )


def load_burden_timepoints(
    path2save: Path, nb_timepoints: int, cells: int, runs: int
) -> Dict[int, Dict[str, snapshot.Histogram]]:
    my_burden = dict()

    for i in range(1, nb_timepoints + 1):
        my_burden[i] = load_burden(path2save, runs=runs, cells=cells, timepoint=i)
    return my_burden


def compute_mean_variance(
    burden: snapshot.Histogram | snapshot.Distribution,
) -> Tuple[float, float]:
    cells = sum(burden.values())
    mean = sum(map(lambda entry: entry[0] * entry[1] / cells, burden.items()))
    variance = sum(
        map(lambda entry: (entry[0] - mean) ** 2 * entry[1] / cells, burden.items())
    )
    return mean, variance


def plot_burden(burden: snapshot.Histogram | snapshot.Distribution, ax, **kwargs):
    ax.bar(
        list(burden.keys()),
        list(burden.values()),
        color=kwargs["color"],
        alpha=kwargs["alpha"],
        width=1,
    )
    ymax = kwargs.get("ymax", max(burden.values()))
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
