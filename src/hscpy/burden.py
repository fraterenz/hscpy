from typing import List, Tuple
from hscpy import load_histogram
import numpy as np
from pathlib import Path
from futils import snapshot

from hscpy.parameters import parameters_from_path


class RealisationBurden:
    def __init__(self, path: Path) -> None:
        assert path.is_file(), f"cannot find burden file {path}"
        self.parameters = parameters_from_path(path)
        self.burden = load_histogram(path)


def compute_mean_variance(
    burden: snapshot.Histogram | snapshot.Distribution,
) -> Tuple[float, float]:
    # cells will be 1 if it's a distribution
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


def average_burden(burdens: List[snapshot.Histogram]):
    # add zeros for values which are not present in all SFS
    burden_uniformised = snapshot.Uniformise.uniformise_histograms(
        [snapshot.Histogram(burden) for burden in burdens]
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


def pooled_burden(burdens: List[snapshot.Histogram]) -> snapshot.Distribution:
    return snapshot.Uniformise.pooled_distribution(burdens)
