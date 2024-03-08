from futils import snapshot
import matplotlib.pyplot as plt
import numpy as np

from hscpy.figures import AgeSims, PlotOptions


def plot_burden(
    ax,
    my_burden: snapshot.Histogram | snapshot.Distribution,
    normalise: bool,
    options: PlotOptions,
    bins: int = 20,
    fancy: bool = True,
    **kwargs,
):
    muts, counts = list(my_burden.keys()), list(my_burden.values())
    min_, max_ = min(muts), max(muts)
    hist, edges = np.histogram(
        muts,
        bins=bins,
        weights=[c / sum(counts) for c in counts] if normalise else counts,
        density=True,
    )
    bin_distance = edges[1] - edges[0]
    try:
        color = kwargs.pop("color")
    except KeyError:
        color = "blue"

    if fancy:
        ax.fill_between(
            edges[:-1] + bin_distance, hist, ls="-", alpha=0.15, color=color
        )
        ax.plot(edges[:-1] + bin_distance, hist, color=color, **kwargs)
    else:
        ax.bar(edges[:-1], hist, width=bin_distance, label=kwargs["label"])
    ax.set_xlabel("Single-cell burden")
    ax.set_ylabel("Density" if normalise else "Counts")
