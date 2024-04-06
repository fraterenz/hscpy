import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from pathlib import Path
from typing import Dict, List, Set
from math import ceil
from scipy import stats
from futils import snapshot

from hscpy import realisation, abc
from hscpy.figures import AgeSims, PlotOptions, ToCellFrequency


def plot_ax_sfs_predictions_data_sims(
    ax,
    donor: pd.Series,
    corrected_one_over_1_squared: realisation.CorrectedVariants,
    mitchell_sfs: snapshot.Histogram,
    plot_options: PlotOptions,
    sfs_sims_donor: List[realisation.RealisationSfs] | None = None,
    one_over_f_csv: Path | None = None,
    idx_sim2plot: int | None = None,
    mew: float | None = None,
    markersize: float | None = None,
    lw: float | None = None,
):
    # 1/f^2 sampled predictions
    normalisation_x = ToCellFrequency(sample_size=donor.cells)
    plot_sfs_correction(
        ax,
        corrected_one_over_1_squared,
        normalise=True,
        options=plot_options,
        normalise_x=normalisation_x,
        linestyle="-",
        color="black",
        alpha=0.8,
        label="growth theory",
        linewidth=lw if lw else 3,
    )

    # 1/f sampled predictions from Nate's simulations
    if one_over_f_csv:
        one_over_f = pd.read_csv(one_over_f_csv)
        one_over_f.drop(index=one_over_f[one_over_f["_f"] == 0.0].index, inplace=True)
        sfs_one_over_f = {
            cell: muts
            for cell, muts in zip(
                (one_over_f["_f"] * normalisation_x.nb_cells).tolist(),
                one_over_f["n_f"].tolist(),
            )
        }
        plot_sfs(
            ax,
            snapshot.Histogram(sfs_one_over_f),
            normalise=True,
            normalise_x=normalisation_x,
            options=plot_options,
            color="black",
            alpha=0.8,
            lw=lw if lw else 3,
            linestyle="--",
            label="constant theory",
        )

    # simulations
    if sfs_sims_donor:
        plot_sfs_avg(
            ax,
            [sfs_.sfs for sfs_ in sfs_sims_donor],
            options_plot=plot_options,
            normalise_x=normalisation_x,
            lw=lw if lw else 3,
            color="grey",
            alpha=0.6,
            label="avg",
        )
    if idx_sim2plot:
        plot_sfs(
            ax,
            [
                sfs_.sfs
                for sfs_ in sfs_sims_donor
                if sfs_.parameters.idx == idx_sim2plot
            ][0],
            normalise=True,
            normalise_x=normalisation_x,
            options=plot_options,
            color="grey",
            mew=mew if mew else 3,
            linestyle="",
            marker=".",
            markersize=markersize if markersize else 10,
            label="simulation",
        )

    # mitchell's data
    plot_sfs(
        ax,
        mitchell_sfs,
        normalise=True,
        normalise_x=normalisation_x,
        options=plot_options,
        color="#d95f0e",
        mew=mew if mew else 3,
        linestyle="",
        marker="x",
        markersize=markersize if markersize else 10,
        label=f"donor {donor.age} y.o.",
    )


def process_sfs(
    my_sfs: snapshot.Histogram, normalise: bool, log_transform: bool
) -> Dict[AgeSims, float]:
    """This modifies the sfs by removing the entry at 0 and log10 transform the
    jcells (keys) and optionally the jmuts (values) i.e. when `log_transform` is
    `True`.

    Normalise means normalise the y axis by dividing all entries by the maximal
    value found on the yaxis.
    """
    my_sfs.pop(0, 0)
    jmuts = list(my_sfs.values())
    if normalise:
        max_ = max(jmuts)
        jmuts = [ele / max_ for ele in jmuts]
    jcells = [np.log10(k) for k in my_sfs.keys()]
    if log_transform:
        jmuts = [np.log10(val) for val in jmuts]
    return {AgeSims(k): float(val) for k, val in zip(jcells, jmuts)}


def plot_sfs_avg_unormalised(
    ax,
    my_sfs: List[snapshot.Histogram],
    options_plot: PlotOptions,
    normalise_x: ToCellFrequency | None,
    **kwargs,
):
    pooled = snapshot.Uniformise.pooled_histogram(my_sfs)
    ax = plot_sfs(
        ax,
        pooled,
        normalise=False,
        normalise_x=normalise_x,
        options=options_plot,
        **kwargs,
    )
    return ax


def plot_sfs_with_avg(
    ax,
    sfs: List[realisation.RealisationSfs],
    sfs_individual: realisation.RealisationSfs,
    age: int,
    cells: int,
    marker: str,
    options: PlotOptions,
    **kwargs,
):
    """Plot individual realisation SFS and normalised avg."""
    l = str(age) + " y.o."

    plot_sfs(
        ax,
        sfs_individual.sfs,
        ls="",
        marker=marker,
        normalise=True,
        normalise_x=ToCellFrequency(cells),
        options=options,
        color=kwargs.get("color", "blue"),
    )

    plot_sfs_avg(
        ax,
        [s.sfs for s in sfs],
        normalise_x=ToCellFrequency(cells),
        options_plot=options,
        label=l,
        **kwargs,
    )
    return ax


def plot_sfs_avg(
    ax,
    my_sfs: List[snapshot.Histogram],
    options_plot: PlotOptions,
    normalise_x: ToCellFrequency | None,
    **kwargs,
):
    pooled = snapshot.Uniformise.pooled_distribution(my_sfs)
    ax = plot_sfs(
        ax,
        pooled,
        normalise=True,
        options=options_plot,
        normalise_x=normalise_x,
        **kwargs,
    )
    return ax


def plot_sfs(
    ax,
    my_sfs: snapshot.Histogram | snapshot.Distribution,
    normalise: bool,
    options: PlotOptions,
    normalise_x: ToCellFrequency | None = None,
    **kwargs,
):
    """This modifies the sfs by removing the entry at 0"""
    my_sfs.pop(0, 0)
    jmuts = list(my_sfs.values())
    jcells = list(my_sfs.keys())
    if normalise:
        max_ = max(jmuts)
        jmuts = [ele / max_ for ele in jmuts]
    if normalise_x:
        jcells = [ele / normalise_x.nb_cells for ele in jcells]
    ax.plot(jcells, jmuts, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if min(jmuts) > 0:
        ax.set_ylim([min(jmuts) / 2, 1.3 if normalise else None])
    ax.set_ylabel(
        "Variant density" if normalise else "number of variants",
    )
    ax.set_xlabel(r"Variant frequency $f$" if normalise_x else "Number of cells")
    ax.tick_params(axis="both", which="both")
    return ax


def plot_sfs_sim_with_id(
    ax,
    my_sfs: snapshot.Histogram,
    normalise: bool,
    options: PlotOptions,
    normalise_x: ToCellFrequency | None,
    **kwargs,
):
    return plot_sfs(
        ax,
        my_sfs,
        normalise=normalise,
        normalise_x=normalise_x,
        options=options,
        **kwargs,
    )


def plot_sfs_correction(
    ax,
    correction: realisation.CorrectedVariants,
    normalise: bool,
    options: PlotOptions,
    normalise_x: ToCellFrequency | None,
    **kwargs,
):
    cells = correction.corrected_variants.shape[0]
    x = correction.frequencies[:cells]
    f_sampled = correction.corrected_variants
    my_sfs = snapshot.Histogram({xx: f for xx, f in zip(x, f_sampled)})
    return plot_sfs(ax, my_sfs, normalise, options, normalise_x, **kwargs)


def plot_sfs_cdf(
    idx2show: Set[int],
    target: snapshot.Histogram,
    sfs_sims: List[realisation.RealisationSfs],
    age: AgeSims,
    donor_name: str,
    donor_cells: int,
    plot_options: PlotOptions,
    markers: List[str] = ["o", "<", "*"],
    colors: List[str] = ["yellowgreen", "cyan", "black"],
    alpha: float = 0.45,
    verbose: bool = True,
):
    fig = plt.figure(layout="constrained", figsize=(7, 4))
    subfigs = fig.subfigures(1, 2, wspace=-0.1, width_ratios=[2.4, 1])

    axes = subfigs[0].subplots(2, 1, height_ratios=[1.4, 1])
    ax3 = subfigs[1].subplots(1, 1)
    normalisation_x = ToCellFrequency(sample_size=donor_cells)
    plot_sfs(
        axes[0],
        target,
        normalise=True,
        normalise_x=normalisation_x,
        options=plot_options,
        color="#d95f0e",
        mew=3,
        linestyle="",
        marker="x",
        # markersize=10,
        label=f"data",
    )

    cdf_x, cdf_y = realisation.cdf_from_dict(target)
    axes[1].plot(cdf_x / donor_cells, cdf_y, color="#d95f0e", alpha=0.9)

    for s_id, marker, color in zip(idx2show, markers, colors):
        run = [ele for ele in sfs_sims if ele.parameters.idx == s_id][0]
        params = abc.sfs_summary_statistic_wasserstein_timepoint(
            [run], target, donor_name, age
        )
        params2plot = run.parameters.stringify({"mu", "s", "std", "idx"})
        print(f"run with params {params2plot}")

        label = (
            f"id: {s_id}, dist: {params2plot['wasserstein']:.2f}"
            if verbose
            else f"best fit"
        )
        plot_sfs(
            axes[0],
            run.sfs,
            normalise=True,
            normalise_x=normalisation_x,
            options=plot_options,
            color="grey",
            mew=3,
            linestyle="",
            marker=".",
            # markersize=1,
            label=label,
        )

        cdf_x, cdf_y = realisation.cdf_from_dict(run.sfs)
        axes[1].plot(cdf_x / donor_cells, cdf_y, color="grey", alpha=0.6)

    axes[0].set_ylabel(axes[0].get_ylabel(), size="small")
    axes[0].set_xlabel(axes[0].get_xlabel(), size="small")
    axes[1].set_ylabel("cdf", size="small")
    axes[1].set_xlabel(r"Variant frequency $f$", size="small")
    axes[1].set_xscale("log")

    tick_width = 1.1
    for ax_ in axes:
        ax_.minorticks_on()
        ax_.tick_params(which="major", width=tick_width, length=5, labelsize=14)
        ax_.tick_params(which="minor", width=tick_width, length=3, labelsize=14)
    ax3.legend(
        *axes[0].get_legend_handles_labels(),
        title=f"{age:.0f} years",
        fontsize="small",
        title_fontsize="small",
        loc=6,
        frameon=False,
        handletextpad=0.3,
    )
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines.right.set_visible(False)
    ax3.spines.left.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax3.spines.bottom.set_visible(False)
    return fig
