import random
import matplotlib.pyplot as plt
from matplotlib import ticker
from pathlib import Path
from typing import Dict, List, Set

from futils import snapshot
from scipy import stats
from hscpy import sfs

from hscpy.figures import AgeSims, Donor, PlotOptions


def plot_sfs_avg_unormalised(
    ax,
    my_sfs: List[snapshot.Histogram],
    options_plot: PlotOptions,
    **kwargs,
):
    pooled = snapshot.Uniformise.pooled_histogram(my_sfs)
    ax = plot_sfs(ax, pooled, normalise=False, options=options_plot, **kwargs)
    return ax


def plot_sfs_avg(
    ax,
    my_sfs: List[snapshot.Histogram],
    options_plot: PlotOptions,
    **kwargs,
):
    pooled = snapshot.Uniformise.pooled_distribution(my_sfs)
    ax = plot_sfs(ax, pooled, normalise=True, options=options_plot, **kwargs)
    return ax


def plot_sfs(
    ax,
    my_sfs: snapshot.Histogram | snapshot.Distribution,
    normalise: bool,
    options: PlotOptions,
    **kwargs,
):
    """This modifies the sfs by removing the entry at 0"""
    my_sfs.pop(0, 0)
    jmuts = list(my_sfs.values())
    if normalise:
        max_ = max(jmuts)
        jmuts = [ele / max_ for ele in jmuts]
    jcells = list(my_sfs.keys())
    ax.plot(jcells, jmuts, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(
        "density of variants" if normalise else "number of variants",
        fontsize="xx-large",
    )
    ax.set_xlabel("number of cells", fontsize="xx-large")
    ax.tick_params(axis="both", which="both", labelsize=14)
    return ax


def plot_sfs_sim_with_id(
    ax,
    my_sfs: snapshot.Histogram,
    normalise: bool,
    options: PlotOptions,
    **kwargs,
):
    return plot_sfs(ax, my_sfs, normalise=normalise, options=options, **kwargs)


def plot_sfs_correction(
    ax,
    correction: sfs.CorrectedVariants,
    normalise: bool,
    options: PlotOptions,
    **kwargs,
):
    cells = correction.corrected_variants.shape[0]
    x = correction.frequencies[:cells]
    f_sampled = correction.corrected_variants
    my_sfs = snapshot.Histogram({xx: f for xx, f in zip(x, f_sampled)})
    return plot_sfs(ax, my_sfs, normalise, options, **kwargs)


def plot_sfs_patient(
    ax,
    donor: Donor,
    path2mitchell: Path,
    remove_indels: bool,
    normalise: bool,
    options: PlotOptions,
    **kwargs,
):
    raise NotImplementedError
    my_sfs = mitchell.sfs_donor_mitchell(donor.name, path2mitchell, remove_indels)
    return plot_sfs(ax, my_sfs, normalise, options, **kwargs)


def plot_simulations_donors_sfs(
    donors: List[Donor],
    burdens: Dict[AgeSims, List[sfs.RealisationSfs]],
    options_plot: PlotOptions,
    normalise: bool,
    corrected_variants: Dict[str, sfs.CorrectedVariants],
    id2show: str | None = None,
):
    raise NotImplementedError
    idx2show = (
        id2show if id2show else random.sample(list(sims.my_sfs[3].keys()), k=1)[0]
    )
    for donor in sims.donors:
        fig, ax = plt.subplots(1, 1, figsize=options_plot.figsize, layout="tight")

        ax = plot_sfs_sim_with_id(
            ax,
            sims.my_sfs[donor.id_timepoint][idx2show],
            normalise=True,
            options=options_plot,
            color="yellowgreen",
            marker="d",
            linestyle="",
            alpha=0.6,
            label=f"1 run with id {idx2show}",
        )

        ax = plot_sfs_avg(
            ax,
            sims.my_sfs[donor.id_timepoint],
            age=0,
            options=options_plot,
            color="blue",
            linestyle="-",
            label=f"avg of {sims.sim_options.runs} runs",
            alpha=0.6,
        )

        ax = plot_sfs_correction(
            ax,
            corrected_variants[donor.name],
            sims.sim_options.sample,
            normalise=True,
            options=options_plot,
            color="grey",
            label=r"$1/f^2$ sampled",
            lw=2,
        )

        ax = plot_sfs_patient(
            ax,
            donor,
            paths2patients,
            remove_indels=False,
            normalise=True,
            options=options_plot,
            color="purple",
            label=f"{donor.name}, age: {donor.age:.0f}",
            marker="x",
            linestyle="",
            mew=2,
        )
        ax.legend(prop={"size": 13}, fancybox=False)
        plt.tight_layout()
        if options_plot.save:
            plt.savefig(
                f"SFS_{sims.sim_options.sample}sample_{sims.sim_options.cells}cells_{idx2show}run_{donor.name}{options_plot.extension}",
                transparent=True,
            )
        plt.show()


def plot_sfs_cdf(
    idx2show: Set[int],
    target: snapshot.Histogram,
    sfs_sims: List[sfs.RealisationSfs],
    age: float,
    markers: List[str] = ["o", "<", "*"],
    colors: List[str] = ["yellowgreen", "cyan", "black"],
    alpha: float = 0.45,
    verbose: bool = True,
):
    fig = plt.figure(layout="constrained", figsize=(7, 4))
    subfigs = fig.subfigures(1, 2, wspace=-0.1, width_ratios=[2.4, 1])

    axes = subfigs[0].subplots(2, 1, height_ratios=[1.4, 1])
    ax3 = subfigs[1].subplots(1, 1)

    target_processed = sfs.process_sfs(target, normalise=False, log_transform=True)
    u_values, u_weights = list(target_processed.keys()), list(target_processed.values())

    axes[0].plot(
        [10**ele for ele in target_processed.keys()],
        [10**ele for ele in target_processed.values()],
        marker="x",
        linestyle="",
        color="purple",
        label=f"Mitchell",
        mew=2,
    )
    cdf_x, cdf_y = sfs.cdf_from_dict(target_processed)
    axes[1].plot([10**ele for ele in cdf_x], cdf_y, color="purple", label="Mitchell")
    axes[1].set_xscale("log")

    for s_id, marker, color in zip(idx2show, markers, colors):
        run = [ele for ele in sfs_sims if ele.parameters.idx == s_id][0]
        params2plot = run.parameters.stringify({"mu", "s", "std", "idx"})
        print(f"run with params {params2plot}")

        sim = sfs.process_sfs(run.sfs, normalise=False, log_transform=True)
        v_values, v_weights = list(sim.keys()), list(sim.values())
        wasserstein_scipy = stats.wasserstein_distance(
            u_values, v_values, u_weights, v_weights
        )
        label = (
            f"id: {s_id}, dist: {wasserstein_scipy:.2f}" if verbose else f"simulation"
        )

        axes[0].loglog(
            [10**ele for ele in sim.keys()],
            [10**ele for ele in sim.values()],
            marker=marker,
            linestyle="",
            mew=1,
            alpha=alpha,
            color=color,
            label=label,
        )

        cdf_x, cdf_y = sfs.cdf_from_dict(sim)

        axes[1].plot(
            [10**ele for ele in cdf_x],
            cdf_y,
            alpha=alpha,
            color=color,
            linestyle="--",
            label=label,
        )
    axes[0].set_ylabel("nb of mutants", size="x-large")
    axes[1].set_ylabel("cdf", size="x-large")
    axes[1].set_xlabel("nb of cells", size="x-large")

    tick_width = 1.1
    for ax_ in axes:
        ax_.minorticks_on()
        ax_.tick_params(which="major", width=tick_width, length=5, labelsize=14)
        ax_.tick_params(which="minor", width=tick_width, length=3, labelsize=14)
    ax3.legend(*axes[0].get_legend_handles_labels(), title=f"{age:.0f} years", fontsize="large", title_fontsize="x-large", loc=6, frameon=False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines.right.set_visible(False)
    ax3.spines.left.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax3.spines.bottom.set_visible(False)
    return fig
