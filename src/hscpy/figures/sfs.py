import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from futils import snapshot

from hscpy import Donor, Measurement, burden, load_measurement, sfs
from hscpy.figures import PlotOptions, mitchell
from hscpy.figures.simulations import Simulations


def plot_sfs_avg(
    ax,
    my_sfs: Dict[str, snapshot.Histogram],
    age: int,
    options: PlotOptions,
    **kwargs,
):
    pooled = burden.pooled_burden(my_sfs)
    ax = plot_sfs(ax, pooled, normalise=True, options=options, **kwargs)
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
    ax.set_ylabel("density of variants", fontsize="xx-large")
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
    cells: int,
    normalise: bool,
    options: PlotOptions,
    **kwargs,
):
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
    my_sfs = mitchell.sfs_donor_mitchell(donor.name, path2mitchell, remove_indels)
    return plot_sfs(ax, my_sfs, normalise, options, **kwargs)


def plot_simulations_donors_sfs(
    sims: Simulations,
    options_plot: PlotOptions,
    paths2patients: Path,
    normalise: bool,
    corrected_variants: Dict[str, sfs.CorrectedVariants],
    id2show: str | None = None,
):
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
