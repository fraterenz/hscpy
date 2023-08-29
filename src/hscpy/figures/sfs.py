from typing import Set
from scipy import stats
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hscpy.figures.options import PlotOptions, SimulationOptions
from hscpy.figures import mitchell
from hscpy import get_idx_timepoint_from_age, sfs


def show_sfs_last_timepoint_plots(
    sim_options: SimulationOptions,
    plot_options: PlotOptions,
    id2plot: str = "3",
):
    sfs_all = sfs.load_sfs(
        sim_options.path2save,
        runs=sim_options.runs,
        cells=sim_options.sample,
        timepoint=1,
    )  # 1 is the last timepoint

    sfs_three = sfs_all[id2plot]
    fig, ax = plt.subplots(1, 1, figsize=plot_options.figsize)
    sfs2plot = pd.Series(sfs_three, dtype=int).value_counts()
    ax.plot(sfs2plot.index, sfs2plot, linestyle="", marker="x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("j cells")
    ax.set_ylabel("# of muts in j cells")
    ax.set_title(
        f"SFS of {id2plot} run at age {sim_options.last_timepoint_years} with {sim_options.sample} cells"
    )
    fig.show()

    sfs_all = sfs.pandafy_sfs_dict(sfs_all)

    fig, ax = plt.subplots(1, 1, figsize=plot_options.figsize)
    x = np.arange(1, sim_options.cells, dtype=int)
    y = 1.0 / x * (sim_options.neutral_rate * sim_options.cells)
    ax.plot(x, y, linestyle="--", color="black", alpha=0.5, label="1/f")
    sns.scatterplot(
        sfs_all,
        x="# of j cells",
        y="# of muts in j cells",
        hue="id",
        ax=ax,
        marker="x",
        alpha=0.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("j cells")
    ax.set_ylabel("# of muts in j cells")
    ax.set_title(
        f"SFS of {sim_options.runs} runs at age {sim_options.last_timepoint_years} with {sim_options.sample} cells"
    )
    if plot_options.save:
        plt.savefig(sim_options.path2save / f"sfs{plot_options.extension}")
    fig.show()


def show_entropy_plots(
    sim_options: SimulationOptions,
    plot_options: PlotOptions,
    ages: Set[int],
    early_variants_only: bool = True,
):
    closest_age = dict.fromkeys(ages)
    simulated = dict()
    for age in ages:
        if age < 2:
            continue
        print(f"\nloading mutational burden for age {age}")
        idx_timepoint, closest_age_ = get_idx_timepoint_from_age(
            age,
            sim_options.last_timepoint_years,
            nb_timepoints=sim_options.nb_timepoints,
        )
        closest_age[age] = closest_age_
        simulated[closest_age[age]] = dict()
        try:
            if early_variants_only:
                for idx_sim, simulation in sfs.load_sfs_entropy(
                    sim_options.path2save,
                    runs=sim_options.runs,
                    cells=sim_options.cells,
                    timepoint=idx_timepoint,
                ).items():
                    simulated[closest_age[age]][idx_sim] = simulation
            else:
                for idx_sim, simulation in sfs.load_sfs(
                    sim_options.path2save,
                    runs=sim_options.runs,
                    cells=sim_options.cells,
                    timepoint=idx_timepoint,
                ).items():
                    simulated[closest_age[age]][idx_sim] = simulation
        except (AssertionError, ValueError):
            print(
                f"skipping timepoint {idx_timepoint} with age {closest_age_} because empty sfs"
            )

    fig, ax = plt.subplots(1, 1, figsize=plot_options.figsize)
    entropies = dict()

    for age, sfs_ in simulated.items():
        if not age:  # do not consider age 0
            continue
        entropies[age] = list()
        for id, sfs_run in sfs_.items():
            entropies[age].append(stats.entropy(sfs_run))

    for age, entropy in entropies.items():
        ax.errorbar(
            age,
            np.mean(entropies[age]),
            yerr=np.std(entropies[age]),
            label=age,
            color="yellowgreen",
            marker="x",
        )
    ax.set_xlabel("age [years]")
    ax.set_ylabel("entropy")
    ax.set_title(f"variant entropy averaged over {sim_options.runs} simulations")
    if plot_options.save:
        plt.savefig(sim_options.path2save / f"entropy{plot_options.extension}")
    fig.show()


# TODO
"""
def get_ymin(min1: float, min2: float) -> float:
    my_min = min1 if min1 < min2 else min2
    return my_min - my_min * 0.2


def get_xmax(max1: int, max2: int) -> float:
    my_max = max1 if max1 > max2 else max2
    return my_max + my_max * 0.2


def plot_sfs_simulations_data(summary, simulated, cells_sampled: int, id2plot: str = "0"):
    # summary: Summary_cut.csv
    # pop size
    N = 200_000
    _f = sfs.compute_frequencies(N)

    for i, donor in enumerate(summary.donor_id.unique()):
        age = summary.loc[summary.donor_id == donor, "age"].iloc[0]

        print(f"donor {donor}")

        filtered_matrix = mitchell.filter_mutations(
            *mitchell.load_patient(
                donor,
                path2data / f"mutMatrix{donor}.csv",
                path2data / f"mutType{donor}.csv",
            )
        )

        sfs_donor = filtered_matrix.sum(axis=1).value_counts(normalize=True)

        sfs_donor.drop(index=sfs_donor[sfs_donor.index == 0].index, inplace=True)

        x_sfs = sfs_donor.index.to_numpy(dtype=int)

        y_sfs = sfs_donor.to_numpy()

        # sample size

        cells = summary.loc[summary.donor_id == donor, "cells"].unique()[0]

        correction = sfs.SamplingCorrection(N, cells)

        assert cells <= 1000

        sampled_f, y = sfs.compute_variants(
            correction, sfs.Correction.ONE_OVER_F, cells
        )

        sampled_f_squared, y_squared = sfs.compute_variants(
            correction, sfs.Correction.ONE_OVER_F_SQUARED, cells
        )

        sfs_simulations = pd.Series(
            simulated[closest_age[age]][id2plot], dtype=int
        ).value_counts()

        sfs_simulations /= sfs_simulations.max()

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

        ax.plot(
            _f[:cells],
            sampled_f,
            label=f"$1/f$ sampled",
            alpha=0.4,
            linestyle="--",
            c="black",
        )

        ax.plot(
            _f[:cells],
            sampled_f_squared / sampled_f_squared.max(),
            label=f"$1/f^2$ sampled",
            alpha=0.4,
            c="black",
        )

        ax.plot(
            x_sfs,
            sfs_donor.to_numpy(),
            label=f"{donor}",
            linestyle="",
            marker="x",
            c="blue",
            alpha=0.7,
        )

        ax.plot(
            sfs_simulations.index,
            sfs_simulations,
            linestyle="",
            marker="o",
            label="simulation",
            c="purple",
            alpha=0.7,
        )

        ax.set_yscale("log")

        ax.set_xscale("log")

        ax.set_xlabel("j cells")

        ax.set_ylabel("normalised nb of muts in j cells")

        ax.set_ylim([get_ymin(y_sfs.min(), sfs_simulations.min()), 2])

        ax.set_xlim(
            [0.8, get_xmax(x_sfs.max(), sfs_simulations.index.to_numpy().max())]
        )

        ax.legend()

        ax.set_title(f"age {age}")

        if SAVE:
            plt.savefig(f"./{donor}_sfs_{cells_sampled}cells{EXTENSION}")

        plt.show()


def load_sfs_simulations(sim_options):
    simulated = dict()

    for age in ages:
        print(f"\nloading sfs for age {age}")

        idx_timepoint, closest_age_ = get_idx_timepoint_from_age(
            age,
            sim_options.last_timepoint_years,
            nb_timepoints=sim_options.nb_timepoints,
        )

        closest_age[age] = closest_age_

        simulated[closest_age[age]] = dict()

        for idx_sim, simulation in sfs.load_sfs(
            sim_options.path2save,
            runs=sim_options.runs,
            cells=sim_options.sample,
            timepoint=idx_timepoint,
        ).items():
            simulated[closest_age[age]][idx_sim] = simulation

    return simulated
"""
