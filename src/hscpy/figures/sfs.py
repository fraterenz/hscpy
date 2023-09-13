from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set
from futils.snapshot import Distribution
from scipy import stats
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hscpy.figures.options import PlotOptions, SimulationOptions
from hscpy.figures import mitchell
from hscpy import get_idx_timepoint_from_age, sfs


@dataclass
class Donor:
    age: int
    closest_age: int
    cells: int
    name: str
    id_timepoint: int


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
    verbosity: bool = False,
):
    closest_age = dict.fromkeys(ages)
    simulated = dict()
    for age in ages:
        if age < 2:
            continue
        idx_timepoint, closest_age_ = get_idx_timepoint_from_age(
            age,
            sim_options.last_timepoint_years,
            nb_timepoints=sim_options.nb_timepoints,
            verbosity=verbosity,
        )
        closest_age[age] = closest_age_
        simulated[closest_age[age]] = dict()
        try:
            if early_variants_only:
                for idx_sim, simulation in sfs.load_sfs_entropy(
                    sim_options.path2save,
                    runs=sim_options.runs,
                    cells=sim_options.sample,
                    timepoint=idx_timepoint,
                ).items():
                    simulated[closest_age[age]][idx_sim] = simulation
            else:
                for idx_sim, simulation in sfs.load_sfs(
                    sim_options.path2save,
                    runs=sim_options.runs,
                    cells=sim_options.sample,
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


def get_ymin(min1: float, min2: float) -> float:
    my_min = min1 if min1 < min2 else min2
    return my_min - my_min * 0.2


def get_xmax(max1: int, max2: int) -> float:
    my_max = max1 if max1 > max2 else max2
    return my_max + my_max * 0.2


def plot_sfs_simulations_data(
    simulated: Dict[int, Dict[str, sfs.Sfs]],
    corrected_variants: Dict[str, sfs.CorrectedVariants],
    pop_size: int,
    sample_size: int,
    donors: List[Donor],
    options: PlotOptions,
    path2mitchell: Path,
    id2plot: str | None = None,
    remove_indels: bool = False,
):
    for donor in donors:
        if remove_indels:
            filtered_matrix = mitchell.filter_mutations(
                *mitchell.load_patient(
                    donor.name,
                    path2mitchell / f"mutMatrix{donor.name}.csv",
                    path2mitchell / f"mutType{donor.name}.csv",
                )
            )
        else:
            filtered_matrix = mitchell.load_patient(
                donor.name,
                path2mitchell / f"mutMatrix{donor.name}.csv",
                path2mitchell / f"mutType{donor.name}.csv",
            )[0]

        sfs_donor = filtered_matrix.sum(axis=1).value_counts(normalize=True)
        sfs_donor.drop(index=sfs_donor[sfs_donor.index == 0].index, inplace=True)
        x_sfs = sfs_donor.index.to_numpy(dtype=int)

        assert donor.cells <= 1000
        correction = corrected_variants[donor.name]
        sampled_f, y = (
            correction.corrected_variants,
            correction.variant2correct,
        )

        # if id2plot is specified, plot the sfs of the simulated run
        if id2plot:
            sfs_simulations = pd.Series(
                simulated[donor.closest_age][id2plot], dtype=int
            ).value_counts(normalize=True)
            sfs_simulations = Distribution(sfs_simulations.to_dict())
            # create a np array just to have the same type for jmuts
            jcells, jmuts = list(sfs_simulations.keys()), np.array(
                list(sfs_simulations.values()), dtype=float
            )
        else:  # average over all simulations otherwise
            sfs_simulations = sfs.pooled_sfs(simulated[donor.closest_age])
            jcells, jmuts = list(sfs_simulations.keys()), np.array(
                list(sfs_simulations.values()), dtype=float
            )
            # normalise the averages
            jmuts /= jmuts.max()

        fig, ax = plt.subplots(1, 1, figsize=options.figsize)
        ax.plot(
            corrected_variants[donor.name].frequencies[: donor.cells],
            sampled_f / sampled_f.max(),
            label="$1/f^2$ sampled"
            if correction.correction == sfs.Correction.ONE_OVER_F_SQUARED
            else "$1/f$ sampled",
            alpha=0.4,
            linestyle="--",
            c="black",
        )

        ax.plot(
            x_sfs,
            sfs_donor.to_numpy(),
            label=f"{donor.name}",
            linestyle="",
            marker="x",
            c="blue",
            alpha=0.7,
        )

        ax.plot(
            jcells,
            jmuts,
            linestyle="",
            marker="o",
            label="simulation"
            if id2plot
            else f"avg over {len(simulated[donor.closest_age])} simulations",
            c="purple",
            alpha=0.7,
        )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("j cells")
        ax.set_ylabel("normalised nb of muts in j cells")
        # ax.set_ylim([get_ymin(y_sfs.min(), min(sfs_simulations.values())), 2])
        ax.set_xlim([0.8, get_xmax(x_sfs.max(), max(sfs_simulations.keys()))])
        ax.legend()
        ax.set_title(f"age {donor.age}")
        if options.save:
            plt.savefig(
                f"./{donor.name}_sfs_{donor.cells}donorcells_{sample_size}cells{options.extension}"
            )
        plt.show()


def donors_from_mitchell(
    mitchell_data: pd.DataFrame,
    sim_options: SimulationOptions,
    verbosity: bool = False,
) -> List[Donor]:
    donors = list()
    for row in mitchell_data[["donor_id", "age", "cells"]].drop_duplicates().iterrows():
        donor_id, age, cells = row[1].donor_id, row[1].age, row[1].cells
        idx_timepoint, closest_age = get_idx_timepoint_from_age(
            age,
            sim_options.last_timepoint_years,
            nb_timepoints=sim_options.nb_timepoints,
            verbosity=verbosity,
        )
        if verbosity:
            print(
                f"\ncreating donor {donor_id} with age {age} and closest age {closest_age}"
            )
        donors.append(
            Donor(
                age=age,
                name=donor_id,
                cells=cells,
                closest_age=closest_age,
                id_timepoint=idx_timepoint,
            )
        )

    return donors


def load_sfs_simulations(
    donors: List[Donor], sim_options: SimulationOptions
) -> Dict[int, Dict[str, sfs.Sfs]]:
    simulated = dict()

    for donor in donors:
        print(f"\nloading sfs for donor {donor.name} with age {donor.age}")

        simulated[donor.closest_age] = dict()

        for idx_sim, simulation in sfs.load_sfs(
            sim_options.path2save,
            runs=sim_options.runs,
            cells=sim_options.sample,
            timepoint=donor.id_timepoint,
        ).items():
            simulated[donor.closest_age][idx_sim] = simulation

    return simulated
