import matplotlib.pyplot as plt
from typing import Set
from hscpy import get_idx_timepoint_from_age
from hscpy.burden import load_burden, plot_burden
from hscpy.figures.options import PlotOptions, SimulationOptions


def show_burden_plots(
    sim_options: SimulationOptions,
    plot_options: PlotOptions,
    ages: Set[int],
    id2plot: str = "3",
):
    closest_age = dict.fromkeys(ages)
    simulated = dict()
    for age in ages:
        print(f"\nloading mutational burden for age {age}")
        idx_timepoint, closest_age_ = get_idx_timepoint_from_age(
            age, sim_options.last_timepoint_years, sim_options.nb_timepoints
        )
        closest_age[age] = closest_age_
        simulated[closest_age[age]] = dict()
        try:
            for idx_sim, simulation in load_burden(
                sim_options.path2save,
                runs=sim_options.runs,
                cells=sim_options.sample,
                timepoint=idx_timepoint,
            ).items():
                simulated[closest_age[age]][idx_sim] = simulation
                print(
                    f"\t total mutational burden of {sum([mut * cells for mut, cells in simulation.items()])} and {sum(simulation.values())} cells"
                )
        except AssertionError:
            print(
                f"skipping timepoint {idx_timepoint} with age {closest_age_} because empty mutational burden"
            )

    id2plot = "3"
    fig, ax = plt.subplots(1, 1, figsize=plot_options.figsize)
    for (age, sfs_dict), c in zip(
        simulated.items(),
        ("red", "grey", "black", "cyan", "yellowgreen", "blue", "pink"),
    ):
        plot_burden(sfs_dict[id2plot], ax, label=age, color=c, alpha=0.8)
    ax.legend(title="age")
    ax.set_xlabel("single nucleotide variant")
    ax.set_ylabel("cell count")
    ax.set_title("single cell mutational burden")
    if plot_options.save:
        plt.savefig(sim_options.path2save / f"burden{plot_options.extension}")
    plt.show()
