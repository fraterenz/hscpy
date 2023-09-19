import matplotlib.pyplot as plt
from typing import Set
from hscpy import get_idx_timepoint_from_age
from hscpy import burden
from hscpy.burden import load_burden, plot_burden
from hscpy.figures.options import PlotOptions, SimulationOptions


def show_burden_plots(
    sim_options: SimulationOptions,
    plot_options: PlotOptions,
    ages: Set[int],
    id2plot: str | None = None,
    verbosity: bool = False,
):
    closest_age = dict.fromkeys(ages)
    simulated = dict()
    for age in ages:
        print(f"\nloading mutational burden for age {age}")
        idx_timepoint, closest_age_ = get_idx_timepoint_from_age(
            age,
            sim_options.last_timepoint_years,
            sim_options.nb_timepoints,
            verbosity,
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
                try:
                    simulation.pop(0)
                except KeyError:
                    pass
                if id2plot:
                    simulation = {
                        k: val / sim_options.sample for k, val in simulation.items()
                    }

                simulated[closest_age[age]][idx_sim] = simulation
                if verbosity:
                    print(
                        f"\t total mutational burden of {sum([mut * cells for mut, cells in simulation.items()])}"
                    )
        except AssertionError:
            print(
                f"skipping timepoint {idx_timepoint} with age {closest_age_} because empty mutational burden"
            )

    fig, ax = plt.subplots(1, 1, figsize=plot_options.figsize)
    if id2plot:
        ymax = max(
            [max(burden_dict[id2plot].values()) for burden_dict in simulated.values()]
        )
    else:
        ymax = 0
    for (age, burden_dict), c in zip(
        simulated.items(),
        (
            "grey",
            "yellowgreen",
            "orange",
            "cyan",
            "blue",
            "pink",
            "purple",
            "green",
            "brown",
        ),
    ):
        if id2plot:
            plot_burden(
                burden_dict[id2plot],
                ax,
                label=f"{age:.1f}",
                color=c,
                alpha=0.5,
                ymax=ymax,
            )
            cells = max(burden_dict[id2plot].values())
            if cells > ymax:
                ymax = cells
        else:
            burdens = burden.pooled_burden(burden_dict)
            ymax = max(burdens.values())
            plot_burden(
                burdens,
                ax,
                label=f"{age:.1f}",
                color=c,
                alpha=0.5,
                ymax=ymax,
            )
    ax.legend(title="age")
    # ax.set_ylim((0, ymax))
    ax.set_xlabel("single nucleotide variant")
    ax.set_ylabel("density")
    if id2plot:
        ax.set_title("single cell mutational burden for one realisation")
    else:
        ax.set_title(
            f"single cell mutational burden averaged over {sim_options.runs} runs"
        )
    if plot_options.save:
        plt.savefig(sim_options.path2save / f"burden{plot_options.extension}")
    plt.show()
