import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from hscpy import variant
from hscpy.figures import PlotOptions
from hscpy.figures.simulations import SimulationOptions


def show_variant_plots(
    sim_options: SimulationOptions,
    plot_options: PlotOptions,
    path2sims: Path,
    detection_threshold: float,
    id2plot: str = "3",
):
    x = np.linspace(0, sim_options.last_timepoint_years, sim_options.nb_timepoints)
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=plot_options.figsize)
    other_sims = pd.read_csv(path2sims).loc[: sim_options.last_timepoint_years, :]

    ax.set_xlabel("time [years]")
    ax.set_ylabel("avg total variant fraction")
    variant_fraction = variant.load_variant_fractions(
        sim_options.path2save,
        nb_timepoints=sim_options.nb_timepoints,
        cells=sim_options.sample,
        runs=sim_options.runs,
        subclones=sim_options.nb_subclones,
    )
    total_fraction = variant_fraction.sum(axis=-1)
    ax.errorbar(
        x,
        total_fraction.mean(axis=-1),
        yerr=total_fraction.std(axis=-1),
        fmt="o",
        alpha=0.8,
        label=f"ABM, avg of {sim_options.runs} runs",
    )
    ax.plot(other_sims.t, other_sims["Expected total variant fraction"], label="theory")
    ax.plot(
        other_sims.t,
        other_sims["Average total variant fraction"],
        linestyle="--",
        label="sims, avg of ?? runs",
    )
    ax.legend(loc="upper left")
    if plot_options.save:
        plt.savefig(sim_options.path2save / f"total_variant{plot_options.extension}")
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=plot_options.figsize)
    clones_abm = (
        np.sum(np.sum(np.where(variant_fraction > 0.0, 1, 0), axis=-1), axis=-1)
        / sim_options.runs
    )
    ax.scatter(x, clones_abm, label=f"ABM, avg {sim_options.runs} runs")
    ax.plot(
        other_sims.t, other_sims[f"Average number of existing clones"], label="sims"
    )
    # ax.set_yscale("log")
    ax.set_xlabel("time [years]")
    ax.set_ylabel("clones")
    ax.legend()
    ax.set_title("avg # of clones")
    fig.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=plot_options.figsize)
    clones_abm = (
        np.sum(
            np.sum(np.where(variant_fraction > detection_threshold, 1, 0), axis=-1),
            axis=-1,
        )
        / sim_options.runs
    )
    ax.scatter(x, clones_abm, label=f"ABM, avg {sim_options.runs} runs")
    ax.plot(
        other_sims.t,
        other_sims[f"Average number of clones above threshold 0.01"],
        label="sims",
    )
    # ax.set_yscale("log")
    ax.set_xlabel("time [years]")
    ax.set_ylabel("clones")
    ax.legend()
    ax.set_title(f"avg # of clones above frequency threshold of {detection_threshold}")
    fig.show()

    # record-format
    df = list()
    for t in range(0, sim_options.nb_timepoints):
        for r in range(sim_options.runs):
            for c in range(0, variant_fraction.shape[-1]):
                df.append((x[t], r, c, variant_fraction[t, r, c]))
    df = pd.DataFrame(
        df, columns=["time [years]", "run", "clone_id", "avg tot variant fraction"]
    )

    rl = sns.relplot(
        data=df.loc[
            df["avg tot variant fraction"] > detection_threshold,
            ["time [years]", "avg tot variant fraction"],
        ],
        x="time [years]",
        y="avg tot variant fraction",
        kind="line",
        errorbar="sd",
        aspect=2,
        height=3,
    )
    rl.fig.suptitle("tot avg fraction for detectable clones")
    rl.fig.show()

    grouped = (
        df[["run", "time [years]", "avg tot variant fraction"]]
        .groupby(["run", "time [years]"])
        .sum()
        .reset_index()
    )
    grouped.rename(
        columns={"avg tot variant fraction": "tot variant fraction"}, inplace=True
    )
    df = df.merge(
        grouped, on=["run", "time [years]"], how="left", validate="many_to_one"
    )
    df["effective fitness"] = (
        sim_options.s * df["avg tot variant fraction"] / df["tot variant fraction"]
    ).fillna(0)

    int_x = [round(x_) for x_ in x]
    clones = []
    for run in range(sim_options.runs):
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=plot_options.figsize)
        try:
            detected = df.loc[
                (df["avg tot variant fraction"] > detection_threshold)
                & (df.run == run),
                :,
            ]
            detected_clones = set(detected.clone_id.tolist())
            sns.barplot(
                detected,
                x="time [years]",
                y="avg tot variant fraction",
                hue="clone_id",
                ax=ax,
                palette="Dark2",
            )
        except ValueError:  # no detectable clone for this run
            continue
        ax.set_ylabel("variant fraction")
        ax.legend(loc="center left", title="clone id")
        ax.set_title(
            f"variant fraction of clones above frequency threshold of {detection_threshold}"
        )
        plt.show()

        pivoted = detected.pivot(
            columns="clone_id", index="time [years]", values="avg tot variant fraction"
        ).fillna(0)

        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=plot_options.figsize)
        pivoted.plot(ax=ax, color=sns.color_palette("Dark2"))
        ax.set_ylabel("variant fraction")
        ax.legend(loc="center left", title="clone id")
        ax.set_title(
            f"variant fraction of clones above frequency threshold of {detection_threshold}"
        )
        ax.set_xlim([0, sim_options.last_timepoint_years])

        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=plot_options.figsize)
        pivoted = (
            df.loc[
                (df.run == run) & (df.clone_id.isin(detected_clones)),
                ["clone_id", "time [years]", "effective fitness"],
            ]
            .pivot(columns="clone_id", index="time [years]", values="effective fitness")
            .fillna(0)
        )
        pivoted.plot(ax=ax, color=sns.color_palette("Dark2"))
        ax.set_ylabel("effective fitness")
        ax.legend(loc="center left", title="clone id")
        ax.set_title(
            f"effective fitness of clones above frequency threshold of {detection_threshold}"
        )

        plt.show
