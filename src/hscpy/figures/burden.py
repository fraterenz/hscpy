from pathlib import Path
from typing import Dict, List
from futils import snapshot
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from hscpy.burden import RealisationBurden

from hscpy.figures import AgeSims, Donor, PlotOptions


def plot_simulations_burden(
    donors: List[Donor],
    burdens: Dict[AgeSims, List[RealisationBurden]],
    options_plot: PlotOptions,
):
    fig, ax = plt.subplots(1, 1, figsize=options_plot.figsize)
    raise NotImplementedError

    ax.plot(
        summary["age"],
        summary["number_mutations"],
        linestyle="",
        marker="o",
        color="purple",
        alpha=0.01,
        label="Mitchell's",
        mew=2,
    )

    ax.plot(
        sims.mean_mutations.age,
        sims.mean_mutations.number_mutations,
        "x",
        c="#00cd00",
        label="avg Mitchell's",
        mew=2,
    )

    ax.plot(
        age_simulations,
        sims.m * age_simulations + sims.c,
        linestyle="--",
        c="#00cd00",
        label=f"fit Mitchell's, m={sims.m:.1f}, c={sims.c:.1f}",
    )

    means = []

    for idx, age in enumerate(age_simulations, 1):
        # plot only the snvs
        pooled = burden.pooled_burden(sims.my_burden[idx])
        mean, _ = burden.compute_mean_variance(pooled)
        means.append(mean)
        snvs = [k for k, i in pooled.items() if i > 0.0]

        ax.plot(
            [age] * len(snvs),
            snvs,
            "o",
            c="blue",
            alpha=0.01,
            label="pooled sims" if idx == 1 else None,
        )

    ax.plot(
        age_simulations,
        means,
        "v",
        label=f"avg from {sims.sim_options.runs} sims",
        c="#00ffff",
        mew=2,
        alpha=1,
    )

    A_sims = np.vstack([age_simulations, np.ones(len(age_simulations))]).T

    m_sims, c_sims = np.linalg.lstsq(A_sims, means, rcond=None)[0]

    ax.plot(
        age_simulations,
        m_sims * age_simulations + c_sims,
        linestyle="--",
        c="#00ffff",
        label=f"fit sim's, m={m_sims:.1f}, c={c_sims:.1f}",
        alpha=1,
    )

    ax.set_xlabel("age [years]", fontsize="xx-large")
    ax.set_ylabel("number of SNVs", fontsize="xx-large")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(which="minor", length=3, width=0.8)
    leg = ax.legend(prop={"size": 13}, fancybox=False)
    for lh in leg.legend_handles:
        lh.set_alpha(0.6)
    fig.tight_layout()

    if options_plot.save:
        plt.savefig(f"burden{options_plot.extension}")

    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 4))
    sns.histplot(
        data=summary[summary.age == 0],
        x="number_mutations",
        hue="donor_id",
        kde=False,
        ax=ax,
        stat="density",
        discrete=True,
        common_norm=False,
    )

    pooled = burden.pooled_burden(sims.my_burden[sims.sim_options.nb_timepoints])
    ax.bar(
        x=list(pooled.keys()),
        height=list(pooled.values()),
        width=1,
        color="purple",
        alpha=0.3,
        edgecolor="black",
        label=f"{sims.sim_options.runs} sims",
    )

    mean = sum([k * v for k, v in pooled.items()])
    mean_cb001 = summary.loc[summary.donor_id == "CB001", "number_mutations"].mean()
    mean_cb002 = summary.loc[summary.donor_id == "CB002", "number_mutations"].mean()

    ax.set_xlabel("number of SNVs", fontsize="xx-large")
    ax.set_ylabel("density", fontsize="xx-large")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.legend(
        [
            f"CB002, mean={mean_cb002:.2f}",
            f"CB001, mean={mean_cb001:.2f}",
            f"sims, mean={mean:.2f}",
        ],
        prop={"size": 12},
        fancybox=False,
    )

    if options_plot.save:
        plt.savefig(f"burden_year0{options_plot.extension}")

    plt.show()
