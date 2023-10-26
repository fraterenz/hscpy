from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hscpy import abc


def lims(mask: pd.DataFrame, col: str) -> Tuple[float, float]:
    min_, max_ = mask[col].min(), mask[col].max()
    shift = 0.1
    return min_ - min_ * shift, max_ + max_ * shift


def plot_prior(prior: pd.Series, ax, **kwargs):
    ax = sns.histplot(prior, ax=ax, **kwargs)
    return ax


def plot_results(
    selected: pd.DataFrame, pair2plot: List[str], lim1, lim2, kwargs1, kwargs2
):
    assert len(pair2plot) == 2

    g = sns.JointGrid(ratio=2, marginal_ticks=True)
    x, y = selected[pair2plot[0]], selected[pair2plot[1]]
    ax = sns.scatterplot(
        x=x,
        y=y,
        marker="o",
        s=100,
        ax=g.ax_joint,
    )
    ax.tick_params(axis="both", which="both", labelsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize="xx-large")
    ax.set_xlabel(ax.get_xlabel(), fontsize="xx-large")

    ax = sns.histplot(x=x, fill=True, linewidth=2, ax=g.ax_marg_x, **kwargs1)
    ax.set_xlim(*lim1)
    ax.tick_params(axis="both", which="both", labelsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize="xx-large")

    ax = sns.histplot(y=y, fill=True, linewidth=2, ax=g.ax_marg_y, **kwargs2)
    ax.set_ylim(*lim2)
    ax.tick_params(axis="both", which="both", labelsize=14)
    ax.set_xlabel(ax.get_xlabel(), fontsize="xx-large")

    plt.show()


def plot_posteriors(abc_results: abc.AbcResults, abc_summary: pd.DataFrame):
    results = abc_summary.loc[
        abc_summary.idx.isin(abc_results.get_idx()), ["mu", "s", "std"]
    ].drop_duplicates()

    priors = abc_summary[["mu", "u", "s", "std"]].drop_duplicates()

    print(f"plotting {results.shape[0]} runs")
    plot_results(
        results,
        ["mu", "s"],
        [0, lims(priors, "mu")[1]],
        [0, lims(priors, "s")[1]],
        {"discrete": True},
        {"binwidth": 0.01},
    )

    plot_results(
        results,
        ["mu", "std"],
        [0, lims(priors, "mu")[1]],
        [0, lims(priors, "std")[1]],
        {"discrete": True},
        {"binwidth": 0.005},
    )

    plot_results(
        results,
        ["s", "std"],
        [0, lims(priors, "s")[1]],
        [0, lims(priors, "std")[1]],
        {"binwidth": 0.01},
        {"binwidth": 0.005},
    )

    plt.show()


def run_abc_filtering_on_clones(
    df, runs2keep: int, single_timepoint: bool, nb_clones_diff: int, minimum_runs: int
):
    idx_abc = dict()

    if single_timepoint:
        for t in df.sort_values(by="timepoint").timepoint.unique():
            print(f"{df.shape[0]} runs before filtering on nb of clones")
            # first filter on clones
            view = df[(df["clones diff"] <= nb_clones_diff) & (df.timepoint == t)]
            nb_runs_after_clone_filtering = view.shape[0]
            print(
                f"{view.shape[0]} runs after filtering on nb of clones {view['clones'].unique()}"
            )

            # then take the quantile, s.t. the same number of runs for every
            # timepoint are kept
            quantile = runs2keep / (nb_runs_after_clone_filtering)
            # run abc with wasserstein metric
            idx_abc[t] = abc.run_abc(view, quantile, minimum_runs, verbose=True)
            plot_posteriors(idx_abc[t], view)
    else:
        view = df[df["clones diff"] <= nb_clones_diff]
        quantile = runs2keep / df.shape[0]
        idx_abc = abc.run_abc(view, quantile, minimum_runs, verbose=True)
        plot_posteriors(idx_abc, view)
    return idx_abc
