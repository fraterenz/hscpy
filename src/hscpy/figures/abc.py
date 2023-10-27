from typing import List, Tuple
from futils import snapshot
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
    selected: pd.DataFrame,
    pair2plot: List[str],
    lim1,
    lim2,
    kwargs1,
    kwargs2,
):
    mapping = {"mu": r"$\mu$", "std": r"$\sigma$"}
    assert len(pair2plot) == 2
    tick_width = 1.1

    g = sns.JointGrid(ratio=2, marginal_ticks=True)
    x, y = selected[pair2plot[0]], selected[pair2plot[1]]
    ax = sns.scatterplot(
        x=x,
        y=y,
        marker="o",
        s=100,
        ax=g.ax_joint,
    )
    label = ax.get_ylabel()
    ax.set_ylabel(mapping.get(label, label), fontsize="xx-large")
    label = ax.get_xlabel()
    ax.set_xlabel(mapping.get(label, label), fontsize="xx-large")
    xlims = ax.get_xlim()
    ax.hlines(y.mean(), xmin=xlims[0], xmax=xlims[1], linestyle="--", color="red")
    ylims = ax.get_ylim()
    ax.vlines(x.mean(), ymin=ylims[0], ymax=ylims[1], linestyle="--", color="red")
    ax.minorticks_on()
    ax.tick_params(which="major", width=tick_width, length=5, labelsize=14)
    ax.tick_params(which="minor", width=tick_width, length=3, labelsize=14)

    ax = sns.histplot(x=x, fill=True, linewidth=1.5, ax=g.ax_marg_x, **kwargs1)
    ax.set_xlim(*lim1)
    ax.set_ylabel(ax.get_ylabel(), fontsize="xx-large")
    ax.tick_params(which="major", width=tick_width, length=5, labelsize=14)
    ax.tick_params(which="minor", width=tick_width, length=3, labelsize=14)

    ax = sns.histplot(y=y, fill=True, linewidth=1.5, ax=g.ax_marg_y, **kwargs2)
    ax.set_ylim(*lim2)
    label = ax.get_xlabel()
    ax.set_xlabel(ax.get_xlabel(), fontsize="xx-large")
    ax.tick_params(which="major", width=tick_width, length=5, labelsize=14)
    ax.tick_params(which="minor", width=tick_width, length=3, labelsize=14)
    return g


def plot_posteriors(abc_results: abc.AbcResults, abc_summary: pd.DataFrame):
    results = abc_summary.loc[
        abc_summary.idx.isin(abc_results.get_idx()), ["mu", "s", "std"]
    ].drop_duplicates()

    priors = abc_summary[["mu", "u", "s", "std"]].drop_duplicates()

    print(f"plotting {results.shape[0]} runs")
    g_mu_s = plot_results(
        results,
        ["mu", "s"],
        [0, lims(priors, "mu")[1]],
        [0, lims(priors, "s")[1]],
        {"discrete": True},
        {"binwidth": 0.01},
    )

    g_mu_std = plot_results(
        results,
        ["mu", "std"],
        [0, lims(priors, "mu")[1]],
        [0, lims(priors, "std")[1]],
        {"discrete": True},
        {"binwidth": 0.002},
    )

    g_s_std = plot_results(
        results,
        ["s", "std"],
        [0, lims(priors, "s")[1]],
        [0, lims(priors, "std")[1]],
        {"binwidth": 0.01},
        {"binwidth": 0.002},
    )
    return g_mu_s, g_mu_std, g_s_std


def run_abc_filtering_on_clones(
    df, quantile: float, nb_clones_diff: int, minimum_runs: int, verbose: bool = True
):
    idx_abc = dict()
    view = df[df["clones diff"] <= nb_clones_diff]
    idx_abc = abc.run_abc(view, quantile, minimum_runs, verbose=verbose)
    g1, g2, g3 = plot_posteriors(idx_abc, view)
    return idx_abc, g1, g2, g3


def get_idx_smaller_distance_clones_from_results(
    abc_summary: pd.DataFrame, abc_results: abc.AbcResults
) -> int:
    """From all runs accepted by ABC, get the index of the run with the minimal
    distance between the number of clones (likely all the runs with a number of
    clones equal to the number of clones of the target).
    Then, among this subset of runs, take the idx of the run with the smaller
    wasserstein distance.
    """
    view = abc_summary[abc_summary.idx.isin(abc_results.get_idx())]
    second_view = view[view["clones diff"] == view["clones diff"].min()]
    return second_view.loc[
        second_view.wasserstein == second_view.wasserstein.min(), "idx"
    ].squeeze()
