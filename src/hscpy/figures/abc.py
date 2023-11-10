from pathlib import Path
from typing import Dict, List, Tuple
from futils import snapshot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from hscpy import abc, realisation
from hscpy.figures import AgeSims


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
    show_mean: bool,
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
    ax.set_ylabel(mapping.get(label, label))  # , fontsize="xx-large")
    label = ax.get_xlabel()
    ax.set_xlabel(mapping.get(label, label))  # , fontsize="xx-large")
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    if show_mean:
        ax.hlines(y.mean(), xmin=xlims[0], xmax=xlims[1], linestyle="--", color="red")
        ax.vlines(x.mean(), ymin=ylims[0], ymax=ylims[1], linestyle="--", color="red")
    ax.minorticks_on()
    ax.tick_params(which="major", width=tick_width, length=5, labelsize=14)
    ax.tick_params(which="minor", width=tick_width, length=3, labelsize=14)

    ax = sns.histplot(x=x, fill=True, linewidth=1.5, ax=g.ax_marg_x, **kwargs1)
    ax.set_xlim(*lim1)
    ax.set_ylabel(ax.get_ylabel())  # , fontsize="xx-large")
    ax.tick_params(which="major", width=tick_width, length=5, labelsize=14)
    ax.tick_params(which="minor", width=tick_width, length=3, labelsize=14)

    ax = sns.histplot(y=y, fill=True, linewidth=1.5, ax=g.ax_marg_y, **kwargs2)
    ax.set_ylim(*lim2)
    label = ax.get_xlabel()
    ax.set_xlabel(ax.get_xlabel())  # , fontsize="xx-large")
    ax.tick_params(
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        width=tick_width,
        length=5,
        labelsize=14,
    )
    ax.tick_params(
        which="minor",
        bottom=True,
        top=False,
        left=True,
        right=False,
        width=tick_width,
        length=3,
        labelsize=14,
    )
    return g


def plot_posteriors(
    results: pd.DataFrame,
    show_mean: bool,
    mu_lims: Tuple[float, float],
    s_lims: Tuple[float, float],
    std_lims: Tuple[float, float],
):
    results = results[["mu", "s", "std"]].drop_duplicates()

    print(f"plotting {results.shape[0]} runs")
    g_mu_s = plot_results(
        results,
        ["mu", "s"],
        mu_lims,
        s_lims,
        {"discrete": True},
        {"binwidth": 0.01},
        show_mean,
    )

    g_mu_std = plot_results(
        results,
        ["mu", "std"],
        mu_lims,
        std_lims,
        {"discrete": True},
        {"binwidth": 0.002},
        show_mean,
    )

    g_s_std = plot_results(
        results,
        ["s", "std"],
        s_lims,
        std_lims,
        {"binwidth": 0.01},
        {"binwidth": 0.002},
        show_mean,
    )
    for g_ in {g_mu_s, g_mu_std, g_s_std}:
        g_.ax_joint.tick_params(
            which="major",
            bottom=True,
            top=False,
            left=True,
            right=False,
            width=1.1,
            length=5,
            labelsize=14,
        )
        g_.ax_joint.tick_params(
            which="minor",
            bottom=True,
            top=False,
            left=True,
            right=False,
            width=1.1,
            length=3,
            labelsize=14,
        )
        g_.ax_marg_x.tick_params(
            which="major",
            bottom=True,
            top=False,
            left=True,
            right=False,
            width=1.1,
            length=3,
            labelsize=14,
        )
        g_.ax_marg_x.tick_params(
            which="minor",
            bottom=True,
            top=False,
            left=True,
            right=False,
            width=1.1,
            length=3,
            labelsize=14,
        )
        g_.ax_marg_y.tick_params(
            which="minor",
            bottom=True,
            top=False,
            left=True,
            right=False,
            width=1.1,
            length=3,
            labelsize=14,
        )
        g_.ax_marg_y.tick_params(
            which="major",
            bottom=True,
            top=False,
            left=True,
            right=False,
            width=1.1,
            length=3,
            labelsize=14,
        )
    return g_mu_s, g_mu_std, g_s_std


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


def abc_simulated_validation(
    target_stem: str,
    sfs_sims: Dict[AgeSims, List[realisation.RealisationSfs]],
    counts: pd.DataFrame,
    thresholds: abc.AbcThresholds,
    show_priors: bool = True,
):
    assert False, "TODO"
    target_sfs_simulated = {
        t: sfs_.sfs
        for t, sfs_donor in sfs_sims.items()
        for sfs_ in sfs_donor
        if sfs_.parameters.path.stem == target_stem
    }

    assert len(target_sfs_simulated), "wrong `target_stem`"

    abc_simulated = abc.sfs_summary_statistic_wasserstein(
        sfs_sims, target_sfs_simulated, target_stem
    )

    abc_simulated["target"] = (
        abc_simulated.path.map(lambda x: Path(x).stem) == target_stem
    )

    abc_simulated = abc_simulated.merge(
        right=counts[["age", "idx", "variant counts detected"]],
        how="left",
        left_on=["idx", "timepoint"],
        right_on=["idx", "age"],
        validate="one_to_one",
    )

    abc_simulated = abc_simulated.merge(
        right=abc_simulated.loc[
            abc_simulated.target, ["variant counts detected", "timepoint"]
        ].rename({"variant counts detected": "clones"}, axis=1),
        how="left",
        on="timepoint",
        validate="many_to_one",
    )

    abc_simulated["clones diff"] = (
        abc_simulated["clones"] - abc_simulated["variant counts detected"]
    ).abs()

    if show_priors:
        priors = abc_simulated[["mu", "u", "s", "std"]].drop_duplicates()

        fig, ax = plt.subplots(1, 1, figsize=[7, 6])
        ax = plot_prior(priors["s"], ax=ax, binwidth=0.01)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=[7, 6])
        ax = plot_prior(priors["std"], ax=ax, binwidth=0.001)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=[7, 6])
        ax = plot_prior(priors["mu"], ax=ax, discrete=True)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=[7, 6])
        ax = plot_prior(priors["u"], ax=ax)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=[7, 6])
        sns.histplot(abc_simulated["wasserstein"], binwidth=0.01, ax=ax)
        plt.show()

    tot_runs = abc_simulated.age.unique().shape[0]
    minimum_runs = tot_runs - round(tot_runs * thresholds.proportion_runs_to_discard)

    results, g1, g2, g3 = run_abc_filtering_on_clones(abc_simulated, thresholds)

    mu_target, s_target, std_target = (
        abc_simulated.loc[abc_simulated.target, "mu"].squeeze(),
        abc_simulated.loc[abc_simulated.target, "s"].squeeze(),
        abc_simulated.loc[abc_simulated.target, "std"].squeeze(),
    )

    g1.ax_joint.plot(mu_target, s_target, marker="x", color="black", mew=2)
    g2.ax_joint.plot(mu_target, std_target, marker="x", color="black", mew=2)
    g3.ax_joint.plot(s_target, std_target, marker="x", color="black", mew=2)

    return abc_simulated, g1, g2, g3


def run_abc_indep(
    summary: pd.DataFrame, metric: str, q: float, counts
) -> List[abc.AbcResults]:
    results_indep_timepoints = []
    for t in summary.timepoint.unique():
        print(
            f"running ABC with metric {metric} on timepoint {t} with q {q} with value of {summary.loc[summary.timepoint == t, metric].quantile(q):.2f}"
        )

        res = abc.run_abc(summary[summary.timepoint == t], q, metric, True)
        (
            _,
            _,
            _,
        ) = plot_posteriors(
            summary[summary.idx.isin(res.get_idx())],
            False,
            (0, 21),
            (0, 0.41),
            (0, 0.11),
        )
        plt.show()
        results_indep_timepoints.append(res)

    fig, ax = plt.subplots(1, 1)
    view = counts[
        counts.idx.isin(
            set([idx for res in results_indep_timepoints for idx in res.get_idx()])
        )
    ]
    sns.lineplot(
        view,
        x="age",
        y="variant counts detected",
        errorbar=lambda x: (np.min(x), np.max(x)),
        ax=ax,
        label="min-max",
    )
    sns.lineplot(
        view,
        x="age",
        y="variant counts detected",
        errorbar="sd",
        ax=ax,
        color="orange",
        label="std",
    )
    sns.scatterplot(
        data=summary[["age", "clones"]].drop_duplicates(),
        x="age",
        y="clones",
        marker="x",
        linewidths=2,
        color="purple",
        label="Mitchell",
    )
    ax.legend()
    plt.show()

    return results_indep_timepoints
