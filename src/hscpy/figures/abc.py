import functools
from pathlib import Path
from typing import Dict, List, Tuple
from futils import snapshot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats

from hscpy import abc, realisation
from hscpy.figures import AgeSims


class Gamma:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def shape_scale_from_mean_std(self):
        return self.mean**2 / self.std**2, self.std**2 / self.mean

    def plot(self, ax, **kwargs):
        shape, scale = self.shape_scale_from_mean_std()
        x = np.linspace(
            stats.gamma.ppf(0.00001, shape, scale=scale),
            stats.gamma.ppf(0.99999, shape, scale=scale),
            100,
        )
        ax.plot(x, stats.gamma.pdf(x, shape, 0, scale), **kwargs)
        return ax


def plot_posteriors_fancy(
    accepted: pd.Series,
    xlabel: str,
    bins,
    ax,
    color,
    fancy: bool,
    legend: bool = False,
):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/stairs_demo.html
    bin_distance = (bins[1] - bins[0]) / 2
    if fancy:
        values, _ = np.histogram(
            accepted, bins=bins, density=True
        )  # TODO density=True is ok?
        ax.fill_between(
            bins[:-1] + bin_distance, values, ls="-", color=color, alpha=0.15
        )
        ax.plot(
            bins[:-1] + bin_distance,
            values,
            ls="-",
            marker=".",
            mew=3,
            color=color,
            alpha=0.5,
        )
    else:
        values, bins, _ = ax.hist(
            accepted,
            align="mid",
            alpha=0.4,
            density=True,
            bins=bins,
            edgecolor="black",
            color=color,
        )

    # MAP
    point_estimate = bins[np.argmax(values)] + bin_distance
    ax.axvline(x=point_estimate, mew=3, ls="--", c=color)
    ax.set_xlim([bins[0] - bin_distance, bins[-1] + bin_distance])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("pdf")
    if legend:
        ax.legend()
    return point_estimate


def fmt_two_digits(x, pos):
    return f"{x:.2f}"


def plot_posteriors_grid_eta_sigma_tau_mu(
    posterior: pd.DataFrame,
    name: str,
    fig,
    color: str,
    bins_eta,
    bins_sigma,
    bins_tau,
    bins_mu,
    fancy: bool = True,
):
    # posteriors
    point_estimate_eta = plot_posteriors_fancy(
        posterior.eta, r"$\eta$", bins_eta, fig.axes[2], color, fancy=fancy
    )
    point_estimate_sigma = plot_posteriors_fancy(
        posterior.sigma, r"$\sigma$", bins_sigma, fig.axes[3], color, fancy=fancy
    )
    _ = plot_posteriors_fancy(
        posterior.tau, r"$\tau$", bins_tau, fig.axes[4], color, fancy=fancy
    )
    _ = plot_posteriors_fancy(
        posterior.mu, r"$\mu$", bins_mu, fig.axes[5], color, fancy=fancy
    )

    fig.axes[3].set_ylabel("")
    fig.axes[5].set_ylabel("")

    # gamma
    gamma = Gamma(point_estimate_eta, point_estimate_sigma)
    gamma.plot(fig.axes[0], label=name, color=color)

    return fig, gamma


def create_posteriors_grid_eta_sigma_tau_mu():
    fig = plt.figure(figsize=[8, 7.5], layout="constrained")
    # init axes
    gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 2])
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs0[0])
    gs01 = gs0[1].subgridspec(2, 2)
    ax1 = fig.add_subplot(gs00[0, :-1])
    # ax1.set_ylim([-1, 22])
    ax1.set_ylabel("pdf")
    ax1.set_xlabel(r"$s$")  # TODO
    ax2 = fig.add_subplot(gs00[0, -1])
    ax2.axis("off")
    ax3 = fig.add_subplot(gs01[0, 0])
    ax4 = fig.add_subplot(gs01[0, 1])
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_two_digits))
    # ax4.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_two_digits))
    ax5 = fig.add_subplot(gs01[1, 0])
    ax6 = fig.add_subplot(gs01[1, 1])

    return fig


class SyntheticValidation:
    def __init__(
        self,
        idx: int,
        sims_sfs: Dict[AgeSims, List[realisation.RealisationSfs]],
        sims_clones: pd.DataFrame,
    ):
        self.target_sfs = {k: sims_sfs[k][idx] for k in sims_sfs.keys()}
        self.params = self.target_sfs[AgeSims(0.0)].parameters.into_dict()

        print(
            f"running abc synthetic with ground truth run {idx} with params: {self.params}"
        )

        target_clones = sims_clones.loc[
            sims_clones.idx == self.params["idx"],
            ["age", "variant counts detected", "mu"],
        ].rename(columns={"variant counts detected": "target clones detected"})
        assert (self.params["mu"] == target_clones.mu).all()
        target_clones.drop(columns=["mu"], inplace=True)
        self.target_clones = target_clones

        abc_mitchell = abc.compute_abc_results(
            self.target_sfs, self.target_clones, sims_sfs, sims_clones
        )
        self.abc = abc_mitchell

    def compute_posteriors(
        self,
        quantile_sfs: float = 0.2,
        quantile_clones: float = 0.2,
        proprtion_runs_disc: float = 0.2,
        bins_s: np.ndarray = np.arange(0, 0.44, 0.02),
        bins_mu: np.ndarray = np.arange(0, 20, 1),
        bins_tau: np.ndarray = np.arange(0, 5.2, 0.2),
        bins_std: np.ndarray = np.arange(0, 0.12, 0.01),
        density: bool = False,
    ):
        runs2keep = abc.run_abc_sfs_clones(
            self.abc, quantile_sfs, quantile_clones, proprtion_runs_disc
        )

        view_synthetic = self.abc[self.abc.idx.isin(runs2keep)].drop_duplicates(
            subset="idx"
        )
        assert not view_synthetic.empty, "empty posterior"

        print(f"ABC combined kept {len(runs2keep)} runs")

        gs = []

        axd = plot_results(view_synthetic, "s", "mu", bins_s, bins_mu, density=density)
        axd["C"].axvline(self.params["s"])
        axd["C"].axhline(self.params["mu"])
        gs.append(axd)
        plt.show()

        axd = plot_results(
            view_synthetic, "s", "std", bins_s, bins_std, density=density
        )
        axd["C"].axvline(self.params["s"])
        axd["C"].axhline(self.params["std"])
        gs.append(axd)
        plt.show()

        axd = plot_results(
            view_synthetic, "s", "tau", bins_s, bins_tau, density=density
        )
        axd["C"].axvline(self.params["s"])
        axd["C"].axhline(self.params["tau"])
        gs.append(axd)
        plt.show()

        axd = plot_results(
            view_synthetic, "mu", "tau", bins_mu, bins_tau, density=density
        )
        axd["C"].axvline(self.params["mu"])
        axd["C"].axhline(self.params["tau"])
        gs.append(axd)
        plt.show()

        axd = plot_results(
            view_synthetic, "mu", "std", bins_mu, bins_std, density=density
        )
        axd["C"].axvline(self.params["mu"])
        axd["C"].axhline(self.params["std"])
        gs.append(axd)
        plt.show()

        axd = plot_results(
            view_synthetic, "tau", "std", bins_tau, bins_std, density=density
        )
        axd["C"].axvline(self.params["tau"])
        axd["C"].axhline(self.params["std"])
        gs.append(axd)

        plt.show()

        return gs


def lims(mask: pd.DataFrame, col: str) -> Tuple[float, float]:
    min_, max_ = mask[col].min(), mask[col].max()
    shift = 0.1
    return min_ - min_ * shift, max_ + max_ * shift


def plot_prior(prior: pd.Series, ax, **kwargs):
    ax = sns.histplot(prior, ax=ax, **kwargs)
    return ax


def plot_results(
    results: pd.DataFrame,
    x: str,
    y: str,
    xbins: np.ndarray,
    ybins: np.ndarray,
    density: bool = False,
) -> Dict:
    mapping = {
        "tau": r"$\tau$",
        "mu": r"$\mu$",
        "eta": r"$\eta$",
        "sigma": r"$\sigma$",
    }
    if density:
        print(
            "WARNING setting `density=True` is buggy: the yaxis of the top plot seems wrong"
        )

    xlims = [xbins.min() - (xbins[1] - xbins[0]), xbins.max() + (xbins[1] - xbins[0])]
    ylims = [ybins.min() - (ybins[1] - ybins[0]), ybins.max() + (ybins[1] - ybins[0])]

    axd = plt.figure(layout="constrained").subplot_mosaic(
        """
        A.
        CD
        """,
        # set the height ratios between the rows
        height_ratios=[1, 3.5],
        # set the width ratios between the columns
        width_ratios=[3.5, 1],
        per_subplot_kw={
            "A": {
                "xticklabels": [],
                "xlim": xlims,
                "ylabel": "pdf" if density else "counts",
            },
            "C": {
                "xlim": xlims,
                "ylim": ylims,
                "xlabel": mapping.get(x, x),
                "ylabel": mapping.get(y, x),
            },
            "D": {
                "yticklabels": [],
                "ylim": ylims,
                "xlabel": "pdf" if density else "counts",
            },
        },
    )

    axd["A"].hist(results[x], density=density, bins=xbins, edgecolor="black")
    axd["D"].hist(
        results[y],
        bins=ybins,
        density=density,
        orientation="horizontal",
        edgecolor="black",
    )
    axd["C"].hist2d(x=results[x], y=results[y], bins=[xbins, ybins], cmap="Greys")

    # force lims after hist2d plot
    axd["C"].set_xlim(xlims)
    axd["C"].set_ylim(ylims)

    return axd


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


def get_idx_smaller_distance_clones_idx(
    abc_summary: pd.DataFrame, idx: List[int]
) -> int:
    """From all runs accepted by ABC, get the index of the run with the minimal
    distance between the number of clones (likely all the runs with a number of
    clones equal to the number of clones of the target).
    Then, among this subset of runs, take the idx of the run with the smaller
    wasserstein distance.
    """
    view = abc_summary[abc_summary.idx.isin(idx)]
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
