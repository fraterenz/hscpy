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
from itertools import permutations
from scipy import stats

from hscpy import abc, realisation
from hscpy.figures import AgeSims


class Estimate:
    def __init__(
        self, name: str, point_estimate, credible_interval_90: Tuple[float, float]
    ):
        """MAP estimate with 90% credibility interval"""
        self.name = name
        self.point_estimate = point_estimate
        if point_estimate < credible_interval_90[0]:
            self.credible_interval_90 = (point_estimate, credible_interval_90[1])
        else:
            self.credible_interval_90 = credible_interval_90

    def to_string(self, precision: str) -> str:
        point_estimate = round_estimates(self.point_estimate, precision)
        interval = round_estimates(
            self.point_estimate - self.credible_interval_90[0], precision
        ), round_estimates(
            self.credible_interval_90[1] - self.point_estimate, precision
        )
        return f"{point_estimate}^{{+{interval[1]}}}_{{-{interval[0]}}}"


class Bin:
    def __init__(self, name: str, bin_: np.ndarray):
        self.name = name
        self.bin = bin_
        self.bin_distance = (self.bin[1] - self.bin[0]) / 2

    def compute_hist(self, accepted: pd.Series) -> np.ndarray:
        values, _ = np.histogram(accepted, bins=self.bin, density=True)
        return values

    def compute_estimate(self, accepted: pd.Series) -> Estimate:
        values = self.compute_hist(accepted)
        point_estimate = self.bin[np.argmax(values)] + self.bin_distance
        return Estimate(
            self.name,
            point_estimate,
            (accepted.quantile((0.10)), accepted.quantile(0.90)),
        )


def plot_results(
    results: pd.DataFrame,
    xbins: Bin,
    ybins: Bin,
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

    xlims = [
        xbins.bin.min() - (xbins.bin_distance),
        xbins.bin.max() + xbins.bin_distance,
    ]
    ylims = [
        ybins.bin.min() - (ybins.bin_distance),
        ybins.bin.max() + ybins.bin_distance,
    ]
    x, y = xbins.name, ybins.name

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

    axd["A"].hist(results[x], density=density, bins=xbins.bin, edgecolor="black")
    axd["D"].hist(
        results[y],
        bins=ybins.bin,
        density=density,
        orientation="horizontal",
        edgecolor="black",
    )
    axd["C"].hist2d(
        x=results[x], y=results[y], bins=[xbins.bin, ybins.bin], cmap="Greys"
    )

    # force lims after hist2d plot
    axd["C"].set_xlim(xlims)
    axd["C"].set_ylim(ylims)

    return axd


class Bins:
    def __init__(self, bins_s, bins_std, bins_mu, bins_tau):
        self.bins = {
            "eta": Bin("eta", bins_s),
            "sigma": Bin("sigma", bins_std),
            "mu": Bin("mu", bins_mu),
            "tau": Bin("tau", bins_tau),
        }
        self.iteration = [
            tuple(ele) for ele in {frozenset(c) for c in permutations(self.bins, r=2)}
        ]

    def iterate(self) -> List[Tuple[str, str]]:
        return self.iteration

    def plot_posterior(self, posterior: pd.DataFrame, density: bool):
        # plots
        axd = list()
        for b in self.iterate():
            xbins, ybins = self.bins[b[0]], self.bins[b[1]]
            ax = plot_results(
                posterior,
                xbins,
                ybins,
                density=density,
            )
            estimate_x, estimate_y = xbins.compute_estimate(
                posterior.loc[:, xbins.name]
            ), ybins.compute_estimate(posterior.loc[:, ybins.name])
            if xbins.name == "sigma":
                precision = "three"
            else:
                precision = "two"
            ax["C"].axvline(estimate_x.point_estimate, alpha=0.8)
            ax["C"].axhline(estimate_y.point_estimate, alpha=0.8)
            ax["C"].text(
                0.6,
                0.85,
                f"$\{xbins.name}={{{estimate_x.to_string(precision)}}}$",
                transform=ax["C"].transAxes,
            )
            ax["C"].text(
                0.6,
                0.7,
                f"$\{ybins.name}={{{estimate_y.to_string(precision)}}}$",
                transform=ax["C"].transAxes,
            )

            axd.append(ax)

        return axd


def posterior_mitchell_quantile(
    abc_mitchell,
    quantiles_sfs: float,
    quantiles_clones: float,
    prop_runs_disc: float,
    bins: Bins,
    density: bool = False,
):
    runs2keep = abc.run_abc_sfs_clones(
        abc_mitchell, quantiles_sfs, quantiles_clones, prop_runs_disc
    )
    posterior_mitchell = abc_mitchell.loc[
        abc_mitchell.idx.isin(runs2keep), :
    ].drop_duplicates(subset="idx")

    assert not posterior_mitchell.empty, "empty posterior"
    tot_runs = abc_mitchell.idx.unique().shape[0]
    print(
        f"ABC combined kept {len(runs2keep) / tot_runs:.2%} of the runs ({len(runs2keep)} runs) over a total of {tot_runs} runs"
    )
    return runs2keep


def round_estimates(estimate: float, significant: str) -> str:
    if significant == "three":
        return str(round(estimate, 3))
    elif significant == "two":
        return str(round(estimate, 2))
    elif significant == "one":
        return str(round(estimate, 1))
    elif significant == "zero":
        return str(int(round(estimate, 0)))
    raise ValueError(
        f"significant must be either 'two' 'one' or 'zero', not '{significant}'"
    )


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
        # pad to zero
        x, y = np.insert(x, 0, 0), np.insert(stats.gamma.pdf(x, shape, 0, scale), 0, 0)
        # pad to 0.4 (max val of s)
        if max(x) < 0.4:
            x, y = np.insert(x, -1, 0.4), np.insert(
                stats.gamma.pdf(x, shape, 0, scale), 0, 0
            )

        ax.plot(x, y, **kwargs)
        return ax


def plot_posteriors_fancy(
    accepted: pd.Series,
    xlabel: str,
    bins: Bin,
    ax,
    color,
    fancy: bool,
    legend: bool = False,
):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/stairs_demo.html
    values = bins.compute_hist(accepted)
    if fancy:
        ax.fill_between(
            bins.bin[:-1] + bins.bin_distance, values, ls="-", color=color, alpha=0.15
        )
        ax.plot(
            bins.bin[:-1] + bins.bin_distance,
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

    ax.set_xlabel(xlabel)
    ax.set_ylabel("pdf")
    if legend:
        ax.legend()
    return ax


def fmt_two_digits(x, pos):
    return f"{x:.2f}"


def plot_posteriors_grid_eta_sigma_tau_mu(
    posterior: pd.DataFrame,
    name: str,
    fig,
    color: str,
    bins_eta: Bin,
    bins_sigma: Bin,
    bins_tau: Bin,
    bins_mu: Bin,
    fancy: bool = True,
):
    # posteriors
    _ = plot_posteriors_fancy(
        posterior.eta, r"$\eta$", bins_eta, fig.axes[2], color, fancy=fancy
    )
    _ = plot_posteriors_fancy(
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
    estimate_eta, estimate_sigma = bins_eta.compute_estimate(posterior.loc[:, "eta"]), bins_sigma.compute_estimate(posterior.loc[:, "sigma"])
    gamma = Gamma(estimate_eta.point_estimate, estimate_sigma.point_estimate)
    gamma.plot(fig.axes[0], label=name, color=color)
    fig.axes[0].set_xlim(bins_eta.bin[0] - 0.01, bins_eta.bin[-1])

    return fig, gamma, [estimate_eta, estimate_sigma]


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
        target_sfs = {
            k: sfs
            for k, s in sims_sfs.items()
            for sfs in s
            if sfs.parameters.idx == idx
        }
        assert target_sfs, f"idx {idx} not found in the simulations"
        self.params = target_sfs[AgeSims(0.0)].parameters.into_dict()
        self.params["eta"] = self.params["s"] / self.params["tau"]
        self.params["sigma"] = self.params["std"] / self.params["tau"]
        assert idx == self.params["idx"]
        self.target_sfs = {k: sfs.sfs for k, sfs in target_sfs.items()}

        print(
            f"running abc synthetic with ground truth run {idx} with params: {self.params}"
        )

        self.target_clones = sims_clones.loc[
            sims_clones.idx == self.params["idx"], ["age", "variant counts detected"]
        ].rename(columns={"variant counts detected": "clones"})

        abc_mitchell = abc.compute_abc_results(
            self.target_sfs,
            self.target_clones,
            sims_sfs,
            sims_clones.loc[:, ["idx", "age", "variant counts detected"]],
            "synthetic",
        )
        self.abc = abc_mitchell

    def compute_posteriors(
        self,
        quantile_sfs: float,
        quantile_clones: float,
        proportion_runs_disc: float,
        bins: Bins,
        density: bool = False,
    ):
        runs2keep = posterior_mitchell_quantile(
            self.abc, quantile_sfs, quantile_clones, proportion_runs_disc, bins, density
        )
        for b in bins.iterate():
            xbins, ybins = bins.bins[b[0]], bins.bins[b[1]]
            axd = plot_results(
                self.abc.loc[self.abc.idx.isin(runs2keep), :], xbins, ybins
            )
            axd["C"].axvline(self.params[xbins.name])
            axd["C"].axhline(self.params[ybins.name])
        return runs2keep, axd


def lims(mask: pd.DataFrame, col: str) -> Tuple[float, float]:
    min_, max_ = mask[col].min(), mask[col].max()
    shift = 0.1
    return min_ - min_ * shift, max_ + max_ * shift


def plot_prior(prior: pd.Series, ax, **kwargs):
    ax = sns.histplot(prior, ax=ax, **kwargs)
    return ax


def get_idx_smaller_distance_clones_idx(
    abc_summary: pd.DataFrame, idx: List[int]
) -> int:
    """From all runs accepted by ABC, get the index of the run with the minimal
    distance between the number of clones (likely all the runs with a number of
    clones equal to the number of clones of the target).
    Then, among this subset of runs, take the idx of the run with the smaller
    wasserstein distance.
    """
    view = abc_summary.loc[abc_summary.idx.isin(idx), :]
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
