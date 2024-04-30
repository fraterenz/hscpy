from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from hscpy import abc, realisation
from hscpy.figures import AgeSims

PRECISION = {"eta": "two", "sigma": "three", "mu": "zero", "tau": "one"}


class Estimate:
    def __init__(
        self,
        name: str,
        point_estimate,
        credible_interval_90: Tuple[float, float],
    ):
        """MAP estimate with 90% credibility interval"""
        self.name = name
        self.point_estimate = point_estimate
        if point_estimate < credible_interval_90[0]:
            self.credible_interval_90 = (
                point_estimate,
                credible_interval_90[1],
            )
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
    results_unique = results.drop_duplicates("idx")

    xlims = [
        xbins.bin.min() - (xbins.bin_distance),
        xbins.bin.max() + xbins.bin_distance,
    ]
    ylims = [
        ybins.bin.min() - (ybins.bin_distance),
        ybins.bin.max() + ybins.bin_distance,
    ]
    x, y = xbins.name, ybins.name

    axd = plt.figure(figsize=(3.5, 3), layout="constrained").subplot_mosaic(
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

    axd["A"].hist(
        results_unique[x], density=density, bins=xbins.bin, edgecolor="black"
    )
    axd["D"].hist(
        results_unique[y],
        bins=ybins.bin,
        density=density,
        orientation="horizontal",
        edgecolor="black",
    )
    axd["C"].hist2d(
        x=results_unique[x],
        y=results_unique[y],
        bins=[xbins.bin, ybins.bin],
        cmap="Greys",
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
            tuple(ele)
            for ele in {frozenset(c) for c in permutations(self.bins, r=2)}
        ]

    def iterate(self) -> List[Tuple[str, str]]:
        return self.iteration

    def plot_posterior(
        self,
        posterior: pd.DataFrame,
        density: bool,
        path2file: Union[Path, None] = None,
    ):
        axd, estimates = list(), dict()
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
            estimates[xbins.name], estimates[ybins.name] = (
                estimate_x,
                estimate_y,
            )
            precision_x, precision_y = PRECISION[xbins.name], PRECISION[ybins.name]
            ax["C"].axvline(estimate_x.point_estimate, alpha=0.8)
            ax["C"].axhline(estimate_y.point_estimate, alpha=0.8)
            ax["C"].text(
                0.45,
                0.85,
                fr"$\{xbins.name}={estimate_x.to_string(precision_x)}$",
                transform=ax["C"].transAxes,
                fontsize=12,
            )
            ax["C"].text(
                0.45,
                0.7,
                fr"$\{ybins.name}={estimate_y.to_string(precision_y)}$",
                transform=ax["C"].transAxes,
                fontsize=12,
            )
            if path2file:
                x = ax["C"].get_xlabel().replace("$", "").replace("\\", "")
                y = ax["C"].get_ylabel().replace("$", "").replace("\\", "")
                plt.savefig(str(path2file) + x + f"_{y}.svg")

            axd.append(ax)

        return axd, estimates


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
    counts = abc_mitchell["idx"].value_counts()
    tot_runs = (counts == counts.max()).sum()
    print(
        "ABC kept {:.2%} of the runs ({} runs) over a total of {} runs".format(
            len(runs2keep) / tot_runs, len(runs2keep), tot_runs
        )
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
        f"significant must be 'two' 'one' or 'zero', not '{significant}'"
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
        x, y = np.insert(x, 0, 0), np.insert(
            stats.gamma.pdf(x, shape, 0, scale), 0, 0
        )
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
            bins.bin[:-1] + np.diff(bins.bin) / 2,
            values,
            ls="-",
            color=color,
            alpha=0.3,
        )
        ax.plot(
            bins.bin[:-1] + np.diff(bins.bin) / 2,
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
            bins=bins.bin,
            edgecolor="black",
            color=color,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("pdf")
    if bins.name == "sigma":
        # ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        ax.set_xlim([0, 0.105])
    if legend:
        ax.legend()
    return ax


def fmt_two_digits(x, pos):
    return f"{x:.2f}"


def plot_gamma_inferred(
    ax,
    posterior: pd.DataFrame,
    name: str,
    color: str,
    bins_eta: Bin,
    bins_sigma: Bin,
    loc: Tuple[float, float],
    seed: int = 26,
    nb2plot: int = 30,
):
    estimate_eta = bins_eta.compute_estimate(posterior.loc[:, "eta"])
    estimate_sigma = bins_sigma.compute_estimate(posterior.loc[:, "sigma"])
    gamma = Gamma(estimate_eta.point_estimate, estimate_sigma.point_estimate)
    gamma.plot(ax, label=name, color=color)
    ax.set_xlim(bins_eta.bin[0] - 0.01, bins_eta.bin[-1])
    ax.text(
        loc[0],
        loc[1],
        r"$\eta={:.2f}\;\sigma={:.3f}$".format(
            estimate_eta.point_estimate, estimate_sigma.point_estimate
        ),
        fontsize=13,
        color=color,
        # bbox=bbox,
        transform=ax.transAxes,
        horizontalalignment="right",
    )
    ax.set_ylabel("pdf")
    ax.set_xlabel(r"Innate clone fitness $s$")
    for row in (
        posterior[["eta", "sigma"]]
        .sample(nb2plot, random_state=seed)
        .itertuples()
    ):
        Gamma(row.eta, row.sigma).plot(ax, color=color, alpha=0.1)
    return ax


def plot_posteriors_with_estimate(
    ax,
    posterior: pd.Series,
    name: str,
    bins: Bin,
    color: str,
    loc: Tuple[float, float],
):
    # posteriors
    estimate_ = bins.compute_estimate(posterior)
    _ = plot_posteriors_fancy(posterior, name, bins, ax, color, fancy=True)
    ax.axvline(estimate_.point_estimate, color=color, ls="--")
    xlabel = ax.get_xlabel().replace("\\", "").replace("$", "")
    precision = PRECISION[xlabel]
    ax.text(
        loc[0],
        loc[1],
        f"$\\{xlabel}={estimate_.to_string(precision)}$",
        fontsize=13,
        color=color,
        # bbox=bbox,
        transform=ax.transAxes,
        horizontalalignment="right",
    )
    return ax


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
            f"running abc synthetic with target run {idx} with: {self.params}"
        )

        self.target_clones = sims_clones.loc[
            sims_clones.idx == self.params["idx"],
            ["age", "variant counts detected"],
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
        path2file: Union[Path, None] = None,
        density: bool = False,
    ):
        runs2keep = posterior_mitchell_quantile(
            self.abc, quantile_sfs, quantile_clones, proportion_runs_disc, bins
        )
        posterior = self.abc.loc[self.abc.idx.isin(runs2keep), :]
        return bins.plot_posterior(posterior, density, path2file)


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
