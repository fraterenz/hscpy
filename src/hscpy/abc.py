from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Union

from futils import snapshot
import numpy as np
import pandas as pd
from scipy import stats

from hscpy import realisation
from hscpy.figures import AgeSims
from hscpy.realisation import RealisationSfs


def compute_abc_results(
    target_sfs: Dict[AgeSims, snapshot.Histogram],
    target_clones: pd.DataFrame,
    sims_sfs: Dict[AgeSims, List[realisation.RealisationSfs]],
    sims_clones: pd.DataFrame,
    experiment: str,
) -> pd.DataFrame:
    """
    Compute the ABC summary statistics.

    Required columns for `target_clones`: `age` and `variant counts detected`.
    Required columns for `sims_clone`: `age`, `idx` and `clones`.
    """
    assert np.all(sims_clones.age.unique() == target_clones.age)
    assert all(
        [ele in target_clones.columns for ele in ["age", "clones"]]
    ), "`target_clones` should have `age` and `clones` as cols"
    assert all(
        [
            ele in sims_clones.columns
            for ele in ["age", "idx", "variant counts detected"]
        ]
    ), "`sims_clones` missing cols `age`, `idx` and `variant counts detected`"
    print("wasserstein metric")
    abc_results = sfs_summary_statistic_wasserstein(
        sims_sfs,
        target_sfs,
        experiment,
    )

    print("clones metric")
    # perform a one to many merge to expand the series
    # since all sims with the same age will match one
    # unique target clone per timepoint
    clones_diff = target_clones[["age", "clones"]].merge(
        right=sims_clones[["age", "idx", "variant counts detected"]],
        how="left",
        on="age",
        validate="one_to_many",
    )  # add data from target
    clones_diff["clones diff"] = (
        clones_diff["variant counts detected"] - clones_diff["clones"]
    ).abs()
    abc_results = abc_results.merge(
        right=clones_diff,
        how="left",
        right_on=["idx", "age"],
        left_on=["idx", "timepoint"],
        validate="one_to_one",
    ).rename(columns={"variant counts detected": "sims clones"})
    abc_results = summary_statistic_relative_diff_clones(abc_results)

    abc_results["eta"] = abc_results.s / abc_results.tau
    abc_results["sigma"] = abc_results["std"] / abc_results.tau

    return abc_results


def run_abc_sfs_clones(
    abc_results: pd.DataFrame,
    quantile_sfs: float,
    quantile_clones: float,
    proportion_runs_to_discard: float,
) -> Set[int]:
    nb_timepoints = abc_results["sample"].unique().shape[0]
    minimum_timepoints = int(
        round(nb_timepoints - nb_timepoints * proportion_runs_to_discard)
    )

    # run abc with different metrics per each timepoint and keep only the runs
    # that are accepted at least for minimum timepoints
    print(
        f"Running ABC with {minimum_timepoints} minimum timepoints"
        f" over {nb_timepoints}"
    )

    wasserstein_idx = set(
        run_abc(
            abc_results,
            quantile=quantile_sfs,
            metric="wasserstein",
        )
        .abc_filter_on_minimum_timepoints(minimum_timepoints)
        .idx.tolist()
    )
    print(f"ABC wasserstein kept {len(wasserstein_idx)} runs")

    clones_idx = set(
        run_abc(
            abc_results,
            quantile=quantile_clones,
            metric="rel clones diff",
        )
        .abc_filter_on_minimum_timepoints(minimum_timepoints)
        .idx.tolist()
    )
    print(f"ABC clones kept {len(clones_idx)} runs")

    return clones_idx.intersection(wasserstein_idx)


def sfs_summary_statistic_ks(
    sims: Dict[AgeSims, List[RealisationSfs]],
    target: Dict[AgeSims, snapshot.Histogram],
    target_name: str,
) -> pd.DataFrame:
    """
    Return a list of records (list of dict) with the summary statistics and
    other quantities such as the parameters used.
    We compute the cumulative distribution function (cdf) for all the SFS from
    the simulations `sims` and compare that against the cdf of simulations' SFS
    `sims`.
    `target` and `sims` must have the same keys, keys being the timepoint
    considered.
    """
    abc_results = []

    for age_sim, sfs_sim in sims.items():
        data1 = np.log10(snapshot.array_from_hist(target[age_sim]))
        for my_sfs in sfs_sim:
            data2 = np.log10(snapshot.array_from_hist(my_sfs.sfs))
            params = my_sfs.parameters.into_dict()
            params["ks"] = stats.ks_2samp(data1, data2).statistic
            params["donor_name"] = target_name
            params["timepoint"] = age_sim
            abc_results.append(params)

    return pd.DataFrame.from_records(abc_results)


def get_values_weights_from_sfs(
    sfs_: snapshot.Histogram,
) -> Tuple[List[float], List[float]]:
    sfs_.pop(0, 0)
    # we dont use the log scale transf anymore because it
    # takes ages to run snapshot.array_from_hist(sfs)
    return list(sfs_.keys()), list(sfs_.values())


def sfs_summary_statistic_wasserstein(
    sims: Dict[AgeSims, List[RealisationSfs]],
    target: Dict[AgeSims, snapshot.Histogram],
    target_name: str,
) -> pd.DataFrame:
    """
    Compute the wasserstein distance between the simulated SFS in `sims` and
    the target SFS in `target`.
    We use the scipy function `stats.wasserstein_distance` with the values
    being the x-axis of the SFS (nb of cells with j muts) and the weights being
    the y-axis of the SFS (the nb of j muts), see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html

    The only processing step we take before computing the SFS is to remove the
    entry 0 when present, we don't log-transform anymore because it took time
    to transform the dict into an array, see `get_values_weigths_from_sfs`.

    Return a list of records (list of dict) with the summary statistics and
    other quantities such as the parameters used.
    `target` and `sims` must have the same keys, keys being the timepoints
    considered.
    """
    abc_results = []

    for age_sim, sfs_sim in sims.items():
        params = sfs_summary_statistic_wasserstein_timepoint(
            sfs_sim, target[age_sim], target_name, age_sim
        )

        abc_results.extend(params)

    return pd.DataFrame.from_records(abc_results)


def sfs_summary_statistic_wasserstein_timepoint(
    sims: List[RealisationSfs],
    target: snapshot.Histogram,
    target_name: str,
    age: Union[int, float],
):
    """
    Return a list of records (list of dict) with the summary statistics and
    other quantities such as the parameters used.
    We compute the SFS from the simulations `sims` and compare that against the
    `v_values`, `v_weights` which are computed from the SFS of the target data.
    """

    all_params = []

    for i, my_sfs in enumerate(sims):
        v_values, v_weights = get_values_weights_from_sfs(target)
        u_values, u_weights = get_values_weights_from_sfs(my_sfs.sfs)
        params = my_sfs.parameters.into_dict()
        # compute the summary statistic
        params["wasserstein"] = stats.wasserstein_distance(
            u_values, v_values, u_weights, v_weights
        )
        params["donor_name"] = target_name
        params["timepoint"] = age
        all_params.append(params)

    return all_params


def summary_statistic_relative_diff_clones(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    summary["rel clones diff"] = np.where(
        summary["clones"] == 0,
        summary["clones diff"],
        summary["clones diff"] / summary["clones"],
    )
    return summary


def filter_run(
    summary_t: pd.DataFrame, quantile: float, metric: str
) -> pd.DataFrame:
    assert metric in set(
        summary_t.columns
    ), f"metric {metric} not found in df with cols {set(summary_t.columns)}"
    df = summary_t[[metric, "idx", "timepoint"]]
    kept = df.loc[
        df[metric] <= df[metric].quantile(quantile), ["idx", "timepoint"]
    ]
    kept["metric"] = metric
    return kept


def filter_per_timepoint(
    summary: pd.DataFrame, quantile: float, metric: str, verbose: bool
) -> pd.DataFrame:
    accepted = []
    for t in summary.timepoint.unique():
        kept = filter_run(
            summary.loc[summary.timepoint == t, :],
            quantile,
            metric,
        )
        accepted.append(kept)
        if verbose:
            print(f"{len(kept)} runs accepted for timepoint {t} with {metric}")
    return pd.concat(accepted)


@dataclass
class AbcThresholds:
    quantile: float
    nb_clones_diff: int
    proportion_runs_to_discard: float


class AbcResults:
    def __init__(
        self,
        accepted: pd.DataFrame,
        quantile: float,
        metric: str,
    ) -> None:
        """The DataFrames with 3 columns:
        - the id of the accepted runs
        - the timepoint (age) at which the run has been accepted
        - the metric used to perform the filtering

        To get a unique list of idx independently of the timepoints, run
        `get_idx`.

        `accepted`: are all the runs that meet the quantile threshold per
        timepoint
        """
        assert quantile <= 1, f"found quantile greater than 1: {quantile}"
        # all the runs that meet the quantile threshold
        self.accepted = accepted
        self.quantile = quantile
        self.metric = metric

    def get_idx(self) -> List[int]:
        return list(self.accepted.idx.unique())

    def abc_filter_on_minimum_timepoints(
        self, minimum_timepoints: int
    ) -> pd.DataFrame:
        timepoints_accepted = self.accepted.timepoint.unique().shape[0]
        assert (
            minimum_timepoints <= timepoints_accepted
        ), "minimum_timepoints greater than the timepoints in the accepted "
        f"runs {minimum_timepoints} vs {timepoints_accepted}"
        runs2keep = (
            self.accepted.groupby("idx").count() >= minimum_timepoints
        ).reset_index()
        runs2keep = [
            int(ele)
            for ele in runs2keep.where(runs2keep.timepoint).dropna().idx
        ]
        return self.accepted.loc[self.accepted.idx.isin(runs2keep), :]


def run_abc(
    summary: pd.DataFrame,
    quantile: float,
    metric: str,
    verbose: bool = False,
) -> AbcResults:
    accepted = filter_per_timepoint(summary, quantile, metric, verbose)
    if verbose:
        print(f"{accepted.shape[0]} runs accepted with metric {metric}")
    return AbcResults(accepted, quantile, metric)


def run_abc_per_single_timepoint(
    summary: pd.DataFrame,
    timepoint: float,
    quantile: float,
    metric: str,
    verbose: bool = False,
) -> AbcResults:
    accepted = filter_run(summary[summary.age == timepoint], quantile, metric)
    if verbose:
        print(f"{accepted.shape[0]} runs accepted with metric {metric}")
    return AbcResults(accepted, quantile, metric)
