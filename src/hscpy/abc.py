import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from futils import snapshot
from hscpy.figures import AgeSims

from hscpy.realisation import RealisationSfs, process_sfs


def get_values_weights_from_sfs(
    sfs_: snapshot.Histogram,
) -> Tuple[List[float], List[float]]:
    sfs_processed = process_sfs(
        snapshot.Histogram(sfs_), normalise=False, log_transform=True
    )
    return list(sfs_processed.keys()), list(sfs_processed.values())


def sfs_summary_statistic_wasserstein(
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
    `target` and `sims` must have the same keys, keys being the timepoint considered.
    """
    abc_results = []

    for age_sim, sfs_sim in sims.items():
        v_values, v_weights = get_values_weights_from_sfs(target[age_sim])
        params = sfs_summary_statistic_wasserstein_timepoint(
            sfs_sim, v_values, v_weights, target_name, age_sim
        )

        abc_results.extend(params)

    return pd.DataFrame.from_records(abc_results)


def sfs_summary_statistic_wasserstein_timepoint(
    sims: List[RealisationSfs],
    v_values: List[float],
    v_weights: List[float],
    target_name: str,
    age: int | float,
):
    """
    Return a list of records (list of dict) with the summary statistics and
    other quantities such as the parameters used.
    We compute the SFS from the simulations `sims` and compare that against the
    `v_values`, `v_weights` which are computed from the SFS of the target data.
    """

    all_params = []

    for i, my_sfs in enumerate(sims):
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


def summary_statistic_relative_diff_clones(summary: pd.DataFrame) -> pd.DataFrame:
    summary["rel clones diff"] = np.where(
        summary["clones"] == 0,
        summary["clones diff"],
        summary["clones diff"] / summary["clones"],
    )
    return summary


def filter_run(summary_t: pd.DataFrame, quantile: float, metric: str) -> pd.DataFrame:
    df = summary_t[[metric, "idx", "timepoint"]]
    kept = df.loc[df[metric] <= df[metric].quantile(quantile), ["idx", "timepoint"]]
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
            print(f"{len(kept)} runs accepted for timepoint {t} with metric {metric}")
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

        To get a unique list of idx independently of the timepoints, run `get_idx`.

        `accepted`: are all the runs that meet the quantile threshold per timepoint
        """
        assert quantile <= 1, f"found quantile greater than 1: {quantile}"
        # all the runs that meet the quantile threshold
        self.accepted = accepted
        self.quantile = quantile
        self.metric = metric

    def get_idx(self) -> List[int]:
        return list(self.accepted.idx.unique())

    def abc_filter_on_minimum_timepoints(self, minimum_timepoints: int) -> pd.DataFrame:
        timepoints_accepted = self.accepted.timepoint.unique().shape[0]
        assert (
            minimum_timepoints <= timepoints_accepted
        ), f"minimum_timepoints greater than the timepoints in the accepted runs {minimum_timepoints} vs {timepoints_accepted}"
        runs2keep = (
            self.accepted.groupby("idx").count() >= minimum_timepoints
        ).reset_index()
        runs2keep = [
            int(ele)
            for ele in runs2keep.where(runs2keep.timepoint).dropna().idx.tolist()
        ]
        return self.accepted.loc[self.accepted.idx.isin(runs2keep), :]


def run_abc(
    summary: pd.DataFrame,
    quantile: float,
    metric: str,
    verbose: bool = False,
) -> AbcResults:
    accepted = filter_run(summary, quantile, metric)
    if verbose:
        print(f"{accepted.shape[0]} runs accepted with metric {metric}")
    return AbcResults(accepted, quantile, metric)
