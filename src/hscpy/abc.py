import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy import stats
from futils import snapshot

from hscpy.sfs import RealisationSfs, process_sfs


def get_values_weights_from_sfs(
    sfs_: snapshot.Histogram,
) -> Tuple[List[int], List[int]]:
    sfs_processed = process_sfs(sfs_, normalise=False, log_transform=True)
    return list(sfs_processed.keys()), list(sfs_processed.values())


def sfs_summary_statistic_wasserstein(
    sims: Dict[int | float, List[RealisationSfs]],
    target: Dict[int | float, snapshot.Histogram],
    target_name: str,
):
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
    v_values: List[int],
    v_weights: List[int],
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


def filter_per_timepoint(
    summary: pd.DataFrame, quantile: float, verbose: bool
) -> pd.DataFrame:
    accepted = []
    for t in summary.timepoint.unique():
        df = summary.loc[summary.timepoint == t, ["wasserstein", "idx", "timepoint"]]
        kept = df.loc[
            df.wasserstein < df.wasserstein.quantile(quantile), ["idx", "timepoint"]
        ]
        accepted.append(kept)
        if verbose:
            print(f"{len(kept)} runs accepted for timepoint {t}")
    return pd.concat(accepted)


class AbcResults:
    def __init__(
        self,
        accepted: pd.DataFrame,
        accepted_quantile: pd.DataFrame,
        quantile: float,
        minimum_timepoints: int,
    ) -> None:
        """The DataFrames with two columns:
        - the id of the accepted runs
        - the timepoint (age) at which the run has been accepted

        To get a unique list of idx independently of the timepoints, run `get_idx`.

        `accepted_quantile`: are all the runs that meet the quantile threshold
        `accepted`: are all the runs that meet both the quantile and the
        timepoint threshold
        """
        assert quantile <= 1, f"found quantile greater than 1: {quantile}"
        assert accepted_quantile.shape[0] >= accepted.shape[0]
        # all the runs that meet the quantile threshold
        self.accepted_quantile = accepted_quantile
        # all the runs that meet both the quantile and the timepoint threshold
        self.accepted = accepted
        self.quantile = quantile
        self.minimum_timepoints = minimum_timepoints

    def get_idx(self) -> List[int]:
        return list(self.accepted.idx.unique())


def run_abc(
    summary: pd.DataFrame,
    quantile: float,
    minimum_timepoints: int,
    verbose: bool = False,
) -> AbcResults:
    accepted = filter_per_timepoint(summary, quantile, verbose)
    # keep only the runs that have at least `minimum_timepoints` good runs
    runs2keep = (accepted.groupby("idx").count() >= minimum_timepoints).reset_index()
    runs2keep = [
        int(ele) for ele in runs2keep.where(runs2keep.timepoint).dropna().idx.tolist()
    ]
    return AbcResults(
        accepted[accepted.idx.isin(runs2keep)], accepted, quantile, minimum_timepoints
    )
