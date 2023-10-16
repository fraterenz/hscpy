import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy import stats
from futils import snapshot

from hscpy.sfs import process_sfs


def heatmap_wasserstein(
    my_sfs: Dict[int, Dict[str, snapshot.Histogram]],
    id1: str,
    id2: str,
    nb_timepoints,
    last_timepoint_years,
    normalise,
    log_transform,
) -> pd.DataFrame:
    my_range = list(range(1, nb_timepoints + 1))[::-1]
    my_stats = list()

    for i in my_range:
        for j in my_range:
            u_values, u_weights = snapshot.cdf_from_histogram(
                process_sfs(my_sfs[i][id1], normalise, log_transform)
            )

            v_values, v_weights = snapshot.cdf_from_histogram(
                process_sfs(my_sfs[j][id2], normalise, log_transform)
            )

            my_stats.append(
                stats.wasserstein_distance(u_values, v_values, u_weights, v_weights)
            )

    my_stats = np.array(my_stats, dtype=float).reshape(nb_timepoints, nb_timepoints)
    my_range_mapped = [
        round(ele)
        for ele in np.linspace(start=0, stop=last_timepoint_years, num=nb_timepoints)
    ]
    my_stats = pd.DataFrame(my_stats, my_range_mapped, my_range_mapped)
    my_stats.rename_axis(f"run id: {id1}", axis=0, inplace=True)
    my_stats.rename_axis(f"run id: {id2}", axis=1, inplace=True)
    return my_stats


class Parameters:
    def __init__(
        self,
        cells: int,
        b: float,
        mu: int,
        u: float,
        mean: float,
        std: float,
        idx: int,
    ):
        self.cells = cells
        self.b0 = b
        self.mu = int(mu)
        self.u = u
        self.s = mean
        self.std = std
        self.idx = int(idx)

    def into_dict(self) -> Dict[str, Any]:
        return self.__dict__


def parse_filename_into_parameters(filename: Path) -> Parameters:
    import re

    match_nb = re.compile(r"(\d+\.?\d*)([a-z]+)", re.IGNORECASE)
    filename_str = filename.stem
    filename_str = filename_str.replace("dot", ".").split("_")
    my_dict = dict()
    for ele in filename_str:
        matched = match_nb.search(ele)
        if matched:
            my_dict[matched.group(2)] = float(matched.group(1))
        else:
            raise ValueError(f"could not parse the filename into parameters {filename}")
    return Parameters(**my_dict)


def params_into_dataframe(params: List[Parameters]) -> pd.DataFrame:
    df = pd.DataFrame.from_records([param.into_dict() for param in params])
    df.idx = df.idx.astype(int)
    df.cells = df.cells.astype(int)
    df.mu = df.mu.astype(int)
    return df


def get_values_weights_from_sfs(
    sfs_: snapshot.Histogram,
) -> Tuple[List[int], List[int]]:
    sfs_processed = process_sfs(sfs_, normalise=False, log_transform=True)
    return list(sfs_processed.keys()), list(sfs_processed.values())


def sfs_summary_statistic_wasserstein(
    sims: Dict[int, Dict[str, snapshot.Histogram]],
    target: Dict[int, snapshot.Histogram],
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

    for timepoint_sim, sfs_sim in sims.items():
        v_values, v_weights = get_values_weights_from_sfs(target[timepoint_sim])
        params = sfs_summary_statistic_wasserstein_timepoint(
            sfs_sim, v_values, v_weights, target_name, timepoint_sim
        )

        abc_results.extend(params)

    return pd.DataFrame.from_records(abc_results)


def sfs_summary_statistic_wasserstein_timepoint(
    sims: Dict[str, snapshot.Histogram],
    v_values: List[int],
    v_weights: List[int],
    target_name: str,
    timepoint: int,
):
    """
    Return a list of records (list of dict) with the summary statistics and
    other quantities such as the parameters used.

    We compute the SFS from the simulations `sims` and compare that against the
    `v_values`, `v_weights` which are computed from the SFS of the target data.

    """

    all_params = []

    for i, (filename, my_sfs) in enumerate(sims.items()):
        u_values, u_weights = get_values_weights_from_sfs(my_sfs)
        params = parse_filename_into_parameters(Path(filename)).into_dict()
        # compute the summary statistic
        params["wasserstein"] = stats.wasserstein_distance(
            u_values, v_values, u_weights, v_weights
        )
        params["donor_name"] = target_name
        params["timepoint"] = timepoint
        params["filename"] = filename
        all_params.append(params)

    return all_params


def run_abc(
    summary: pd.DataFrame, quantile: float, minimum_timepoints: int, max_years: int, min_years: int = 0
) -> List[int]:
    runs2keep = []

    proportions_accepted_per_timepoint = {t: 0 for t in summary.timepoint.unique()}

    for idx in summary.idx.unique():
        tmp = summary.loc[
            (summary.idx == idx)
            & (summary.wasserstein < summary.wasserstein.quantile(quantile))
        ]

        for t in tmp.timepoint.tolist():
            proportions_accepted_per_timepoint[t] += 1

        if tmp.shape[0] >= minimum_timepoints:
            runs2keep.append(idx)

    print(f"found {len(runs2keep)} runs")

    tot_runs, tot_timepoints = (
        summary.idx.unique().shape[0],
        summary.timepoint.unique().shape[0],
    )

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        np.linspace(min_years, max_years, tot_timepoints),
        [ele / tot_runs for ele in proportions_accepted_per_timepoint.values()][::-1],
    )
    ax.set_xlabel("years", fontsize="xx-large")
    ax.set_ylabel(f"proportion of accepted runs (tot: {tot_runs})", fontsize="xx-large")
    ax.tick_params(axis="both", which="both", labelsize=14)
    plt.show()

    return runs2keep

