import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List
from matplotlib import colors
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
