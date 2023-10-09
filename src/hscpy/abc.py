import numpy as np
import pandas as pd
from typing import Dict
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
