import numpy as np
from futils import snapshot


def prepare_sfs_with_uniformisation_for_test(
    sfs_target: snapshot.Histogram, sfs_sim: snapshot.Histogram
):
    f_obs, f_exp = snapshot.Uniformise.uniformise_histograms(
        [sfs_target, sfs_sim]
    ).make_histograms()
    f_obs, f_exp = np.fromiter(f_obs.values(), dtype=float), np.fromiter(
        f_exp.values(), dtype=float
    )

    # rm state 0 and
    f_obs, f_exp = f_obs[1:], f_exp[1:]

    idx_lower_bound = 0

    # find the first ele that is 0
    idx_obs, idx_exp = np.argmin(f_obs), np.argmin(f_exp)
    idx_upper_bound = min([idx_obs, idx_exp])

    f_obs, f_exp = (
        f_obs[idx_lower_bound:idx_upper_bound],
        f_exp[idx_lower_bound:idx_upper_bound],
    )
    f_obs /= f_obs.sum()
    f_exp /= f_exp.sum()
    assert len(f_obs) == len(f_exp)

    mean_squared_log_error = np.mean(np.power(np.log(f_obs + 1) - np.log(f_exp + 1), 2))
    rmsre = np.mean(np.power((f_obs - f_exp) / f_obs, 2))
    mape = np.mean(np.abs(f_obs - f_exp) / f_obs)

    return (
        f_obs,
        f_exp,
        idx_lower_bound,
        idx_upper_bound,
        mean_squared_log_error,
        rmsre,
        mape,
    )
