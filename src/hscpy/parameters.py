import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


# Ben's nature communications, another estimate is 1.2 (Lee-Six et al. 2018)
MUT_PER_DIV = 1.14
CELLS = 100_000
# data obtained from regressing time vs mut burden for donors without any
# expanded clone (i.e.  {"CB002", "KX001", "SX001"})
SLOPE = 14.35
# the duration of the exp. growing phase in years
EXP_TIME = 5 / 12
# time at birth
TIME_AT_BIRTH = 9 / 12


def tau_exp(cells: int):
    return EXP_TIME / np.log(cells)


def m_background(tau) -> float:
    return SLOPE - MUT_PER_DIV / tau


def m_background_exp(mean, time) -> float:
    a = 2 * np.log((CELLS + 1) / 2)
    return (mean - MUT_PER_DIV * a) / time


def compute_m_background_exp() -> float:
    # data comes from Mitchell et al. 2022: the mean of the single-cell
    # mutational burden for the newborns is computed from the genotype matrix
    # MutMatrix.csv
    return m_background_exp(np.mean([52.33, 50.47]), TIME_AT_BIRTH)


def compute_std_per_division_from_std_per_year(
    std_per_year: float, tau: float
) -> float:
    return std_per_year * tau


def compute_s_per_division_from_s_per_year(
    s_per_year: float, tau: float
) -> float:
    return s_per_year * tau


class Parameters:
    def __init__(
        self,
        path: Path,
        sample: int,
        cells: int,
        tau: float,
        mu: float,
        s: float,
        std: float,
        idx: int,
    ):
        self.sample = sample
        self.path = path
        self.cells = cells
        self.tau = tau
        self.mu = mu
        self.s = s
        self.std = std
        self.idx = idx

    def into_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def stringify(self, some_params: Set[str]) -> str:
        return ", ".join(
            [
                f"{k}={v}"
                for k, v in self.into_dict().items()
                if k in some_params
            ]
        )


class ParametersFile:
    def __init__(
        self,
        cells: int,
        tau: float,
        mu: float,
        mean: float,
        std: float,
        idx: int,
    ):
        self.cells = int(cells)
        self.tau = tau
        self.mu = mu
        self.s = mean
        self.std = std
        self.idx = int(idx)

    def into_dict(self) -> Dict[str, Any]:
        return self.__dict__


def parameters_from_path(path: Path) -> Parameters:
    """Assume something like
    test1/20cells/sfs/0dot0years/13dot26541mu0_0dot034741633mean_0dot013301114std_1tau_20cells_270idx.json
    """
    parts = path.parts
    match_sample = re.compile(r"^(\d+)(cells)$", re.IGNORECASE)
    sample = 0
    for part in parts:
        matched = match_sample.search(part)
        if matched:
            sample = int(matched.group(1))
    assert sample > 0, f"cannot find a value for sample from {path}"

    params_file = parse_filename_into_parameters(path)
    return Parameters(path, sample, **params_file.__dict__)


def parse_filename_into_parameters(filename: Path) -> ParametersFile:
    match_nb = re.compile(r"(\d+\.?\d*)([a-z]+)", re.IGNORECASE)
    filename_str = filename.stem
    filename_str = filename_str.replace("dot", ".").split("_")

    my_dict = dict()
    for ele in filename_str:
        matched = match_nb.search(ele)
        if matched:
            my_dict[matched.group(2)] = float(matched.group(1))
        else:
            raise ValueError(
                f"could not parse the filename into parameters {filename}"
            )
    return ParametersFile(**my_dict)


def params_into_dataframe(params: List[Parameters]) -> pd.DataFrame:
    df = pd.DataFrame.from_records([param.into_dict() for param in params])
    df.idx = df.idx.astype(int)
    df.cells = df.cells.astype(int)
    df["sample"] = df["sample"].astype(int)
    df.mu = df.mu.astype(int)
    return df


def params_files_into_dataframe(params: List[ParametersFile]) -> pd.DataFrame:
    df = pd.DataFrame.from_records([param.into_dict() for param in params])
    df.idx = df.idx.astype(int)
    df.cells = df.cells.astype(int)
    df.mu = df.mu.astype(int)
    return df


def is_filtered_sim(
    sim: Path, mu: float, mean: float, std: float, tau: float
) -> Tuple[bool, ParametersFile]:
    params = parse_filename_into_parameters(sim)
    my_dict = params.into_dict()
    return (
        (
            abs(my_dict["s"] - mean) <= 0.02
            and abs(my_dict["mu"] - mu) <= 0.1
            and abs(my_dict["std"] - std) <= 0.02
            and abs(my_dict["tau"] - tau) <= 0.1
        ),
        params,
    )


def get_params_filtered_sim(
    path2dir: Path, mu: float, mean: float, std: float, tau: float
) -> List[ParametersFile]:
    params = list()
    for path in path2dir.iterdir():
        for sim in path.glob("*.json"):
            is_ok, param = is_filtered_sim(sim, mu, mean, std, tau)
            if is_ok:
                params.append(param)
        break
    return params


def filter_simulations(
    path2dir: Path,
    mu: float,
    mean: float,
    std: float,
    tau: float,
) -> pd.DataFrame:
    return params_files_into_dataframe(
        get_params_filtered_sim(path2dir, mu, mean, std, tau)
    )
