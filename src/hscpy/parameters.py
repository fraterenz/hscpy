import re
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Set


class Parameters:
    def __init__(
        self,
        path: Path,
        sample: int,
        cells: int,
        b0: float,
        mu: float,
        u: float,
        s: float,
        std: float,
        idx: int,
    ):
        self.sample = sample
        self.path = path
        self.cells = cells
        self.b0 = b0
        self.mu = mu
        self.u = u
        self.s = s
        self.std = std
        self.idx = idx

    def into_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def stringify(self, some_params: Set[str]) -> str:
        return ", ".join(
            [f"{k}={v}" for k, v in self.into_dict().items() if k in some_params]
        )


class ParametersFile:
    def __init__(
        self,
        cells: int,
        b: float,
        mu: float,
        u: float,
        mean: float,
        std: float,
        idx: int,
    ):
        self.cells = int(cells)
        self.b0 = b
        self.mu = mu
        self.u = u
        self.s = mean
        self.std = std
        self.idx = int(idx)

    def into_dict(self) -> Dict[str, Any]:
        return self.__dict__


def parameters_from_path(path: Path) -> Parameters:
    """Assume something like
    test1/20cells/sfs/0dot0years/13dot26541mu0_0dot3158431053161621u_0dot034741633mean_0dot013301114std_1b0_20cells_270idx.json
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
            raise ValueError(f"could not parse the filename into parameters {filename}")
    return ParametersFile(**my_dict)


def params_into_dataframe(params: List[Parameters]) -> pd.DataFrame:
    df = pd.DataFrame.from_records([param.into_dict() for param in params])
    df.idx = df.idx.astype(int)
    df.cells = df.cells.astype(int)
    df["sample"] = df["sample"].astype(int)
    df.mu = df.mu.astype(int)
    return df
