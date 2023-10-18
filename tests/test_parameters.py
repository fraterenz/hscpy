from pathlib import Path
import pytest
from hscpy import parameters

PATH2FILE = Path(
    "/data/v1.3.0/368cells/sfs/0dot0years/5dot649074mu0_0dot000014122614629741292u_0dot040376373mean_0dot034369513std_1b0_200000cells_147380idx.json"
)


def test_parameters_from_path():
    params = parameters.parameters_from_path(PATH2FILE)
    expected = parameters.Parameters(
        b0=1.0,
        cells=200000,
        idx=147380,
        mu=5.649074,
        path=PATH2FILE,
        s=0.040376373,
        sample=368,
        std=0.034369513,
        u=0.000014122614629741292,
    )
    assert params.into_dict() == expected.into_dict()


def test_parameters_from_file():
    params = parameters.parse_filename_into_parameters(Path(PATH2FILE.name))
    expected = parameters.ParametersFile(
        b=1.0,
        cells=200000,
        idx=147380,
        mu=5.649074,
        mean=0.040376373,
        std=0.034369513,
        u=0.000014122614629741292,
    )
    assert params.into_dict() == expected.into_dict()
