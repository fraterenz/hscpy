from pathlib import Path

from hscpy import parameters

FILE = "0dot5mu_0dot8mean_0dot01std_1tau_100000cells_1000idx.json"
PATH2FILE = Path(f"/data/v1.3.0/368cells/sfs/0dot0years/{FILE}")


def test_parameters_from_path():
    params = parameters.parameters_from_path(PATH2FILE)
    expected = parameters.Parameters(
        cells=100000,
        idx=1000,
        mu=0.5,
        path=PATH2FILE,
        s=0.8,
        sample=368,
        std=0.01,
        tau=1
    )
    assert params.into_dict() == expected.into_dict()


def test_parameters_from_file():
    params = parameters.parse_filename_into_parameters(Path(PATH2FILE.name))
    expected = parameters.ParametersFile(
        cells=100000,
        idx=1000,
        mu=0.5,
        mean=0.8,
        std=0.01,
        tau=1
    )
    assert params.into_dict() == expected.into_dict()
