from typing import Dict, List
import numpy as np
from pathlib import Path
from hscpy import load_variant, parse_path2folder_xdoty_years

from hscpy.parameters import parameters_from_path


class RealisationVariantCounts:
    def __init__(self, path: Path) -> None:
        assert path.is_file(), f"cannot find file {path}"
        self.parameters = parameters_from_path(path)
        self.variant_counts = len([ele for ele in load_variant(path) if ele > 0])


def load_all_var_counts_by_age(
    path2dir: Path,
) -> Dict[float, List[RealisationVariantCounts]]:
    assert path2dir.is_dir()
    var_counts_sims = dict()

    for path in path2dir.iterdir():
        i = 0
        if path.is_dir():
            var_counts_sims[parse_path2folder_xdoty_years(path)] = list()
            for i, p in enumerate(path.glob("*.csv")):
                var_counts_sims[parse_path2folder_xdoty_years(p.parent)].append(
                    RealisationVariantCounts(p)
                )

            print(f"loaded {i + 1} files from {path}")

    return var_counts_sims


class RealisationVariantFractions:
    def __init__(self, path: Path) -> None:
        assert path.is_file(), f"cannot find file {path}"
        self.parameters = parameters_from_path(path)
        self.variant_fractions = load_variant(path)


def load_all_var_frac_by_age(
    path2dir: Path,
) -> Dict[float, List[RealisationVariantFractions]]:
    assert path2dir.is_dir()
    var_frac_sims = dict()

    for path in path2dir.iterdir():
        i = 0
        if path.is_dir():
            var_frac_sims[parse_path2folder_xdoty_years(path)] = list()
            for i, p in enumerate(path.glob("*.csv")):
                var_frac_sims[parse_path2folder_xdoty_years(p.parent)].append(
                    RealisationVariantFractions(p)
                )

            print(f"loaded {i + 1} files from {path}")

    return var_frac_sims
