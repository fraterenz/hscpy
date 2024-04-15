from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from hscpy import load_variant, parse_path2folder_xdoty_years
from hscpy.figures import AgeSims
from hscpy.parameters import parameters_from_path


class RealisationVariantCountsDetected:
    def __init__(self, path: Path, detection_threshold: float) -> None:
        """All the clones that have a frequency greater than 0"""
        assert path.is_file(), f"cannot find file {path}"
        self.parameters = parameters_from_path(path)
        self.variant_counts = len(
            [ele for ele in load_variant(path) if ele > detection_threshold]
        )
        self.detection_threshold = detection_threshold


class RealisationVariantCounts:
    def __init__(self, path: Path) -> None:
        """All the clones that have a frequency greater than 0"""
        assert path.is_file(), f"cannot find file {path}"
        self.parameters = parameters_from_path(path)
        self.variant_counts = len([ele for ele in load_variant(path) if ele])


def _variant_counts_df(
    variants: Union[
        Dict[AgeSims, List[RealisationVariantCounts]],
        Dict[AgeSims, List[RealisationVariantCountsDetected]],
    ]
) -> List[Dict[str, Any]]:
    var = []
    for age, variant in variants.items():
        for v in variant:
            params = v.parameters.into_dict()
            params["age"] = age
            params["counts"] = v.variant_counts
            var.append(params)
    return var


def variant_counts_from_dectected(
    realisation: RealisationVariantCountsDetected,
) -> RealisationVariantCounts:
    return RealisationVariantCounts(realisation.parameters.path)


def variant_counts_detected_df(
    variants: Dict[AgeSims, List[RealisationVariantCountsDetected]]
) -> pd.DataFrame:
    df = pd.DataFrame.from_records(_variant_counts_df(variants))
    return df.rename({"counts": "variant counts detected"}, axis=1)


def load_all_detected_var_counts_by_age(
    path2dir: Path,
    detection_threshold: float,
) -> Dict[float, List[RealisationVariantCountsDetected]]:
    assert path2dir.is_dir()
    var_counts_sims = dict()
    i = 0

    for path in path2dir.iterdir():
        if path.is_dir():
            var_counts_sims[parse_path2folder_xdoty_years(path)] = list()
            for i, p in enumerate(path.glob("*.csv")):
                var_counts_sims[
                    parse_path2folder_xdoty_years(p.parent)
                ].append(
                    RealisationVariantCountsDetected(p, detection_threshold)
                )

            print(f"loaded {i + 1} files from {path}")

    return var_counts_sims


def load_all_var_counts_by_age(
    path2dir: Path,
) -> Dict[float, List[RealisationVariantCounts]]:
    assert path2dir.is_dir()
    return {
        k: list(map(variant_counts_from_dectected, realisations))
        for k, realisations in load_all_detected_var_counts_by_age(
            path2dir, detection_threshold=0.0
        ).items()
    }


def variant_counts_df(
    variants: Dict[AgeSims, List[RealisationVariantCounts]]
) -> pd.DataFrame:
    df = pd.DataFrame.from_records(_variant_counts_df(variants))
    return df.rename({"counts": "variant counts"}, axis=1)


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
    i = 0

    for path in path2dir.iterdir():
        if path.is_dir():
            var_frac_sims[parse_path2folder_xdoty_years(path)] = list()
            for i, p in enumerate(path.glob("*.csv")):
                var_frac_sims[parse_path2folder_xdoty_years(p.parent)].append(
                    RealisationVariantFractions(p)
                )

            print(f"loaded {i + 1} files from {path}")

    return var_frac_sims


def variant_fractions_df(
    variants_frac: Dict[float, List[RealisationVariantFractions]],
    runs2load: Union[int , None] = None,
) -> pd.DataFrame:
    vs = []
    for age, var in variants_frac.items():
        for i, v in enumerate(var):
            if runs2load and i >= runs2load:
                break
            params = v.parameters.into_dict()
            params["age"] = age
            params["variant fraction"] = sum(v.variant_fractions)
            vs.append(params)
    return pd.DataFrame.from_records(vs)
