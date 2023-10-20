"""Load patient's data"""
import pandas as pd
from futils import snapshot
from pathlib import Path
from typing import Dict, List, NewType, Set, Tuple

from hscpy import sfs

Mitchell = NewType("Mitchell", pd.DataFrame)


def load_patient(
    patient: str, path2matrix: Path, path2type: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # f"../mutMatrix{patient}.csv"
    # f"../mutType{patient}.csv"
    mut_matrix = pd.read_csv(path2matrix, index_col=0)
    mut_type = pd.read_csv(path2type, usecols=[1], dtype="category")
    mut_matrix = mut_matrix.map(int)  # map 0.5 to 0
    mut_matrix = pd.DataFrame(mut_matrix, dtype=int)
    return mut_matrix, mut_type


def filter_mutations(m_matrix: pd.DataFrame, m_type: pd.DataFrame) -> pd.DataFrame:
    return m_matrix.iloc[m_type[m_type == "SNV"].dropna().index, :]


def load_and_process_mitchell(path2sims: Path, drop_donor_KX007: bool) -> Mitchell:
    assert (
        path2sims.name == "Summary_cut.csv"
    ), f"wrong arg `path2sims` {path2sims}, should be the path to the file `Summary_cut.csv`"
    summary = pd.read_csv(path2sims, index_col=0)
    summary.cell_type = summary.cell_type.astype("category")
    summary.sample_type = summary.sample_type.astype("category")
    summary.sort_values(by="age", inplace=True)
    summary.reset_index(inplace=True)
    # drop this donor because they have updated it twice
    summary.drop(index=summary[summary.donor_id == "KX007"].index, inplace=True)

    # neglect some duplicated colonies e.g. summary.colony_ID == "11_E07"
    summary = summary.merge(
        summary[["donor_id", "age"]]
        .groupby("donor_id")
        .count()
        .reset_index()
        .rename(columns={"age": "cells"}),
        on="donor_id",
        validate="many_to_one",
        how="left",
    )
    return Mitchell(summary)


def sfs_donor_mitchell(
    name: str, path2mitchell: Path, remove_indels: bool
) -> snapshot.Histogram:
    if remove_indels:
        filtered_matrix = filter_mutations(
            *load_patient(
                name,
                path2mitchell / f"mutMatrix{name}.csv",
                path2mitchell / f"mutType{name}.csv",
            )
        )

    else:
        filtered_matrix = load_patient(
            name,
            path2mitchell / f"mutMatrix{name}.csv",
            path2mitchell / f"mutType{name}.csv",
        )[0]

    sfs_donor = filtered_matrix.sum(axis=1).value_counts()
    sfs_donor.drop(index=sfs_donor[sfs_donor.index == 0].index, inplace=True)
    x_sfs = sfs_donor.index.to_numpy(dtype=int)
    my_sfs = snapshot.Histogram({x: y for x, y in zip(x_sfs, sfs_donor.to_numpy())})
    return my_sfs


def load_all_sfs_by_age(path2dir: Path) -> Dict[float, List[sfs.RealisationSfs]]:
    assert path2dir.is_dir()
    sfs_sims = dict()

    for path in path2dir.iterdir():
        i = 0
        if path.is_dir():
            sfs_sims[parse_path2folder_xdoty_years(path)] = list()
            for i, p in enumerate(path.glob("*.json")):
                sfs_sims[parse_path2folder_xdoty_years(p.parent)].append(
                    sfs.RealisationSfs(p)
                )

            print(f"loaded {i + 1} files from {path}")

    return sfs_sims


def parse_path2folder_xdoty_years(path2folder: Path) -> int:
    """assume `path2folder` exists and is of the form: `/path/to/data/10dot5years`"""
    assert path2folder.is_dir()
    return int(round(float(path2folder.stem.replace("dot", ".").replace("years", ""))))
