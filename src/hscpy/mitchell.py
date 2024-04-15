"""Load patient's data"""

from pathlib import Path
from typing import NewType, Tuple

from futils import snapshot
import pandas as pd


Mitchell = NewType("Mitchell", pd.DataFrame)


def donors() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {"name": "CB001", "age": 0, "cells": 216, "clones": 0},
            {"name": "CB002", "age": 0, "cells": 390, "clones": 0},
            {"name": "KX001", "age": 29, "cells": 407, "clones": 0},
            {"name": "KX002", "age": 38, "cells": 380, "clones": 1},
            {"name": "SX001", "age": 48, "cells": 362, "clones": 0},
            {"name": "AX001", "age": 63, "cells": 361, "clones": 1},
            {"name": "KX008", "age": 76, "cells": 367, "clones": 12},
            {"name": "KX004", "age": 77, "cells": 451, "clones": 15},
            {"name": "KX003", "age": 81, "cells": 328, "clones": 13},
        ]
    )


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


def filter_mutations(
    m_matrix: pd.DataFrame, m_type: pd.DataFrame
) -> pd.DataFrame:
    return m_matrix.iloc[m_type[m_type == "SNV"].dropna().index, :]


def load_and_process_mitchell(
    path2sims: Path, drop_donor_KX007: bool
) -> Mitchell:
    assert (
        path2sims.name == "Summary_cut.csv"
    ), f"wrong arg `path2sims` {path2sims}, expected path to file "
    "`Summary_cut.csv`"
    summary = pd.read_csv(path2sims, index_col=0)
    summary.cell_type = summary.cell_type.astype("category")
    summary.sample_type = summary.sample_type.astype("category")
    summary.sort_values(by="age", inplace=True)
    summary.reset_index(inplace=True)
    # drop this donor because they have updated it twice
    summary.drop(
        index=summary[summary.donor_id == "KX007"].index, inplace=True
    )

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


def donor_mut_matrix(
    name: str, path2mitchell: Path, remove_indels: bool
) -> Tuple[str, pd.DataFrame, int]:
    """Load the genotype matrix for donor `name`.

    Each column represents a cell and each row represents a mutation.
    """
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

    cells = filtered_matrix.shape[1]
    return name, filtered_matrix, cells


def sfs_donor_mitchell(
    name: str, age: int, path2mitchell: Path, remove_indels: bool
) -> Tuple[str, int, int, snapshot.Histogram]:
    _, filtered_matrix, cells = donor_mut_matrix(
        name, path2mitchell, remove_indels
    )
    sfs_donor = filtered_matrix.sum(axis=1).value_counts()
    sfs_donor.drop(index=sfs_donor[sfs_donor.index == 0].index, inplace=True)
    x_sfs = sfs_donor.index.to_numpy(dtype=int)
    my_sfs = snapshot.histogram_from_dict(
        {x: y for x, y in zip(x_sfs, sfs_donor.to_numpy())}
    )
    return (name, age, cells, my_sfs)


def burden_donor_mitchell(
    name: str, age: int, path2mitchell: Path, remove_indels: bool
) -> Tuple[str, int, int, snapshot.Histogram]:
    _, filtered_matrix, cells = donor_mut_matrix(
        name, path2mitchell, remove_indels
    )
    burden_donor = filtered_matrix.sum(axis=0).value_counts()
    x_burden, burden_donor = (
        burden_donor.index.to_numpy(),
        burden_donor.to_numpy(),
    )
    my_burden = snapshot.histogram_from_dict(
        {x: y for x, y in zip(x_burden, burden_donor)}
    )
    return (name, age, cells, my_burden)
