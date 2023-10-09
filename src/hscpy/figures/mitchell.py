"""Load patient's data"""
import pandas as pd
from pathlib import Path
from typing import NewType, Tuple


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


def load_and_process_mitchell(path2sims: Path) -> Mitchell:
    assert (
        path2sims.name == "Summary_cut.csv"
    ), f"wrong arg `path2sims` {path2sims}, should be the path to the file `Summary_cut.csv`"
    summary = pd.read_csv(path2sims, index_col=0)
    summary.cell_type = summary.cell_type.astype("category")
    summary.sample_type = summary.sample_type.astype("category")
    summary.sort_values(by="age", inplace=True)
    summary.reset_index(inplace=True)

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
