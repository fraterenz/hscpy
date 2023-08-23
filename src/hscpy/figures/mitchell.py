"""Load patient's data"""
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_patient(
    patient: str, path2matrix: Path, path2type: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # f"../mutMatrix{patient}.csv"
    # f"../mutType{patient}.csv"
    mut_matrix = pd.read_csv(path2matrix, index_col=0)
    mut_type = pd.read_csv(path2type, usecols=[1], dtype="category")
    mut_matrix = mut_matrix.applymap(int)  # map 0.5 to 0
    mut_matrix = pd.DataFrame(mut_matrix, dtype=int)
    return mut_matrix, mut_type


def filter_mutations(m_matrix: pd.DataFrame, m_type: pd.DataFrame) -> pd.DataFrame:
    return m_matrix
    # return m_matrix.iloc[m_type[m_type == "SNV"].dropna().index, :]
