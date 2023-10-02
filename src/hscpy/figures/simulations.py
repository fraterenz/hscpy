from futils import snapshot
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from hscpy import Donor, get_idx_timepoint_from_age, sfs
from hscpy.burden import load_burden_timepoints
from hscpy.figures import mitchell


@dataclass
class SimulationOptions:
    runs: int
    cells: int
    sample: int
    path2save: Path
    neutral_rate: float
    nb_timepoints: int
    last_timepoint_years: int
    nb_subclones: int
    s: float


def donors_from_mitchell(
    mitchell_data: pd.DataFrame,
    sim_options: SimulationOptions,
    verbosity: bool = False,
) -> List[Donor]:
    donors = list()
    for row in mitchell_data[["donor_id", "age", "cells"]].drop_duplicates().iterrows():
        donor_id, age, cells = row[1].donor_id, row[1].age, row[1].cells
        idx_timepoint, closest_age = get_idx_timepoint_from_age(
            age=age,
            years=sim_options.last_timepoint_years,
            nb_timepoints=sim_options.nb_timepoints,
            verbosity=verbosity,
        )
        if verbosity:
            print(
                f"\ncreating donor {donor_id} with age {age} and closest age {closest_age} using {sim_options.nb_subclones} timepoints and year {sim_options.last_timepoint_years}"
            )
        donors.append(
            Donor(
                age=age,
                name=donor_id,
                cells=cells,
                closest_age=closest_age,
                id_timepoint=idx_timepoint,
            )
        )
    return donors


class Simulations:
    def __init__(self, summary: pd.DataFrame, sim_options: SimulationOptions):
        self.my_burden: Dict[
            int, Dict[str, snapshot.Histogram]
        ] = load_burden_timepoints(
            sim_options.path2save,
            sim_options.nb_timepoints,
            sim_options.sample,
            sim_options.runs,
        )
        self.my_sfs: Dict[int, Dict[str, snapshot.Histogram]] = sfs.load_sfs_timepoints(
            sim_options.path2save,
            sim_options.nb_timepoints,
            sim_options.sample,
            sim_options.runs,
        )
        self.donors: List[Donor] = donors_from_mitchell(summary, sim_options)
        self.sim_options: SimulationOptions = sim_options
        self.mean_mutations = (
            summary[["donor_id", "number_mutations"]]
            .groupby("donor_id")
            .mean()
            .reset_index()
            .merge(
                summary[["donor_id", "age"]].drop_duplicates(),
                on="donor_id",
                how="inner",
                validate="one_to_one",
            )
            .sort_values(by="age")
        )

        # fit only the neutral ones (without clonal exp)
        x = self.mean_mutations[self.mean_mutations.age < 70].age.to_numpy()
        y = self.mean_mutations[
            self.mean_mutations.age < 70
        ].number_mutations.to_numpy()
        A = np.vstack([x, np.ones(len(x))]).T
        self.m, self.c = np.linalg.lstsq(A, y, rcond=None)[0]
