from dataclasses import dataclass
from typing import NewType, Tuple


AgeSims = NewType("AgeSims", float)


@dataclass
class Donor:
    age: int
    age_sims: AgeSims
    cells: int
    name: str
    number_mutations: int


@dataclass
class PlotOptions:
    figsize: Tuple[float, float]
    extension: str
    save: bool
