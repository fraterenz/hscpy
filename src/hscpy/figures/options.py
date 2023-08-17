from pathlib import Path
from typing import Tuple
from dataclasses import dataclass


@dataclass
class PlotOptions:
    figsize: Tuple[float, float]
    extension: str
    save: bool


@dataclass
class SimulationOptions:
    runs: int
    cells: int
    path2save: Path
    neutral_rate: float
    nb_timepoints: int
    last_timepoint_years: int
    nb_subclones: int
    s: float
