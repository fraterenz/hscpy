from dataclasses import dataclass
from typing import Tuple


@dataclass
class PlotOptions:
    figsize: Tuple[float, float]
    extension: str
    save: bool
