from dataclasses import dataclass
from typing import NewType, Tuple


AgeSims = NewType("AgeSims", float)


@dataclass
class PlotOptions:
    figsize: Tuple[float, float]
    extension: str
    save: bool
