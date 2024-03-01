from dataclasses import dataclass
from typing import NewType, Tuple


AgeSims = NewType("AgeSims", float)


class ToCellFrequency:
    def __init__(self, sample_size: int) -> None:
        self.nb_cells = sample_size


@dataclass
class PlotOptions:
    figsize: Tuple[float, float]
    extension: str
    save: bool
