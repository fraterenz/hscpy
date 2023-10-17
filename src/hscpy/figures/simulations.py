from typing import Tuple
import pandas as pd
from pathlib import Path


def plot_rates(ax, path2dir: Path, idx: int, xlims: Tuple[float, float] | None = None):
    pd.read_csv(path2dir / f"rates/{idx}.csv", header=None).squeeze().plot(
        kind="hist", ax=ax, bins=35
    )
    if xlims:
        ax.set_xlim(xlims[0], xlims[1])
    ax.set_title(f"simulation id: {idx}")
    return ax
