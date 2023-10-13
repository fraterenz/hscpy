from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def lims(mask: pd.DataFrame, col: str) -> Tuple[float, float]:
    min_, max_ = mask[col].min(), mask[col].max()
    shift = 0.1
    return min_ - min_ * shift, max_ + max_ * shift


def plot_prior(prior: pd.Series, ax, **kwargs):
    ax = sns.histplot(prior, ax=ax, **kwargs)
    return ax


def plot_results(
    selected: pd.DataFrame, pair2plot: List[str], lim1, lim2, kwargs1, kwargs2
):
    assert len(pair2plot) == 2

    g = sns.JointGrid(ratio=2, marginal_ticks=True)
    x, y = selected[pair2plot[0]], selected[pair2plot[1]]
    ax = sns.scatterplot(
        x=x,
        y=y,
        marker="o",
        s=100,
        ax=g.ax_joint,
    )
    ax.tick_params(axis="both", which="both", labelsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize="xx-large")
    ax.set_xlabel(ax.get_xlabel(), fontsize="xx-large")

    ax = sns.histplot(x=x, fill=True, linewidth=2, ax=g.ax_marg_x, **kwargs1)
    ax.set_xlim(*lim1)
    ax.tick_params(axis="both", which="both", labelsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize="xx-large")

    ax = sns.histplot(y=y, fill=True, linewidth=2, ax=g.ax_marg_y, **kwargs2)
    ax.set_ylim(*lim2)
    ax.tick_params(axis="both", which="both", labelsize=14)
    ax.set_xlabel(ax.get_xlabel(), fontsize="xx-large")

    plt.show()
