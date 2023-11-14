from typing import Dict, List, Mapping, Sequence, Set, Tuple, Union
import numpy as np
from scipy import stats
from enum import StrEnum, auto
from pathlib import Path
from dataclasses import dataclass
from futils import snapshot
from hscpy.figures import AgeSims
from hscpy.parameters import parameters_from_path, parse_filename_into_parameters
from hscpy import load_histogram, parse_path2folder_xdoty_years

from enum import Enum, auto


class RealisationKind(Enum):
    BURDEN = auto()
    SFS = auto()


class RealisationSfs:
    def __init__(self, path: Path) -> None:
        assert path.is_file(), f"cannot find SFS file {path}"
        self.parameters = parameters_from_path(path)
        self.sfs = load_histogram(path)


def load_all_sfs_by_age(path2dir: Path, filtering: Set[int] | None = None):
    return load_realisations(path2dir, RealisationKind.SFS, filtering)


def cdf_from_dict(my_dict: Dict[AgeSims, float]) -> Tuple[np.ndarray, np.ndarray]:
    ordered_distr = dict(sorted(my_dict.items()))
    tot = sum(ordered_distr.values())
    probs = np.array(
        np.cumsum([ele / tot for ele in ordered_distr.values()], dtype=float),
        dtype=float,
    )
    return np.array(list(ordered_distr.keys())), probs


def process_sfs(
    my_sfs: snapshot.Histogram, normalise: bool, log_transform: bool
) -> Dict[AgeSims, float]:
    """This modifies the sfs by removing the entry at 0 and log10 transform the
    jcells (keys) and optionally the jmuts (values) i.e. when `log_transform` is
    `True`.

    Normalise means normalise the y axis by dividing all entries by the maximal
    value found on the yaxis.
    """
    my_sfs.pop(0, 0)
    jmuts = list(my_sfs.values())
    if normalise:
        max_ = max(jmuts)
        jmuts = [ele / max_ for ele in jmuts]
    jcells = [np.log10(k) for k in my_sfs.keys()]
    if log_transform:
        jmuts = [np.log10(val) for val in jmuts]
    return {AgeSims(k): float(val) for k, val in zip(jcells, jmuts)}


class Correction(StrEnum):
    ONE_OVER_F = auto()
    ONE_OVER_F_SQUARED = auto()


def compute_frequencies(pop_size: int) -> np.ndarray:
    return np.arange(1, pop_size + 1, step=1, dtype=int)


def compute_sampling_correction(n: int, s: int) -> np.ndarray:
    return np.array(
        [
            stats.binom(s, (v + 1) / n).pmf(np.arange(1, s + 1, step=1))
            for v in range(0, n)
        ],
        dtype=float,
    ).T


@dataclass
class CorrectedVariants:
    correction: Correction
    corrected_variants: np.ndarray
    variant2correct: np.ndarray
    frequencies: np.ndarray


def compute_variants(
    correction: Correction, pop_size: int, sample_size: int
) -> CorrectedVariants:
    """
    _f = range(0,1,length=N+1)
    nThGr_f = (1 ./ _f).^2
    nSample_f = Vector{Float64}(undef, S+1)
    for u in 1:S
        nSample_f[1+u] =
            sum(
                [ nTrue_f[1+v] * pdf(Binomial(S, v/N), u) for v=1:N ]
            )
    end
    """
    frequencies = compute_frequencies(pop_size)
    if correction == Correction.ONE_OVER_F:
        variants2correct = 1 / frequencies
    elif correction == Correction.ONE_OVER_F_SQUARED:
        variants2correct = 1 / frequencies**2
    else:
        raise ValueError

    assert variants2correct.shape[0] == pop_size, f"{variants2correct.shape[0]}"
    corrected = (
        compute_sampling_correction(pop_size, sample_size)[:sample_size, :]
        @ variants2correct
    )
    return CorrectedVariants(correction, corrected, variants2correct, frequencies)


class RealisationBurden:
    def __init__(self, path: Path) -> None:
        assert path.is_file(), f"cannot find burden file {path}"
        self.parameters = parameters_from_path(path)
        self.burden = load_histogram(path)


def load_all_burden_by_age(path2dir: Path, filtering: Set[int] | None = None):
    return load_realisations(path2dir, RealisationKind.BURDEN, filtering)


def compute_mean_variance(
    burden: snapshot.Histogram | snapshot.Distribution,
) -> Tuple[float, float]:
    # cells will be 1 if it's a distribution
    cells = sum(burden.values())
    mean = sum(map(lambda entry: entry[0] * entry[1] / cells, burden.items()))
    variance = sum(
        map(lambda entry: (entry[0] - mean) ** 2 * entry[1] / cells, burden.items())
    )
    return mean, variance


def plot_burden(burden: snapshot.Histogram | snapshot.Distribution, ax, **kwargs):
    # remove empty entries
    cleaned = {k: v for k, v in sorted(burden.items()) if v > 0 }
    
    ax.bar(
        list(cleaned.keys()),
        list(cleaned.values()),
        color=kwargs["color"],
        alpha=kwargs["alpha"],
        width=1,
    )
    ymax = kwargs.get("ymax", max(cleaned.values()))
    mean, var = compute_mean_variance(cleaned)
    ax.axvline(
        x=mean,
        ymin=0,
        ymax=ymax,
        linestyle="--",
        c=kwargs["color"],
        label=f"{kwargs['label']} years, m={mean:.2f}, var={var:.2f}",
    )
    return ax


def average_burden(burdens: List[snapshot.Histogram]):
    # add zeros for values which are not present in all SFS
    burden_uniformised = snapshot.Uniformise.uniformise_histograms(
        [snapshot.Histogram(burden) for burden in burdens]
    )
    jcells = burden_uniformised.create_x_array()
    # take the Nj mutations for all simulations
    avg_burden = burden_uniformised.y
    # compute the average, not pooling
    return jcells, np.mean(avg_burden, axis=0)


def single_cell_mutations_from_burden(burden: snapshot.Histogram) -> np.ndarray:
    """From the SFS, create an array where each entry is the number of
    mutations found in a cell"""
    muts = []
    for jmuts, jcells in burden.items():
        for cell in range(0, jcells):
            muts.append(jmuts)
    return np.array(muts, dtype=int)


def pooled_burden(burdens: List[snapshot.Histogram]) -> snapshot.Distribution:
    return snapshot.Uniformise.pooled_distribution(burdens)


def load_realisations(
    path2dir: Path, realisation: RealisationKind, filtering: Set[int] | None
) -> Mapping[AgeSims, Sequence[Union[RealisationBurden, RealisationSfs]]]:
    assert path2dir.is_dir()
    realisations = dict()

    for path in path2dir.iterdir():
        i = 0
        if path.is_dir():
            realisations[parse_path2folder_xdoty_years(path)] = list()
            for p in path.glob("*.json"):
                # avoid loading sim if not needed
                if filtering:
                    idx = parse_filename_into_parameters(p).idx
                    if idx not in filtering:
                        continue
                    else:
                        i += 1
                else:
                    i += 1
                if realisation == RealisationKind.BURDEN:
                    realisations[parse_path2folder_xdoty_years(p.parent)].append(
                        RealisationBurden(p)
                    )
                elif realisation == RealisationKind.SFS:
                    realisations[parse_path2folder_xdoty_years(p.parent)].append(
                        RealisationSfs(p)
                    )
                else:
                    raise ValueError(
                        f"realisation {realisation} not recognized. Must be `SFS` or `BURDEN`"
                    )

            print(f"loaded {i + 1} files from {path}")

    return dict(sorted(realisations.items()))
