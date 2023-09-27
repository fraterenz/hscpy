import numpy as np
from pathlib import Path


def load_variant_fractions(
    path2dir: Path, nb_timepoints: int, cells: int, runs: int, subclones: int
):
    assert path2dir.is_dir(), "must be dir"
    path2variants = path2dir / f"{cells}cells" / "variant_fraction"
    assert (
        len([x for x in path2variants.iterdir() if x.is_dir()]) == nb_timepoints
    ), "Wrong number of timepoints saved"

    data = []
    found = 0
    for path2snapshot in sorted(
        list(path2variants.iterdir()),
        key=lambda path2name: int(path2name.name),
        reverse=True,
    ):  # need to reverse because rust saves from the last timepoint
        found = 0
        for file in path2snapshot.iterdir():
            with open(file, "r") as f:
                for i, ele in enumerate(f.read().split(",")):
                    # remove wild type clone
                    if i > 0 and ele:
                        data.append(float(ele))
            found += 1
        assert found == runs, f"for timepoint {path2snapshot} found {found} runs instead of {runs}"

    return np.array(data, dtype=float).reshape(
        nb_timepoints, runs, subclones - 1
    )  # timepoints x RUNS x MAX_SUBCLONES - 1 (rm wildtype)
