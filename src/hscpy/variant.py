
def load_variant_fractions(path2dir: Path):

    assert path2dir.is_dir(), "must be dir"

    path2variants = path2dir / "variant_fraction"

    assert (

        len([x for x in path2variants.iterdir() if x.is_dir()]) == NB_TIMEPOINTS

    ), "Wrong number of timepoints saved"



    data = []

    for path2snapshot in sorted(

        list(path2variants.iterdir()),

        key=lambda path2name: int(path2name.name),

        reverse=True,

    ):  # need to reverse because rust saves from the last timepoint

        for file in path2snapshot.iterdir():

            with open(file, "r") as f:

                # remove wild type clone

                data.extend(

                    [

                        float(ele)

                        for i, ele in enumerate(f.read().split(","))

                        if ele and i > 0

                    ]

                )

    return np.array(data, dtype=float).reshape(

        NB_TIMEPOINTS, RUNS, SUBCLONES - 1

    )  # timepoints x RUNS x MAX_SUBCLONES - 1 (rm wildtype)
