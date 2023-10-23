import sys
import json
from pathlib import Path

from futils import snapshot, timeserie


def load_histogram(path: Path) -> snapshot.Histogram:
    try:
        hist = snapshot.histogram_from_file(path)
    except json.JSONDecodeError as e:
        print(f"Error in opening {path} {e}")
        sys.exit(1)
    return hist


def load_variant(path: Path) -> timeserie.Timeserie:
    """Load the variants discarding the neutral clone"""
    return timeserie.Timeserie(
        [ele for i, ele in enumerate(timeserie.load_timeserie(path)) if i > 0]
    )


def parse_path2folder_xdoty_years(path2folder: Path) -> int:
    """assume `path2folder` exists and is of the form: `/path/to/data/10dot5years`"""
    assert path2folder.is_dir()
    return int(round(float(path2folder.stem.replace("dot", ".").replace("years", ""))))
