import sys
import json
from pathlib import Path

from futils import snapshot


def load_histogram(path: Path) -> snapshot.Histogram:
    try:
        hist = snapshot.histogram_from_file(path)
    except json.JSONDecodeError as e:
        print(f"Error in opening {path} {e}")
        sys.exit(1)
    return hist
