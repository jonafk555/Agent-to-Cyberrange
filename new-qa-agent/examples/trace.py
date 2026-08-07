"""Print the artifacts created by a completed Cochise-compatible run."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run_directory = args.run_directory.expanduser()
    print(f"run: {run_directory.resolve()}")
    for path in sorted(run_directory.rglob("*")):
        if path.is_file():
            print(path.relative_to(run_directory))


if __name__ == "__main__":
    main()
