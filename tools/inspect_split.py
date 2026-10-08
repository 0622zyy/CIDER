#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    with args.path.open("rb") as stream:
        split = pickle.load(stream)
    print({str(key): len(value) for key, value in split.items()})


if __name__ == "__main__":
    main()
