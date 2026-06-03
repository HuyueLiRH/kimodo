# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sync a remote prior run folder into a lightweight local review folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from kimodo.demo.prior_run import sync_local_review_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy lightweight prior-run artifacts for local review.")
    parser.add_argument("--source", required=True, help="Source prior run folder.")
    parser.add_argument("--destination", required=True, help="Destination local review folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = sync_local_review_folder(Path(args.source), Path(args.destination))
    print(destination.resolve())


if __name__ == "__main__":
    main()
