# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for the local prior-run review viewer."""

from __future__ import annotations

import argparse
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a local review viewer for a KIMODO prior run folder.")
    parser.add_argument("--run-folder", required=True, help="Path to a prior_run output folder.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the local viewer.")
    parser.add_argument("--port", type=int, default=7861, help="Port for the local viewer.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from kimodo.demo.prior_viewer import launch_prior_viewer

    viewer = launch_prior_viewer(args.run_folder, host=args.host, port=args.port)
    viewer.run()


if __name__ == "__main__":
    main()
