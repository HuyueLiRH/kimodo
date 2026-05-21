# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate repo-persisted prior recipe files and Markdown note drafts."""

from __future__ import annotations

import argparse
from pathlib import Path

from kimodo.demo.prior_run import write_recipe_note_draft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a prior recipe JSON and Markdown note draft.")
    parser.add_argument("--run-folder", required=True, help="Prior run folder containing manifest/review/metrics.")
    parser.add_argument("--candidate", required=True, help="Candidate name to persist.")
    parser.add_argument("--output-root", required=True, help="Repository root or output root for examples/docs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe_path, note_path = write_recipe_note_draft(args.run_folder, args.candidate, args.output_root)
    print(recipe_path.resolve())
    print(note_path.resolve())


if __name__ == "__main__":
    main()
