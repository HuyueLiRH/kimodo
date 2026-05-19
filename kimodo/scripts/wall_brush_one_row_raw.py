# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the validated one-row wall-brushing raw KIMODO/G1 motion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from kimodo import load_model
from kimodo.demo.wall_brush import (
    WALL_BRUSH_ONE_ROW_PRESET,
    build_wall_brush_generation_kwargs,
    wall_brush_preset_metadata,
)
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.model.registry import get_model_info
from kimodo.tools import save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the raw one-row wall-brushing motion with native RightHand constraints."
    )
    parser.add_argument("--model", default=WALL_BRUSH_ONE_ROW_PRESET["model"], help="Model short key or repo name.")
    parser.add_argument("--output", default="outputs/wall_brush_one_row_raw", help="Output directory.")
    parser.add_argument("--seed", type=int, default=WALL_BRUSH_ONE_ROW_PRESET["seed"], help="Random seed.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=WALL_BRUSH_ONE_ROW_PRESET["num_samples"],
        help="Number of raw samples to generate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the preset and generation settings without loading the model.",
    )
    return parser.parse_args()


def _single_sample(output: dict[str, Any], index: int, n_samples: int) -> dict[str, Any]:
    return {
        key: (value[index] if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == n_samples else value)
        for key, value in output.items()
    }


def run_generation(args: argparse.Namespace) -> Path:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, resolved_model = load_model(
        args.model,
        device=device,
        default_family="Kimodo",
        return_resolved_name=True,
    )

    info = get_model_info(resolved_model)
    display_name = info.display_name if info else resolved_model
    print(f"Loaded model: {display_name} ({resolved_model}) on {device}")
    print("Using raw generation: post_processing=False, no filter, no IK, no reranking.")

    seed_everything(args.seed)
    kwargs = build_wall_brush_generation_kwargs(model, num_samples=args.num_samples)
    prompts = kwargs.pop("prompts")
    kwargs.pop("generation_prompt")
    segments = kwargs.pop("segments")
    output = model(prompts, segments, **kwargs)
    n_samples = int(output["posed_joints"].shape[0])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_index in range(n_samples):
        sample_dir = output_dir / f"sample_{sample_index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_kimodo_npz(str(sample_dir / "motion.npz"), _single_sample(output, sample_index, n_samples))

    metadata = wall_brush_preset_metadata()
    metadata.update(
        {
            "seed": args.seed,
            "num_samples": args.num_samples,
            "resolved_model": resolved_model,
            "device": device,
            "raw_generation": True,
            "output_dir": os.fspath(output_dir),
        }
    )
    save_json(str(output_dir / "right_hand_targets.json"), metadata["right_hand_targets"])
    save_json(str(output_dir / "metadata.json"), metadata)
    print(f"Saved {n_samples} sample(s) to {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    if args.dry_run:
        metadata = wall_brush_preset_metadata()
        metadata.update({"seed": args.seed, "num_samples": args.num_samples, "model": args.model})
        print(json.dumps(metadata, indent=2))
        return
    run_generation(args)


if __name__ == "__main__":
    main()
