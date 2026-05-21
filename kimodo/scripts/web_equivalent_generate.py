# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate raw motions from web-equivalent end-effector task specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from kimodo import load_model
from kimodo.demo.web_equivalent import (
    CachedDemoTextEncoder,
    WebEquivalentTaskSpec,
    build_generation_kwargs,
    load_task_spec,
)
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.tools import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw KIMODO motions with demo/web-equivalent end-effector constraints."
    )
    parser.add_argument("--task", required=True, help="Path to a web-equivalent task spec JSON.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda:0 when available.")
    parser.add_argument(
        "--use-demo-embedding-cache",
        action="store_true",
        help="Use cached demo text embeddings instead of loading the local LLM text encoder.",
    )
    parser.add_argument(
        "--embedding-cache-root",
        default="/root/.cache/kimodo_demo/embeddings",
        help="Root directory for demo text embedding cache.",
    )
    parser.add_argument(
        "--embedding-cache-model-name",
        action="append",
        default=[],
        help="Model cache directory name to search. Can be repeated.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export.")
    return parser.parse_args()


def run_generation_with_model(
    model,
    spec: WebEquivalentTaskSpec,
    output_dir: str | Path,
    *,
    save_csv: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(spec.seed))
    kwargs = build_generation_kwargs(model, spec)
    prompts = kwargs.pop("prompts")
    kwargs.pop("generation_prompt")
    segments = kwargs.pop("segments")
    output = model(prompts, segments, **kwargs)

    n_samples = int(output["posed_joints"].shape[0])
    for sample_index in range(n_samples):
        sample_dir = output_dir / f"sample_{sample_index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_kimodo_npz(str(sample_dir / "motion.npz"), _single_sample(output, sample_index, n_samples))

    if save_csv:
        _save_csv_if_available(model, output, output_dir)

    metadata = asdict(spec)
    metadata.update(
        {
            "raw_generation": True,
            "output_dir": str(output_dir),
            "sample_count": n_samples,
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def run_generation_from_task(
    task_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | None = None,
    use_demo_embedding_cache: bool = False,
    embedding_cache_root: str | Path = "/root/.cache/kimodo_demo/embeddings",
    embedding_cache_model_names: list[str] | None = None,
    save_csv: bool = True,
) -> Path:
    spec = load_task_spec(task_path)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    text_encoder = None
    if use_demo_embedding_cache:
        text_encoder = CachedDemoTextEncoder(
            _cache_model_names(spec, embedding_cache_model_names),
            cache_root=embedding_cache_root,
        )

    model_kwargs: dict[str, Any] = {
        "device": device,
        "default_family": "Kimodo",
        "return_resolved_name": True,
    }
    if text_encoder is not None:
        model_kwargs["text_encoder"] = text_encoder
    model, resolved_model = load_model(spec.model, **model_kwargs)

    output_path = run_generation_with_model(model, spec, output_dir, save_csv=save_csv)
    metadata_path = output_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "resolved_model": resolved_model,
            "device": device,
            "used_demo_embedding_cache": use_demo_embedding_cache,
            "embedding_cache_root": str(embedding_cache_root) if use_demo_embedding_cache else None,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def _single_sample(output: dict[str, Any], index: int, n_samples: int) -> dict[str, Any]:
    return {
        key: (value[index] if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == n_samples else value)
        for key, value in output.items()
    }


def _save_csv_if_available(model, output: dict[str, Any], output_dir: Path) -> None:
    try:
        from kimodo.exports.mujoco import MujocoQposConverter
    except Exception as exc:
        print(f"Skipping CSV export because MuJoCo converter is unavailable: {exc}")
        return

    converter = MujocoQposConverter(model.skeleton)
    qpos = converter.dict_to_qpos(output, getattr(model, "device", "cpu"))
    converter.save_csv(qpos, str(output_dir / "motion.csv"))


def _cache_model_names(spec: WebEquivalentTaskSpec, explicit_names: list[str] | None) -> list[str]:
    names = list(explicit_names or [])
    if spec.model not in names:
        names.append(spec.model)
    if spec.model == "kimodo-g1-rp" and "Kimodo-G1-RP-v1" not in names:
        names.append("Kimodo-G1-RP-v1")
    return names


def main() -> None:
    args = parse_args()
    output_dir = run_generation_from_task(
        args.task,
        args.output,
        device=args.device,
        use_demo_embedding_cache=args.use_demo_embedding_cache,
        embedding_cache_root=args.embedding_cache_root,
        embedding_cache_model_names=args.embedding_cache_model_name,
        save_csv=not args.no_csv,
    )
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
