# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run first-stage building-motion prior generation from an executable task spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from kimodo import load_model
from kimodo.demo.prior_run import load_prior_task_spec, run_prior_with_model
from kimodo.demo.web_equivalent import CachedDemoTextEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a task-spec driven KIMODO prior generation folder.")
    parser.add_argument("--task", required=True, help="Path to executable task spec JSON.")
    parser.add_argument("--output", required=True, help="Output prior run folder.")
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


def run_prior_from_task(
    task_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | None = None,
    use_demo_embedding_cache: bool = False,
    embedding_cache_root: str | Path = "/root/.cache/kimodo_demo/embeddings",
    embedding_cache_model_names: list[str] | None = None,
    save_csv: bool = False,
) -> Path:
    task_path = Path(task_path)
    task_source: dict[str, Any] = json.loads(task_path.read_text(encoding="utf-8"))
    spec = load_prior_task_spec(task_path)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    text_encoder = None
    if use_demo_embedding_cache:
        text_encoder = CachedDemoTextEncoder(
            _cache_model_names(spec.model, embedding_cache_model_names),
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
    output_path = run_prior_with_model(model, spec, output_dir, task_source=task_source, save_csv=save_csv)

    metadata_path = output_path / "run_metadata.json"
    metadata = {
        "model": spec.model,
        "resolved_model": resolved_model,
        "device": device,
        "used_demo_embedding_cache": use_demo_embedding_cache,
        "embedding_cache_root": str(embedding_cache_root) if use_demo_embedding_cache else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def _cache_model_names(model_name: str, explicit_names: list[str] | None) -> list[str]:
    names = list(explicit_names or [])
    if model_name not in names:
        names.append(model_name)
    if model_name == "kimodo-g1-rp" and "Kimodo-G1-RP-v1" not in names:
        names.append("Kimodo-G1-RP-v1")
    return names


def main() -> None:
    args = parse_args()
    output_dir = run_prior_from_task(
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
