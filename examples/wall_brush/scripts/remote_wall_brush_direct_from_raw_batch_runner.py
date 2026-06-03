#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from direct_wall_brush_g1_from_raw import (  # noqa: E402
    DIRECT_REFIT_PARAMS,
    DIRECT_VARIANT,
    TARGET_VARIANT,
    compute_basic_metrics,
    process_candidate,
)


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def flatten_metrics(prefix: str, metrics: dict[str, Any], row: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, dict):
            flatten_metrics(f"{prefix}{key}_", value, row)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            row[f"{prefix}{key}"] = value


def raw_paths(remote_root: Path, name: str) -> dict[str, Path]:
    raw_dir = remote_root / "raw" / name
    generation_dir = remote_root / "_generation" / name / "raw_generation"
    return {
        "raw_dir": raw_dir,
        "task_spec": raw_dir / "task_spec.json",
        "recipe": raw_dir / "recipe.json",
        "line_metrics": raw_dir / "line_metrics.json",
        "raw_motion": raw_dir / "motion.npz",
        "raw_metadata": raw_dir / "metadata.json",
        "generation_dir": generation_dir,
        "generated_motion": generation_dir / "sample_00" / "motion.npz",
        "generated_metadata": generation_dir / "metadata.json",
    }


def post_paths(remote_root: Path, name: str) -> dict[str, Path]:
    post_root = remote_root / "postprocessed" / name
    return {
        "target_motion": post_root / TARGET_VARIANT / "motion.npz",
        "target_report": post_root / TARGET_VARIANT / "report.json",
        "final_motion": post_root / DIRECT_VARIANT / "motion.npz",
        "final_report": post_root / DIRECT_VARIANT / "report.json",
    }


def web_generation(
    task_spec: Path,
    output_dir: Path,
    *,
    kimodo_repo: Path,
    python: str,
    device: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": "/root/autodl-tmp/huggingface",
            "HUGGINGFACE_CACHE_DIR": "/root/autodl-tmp/huggingface",
            "TRANSFORMERS_CACHE": "/root/autodl-tmp/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LOCAL_CACHE": "True",
            "PYTHONPATH": str(kimodo_repo),
        }
    )
    cmd = [
        python,
        "-m",
        "kimodo.scripts.web_equivalent_generate",
        "--task",
        str(task_spec),
        "--output",
        str(output_dir),
        "--device",
        device,
        "--use-demo-embedding-cache",
        "--embedding-cache-model-name",
        "kimodo-g1-rp",
        "--embedding-cache-model-name",
        "Kimodo-G1-RP-v1",
        "--no-csv",
    ]
    subprocess.run(cmd, cwd=kimodo_repo, env=env, check=True)


def write_target_files(paths: dict[str, Path], target: dict[str, Any]) -> None:
    write_json(paths["task_spec"], target["task_spec"])
    write_json(paths["recipe"], target["recipe"])
    write_json(paths["line_metrics"], target["line_metrics"])


def raw_generation_step(
    paths: dict[str, Path],
    *,
    kimodo_repo: Path,
    python: str,
    device: str,
    resume: bool,
) -> None:
    if resume and paths["raw_motion"].exists():
        return
    web_generation(paths["task_spec"], paths["generation_dir"], kimodo_repo=kimodo_repo, python=python, device=device)
    if not paths["generated_motion"].exists():
        raise FileNotFoundError(f"Missing generated motion: {paths['generated_motion']}")
    paths["raw_motion"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["generated_motion"], paths["raw_motion"])
    if paths["generated_metadata"].exists():
        shutil.copy2(paths["generated_metadata"], paths["raw_metadata"])


def summarize_record(remote_root: Path, batch: dict[str, Any], target: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    name = target["name"]
    grid = target.get("grid_variant", {})
    raw = raw_paths(remote_root, name)
    post = post_paths(remote_root, name)
    metrics = {
        "raw": compute_basic_metrics(raw["raw_motion"], raw["recipe"]),
        DIRECT_VARIANT: {
            **record["metrics"],
            "report": record["report"],
        },
    }
    return {
        "name": name,
        "recorded_seed": batch.get("seed"),
        "prompt_strategy": "current_best_outside_surface_single_prompt_three_point",
        "prompt_segments": batch.get("prompt_segments", []),
        "constraints": target.get("constraints", []),
        "grid_variant": grid,
        "endpoint_info": target.get("endpoint_info", {}),
        "paths": {
            "task_spec": rel(raw["task_spec"], remote_root),
            "recipe": rel(raw["recipe"], remote_root),
            "line_metrics": rel(raw["line_metrics"], remote_root),
            "raw_motion": rel(raw["raw_motion"], remote_root),
            "target_motion": rel(post["target_motion"], remote_root),
            "target_report": rel(post["target_report"], remote_root),
            "final_motion": rel(post["final_motion"], remote_root),
            "final_report": rel(post["final_report"], remote_root),
        },
        "metrics": metrics,
    }


def write_artifacts(remote_root: Path, batch: dict[str, Any], records: list[dict[str, Any]]) -> None:
    candidates: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    review: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for record in records:
        name = record["name"]
        paths = record["paths"]
        candidates[name] = {
            "name": name,
            "recorded_seed": record["recorded_seed"],
            "prompt_strategy": record["prompt_strategy"],
            "prompt_segments": record["prompt_segments"],
            "constraints": record["constraints"],
            "raw_motion": paths["raw_motion"],
            "raw_recipe": paths["recipe"],
            "task_spec": paths["task_spec"],
            "grid_variant": record["grid_variant"],
            "endpoint_info": record["endpoint_info"],
            "postprocessed": {
                TARGET_VARIANT: {
                    "name": TARGET_VARIANT,
                    "output_motion": paths["target_motion"],
                    "report": paths["target_report"],
                    "note": "Direct target: line stroke plus body-relative default return from raw.",
                },
                DIRECT_VARIANT: {
                    "name": DIRECT_VARIANT,
                    "source_raw_motion": paths["raw_motion"],
                    "target_motion": paths["target_motion"],
                    "output_motion": paths["final_motion"],
                    "report": paths["final_report"],
                    "params": DIRECT_REFIT_PARAMS,
                    "note": "Accepted direct-from-raw G1 hinge refit for the redesigned stable-range 108 batch.",
                },
            },
        }
        metrics[name] = record["metrics"]
        review[name] = {
            "status": "needs_review",
            "notes": "Review raw against accepted direct-from-raw G1 hinge default-return postprocess.",
            "current_best_variant": DIRECT_VARIANT,
        }
        grid = record["grid_variant"]
        row = {
            "variant": name,
            "height_y": grid.get("height_y"),
            "center_x": grid.get("center_x"),
            "width_x": grid.get("width_x"),
            "wall_z": grid.get("wall_z"),
            "constraint_z": grid.get("constraint_z"),
            "left_x": grid.get("left_x"),
            "right_x": grid.get("right_x"),
            "endpoint_abs_x_max": grid.get("endpoint_abs_x_max"),
            "risk_tag": grid.get("risk_tag"),
            "raw_motion": paths["raw_motion"],
            "final_motion": paths["final_motion"],
        }
        flatten_metrics("final_", record["metrics"][DIRECT_VARIANT], row)
        rows.append(row)

    manifest = {
        "schema_version": 1,
        "run_kind": "wall_brush_generalization_review",
        "task_name": batch["task_name"],
        "model": batch.get("model", "kimodo-g1-rp"),
        "duration_frames": batch.get("duration_frames"),
        "candidate_count": len(candidates),
        "grid": batch.get("grid", {}),
        "current_best_pipeline": {
            "raw_generation": "kimodo.scripts.web_equivalent_generate",
            "final_variant": DIRECT_VARIANT,
            "target_variant": TARGET_VARIANT,
            "batch_targets": "batch_targets.json",
            "note": "Direct-from-raw line-stroke + body-default-return target with one G1 right-arm hinge refit.",
        },
        "candidates": candidates,
    }
    task = {
        "task_name": batch["task_name"],
        "model": batch.get("model", "kimodo-g1-rp"),
        "purpose": "Redesigned stable-range 108 wall-brush experiment using direct-from-raw postprocess.",
        "prompt_segments": batch.get("prompt_segments", []),
        "grid": batch.get("grid", {}),
    }
    write_json(remote_root / "manifest.json", manifest)
    write_json(remote_root / "task.json", task)
    write_json(remote_root / "metrics.json", {"schema_version": 1, "candidates": metrics})
    write_json(
        remote_root / "review.json",
        {
            "schema_version": 1,
            "review_statuses": ["needs_review", "accepted", "borderline", "rejected"],
            "candidates": review,
        },
    )
    write_json(remote_root / "summary.json", {"rows": rows})
    write_json(remote_root / "direct_from_raw_summary.json", {"rows": rows})
    if rows:
        fieldnames = list(rows[0].keys())
        for csv_name in ("summary.csv", "direct_from_raw_summary.csv"):
            with (remote_root / csv_name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    log_text = f"""# {batch['task_name']}

Date: 2026-05-26

Status: {len(records)}/{len(batch.get('targets', []))} variants generated.

This run uses a single remote batch target file:

- Batch file: `batch_targets.json`
- Raw generation: KIMODO/G1-RP web-equivalent generation path
- Final postprocess: `{DIRECT_VARIANT}`
- Grid: `{batch.get('grid', {})}`

Review expectation: inspect `raw` versus `{DIRECT_VARIANT}` in the local prior viewer.
"""
    (remote_root / "experiment_log.md").write_text(log_text, encoding="utf-8")


def create_archive(remote_root: Path, archive_path: Path) -> None:
    members = [
        "batch_targets.json",
        "raw",
        "postprocessed",
        "manifest.json",
        "task.json",
        "metrics.json",
        "review.json",
        "summary.json",
        "summary.csv",
        "direct_from_raw_summary.json",
        "direct_from_raw_summary.csv",
        "experiment_log.md",
    ]
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for member in members:
            path = remote_root / member
            if path.exists():
                tar.add(path, arcname=member)


def process_target(index: int, total: int, target: dict[str, Any], args: argparse.Namespace, batch: dict[str, Any]) -> dict[str, Any]:
    name = target["name"]
    paths = raw_paths(args.remote_root, name)
    write_target_files(paths, target)
    log(f"[{index}/{total}] {name}: raw generation")
    raw_generation_step(
        paths,
        kimodo_repo=args.kimodo_repo,
        python=args.python,
        device=args.device,
        resume=args.resume,
    )
    log(f"[{index}/{total}] {name}: direct-from-raw refit")
    params = dict(DIRECT_REFIT_PARAMS)
    params["steps"] = int(args.steps)
    candidate_record = process_candidate(args.remote_root, name, device=args.device, resume=args.resume, params=params)
    log(f"[{index}/{total}] {name}: done")
    return summarize_record(args.remote_root, batch, target, candidate_record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a direct-from-raw wall-brush generalization batch.")
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--remote_root", required=True, type=Path)
    parser.add_argument("--kimodo_repo", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=DIRECT_REFIT_PARAMS["steps"])
    parser.add_argument("--archive", type=Path, default=None)
    args = parser.parse_args()

    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    targets = list(batch.get("targets", []))
    if args.limit is not None:
        targets = targets[: args.limit]

    args.remote_root.mkdir(parents=True, exist_ok=True)
    batch_copy = args.remote_root / "batch_targets.json"
    if args.batch.resolve() != batch_copy.resolve():
        shutil.copy2(args.batch, batch_copy)

    records: list[dict[str, Any]] = []
    total = len(targets)
    log(f"targets={total} device={args.device} steps={args.steps} variant={DIRECT_VARIANT}")
    for index, target in enumerate(targets, start=1):
        records.append(process_target(index, total, target, args, batch))
        write_artifacts(args.remote_root, batch, records)

    write_artifacts(args.remote_root, batch, records)
    if args.archive:
        create_archive(args.remote_root, args.archive)
        log(f"archive: {args.archive}")
    log(f"done: {args.remote_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
