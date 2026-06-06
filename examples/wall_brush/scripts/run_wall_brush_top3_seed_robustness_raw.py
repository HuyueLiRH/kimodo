#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run(cmd: list[str | Path], execute: bool) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(printable, flush=True)
    if execute:
        subprocess.run([str(part) for part in cmd], check=True, cwd=ROOT)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(output_root: Path) -> Path:
    rows = []
    for metrics_path in sorted(output_root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        prompts_path = run_dir / "prompts.json"
        if not prompts_path.exists():
            continue
        metrics = load_json(metrics_path)
        prompts = load_json(prompts_path)
        parts = run_dir.relative_to(output_root).parts
        best = metrics.get("best") or {}
        rows.append(
            {
                "variant_name": parts[0] if len(parts) > 0 else "",
                "seed": parts[1].removeprefix("seed_") if len(parts) > 1 and parts[1].startswith("seed_") else "",
                "run_name": metrics.get("run_name", run_dir.name),
                "cfg_text": (metrics.get("cfg_weight") or ["", ""])[0],
                "cfg_constraint": (metrics.get("cfg_weight") or ["", ""])[1],
                "num_denoising_steps": metrics.get("num_denoising_steps", ""),
                "constraint_variant": metrics.get("variant", ""),
                "raw_kimodo_no_postprocess": not bool(metrics.get("post_processing")),
                "validate_mean_keyframe_error_m": best.get("mean_keyframe_error_m", ""),
                "validate_max_keyframe_error_m": best.get("max_keyframe_error_m", ""),
                "validate_max_row_line_error_m": best.get("max_row_line_error_m", ""),
                "validate_mean_wall_contact_error_m": best.get("mean_wall_contact_error_m", ""),
                "motion_path": str(run_dir / "sample_00" / "motion.npz"),
                "prompts": " | ".join(prompts.get("texts", [])),
            }
        )

    summary_path = output_root / "top3_seed_robustness_raw_summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw KIMODO seed robustness check for the top wall-brush prompt candidates.")
    parser.add_argument("--base_task_root", type=Path, default=Path("task_specs/top3_raw_pipelines"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs/wall_brush_prompt_ablation_top3_seed_raw"))
    parser.add_argument("--variants", nargs="*", default=["outside_surface", "left_arm_relaxed", "seed_upright_style"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[7023, 8023, 9023])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--cfg_text", type=float, default=2.4)
    parser.add_argument("--cfg_constraint", type=float, default=4.0)
    parser.add_argument("--diffusion_steps", type=int, default=200)
    args = parser.parse_args()

    base_task_root = args.base_task_root if args.base_task_root.is_absolute() else ROOT / args.base_task_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if not args.skip_generation:
        for variant_name in args.variants:
            task_spec = base_task_root / variant_name / "task_spec.json"
            if not task_spec.exists():
                raise FileNotFoundError(task_spec)
            for seed in args.seeds:
                seed_dir = output_root / variant_name / f"seed_{seed}"
                cmd: list[str | Path] = [
                    args.python,
                    "scripts/remote_wall_brush_multiprompt_constraint_v34_flat.py",
                    "--output_dir",
                    rel(seed_dir),
                    "--seed",
                    str(seed),
                    "--num_samples",
                    "1",
                    "--frame_plan",
                    "210",
                    "--variant",
                    "endpoint_wrist",
                    "--heading_mode",
                    "none",
                    "--preemphasis_preset",
                    "flat_B",
                    "--prompt_variant",
                    "level_full_width",
                    "--num_transition_frames",
                    "5",
                    "--task_spec",
                    rel(task_spec),
                    "--active_hand",
                    "right",
                    "--disable_y_closed_loop",
                    "--disable_return_constraint",
                    "--cfg_text",
                    str(args.cfg_text),
                    "--cfg_constraint",
                    str(args.cfg_constraint),
                    "--diffusion_steps",
                    str(args.diffusion_steps),
                ]
                run(cmd, args.execute)

    if args.execute:
        summary_path = write_summary(output_root)
        run(
            [
                args.python,
                "scripts/create_kimodo_motion_gallery.py",
                "--logs_dir",
                rel(output_root),
                "--output",
                rel(output_root / "top3_seed_robustness_raw_gallery.html"),
                "--direct_generated_only",
            ],
            True,
        )
        print(summary_path)
        print(output_root / "top3_seed_robustness_raw_gallery.html")


if __name__ == "__main__":
    main()
