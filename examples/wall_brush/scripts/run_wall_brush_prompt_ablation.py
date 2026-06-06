#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

from create_wall_brush_task_spec import build_rows


ROOT = Path(__file__).resolve().parents[1]


PROMPT_VARIANTS: dict[str, list[str]] = {
    "baseline_level_full_width": [
        "A person stands still close to a wall and raises the right hand toward a small patch on the wall, preparing to wipe the wall with the right hand.",
        "A person stands still facing a wall with the right hand touching the left edge of a small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from left to right.",
        "A person stands still facing a wall with the right hand touching the right edge of the same small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from right to left slightly lower on the wall.",
        "A person stands still facing a wall with the right hand touching the left edge of the same small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from left to right slightly lower on the wall.",
        "A person stands still facing a wall with the right hand near the wall, gently lowering the right hand down beside the right thigh and relaxing the arm.",
    ],
    "simple_facing_wall": [
        "A person stands still facing a small wall patch and raises the right hand toward it.",
        "A person stands still facing the wall and slides the right hand in a straight horizontal stroke from left to right on the wall.",
        "A person stands still facing the wall and slides the right hand in a straight horizontal stroke from right to left on the wall.",
        "A person stands still facing the wall and slides the right hand in a straight horizontal stroke from left to right on the wall.",
        "A person remains facing the wall and lowers the right hand down beside the torso.",
    ],
    "feet_planted": [
        "A person keeps both feet planted, faces a small wall patch, and raises the right hand toward it.",
        "A person keeps both feet planted and faces the wall while sliding the right hand in a straight horizontal stroke from left to right on the wall.",
        "A person keeps both feet planted and faces the wall while sliding the right hand in a straight horizontal stroke from right to left on the wall.",
        "A person keeps both feet planted and faces the wall while sliding the right hand in a straight horizontal stroke from left to right on the wall.",
        "A person keeps both feet planted and the torso facing the wall while smoothly lowering the right hand beside the torso.",
    ],
    "seed_upright_style": [
        "A person is standing in an upright stance in front of a wall.",
        "A person standing in an upright stance leans forward and moves the right hand across the wall from left to right.",
        "A person leaning forward moves the right hand across the wall from right to left.",
        "A person leaning forward moves the right hand across the wall from left to right.",
        "A person leaning forward lowers the right hand from the wall.",
    ],
    "outside_surface": [
        "A person stands still and raises the right hand toward the outside surface of a small wall patch.",
        "A person stands still and slides the right palm flat on the outside surface of a wall from left to right.",
        "A person stands still and slides the right palm flat on the outside surface of a wall from right to left.",
        "A person stands still and keeps the right hand on the outside surface while making three straight horizontal brush strokes.",
        "A person stands still and lowers the right hand away from the outside surface of the wall.",
    ],
    "left_arm_relaxed": [
        "A person stands balanced in place in front of a small wall patch, with the left arm relaxed by the side and the right hand ready to brush.",
        "A person keeps the left arm relaxed by the side while the right hand brushes one straight horizontal stroke from left to right on a wall.",
        "A person keeps the left arm relaxed by the side while the right hand brushes one straight horizontal stroke from right to left on a wall.",
        "A person keeps the left arm relaxed by the side while the right hand brushes one straight horizontal stroke from left to right on a wall.",
        "A person relaxes the left arm by the side and smoothly lowers the right hand after brushing.",
    ],
    "natural_torso_passive_left": [
        "A person stands balanced in front of a small wall patch, reaching forward with the right hand while the left arm stays relaxed as a passive counterbalance.",
        "A person uses small natural torso and shoulder motion while the right hand slides in a straight horizontal stroke from left to right on a wall, with the left arm passive.",
        "A person uses small natural torso and shoulder motion while the right hand slides in a straight horizontal stroke from right to left on a wall, with the left arm passive.",
        "A person uses small natural torso and shoulder motion while the right hand slides in a straight horizontal stroke from left to right on a wall, with the left arm passive.",
        "A person smoothly lowers the right hand after brushing, keeping a relaxed balanced stance and passive left arm.",
    ],
    "hold_near_wall_finish": [
        "A person stands still close to a wall and raises the right hand toward a small patch on the wall, preparing to wipe the wall with the right hand.",
        "A person stands still facing a wall with the right hand touching the left edge of a small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from left to right.",
        "A person stands still facing a wall with the right hand touching the right edge of the same small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from right to left slightly lower on the wall.",
        "A person stands still facing a wall with the right hand touching the left edge of the same small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from left to right slightly lower on the wall.",
        "A person remains standing close to the wall with the right hand near the wall after the last wiping stroke.",
    ],
}


SUMMARY_KEYS = [
    "variant_name",
    "filtered",
    "motion_path",
    "composite_score",
    "turn_score",
    "natural_score",
    "stroke_score",
    "tail_root_yaw_delta",
    "tail_root_yaw_path",
    "tail_root_displacement",
    "left_arm_noise",
    "max_left_arm_velocity",
    "transition_boundary_step",
    "stroke_line_mean_error",
    "stroke_line_max_error",
    "avg_x_progress",
    "avg_x_coverage",
    "dead_stroke_count",
    "row_y_range_mean",
    "row_y_curvature_mean",
    "wall_penetration",
    "final_right_hand_distance_to_neutral",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def command_line(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def run(parts: list[str | Path], execute: bool) -> None:
    print(command_line(parts), flush=True)
    if execute:
        subprocess.run([str(part) for part in parts], cwd=ROOT, check=True)


def hand_prompts(prompts: list[str], active_hand: str) -> list[str]:
    if active_hand == "right":
        return prompts
    swapped = []
    replacements = {
        "right hand": "__ACTIVE_HAND__",
        "right palm": "__ACTIVE_PALM__",
        "right arm": "__ACTIVE_ARM__",
        "right thigh": "__ACTIVE_THIGH__",
        "right shoulder": "__ACTIVE_SHOULDER__",
        "left arm": "__PASSIVE_ARM__",
        "left hand": "__PASSIVE_HAND__",
    }
    restore = {
        "__ACTIVE_HAND__": "left hand",
        "__ACTIVE_PALM__": "left palm",
        "__ACTIVE_ARM__": "left arm",
        "__ACTIVE_THIGH__": "left thigh",
        "__ACTIVE_SHOULDER__": "left shoulder",
        "__PASSIVE_ARM__": "right arm",
        "__PASSIVE_HAND__": "right hand",
    }
    for prompt in prompts:
        text = prompt
        for old, new in replacements.items():
            text = text.replace(old, new)
        for old, new in restore.items():
            text = text.replace(old, new)
        swapped.append(text)
    return swapped


def write_task_spec(path: Path, variant_name: str, prompts: list[str], args: argparse.Namespace) -> None:
    rows = build_rows(args.center_x, args.width, args.top_y, args.row_gap, args.z_contact, args.points_per_row)
    first = rows[0]["points"][0]
    spec = {
        "task_text": args.task_text,
        "task_type": "wall_brush",
        "active_hand": args.active_hand,
        "generated_from_scratch": True,
        "total_frames": 210,
        "segments": [36, 42, 42, 42, 48],
        "boundaries": [36, 78, 120, 162],
        "prompt_variant": variant_name,
        "prompts": hand_prompts(prompts, args.active_hand),
        "wall_z": args.z_contact,
        "wall_patch": {
            "center_x": args.center_x,
            "top_y": args.top_y,
            "row_gap": args.row_gap,
            "width": args.width,
            "z_contact": args.z_contact,
        },
        "approach": [
            {
                "frame": 26,
                "endpoint": [round(first[0] - 0.06, 6), round(first[1] - 0.04, 6), round(first[2] - 0.06, 6)],
                "wrist": [round(first[0] - 0.06, 6), round(first[1] - 0.06, 6), round(args.z_wrist - 0.06, 6)],
            },
            {
                "frame": 35,
                "endpoint": first,
                "wrist": [first[0], round(first[1] - 0.02, 6), round(args.z_wrist, 6)],
            },
        ],
        "stroke_rows": rows,
        "return_frames": [186, 209],
        "constraint_route": {
            "type": "native_generation_time_position_only",
            "active_hand": args.active_hand,
            "active_hand_endpoint_points_per_row": args.points_per_row,
            "active_wrist_points_per_row": 2,
            "wrist_frames": [46, 76, 88, 118, 130, 160],
            "z_contact": args.z_contact,
            "z_wrist": args.z_wrist,
            "no_rotation_constraints": True,
            "no_active_arm_ik": True,
            "no_fabrik": True,
        },
        "preemphasis": {
            "preset": "flat_B",
            "x_scale": args.x_scale,
            "x_offset": args.x_offset,
            "base_y_offset": args.y_offset,
            "z_offset": args.z_offset,
            "z_wrist": args.z_wrist,
            "use_y_closed_loop": False,
            "y_reference_motion": None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_spec": spec}, indent=2, ensure_ascii=False), encoding="utf-8")


def cache_prompts(args: argparse.Namespace, prompts_by_variant: dict[str, list[str]]) -> None:
    prompts = []
    seen = set()
    for variant_prompts in prompts_by_variant.values():
        for prompt in hand_prompts(variant_prompts, args.active_hand):
            if prompt not in seen:
                prompts.append(prompt)
                seen.add(prompt)
    cmd: list[str | Path] = [args.python, "scripts/remote_cache_kimodo_prompts.py", "--model_name", "kimodo-g1-rp", "--device", args.cache_device]
    for prompt in prompts:
        cmd.extend(["--prompt", prompt])
    run(cmd, args.execute and not args.skip_cache)


def read_best_filtered(summary_path: Path, variant_name: str) -> dict[str, str] | None:
    if not summary_path.exists():
        return None
    rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8")))
    filtered_rows = [row for row in rows if str(row.get("filtered", "")).lower() in {"true", "1"}]
    candidates = filtered_rows or rows
    if not candidates:
        return None
    row = min(candidates, key=lambda item: float(item.get("composite_score") or "inf"))
    row["variant_name"] = variant_name
    return row


def write_ablation_summary(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_KEYS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-sample prompt ablations for KIMODO/G1 wall brushing.")
    parser.add_argument("--output_root", type=Path, default=Path("logs/wall_brush_prompt_ablation"))
    parser.add_argument("--task_text", default="A person brushes a wall")
    parser.add_argument("--active_hand", choices=["right", "left"], default="right")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6023)
    parser.add_argument("--cfg_text", type=float, default=2.4)
    parser.add_argument("--cfg_constraint", type=float, default=4.0)
    parser.add_argument("--diffusion_steps", type=int, default=200)
    parser.add_argument(
        "--disable_return_constraint",
        action="store_true",
        help="Keep stroke constraints but remove the final neutral-hand endpoint constraint so the return segment is prompt-driven.",
    )
    parser.add_argument("--center_x", type=float, default=0.0)
    parser.add_argument("--top_y", type=float, default=0.92)
    parser.add_argument("--row_gap", type=float, default=0.03)
    parser.add_argument("--width", type=float, default=0.24)
    parser.add_argument("--z_contact", type=float, default=0.32)
    parser.add_argument("--z_wrist", type=float, default=0.24)
    parser.add_argument("--points_per_row", type=int, default=6)
    parser.add_argument("--x_scale", type=float, default=1.9)
    parser.add_argument("--x_offset", type=float, default=0.04)
    parser.add_argument("--y_offset", type=float, default=-0.07)
    parser.add_argument("--z_offset", type=float, default=0.03)
    parser.add_argument("--variants", nargs="*", choices=sorted(PROMPT_VARIANTS), default=sorted(PROMPT_VARIANTS))
    args = parser.parse_args()

    out = (ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    selected = {name: PROMPT_VARIANTS[name] for name in args.variants}
    cache_prompts(args, selected)

    rows = []
    for variant_name, prompts in selected.items():
        variant_root = out / variant_name
        spec_path = variant_root / "task_spec.json"
        write_task_spec(spec_path, variant_name, prompts, args)

        v34_dir = variant_root / "v34_flat_native"
        generation_cmd: list[str | Path] = [
            args.python,
            "scripts/remote_wall_brush_multiprompt_constraint_v34_flat.py",
            "--output_dir",
            rel(v34_dir),
            "--seed",
            str(args.seed),
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
            rel(spec_path),
            "--active_hand",
            args.active_hand,
            "--disable_y_closed_loop",
            "--cfg_text",
            str(args.cfg_text),
            "--cfg_constraint",
            str(args.cfg_constraint),
            "--diffusion_steps",
            str(args.diffusion_steps),
        ]
        if args.disable_return_constraint:
            generation_cmd.append("--disable_return_constraint")
        run(generation_cmd, args.execute)
        run(
            [
                args.python,
                "scripts/score_wall_brush_mp_constraint_native.py",
                "--root",
                rel(v34_dir),
                "--make_filtered",
                "--radius",
                "10",
                "--kernel_size",
                "13",
                "--sigma",
                "3.0",
                "--strength",
                "0.85",
                "--passes",
                "3",
                "--constrained_right_arm_strength",
                "0.05",
            ],
            args.execute,
        )
        if args.execute:
            row = read_best_filtered(v34_dir / "summary.csv", variant_name)
            if row:
                rows.append(row)

    if args.execute:
        rows = sorted(rows, key=lambda row: float(row.get("composite_score") or "inf"))
        summary_path = out / "prompt_ablation_summary.csv"
        write_ablation_summary(summary_path, rows)
        (out / "prompt_variants.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
        print(summary_path)
        for row in rows:
            print(
                json.dumps(
                    {
                        "variant_name": row["variant_name"],
                        "composite_score": row.get("composite_score"),
                        "tail_root_yaw_path": row.get("tail_root_yaw_path"),
                        "tail_root_displacement": row.get("tail_root_displacement"),
                        "left_arm_noise": row.get("left_arm_noise"),
                        "stroke_line_mean_error": row.get("stroke_line_mean_error"),
                        "avg_x_coverage": row.get("avg_x_coverage"),
                        "wall_penetration": row.get("wall_penetration"),
                        "motion_path": row.get("motion_path"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
