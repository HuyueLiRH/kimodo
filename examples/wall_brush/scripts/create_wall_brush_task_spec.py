from __future__ import annotations

import argparse
import json
from pathlib import Path


RIGHT_PROMPTS = [
    "A person stands still close to a wall and raises the right hand toward a small patch on the wall, preparing to wipe the wall with the right hand.",
    "A person stands still facing a wall with the right hand touching the left edge of a small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from left to right.",
    "A person stands still facing a wall with the right hand touching the right edge of the same small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from right to left slightly lower on the wall.",
    "A person stands still facing a wall with the right hand touching the left edge of the same small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from left to right slightly lower on the wall.",
    "A person stands still facing a wall with the right hand near the wall, gently lowering the right hand down beside the right thigh and relaxing the arm.",
]


LEFT_PROMPTS = [
    prompt.replace("right hand", "left hand").replace("right thigh", "left thigh")
    for prompt in RIGHT_PROMPTS
]


def classify_task(text: str) -> str:
    lower = text.lower()
    wall_words = ("wall", "墙", "墙面")
    brush_words = ("brush", "paint", "wipe", "scrub", "刷", "平刷", "刷墙", "擦")
    if any(word in lower for word in wall_words) and any(word in lower for word in brush_words):
        return "wall_brush"
    raise ValueError(f"Unsupported task text for this pipeline: {text!r}")


def linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    step = (end - start) / float(count - 1)
    return [float(start + step * i) for i in range(count)]


def build_rows(center_x: float, width: float, top_y: float, row_gap: float, z_contact: float, points_per_row: int) -> list[dict]:
    frame_rows = [
        [46, 52, 58, 64, 70, 76],
        [88, 94, 100, 106, 112, 118],
        [130, 136, 142, 148, 154, 160],
    ]
    if points_per_row != 6:
        raise ValueError("The current default timing expects points_per_row=6.")
    left = center_x - width / 2.0
    right = center_x + width / 2.0
    rows = []
    for idx, frames in enumerate(frame_rows):
        y = top_y - row_gap * idx
        if idx % 2 == 0:
            xs = linspace(left, right, points_per_row)
        else:
            xs = linspace(right, left, points_per_row)
        rows.append(
            {
                "row": idx + 1,
                "frames": frames,
                "points": [[round(x, 6), round(y, 6), round(z_contact, 6)] for x in xs],
            }
        )
    return rows


def create_spec(args) -> dict:
    task_type = classify_task(args.task_text)
    rows = build_rows(args.center_x, args.width, args.top_y, args.row_gap, args.z_contact, args.points_per_row)
    first = rows[0]["points"][0]
    active_hand = str(args.active_hand).lower()
    prompts = LEFT_PROMPTS if active_hand == "left" else RIGHT_PROMPTS
    spec = {
        "task_text": args.task_text,
        "task_type": task_type,
        "active_hand": active_hand,
        "generated_from_scratch": True,
        "total_frames": 210,
        "segments": [36, 42, 42, 42, 48],
        "boundaries": [36, 78, 120, 162],
        "prompt_variant": "level_full_width",
        "prompts": prompts,
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
            "active_hand": active_hand,
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
    return {"task_spec": spec}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_text", default="一个人在刷墙")
    parser.add_argument("--task_text_file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--center_x", type=float, default=0.0)
    parser.add_argument("--top_y", type=float, default=0.92)
    parser.add_argument("--row_gap", type=float, default=0.03)
    parser.add_argument("--width", type=float, default=0.24)
    parser.add_argument("--z_contact", type=float, default=0.32)
    parser.add_argument("--z_wrist", type=float, default=0.24)
    parser.add_argument("--active_hand", choices=["right", "left"], default="right")
    parser.add_argument("--points_per_row", type=int, default=6)
    parser.add_argument("--x_scale", type=float, default=1.9)
    parser.add_argument("--x_offset", type=float, default=0.04)
    parser.add_argument("--y_offset", type=float, default=-0.07)
    parser.add_argument("--z_offset", type=float, default=0.03)
    args = parser.parse_args()
    if args.task_text_file:
        args.task_text = args.task_text_file.read_text(encoding="utf-8").strip()

    spec = create_spec(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
