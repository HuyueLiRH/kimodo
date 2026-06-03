#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np


RIGHT_ARM = [26, 27, 28, 29, 30, 31, 32, 33]
RIGHT_ARM_PHYSICAL = [26, 27, 28, 29, 30, 31, 32]
RIGHT_HAND = 33

TARGET_VARIANT = "right_arm_g1_direct_from_raw_line_default_return_target"
DIRECT_VARIANT = "right_arm_g1_direct_from_raw_line_default_return"

DIRECT_REFIT_PARAMS = {
    "steps": 1600,
    "lr": 0.012,
    "pad": 8,
    "threshold": 0.0001,
    "taper_frames": 0,
    "fps": 30.0,
    "opt_joints": RIGHT_ARM_PHYSICAL,
    "target_loss_weight": 520.0,
    "hand_loss_weight": 17000.0,
    "pose_prior_weight": 8.0,
    "angle_prior_weight": 10.0,
    "angle_vel_weight": 180.0,
    "angle_acc_weight": 340.0,
    "hand_acc_weight": 5200.0,
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def smootherstep(value: float) -> float:
    t = min(max(float(value), 0.0), 1.0)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def load_constraint_points(recipe_path: Path) -> list[dict[str, Any]]:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    candidate = recipe.get("candidate", recipe)
    points = []
    for constraint in candidate.get("constraints", []):
        if constraint.get("end_effector") != "right-hand":
            continue
        if not constraint.get("used_for_postprocess", True):
            continue
        role = str(constraint.get("role", ""))
        label = str(constraint.get("label", ""))
        if "brush_stroke" not in role and "row_" not in label:
            continue
        points.append(
            {
                "label": constraint.get("label", f"frame_{constraint.get('frame')}"),
                "frame": int(constraint["frame"]),
                "point": [float(value) for value in constraint["position"]],
            }
        )
    points.sort(key=lambda item: item["frame"])
    if len(points) < 2:
        raise ValueError(f"Need at least two right-hand brush constraints in {recipe_path}")
    return points


def interpolate_stroke_hand_targets(
    *,
    total_frames: int,
    constraints: list[dict[str, Any]],
) -> dict[int, np.ndarray]:
    points = sorted(constraints, key=lambda item: int(item["frame"]))
    targets: dict[int, np.ndarray] = {}
    for start_item, end_item in zip(points[:-1], points[1:]):
        start_frame = int(start_item["frame"])
        end_frame = int(end_item["frame"])
        start = np.asarray(start_item["point"], dtype=np.float64)
        end = np.asarray(end_item["point"], dtype=np.float64)
        if not (0 <= start_frame < total_frames and 0 <= end_frame < total_frames):
            raise ValueError(f"Constraint frame outside motion length {total_frames}: {start_frame}, {end_frame}")
        for frame in range(start_frame, end_frame + 1):
            ratio = (frame - start_frame) / max(end_frame - start_frame, 1)
            targets[frame] = start + ratio * (end - start)
    return targets


def make_body_default_right_arm_target(raw: dict[str, np.ndarray], constraints: list[dict[str, Any]], device: str) -> np.ndarray:
    import torch
    from kimodo.exports.motion_io import complete_motion_dict
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.skeleton.definitions import G1Skeleton34
    from refit_g1_right_arm_hinge_to_target import (
        extract_hinge_angles,
        hinge_indices_for_joints,
        reconstruct_with_opt_hinges,
    )

    import kimodo

    kimodo_root = Path(inspect.getfile(kimodo)).resolve().parent
    skeleton = G1Skeleton34(folder=str(kimodo_root / "assets" / "skeletons" / "g1skel34"), load=True).to(device)
    converter = MujocoQposConverter(skeleton)

    source_local = torch.tensor(raw["local_rot_mats"], dtype=torch.float32, device=device)
    source_root = torch.tensor(raw["root_positions"], dtype=torch.float32, device=device)
    angles = extract_hinge_angles(converter, source_local).detach()
    opt_hinges = hinge_indices_for_joints(converter, RIGHT_ARM_PHYSICAL)

    rest = converter._rest_dofs_axis_angle.to(device=device, dtype=torch.float32)
    lo = converter._joint_limits_min.to(device=device, dtype=torch.float32) + rest
    hi = converter._joint_limits_max.to(device=device, dtype=torch.float32) + rest
    angles[:, opt_hinges] = torch.clamp(angles[:, opt_hinges], lo[opt_hinges], hi[opt_hinges])

    end_frame = int(constraints[-1]["frame"])
    final_frame = int(source_local.shape[0] - 1)
    end_angles = angles[end_frame, opt_hinges].clone()
    default_angles = angles[0, opt_hinges].clone()
    for frame in range(end_frame + 1, final_frame + 1):
        alpha = smootherstep((frame - end_frame) / max(final_frame - end_frame, 1))
        angles[frame, opt_hinges] = (1.0 - alpha) * end_angles + alpha * default_angles
    angles[final_frame, opt_hinges] = default_angles

    local_target = reconstruct_with_opt_hinges(converter, source_local, angles, RIGHT_ARM_PHYSICAL)
    out = complete_motion_dict(local_target, source_root, skeleton, fps=30.0)
    return out["posed_joints"].detach().cpu().numpy()


def build_direct_target(
    raw_motion: Path,
    recipe: Path,
    output: Path,
    report: Path,
    *,
    device: str,
) -> None:
    raw = load_npz(raw_motion)
    target = {key: value.copy() for key, value in raw.items()}
    posed = target["posed_joints"].astype(np.float64, copy=True)
    constraints = load_constraint_points(recipe)
    start_frame = int(constraints[0]["frame"])
    end_frame = int(constraints[-1]["frame"])
    total_frames = int(posed.shape[0])

    stroke_targets = interpolate_stroke_hand_targets(total_frames=total_frames, constraints=constraints)
    body_default_posed = make_body_default_right_arm_target(raw, constraints, device)

    for frame, point in stroke_targets.items():
        posed[frame, RIGHT_HAND] = point

    # Return target uses G1 hinge-space default return for the arm, but its hand path starts
    # from the requested stroke endpoint to avoid a hard target switch at the last constraint.
    line_end = np.asarray(constraints[-1]["point"], dtype=np.float64)
    default_final_hand = body_default_posed[-1, RIGHT_HAND].astype(np.float64)
    for frame in range(end_frame + 1, total_frames):
        alpha = smootherstep((frame - end_frame) / max((total_frames - 1) - end_frame, 1))
        posed[frame, RIGHT_ARM] = body_default_posed[frame, RIGHT_ARM]
        posed[frame, RIGHT_HAND] = (1.0 - alpha) * line_end + alpha * default_final_hand
    posed[-1, RIGHT_ARM] = body_default_posed[-1, RIGHT_ARM]
    posed[-1, RIGHT_HAND] = default_final_hand

    target["posed_joints"] = posed.astype(target["posed_joints"].dtype, copy=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **target)

    write_json(
        report,
        {
            "method": "DIRECT_FROM_RAW_LINE_AND_BODY_DEFAULT_RETURN_TARGET",
            "raw_motion": str(raw_motion),
            "recipe": str(recipe),
            "output_motion": str(output),
            "stroke_frames": [start_frame, end_frame],
            "return_frames": [end_frame + 1, total_frames - 1],
            "constraints": constraints,
            "semantics": (
                "One direct target for raw-to-final G1 hinge refit: right hand follows the brush "
                "constraint line during the stroke, then returns to the body-relative frame-0 "
                "G1 right-arm default pose."
            ),
            "final_hand_target": default_final_hand.astype(float).tolist(),
        },
    )


def line_distance(hand: np.ndarray, constraints: list[dict[str, Any]]) -> dict[str, float]:
    start_frame = int(constraints[0]["frame"])
    end_frame = int(constraints[-1]["frame"])
    start = np.asarray(constraints[0]["point"], dtype=np.float64)
    end = np.asarray(constraints[-1]["point"], dtype=np.float64)
    segment = hand[start_frame : end_frame + 1]
    direction = end - start
    denom = float(direction @ direction)
    if denom < 1e-12:
        distances = np.linalg.norm(segment - start[None], axis=-1)
    else:
        t = np.clip(((segment - start[None]) @ direction) / denom, 0.0, 1.0)
        projected = start[None] + t[:, None] * direction[None]
        distances = np.linalg.norm(segment - projected, axis=-1)
    return {"mean_m": float(distances.mean()), "max_m": float(distances.max())}


def stroke_progress_metrics(hand: np.ndarray, constraints: list[dict[str, Any]]) -> dict[str, Any]:
    start_frame = int(constraints[0]["frame"])
    end_frame = int(constraints[-1]["frame"])
    start = np.asarray(constraints[0]["point"], dtype=np.float64)
    end = np.asarray(constraints[-1]["point"], dtype=np.float64)
    stroke = hand[start_frame : end_frame + 1].astype(np.float64)
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        progress = np.zeros((stroke.shape[0],), dtype=np.float64)
        unit = np.zeros((3,), dtype=np.float64)
    else:
        unit = direction / norm
        progress = (stroke - start[None]) @ unit
    dprogress = np.diff(progress)
    backsteps = np.maximum(-dprogress, 0.0)
    axis_names = ["x", "y", "z"]
    axis_idx = int(np.argmax(np.abs(unit)))
    axis = axis_names[axis_idx] if abs(float(unit[axis_idx])) > 0.999 else "custom"
    return {
        "progress_axis": axis,
        "progress_direction": unit.astype(float).tolist(),
        "progress_start_m": float(progress[0]) if progress.size else 0.0,
        "progress_end_m": float(progress[-1]) if progress.size else 0.0,
        "progress_span_m": float(norm),
        "progress_backstep_count": int(np.sum(dprogress < -1e-5)),
        "progress_backstep_total_m": float(backsteps.sum()),
    }


def compute_basic_metrics(motion_path: Path, recipe_path: Path) -> dict[str, Any]:
    motion = load_npz(motion_path)
    hand = motion["posed_joints"][:, RIGHT_HAND].astype(np.float64)
    root = motion["root_positions"].astype(np.float64)
    constraints = load_constraint_points(recipe_path)
    frames = [int(item["frame"]) for item in constraints]
    targets = np.asarray([item["point"] for item in constraints], dtype=np.float64)
    errors = np.linalg.norm(hand[frames] - targets, axis=-1)
    start = frames[0]
    end = frames[-1]
    stroke = hand[start : end + 1]
    steps = np.linalg.norm(np.diff(stroke, axis=0), axis=-1)
    dx = np.diff(stroke[:, 0])
    return {
        "constraint_error": {
            "mean_m": float(errors.mean()),
            "max_m": float(errors.max()),
            "per_point_m": {item["label"]: float(errors[idx]) for idx, item in enumerate(constraints)},
        },
        "line_distance": line_distance(hand, constraints),
        "stroke_hand_speed": {
            "mean_step_m": float(steps.mean()) if steps.size else 0.0,
            "std_step_m": float(steps.std()) if steps.size else 0.0,
            "cv": float(steps.std() / max(steps.mean(), 1e-8)) if steps.size else 0.0,
            "max_step_m": float(steps.max()) if steps.size else 0.0,
            "x_backstep_count": int(np.sum(dx < -1e-5)),
            "x_backstep_total_m": float(np.maximum(-dx, 0.0).sum()),
            **stroke_progress_metrics(hand, constraints),
        },
        "root_drift": float(np.linalg.norm(root[-1] - root[0])) if root.shape[0] > 1 else 0.0,
        "final_hand_distance_to_initial_m": float(np.linalg.norm(hand[-1] - hand[0])),
    }


def run_direct_refit(
    raw_motion: Path,
    target_motion: Path,
    recipe: Path,
    output: Path,
    report: Path,
    *,
    device: str,
    params: dict[str, Any],
) -> None:
    from refit_g1_right_arm_hinge_to_target import refit

    refit(
        source_motion=raw_motion,
        target_motion=target_motion,
        recipe=recipe,
        output=output,
        report=report,
        device=device,
        **params,
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    data["direct_from_raw"] = {
        "target_variant": TARGET_VARIANT,
        "output_variant": DIRECT_VARIANT,
        "semantics": (
            "Single G1 hinge refit from the raw KIMODO motion to a line stroke plus "
            "body-relative default return target. This skips the staged FABRIK/lightlock/"
            "intermediate G1 line refit pipeline."
        ),
    }
    write_json(report, data)


def candidate_names(run_root: Path) -> list[str]:
    raw_root = run_root / "raw"
    return sorted(path.name for path in raw_root.iterdir() if (path / "motion.npz").exists())


def process_candidate(run_root: Path, name: str, *, device: str, resume: bool, params: dict[str, Any]) -> dict[str, Any]:
    raw_motion = run_root / "raw" / name / "motion.npz"
    recipe = run_root / "raw" / name / "recipe.json"
    post_root = run_root / "postprocessed" / name
    target_motion = post_root / TARGET_VARIANT / "motion.npz"
    target_report = post_root / TARGET_VARIANT / "report.json"
    output_motion = post_root / DIRECT_VARIANT / "motion.npz"
    output_report = post_root / DIRECT_VARIANT / "report.json"

    if not (resume and target_motion.exists() and target_report.exists()):
        build_direct_target(raw_motion, recipe, target_motion, target_report, device=device)
    if not (resume and output_motion.exists() and output_report.exists()):
        run_direct_refit(raw_motion, target_motion, recipe, output_motion, output_report, device=device, params=params)

    return {
        "name": name,
        "raw_motion": raw_motion,
        "recipe": recipe,
        "target_motion": target_motion,
        "target_report": target_report,
        "output_motion": output_motion,
        "output_report": output_report,
        "metrics": compute_basic_metrics(output_motion, recipe),
        "report": json.loads(output_report.read_text(encoding="utf-8")),
    }


def update_review_artifacts(run_root: Path, records: list[dict[str, Any]]) -> None:
    manifest_path = run_root / "manifest.json"
    metrics_path = run_root / "metrics.json"
    review_path = run_root / "review.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))

    rows = []
    for record in records:
        name = record["name"]
        candidate = manifest["candidates"][name]
        post = candidate.setdefault("postprocessed", {})
        post[DIRECT_VARIANT] = {
            "name": DIRECT_VARIANT,
            "source_raw_motion": rel(record["raw_motion"], run_root),
            "target_motion": rel(record["target_motion"], run_root),
            "output_motion": rel(record["output_motion"], run_root),
            "report": rel(record["output_report"], run_root),
            "params": DIRECT_REFIT_PARAMS,
            "note": "Experimental direct-from-raw G1 hinge refit; compare against staged final postprocess.",
        }
        metrics["candidates"].setdefault(name, {})
        metrics["candidates"][name][DIRECT_VARIANT] = {
            **record["metrics"],
            "report": record["report"],
        }
        review["candidates"].setdefault(name, {})
        review["candidates"][name]["current_best_variant"] = DIRECT_VARIANT
        review["candidates"][name]["notes"] = (
            "Review experimental direct-from-raw variant against raw and staged final postprocess."
        )
        rows.append(
            {
                "variant": name,
                "constraint_max_m": record["metrics"]["constraint_error"]["max_m"],
                "line_max_m": record["metrics"]["line_distance"]["max_m"],
                "x_backstep_total_m": record["metrics"]["stroke_hand_speed"]["x_backstep_total_m"],
                "progress_axis": record["metrics"]["stroke_hand_speed"]["progress_axis"],
                "progress_backstep_total_m": record["metrics"]["stroke_hand_speed"]["progress_backstep_total_m"],
                "speed_cv": record["metrics"]["stroke_hand_speed"]["cv"],
                "motion": rel(record["output_motion"], run_root),
            }
        )

    manifest["current_best_pipeline"] = {
        **manifest.get("current_best_pipeline", {}),
        "experimental_direct_from_raw_variant": DIRECT_VARIANT,
        "direct_from_raw_note": "Raw motion is refit once to a line-stroke + body-default-return target using legal G1 right-arm hinge DoFs.",
    }
    write_json(manifest_path, manifest)
    write_json(metrics_path, metrics)
    write_json(review_path, review)
    write_json(run_root / "direct_from_raw_summary.json", {"rows": rows})
    with (run_root / "direct_from_raw_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct raw-to-final G1 wall-brush refit for one or more candidates.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=DIRECT_REFIT_PARAMS["steps"])
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    names = args.candidate or candidate_names(run_root)
    if args.limit is not None:
        names = names[: args.limit]
    params = dict(DIRECT_REFIT_PARAMS)
    params["steps"] = int(args.steps)

    records = []
    for index, name in enumerate(names, start=1):
        print(f"[{index}/{len(names)}] direct-from-raw {name}", flush=True)
        records.append(process_candidate(run_root, name, device=args.device, resume=args.resume, params=params))
        update_review_artifacts(run_root, records)

    update_review_artifacts(run_root, records)
    print(json.dumps({"run_root": str(run_root), "variant": DIRECT_VARIANT, "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
