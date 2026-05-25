#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kimodo.exports.motion_io import complete_motion_dict, save_kimodo_npz
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton.definitions import G1Skeleton34

from refit_g1_right_arm_hinge_to_target import (
    RIGHT_ARM,
    RIGHT_ARM_PHYSICAL,
    RIGHT_HAND,
    extract_hinge_angles,
    hinge_indices_for_joints,
    reconstruct_with_opt_hinges,
    refit,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "logs" / "wall_brush_generalization_27_smooth_return_20260521"
TOTAL_FRAMES = 102

SOURCE_VARIANT = "right_arm_fabrik_fk_refit_lightlock_smooth_return_initial_slow"
LINE_TARGET_VARIANT = "right_arm_line_lock_after_smooth_target"
G1_LINE_VARIANT = "right_arm_g1_hinge_line_refit_smooth"
BODY_DEFAULT_TARGET_VARIANT = "right_arm_g1_hinge_line_refit_smooth_body_default_return_target"
FINAL_VARIANT = "right_arm_g1_hinge_line_refit_smooth_body_default_return_no_taper"
UNIFORM_LINE_VARIANT = "right_arm_fabrik_uniform_line"

LINE_REFIT_PARAMS = {
    "steps": 950,
    "lr": 0.012,
    "pad": 8,
    "threshold": 0.0001,
    "taper_frames": 10,
    "fps": 30.0,
    "opt_joints": RIGHT_ARM_PHYSICAL,
    "target_loss_weight": 350.0,
    "hand_loss_weight": 9000.0,
    "pose_prior_weight": 10.0,
    "angle_prior_weight": 12.0,
    "angle_vel_weight": 160.0,
    "angle_acc_weight": 260.0,
    "hand_acc_weight": 4200.0,
}

FINAL_REFIT_PARAMS = {
    "steps": 1400,
    "lr": 0.012,
    "pad": 4,
    "threshold": 0.0001,
    "taper_frames": 0,
    "fps": 30.0,
    "opt_joints": RIGHT_ARM_PHYSICAL,
    "target_loss_weight": 700.0,
    "hand_loss_weight": 16000.0,
    "pose_prior_weight": 4.0,
    "angle_prior_weight": 8.0,
    "angle_vel_weight": 180.0,
    "angle_acc_weight": 320.0,
    "hand_acc_weight": 4500.0,
}


def log(message: str) -> None:
    print(message, flush=True)


def rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def smootherstep(value: float) -> float:
    t = min(max(float(value), 0.0), 1.0)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def constraint_points(recipe_path: Path) -> list[dict[str, Any]]:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    constraints = recipe.get("candidate", recipe).get("constraints", [])
    out = []
    for constraint in constraints:
        if constraint.get("end_effector") != "right-hand":
            continue
        if not constraint.get("used_for_postprocess", False):
            continue
        out.append(
            {
                "label": constraint["label"],
                "frame": int(constraint["frame"]),
                "point": [float(value) for value in constraint["position"]],
            }
        )
    out.sort(key=lambda item: item["frame"])
    return out


def build_line_target(
    source_motion: Path,
    uniform_line_motion: Path,
    recipe_path: Path,
    output_motion: Path,
    report_path: Path,
) -> None:
    source = load_npz(source_motion)
    uniform = load_npz(uniform_line_motion)
    target = {key: value.copy() for key, value in source.items()}
    posed = target["posed_joints"].copy()
    uniform_posed = uniform["posed_joints"]
    points = constraint_points(recipe_path)
    start = points[0]["frame"]
    end = points[-1]["frame"]
    posed[start : end + 1, RIGHT_ARM] = uniform_posed[start : end + 1, RIGHT_ARM]
    target["posed_joints"] = posed
    output_motion.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_motion, **target)
    hand = posed[:, RIGHT_HAND]
    target_points = np.asarray([item["point"] for item in points], dtype=np.float64)
    frames = [item["frame"] for item in points]
    errors = np.linalg.norm(hand[frames] - target_points, axis=-1)
    write_json(
        report_path,
        {
            "method": "LINE_TARGET_FROM_FABRIK_UNIFORM_LINE_STROKE_ONLY",
            "source_motion": str(source_motion),
            "uniform_line_motion": str(uniform_line_motion),
            "output_motion": str(output_motion),
            "stroke_frames": [start, end],
            "constraint_key_error_m": {
                item["label"]: float(errors[index]) for index, item in enumerate(points)
            },
        },
    )


def make_skeleton_and_converter(device: str) -> tuple[G1Skeleton34, MujocoQposConverter]:
    import kimodo

    kimodo_root = Path(inspect.getfile(kimodo)).resolve().parent
    skeleton = G1Skeleton34(folder=str(kimodo_root / "assets" / "skeletons" / "g1skel34"), load=True).to(device)
    return skeleton, MujocoQposConverter(skeleton)


def build_body_default_target(
    source_motion: Path,
    line_target_motion: Path,
    recipe_path: Path,
    output_motion: Path,
    report_path: Path,
    *,
    device: str,
) -> None:
    source = load_npz(source_motion)
    line_target = load_npz(line_target_motion)
    points = constraint_points(recipe_path)
    start = points[0]["frame"]
    end = points[-1]["frame"]
    final = source["posed_joints"].shape[0] - 1

    skeleton, converter = make_skeleton_and_converter(device)
    source_local = torch.tensor(source["local_rot_mats"], dtype=torch.float32, device=device)
    source_root = torch.tensor(source["root_positions"], dtype=torch.float32, device=device)
    source_angles = extract_hinge_angles(converter, source_local).detach()
    opt_hinges = hinge_indices_for_joints(converter, RIGHT_ARM_PHYSICAL)
    rest = converter._rest_dofs_axis_angle.to(device=device, dtype=torch.float32)
    lo = converter._joint_limits_min.to(device=device, dtype=torch.float32) + rest
    hi = converter._joint_limits_max.to(device=device, dtype=torch.float32) + rest
    angles = source_angles.clone()
    angles[:, opt_hinges] = torch.clamp(angles[:, opt_hinges], lo[opt_hinges], hi[opt_hinges])

    stroke_end_angles = angles[end, opt_hinges].clone()
    default_angles = angles[0, opt_hinges].clone()
    for frame in range(end + 1, final + 1):
        alpha = smootherstep((frame - end) / max(final - end, 1))
        angles[frame, opt_hinges] = (1.0 - alpha) * stroke_end_angles + alpha * default_angles
    angles[final, opt_hinges] = default_angles

    local_target = reconstruct_with_opt_hinges(converter, source_local, angles, RIGHT_ARM_PHYSICAL)
    out_tensors = complete_motion_dict(local_target, source_root, skeleton, fps=30.0)
    out = {
        key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        for key, value in out_tensors.items()
    }
    out["posed_joints"][start : end + 1, RIGHT_ARM] = line_target["posed_joints"][start : end + 1, RIGHT_ARM]
    output_motion.parent.mkdir(parents=True, exist_ok=True)
    save_kimodo_npz(str(output_motion), out)

    hand = out["posed_joints"][:, RIGHT_HAND]
    source_hand = source["posed_joints"][:, RIGHT_HAND]
    target_points = np.asarray([item["point"] for item in points], dtype=np.float64)
    frames = [item["frame"] for item in points]
    errors = np.linalg.norm(hand[frames] - target_points, axis=-1)
    write_json(
        report_path,
        {
            "method": "TARGET_G1_HINGE_BODY_DEFAULT_RETURN",
            "source_motion": str(source_motion),
            "line_target_motion": str(line_target_motion),
            "output_motion": str(output_motion),
            "default_source": "source frame 0 physical right-arm hinge angles, replayed at each return frame with current root/torso",
            "return_frames": [end + 1, final],
            "return_curve": "smootherstep in G1 right-arm hinge angle space",
            "constraint_key_error_m": {
                item["label"]: float(errors[index]) for index, item in enumerate(points)
            },
            "final_hand_distance_to_source_final_m": float(np.linalg.norm(hand[final] - source_hand[final])),
            "final_hand_distance_to_source_frame0_world_m": float(np.linalg.norm(hand[final] - source_hand[0])),
            "target_final_hand": hand[final].astype(float).tolist(),
            "source_final_hand": source_hand[final].astype(float).tolist(),
            "source_frame0_hand": source_hand[0].astype(float).tolist(),
            "default_hinge_angles_frame0": default_angles.detach().cpu().numpy().astype(float).tolist(),
            "stroke_end_hinge_angles": stroke_end_angles.detach().cpu().numpy().astype(float).tolist(),
        },
    )


def enrich_final_report(final_report: Path, source_motion: Path, target_motion: Path) -> dict[str, Any]:
    report = json.loads(final_report.read_text(encoding="utf-8"))
    source_hand = load_npz(source_motion)["posed_joints"][:, RIGHT_HAND]
    target_hand = load_npz(target_motion)["posed_joints"][:, RIGHT_HAND]
    out_hand = load_npz(final_report.parent / "motion.npz")["posed_joints"][:, RIGHT_HAND]
    return_steps = np.linalg.norm(np.diff(out_hand[66:], axis=0), axis=1)
    report["default_return"] = {
        "semantics": (
            "Return the right arm to the frame-0 G1 hinge default relative to the robot body/root, "
            "not to the absolute world-space frame-0 hand position."
        ),
        "target_motion": str(target_motion),
        "final_hand_distance_to_body_default_target_m": float(np.linalg.norm(out_hand[-1] - target_hand[-1])),
        "final_hand_distance_to_previous_bad_final_m": float(np.linalg.norm(out_hand[-1] - source_hand[-1])),
        "final_hand_distance_to_source_frame0_world_m": float(np.linalg.norm(out_hand[-1] - source_hand[0])),
        "return_step_mean_m": float(return_steps.mean()) if return_steps.size else 0.0,
        "return_step_max_m": float(return_steps.max()) if return_steps.size else 0.0,
        "return_step_max_frame": int(np.argmax(return_steps) + 66) if return_steps.size else 66,
        "reason_no_taper": (
            "The return segment includes the final frame; tapering the edit mask at the end pulls "
            "frame 101 back toward the old bad final pose."
        ),
    }
    final_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compute_basic_metrics(motion_path: Path, recipe_path: Path) -> dict[str, Any]:
    motion = load_npz(motion_path)
    hand = motion["posed_joints"][:, RIGHT_HAND].astype(np.float64)
    root = motion["root_positions"].astype(np.float64)
    points = constraint_points(recipe_path)
    frames = [item["frame"] for item in points]
    targets = np.asarray([item["point"] for item in points], dtype=np.float64)
    errors = np.linalg.norm(hand[frames] - targets, axis=-1)
    start = frames[0]
    end = frames[-1]
    stroke = hand[start : end + 1]
    steps = np.linalg.norm(np.diff(stroke, axis=0), axis=-1)
    dx = np.diff(stroke[:, 0])
    direction = targets[-1] - targets[0]
    denom = float(direction @ direction)
    if denom < 1e-12:
        line_dist = np.linalg.norm(stroke - targets[0][None], axis=-1)
    else:
        t = np.clip(((stroke - targets[0][None]) @ direction) / denom, 0.0, 1.0)
        projected = targets[0][None] + t[:, None] * direction[None]
        line_dist = np.linalg.norm(stroke - projected, axis=-1)
    return {
        "constraint_error": {
            "mean_m": float(errors.mean()),
            "max_m": float(errors.max()),
            "per_point_m": {item["label"]: float(errors[index]) for index, item in enumerate(points)},
        },
        "line_distance": {
            "mean_m": float(line_dist.mean()),
            "max_m": float(line_dist.max()),
        },
        "stroke_hand_speed": {
            "mean_step_m": float(steps.mean()) if steps.size else 0.0,
            "std_step_m": float(steps.std()) if steps.size else 0.0,
            "cv": float(steps.std() / max(steps.mean(), 1e-8)) if steps.size else 0.0,
            "max_step_m": float(steps.max()) if steps.size else 0.0,
            "x_backstep_count": int(np.sum(dx < -1e-5)),
            "x_backstep_total_m": float(np.maximum(-dx, 0.0).sum()),
        },
        "root_drift": float(np.linalg.norm(root[-1] - root[0])) if root.shape[0] > 1 else 0.0,
        "final_hand_distance_to_initial_m": float(np.linalg.norm(hand[-1] - hand[0])),
    }


def candidate_dirs(run_root: Path) -> list[Path]:
    return sorted(path for path in (run_root / "raw").iterdir() if path.is_dir() and (path / "motion.npz").exists())


def process_candidate(run_root: Path, raw_dir: Path, *, device: str, resume: bool) -> dict[str, Any]:
    name = raw_dir.name
    post_root = run_root / "postprocessed" / name
    recipe = raw_dir / "recipe.json"
    source = post_root / SOURCE_VARIANT / "motion.npz"
    uniform = post_root / UNIFORM_LINE_VARIANT / "motion.npz"
    line_target = post_root / LINE_TARGET_VARIANT / "motion.npz"
    line_target_report = post_root / LINE_TARGET_VARIANT / "report.json"
    g1_line = post_root / G1_LINE_VARIANT / "motion.npz"
    g1_line_report = post_root / G1_LINE_VARIANT / "report.json"
    body_target = post_root / BODY_DEFAULT_TARGET_VARIANT / "motion.npz"
    body_target_report = post_root / BODY_DEFAULT_TARGET_VARIANT / "report.json"
    final = post_root / FINAL_VARIANT / "motion.npz"
    final_report = post_root / FINAL_VARIANT / "report.json"

    if not source.exists():
        raise FileNotFoundError(f"Missing source postprocess motion for {name}: {source}")
    if not uniform.exists():
        raise FileNotFoundError(f"Missing uniform line target for {name}: {uniform}")

    if not (resume and line_target.exists() and line_target_report.exists()):
        log(f"{name}: build line target")
        build_line_target(source, uniform, recipe, line_target, line_target_report)

    if not (resume and g1_line.exists() and g1_line_report.exists()):
        log(f"{name}: G1 hinge line refit smooth")
        refit(
            source_motion=source,
            target_motion=line_target,
            recipe=recipe,
            output=g1_line,
            report=g1_line_report,
            device=device,
            **LINE_REFIT_PARAMS,
        )

    if not (resume and body_target.exists() and body_target_report.exists()):
        log(f"{name}: build body-default return target")
        build_body_default_target(g1_line, line_target, recipe, body_target, body_target_report, device=device)

    if not (resume and final.exists() and final_report.exists()):
        log(f"{name}: G1 hinge default-return no-taper")
        refit(
            source_motion=g1_line,
            target_motion=body_target,
            recipe=recipe,
            output=final,
            report=final_report,
            device=device,
            **FINAL_REFIT_PARAMS,
        )

    final_summary = enrich_final_report(final_report, g1_line, body_target)
    return {
        "name": name,
        "recipe": recipe,
        "raw_motion": raw_dir / "motion.npz",
        "source_motion": source,
        "line_target": line_target,
        "g1_line": g1_line,
        "g1_line_report": g1_line_report,
        "body_target": body_target,
        "body_target_report": body_target_report,
        "final_motion": final,
        "final_report": final_report,
        "final_summary": final_summary,
        "metrics": {
            "raw": compute_basic_metrics(raw_dir / "motion.npz", recipe),
            SOURCE_VARIANT: compute_basic_metrics(source, recipe),
            G1_LINE_VARIANT: compute_basic_metrics(g1_line, recipe),
            FINAL_VARIANT: compute_basic_metrics(final, recipe),
        },
    }


def update_run_artifacts(run_root: Path, records: list[dict[str, Any]]) -> None:
    manifest_path = run_root / "manifest.json"
    metrics_path = run_root / "metrics.json"
    review_path = run_root / "review.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))

    manifest["current_best_pipeline"] = {
        **manifest.get("current_best_pipeline", {}),
        "final_variant": FINAL_VARIANT,
        "g1_hinge_default_return": {
            "line_refit_params": {key: value for key, value in LINE_REFIT_PARAMS.items() if key != "opt_joints"},
            "final_refit_params": {key: value for key, value in FINAL_REFIT_PARAMS.items() if key != "opt_joints"},
            "opt_joints": RIGHT_ARM_PHYSICAL,
            "note": "Accepted baseline postprocess generalized to the 27 already generated motions.",
        },
    }

    rows: list[dict[str, Any]] = []
    for record in records:
        name = record["name"]
        candidate = manifest["candidates"][name]
        post = candidate.setdefault("postprocessed", {})
        post[G1_LINE_VARIANT] = {
            "name": G1_LINE_VARIANT,
            "source_raw_motion": rel(record["source_motion"], run_root),
            "source_target_motion": rel(record["line_target"], run_root),
            "output_motion": rel(record["g1_line"], run_root),
            "report": rel(record["g1_line_report"], run_root),
            "params": {**LINE_REFIT_PARAMS, "opt_joints": RIGHT_ARM_PHYSICAL},
            "note": "Intermediate G1-valid hinge-space line refit before body-relative default return.",
        }
        post[FINAL_VARIANT] = {
            "name": FINAL_VARIANT,
            "source_raw_motion": rel(record["g1_line"], run_root),
            "source_target_motion": rel(record["body_target"], run_root),
            "output_motion": rel(record["final_motion"], run_root),
            "report": rel(record["final_report"], run_root),
            "params": {**FINAL_REFIT_PARAMS, "opt_joints": RIGHT_ARM_PHYSICAL},
            "note": "Accepted G1-valid hinge-space wall-brush postprocess with no-taper body-relative default return.",
        }
        metrics["candidates"].setdefault(name, {})
        metrics["candidates"][name][G1_LINE_VARIANT] = {
            **record["metrics"][G1_LINE_VARIANT],
            "report": json.loads(record["g1_line_report"].read_text(encoding="utf-8")),
        }
        metrics["candidates"][name][FINAL_VARIANT] = {
            **record["metrics"][FINAL_VARIANT],
            "report": record["final_summary"],
        }
        review["candidates"].setdefault(name, {})
        review["candidates"][name]["current_best_variant"] = FINAL_VARIANT
        review["candidates"][name]["notes"] = (
            "Review raw, lightlock smooth-return, G1 line refit, and accepted G1 hinge default-return no-taper postprocess."
        )
        final_metrics = record["metrics"][FINAL_VARIANT]
        rows.append(
            {
                "variant": name,
                "constraint_max_m": final_metrics["constraint_error"]["max_m"],
                "line_max_m": final_metrics["line_distance"]["max_m"],
                "final_to_default_target_m": record["final_summary"]["default_return"][
                    "final_hand_distance_to_body_default_target_m"
                ],
                "return_max_step_m": record["final_summary"]["default_return"]["return_step_max_m"],
                "final_motion": rel(record["final_motion"], run_root),
            }
        )

    write_json(manifest_path, manifest)
    write_json(metrics_path, metrics)
    write_json(review_path, review)
    write_json(run_root / "g1_hinge_default_return_summary.json", {"rows": rows})
    with (run_root / "g1_hinge_default_return_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    log_path = run_root / "experiment_log.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    section = f"""

## G1 Hinge Default Return Generalization

Date: 2026-05-21

Applied accepted baseline postprocess `{FINAL_VARIANT}` to {len(records)} already generated motions.

Pipeline:
- source: `{SOURCE_VARIANT}`
- intermediate: `{G1_LINE_VARIANT}`
- target: right-hand straight line during the brush segment, then body-relative frame-0 G1 hinge default return
- final: `{FINAL_VARIANT}`

Final pass uses only legal physical G1 right-arm hinge DoFs `{RIGHT_ARM_PHYSICAL}` and `taper_frames=0` so frame 101 reaches the default-return target.

Summary files:
- `g1_hinge_default_return_summary.csv`
- `g1_hinge_default_return_summary.json`
"""
    if "## G1 Hinge Default Return Generalization" not in existing:
        log_path.write_text(existing.rstrip() + section, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply accepted G1 hinge default-return wall-brush postprocess to 27 motions.")
    parser.add_argument("--run_root", type=Path, default=RUN_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variant", action="append", default=[])
    args = parser.parse_args()

    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    raw_dirs = candidate_dirs(run_root)
    if args.variant:
        wanted = set(args.variant)
        raw_dirs = [path for path in raw_dirs if path.name in wanted]
        missing = sorted(wanted - {path.name for path in raw_dirs})
        if missing:
            raise ValueError(f"Unknown variants: {missing}")
    if args.limit is not None:
        raw_dirs = raw_dirs[: args.limit]

    if not raw_dirs:
        raise ValueError(f"No raw motion folders found in {run_root / 'raw'}")

    records = []
    for index, raw_dir in enumerate(raw_dirs, start=1):
        log(f"[{index}/{len(raw_dirs)}] {raw_dir.name}")
        records.append(process_candidate(run_root, raw_dir, device=args.device, resume=args.resume))
        update_run_artifacts(run_root, records)

    update_run_artifacts(run_root, records)
    log(f"done: {run_root / 'g1_hinge_default_return_summary.csv'}")


if __name__ == "__main__":
    main()
