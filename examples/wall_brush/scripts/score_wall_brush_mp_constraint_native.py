#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


RIGHT_ARM = [26, 27, 28, 29, 30, 31, 32, 33]
LEFT_ARM = [18, 19, 20, 21, 22, 23, 24, 25]
RIGHT_HAND = 33
RIGHT_WRIST = 32
RIGHT_ELBOW = 29
RIGHT_SHOULDER = 26
LEFT_HAND = 25
LEFT_WRIST = 24
LEFT_ELBOW = 21
LEFT_HIP = 1
RIGHT_HIP = 8
LEFT_SHOULDER = 18
ROOT = 0

SMOOTH_KEYS = ("posed_joints", "root_positions", "smooth_root_pos", "global_root_heading")


def active_hand_name(meta: dict) -> str:
    return str(meta.get("active_hand", "right")).lower()


def active_indices(meta: dict) -> dict:
    if active_hand_name(meta) == "left":
        return {
            "hand": LEFT_HAND,
            "wrist": LEFT_WRIST,
            "elbow": LEFT_ELBOW,
            "shoulder": LEFT_SHOULDER,
            "arm": LEFT_ARM,
            "passive_arm": RIGHT_ARM,
        }
    return {
        "hand": RIGHT_HAND,
        "wrist": RIGHT_WRIST,
        "elbow": RIGHT_ELBOW,
        "shoulder": RIGHT_SHOULDER,
        "arm": RIGHT_ARM,
        "passive_arm": LEFT_ARM,
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_intervals(meta: dict) -> list[dict]:
    rows = meta.get("row_specs") or []
    if rows:
        return [
            {
                "row": int(row["row"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "start_point": np.asarray(row["start_point"], dtype=np.float64),
                "end_point": np.asarray(row["end_point"], dtype=np.float64),
            }
            for row in rows
        ]
    targets = meta["target_points"]
    intervals = []
    for row in sorted({point["row"] for point in targets if int(point["row"]) in (1, 2, 3)}):
        pts = sorted([point for point in targets if point["row"] == row], key=lambda p: p["frame"])
        intervals.append(
            {
                "row": int(row),
                "start_frame": int(pts[0]["frame"]),
                "end_frame": int(pts[-1]["frame"]),
                "start_point": np.asarray(pts[0]["point"], dtype=np.float64),
                "end_point": np.asarray(pts[-1]["point"], dtype=np.float64),
            }
        )
    return intervals


def target_frames(meta: dict) -> list[int]:
    return [int(point["frame"]) for point in meta.get("target_points", [])]


def stroke_target_points(meta: dict) -> list[dict]:
    return [
        point
        for point in meta.get("target_points", [])
        if point.get("phase") == "stroke" and int(point.get("row", -1)) in (1, 2, 3)
    ]


def line_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-10:
        closest = np.repeat(start[None, :], len(points), axis=0)
    else:
        alpha = np.clip(((points - start[None, :]) @ segment) / denom, 0.0, 1.0)
        closest = start[None, :] + alpha[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=1)


def path_len(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def step_and_accel(points: np.ndarray) -> tuple[float, float, float, float]:
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 0.0
    steps = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    if len(points) < 3:
        accels = np.zeros((1,), dtype=np.float64)
    else:
        accels = np.linalg.norm(np.diff(points, n=2, axis=0), axis=-1)
    return (
        float(np.max(steps)),
        float(np.percentile(steps, 95)),
        float(np.max(accels)),
        float(np.percentile(accels, 95)),
    )


def body_yaw_xz(posed: np.ndarray) -> np.ndarray:
    lateral = 0.5 * ((posed[:, RIGHT_HIP] - posed[:, LEFT_HIP]) + (posed[:, RIGHT_SHOULDER] - posed[:, LEFT_SHOULDER]))
    v = lateral[:, [0, 2]]
    norms = np.linalg.norm(v, axis=1)
    good = norms > 1e-6
    yaw = np.zeros((len(posed),), dtype=np.float64)
    yaw[good] = np.arctan2(v[good, 1], v[good, 0])
    last = 0.0
    for i in range(len(yaw)):
        if good[i]:
            last = yaw[i]
        else:
            yaw[i] = last
    return np.unwrap(yaw)


def root_heading_yaw(data: np.lib.npyio.NpzFile, posed: np.ndarray) -> tuple[np.ndarray, str]:
    if "global_root_heading" in data.files:
        heading = data["global_root_heading"].astype(np.float64)
        if heading.ndim == 2 and heading.shape[1] >= 2:
            return np.unwrap(np.arctan2(heading[:, 1], heading[:, 0])), "global_root_heading"
    return body_yaw_xz(posed), "body_axis_fallback"


def elbow_angle_deg(posed: np.ndarray, indices: dict | None = None) -> np.ndarray:
    indices = indices or active_indices({})
    shoulder = posed[:, indices["shoulder"]]
    elbow = posed[:, indices["elbow"]]
    wrist = posed[:, indices["wrist"]]
    a = shoulder - elbow
    b = wrist - elbow
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)
    cosang = np.clip(np.sum(a * b, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def local_rotation_angle(local_rot_mats: np.ndarray, indices: list[int]) -> np.ndarray:
    rots = local_rot_mats[:, indices]
    trace = np.trace(rots, axis1=-2, axis2=-1)
    cosang = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosang)


def boundary_stats(posed: np.ndarray, boundaries: list[int], radius: int = 4) -> tuple[float, float]:
    steps = np.linalg.norm(np.diff(posed, axis=0), axis=2)
    accels = np.linalg.norm(np.diff(posed, n=2, axis=0), axis=2)
    step_values = []
    accel_values = []
    for boundary in boundaries:
        start = max(0, int(boundary) - radius)
        end = min(len(steps) - 1, int(boundary) + radius)
        if end >= start:
            step_values.extend(steps[start : end + 1].reshape(-1).tolist())
        a_start = max(0, int(boundary) - radius)
        a_end = min(len(accels) - 1, int(boundary) + radius)
        if a_end >= a_start:
            accel_values.extend(accels[a_start : a_end + 1].reshape(-1).tolist())
    return (
        float(np.max(step_values)) if step_values else 0.0,
        float(np.max(accel_values)) if accel_values else 0.0,
    )


def monotonicity(points: np.ndarray, direction: int) -> float:
    dx = np.diff(points[:, 0])
    if len(dx) == 0:
        return 0.0
    wrong = np.maximum(-direction * dx, 0.0).sum()
    total = np.abs(dx).sum()
    return float(1.0 - wrong / max(total, 1e-8))


def row_flatness(points: np.ndarray) -> dict:
    if len(points) == 0:
        return {
            "y_range": 0.0,
            "y_std": 0.0,
            "y_endpoint_delta": 0.0,
            "y_slope": 0.0,
            "y_curvature": 0.0,
        }
    y = points[:, 1]
    y_range = float(np.max(y) - np.min(y))
    y_std = float(np.std(y))
    y_endpoint_delta = float(abs(y[-1] - y[0]))
    x_delta = float(abs(points[-1, 0] - points[0, 0]))
    y_slope = y_endpoint_delta / max(x_delta, 1e-6)
    if len(points) >= 2:
        linear_y = np.linspace(float(y[0]), float(y[-1]), len(points))
        y_curvature = float(np.max(np.abs(y - linear_y)))
    else:
        y_curvature = 0.0
    return {
        "y_range": y_range,
        "y_std": y_std,
        "y_endpoint_delta": y_endpoint_delta,
        "y_slope": y_slope,
        "y_curvature": y_curvature,
    }


def wall_penetration(posed: np.ndarray, wall_z: float, meta: dict) -> float:
    active = active_indices(meta)
    idx = [active["hand"], active["wrist"], *active["arm"][4:7]]
    return float(np.max(np.maximum(posed[:, idx, 2] - wall_z, 0.0)))


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
        .replace(" ", "_")
    )


def score_motion(path: Path, meta: dict, filtered: bool) -> dict:
    data = np.load(path, allow_pickle=True)
    posed = data["posed_joints"].astype(np.float64)
    local_rot = data["local_rot_mats"].astype(np.float64) if "local_rot_mats" in data.files else None
    active = active_indices(meta)
    hand = posed[:, active["hand"]]
    wrist = posed[:, active["wrist"]]
    root = posed[:, ROOT]
    rows = row_intervals(meta)
    wall_z = float(meta.get("wall_z", 0.32))
    z_wrist = float(meta.get("z_wrist", 0.20))
    boundaries = [int(x) for x in meta.get("boundaries", [])]
    return_start = int(boundaries[-1]) if boundaries else rows[-1]["end_frame"] + 1
    final_neutral = np.asarray(
        meta.get("neutral_final_active_hand_side_pos", meta.get("neutral_final_right_hand_side_pos", hand[0].tolist())),
        dtype=np.float64,
    )

    line_errors = []
    path_ratios = []
    mono_scores = []
    wall_contact_errors = []
    wrist_clearance_errors = []
    x_range_coverages = []
    y_offset_errors = []
    row_x_progress = {}
    row_x_coverage = {}
    row_y_metrics = {}
    stroke_frames = []
    for row in rows:
        start = row["start_frame"]
        end = row["end_frame"]
        pts = hand[start : end + 1]
        line_errors.extend(line_distance(pts, row["start_point"], row["end_point"]).tolist())
        direct = float(np.linalg.norm(row["end_point"] - row["start_point"]))
        path_ratios.append(path_len(pts) / max(direct, 1e-8))
        direction = 1 if row["end_point"][0] >= row["start_point"][0] else -1
        mono_scores.append(monotonicity(pts, direction))
        wall_contact_errors.extend(np.abs(pts[:, 2] - wall_z).tolist())
        target_x_range = abs(float(row["end_point"][0] - row["start_point"][0]))
        direction = 1 if row["end_point"][0] >= row["start_point"][0] else -1
        progress = direction * float(pts[-1, 0] - pts[0, 0]) / max(target_x_range, 1e-8)
        coverage = (float(np.max(pts[:, 0]) - np.min(pts[:, 0]))) / max(target_x_range, 1e-8)
        row_x_progress[int(row["row"])] = progress
        row_x_coverage[int(row["row"])] = coverage
        row_y_metrics[int(row["row"])] = row_flatness(pts)
        x_range_coverages.append(coverage)
        y_offset_errors.append(abs(float(np.mean(pts[:, 1]) - row["start_point"][1])))
        wrist_pts = wrist[start : end + 1]
        wrist_clearance_errors.extend(np.abs(wrist_pts[:, 2] - z_wrist).tolist())
        stroke_frames.extend(range(start, end + 1))

    pointwise_keyframe_errors = []
    for target in stroke_target_points(meta):
        frame = int(target["frame"])
        point = np.asarray(target.get("true_point", target["point"]), dtype=np.float64)
        if 0 <= frame < len(hand):
            pointwise_keyframe_errors.append(float(np.linalg.norm(hand[frame] - point)))

    stroke_frames_arr = np.asarray(stroke_frames, dtype=np.int64)
    active_arm = posed[:, active["arm"]]
    passive_arm = posed[:, active["passive_arm"]]
    max_ra_step, p95_ra_step, max_ra_accel, p95_ra_accel = step_and_accel(active_arm)
    max_la_step, p95_la_step, max_la_accel, p95_la_accel = step_and_accel(passive_arm)
    max_hand_step, _, max_hand_accel, _ = step_and_accel(hand)
    boundary_step, boundary_accel = boundary_stats(posed, boundaries)
    yaw, yaw_source = root_heading_yaw(data, posed)
    body_yaw = body_yaw_xz(posed)
    tail_yaw = yaw[return_start:]
    tail_body_yaw = body_yaw[return_start:]
    tail_root_yaw_delta = abs(math.degrees(tail_yaw[-1] - tail_yaw[0])) if len(tail_yaw) > 1 else 0.0
    tail_root_yaw_path = float(np.sum(np.abs(np.diff(tail_yaw))) * 180.0 / math.pi) if len(tail_yaw) > 1 else 0.0
    tail_body_yaw_delta = abs(math.degrees(tail_body_yaw[-1] - tail_body_yaw[0])) if len(tail_body_yaw) > 1 else 0.0
    tail_body_yaw_path = float(np.sum(np.abs(np.diff(tail_body_yaw))) * 180.0 / math.pi) if len(tail_body_yaw) > 1 else 0.0
    tail_root_displacement = float(np.linalg.norm(root[-1, [0, 2]] - root[return_start, [0, 2]]))
    final_hand_distance = float(np.linalg.norm(hand[-1] - final_neutral))
    final_hand_height = float(hand[-1, 1])
    final_lower_than_stroke = float(np.mean(hand[stroke_frames_arr, 1]) - final_hand_height) if len(stroke_frames_arr) else 0.0

    elbow = elbow_angle_deg(posed, active)
    elbow_extreme_penalty = float(np.mean(np.maximum(25.0 - elbow, 0.0) / 25.0 + np.maximum(elbow - 170.0, 0.0) / 20.0))
    if local_rot is not None:
        rot_angles = local_rotation_angle(local_rot, active["arm"])
        max_right_arm_local_rot_angle = float(np.max(rot_angles))
        rot_extreme_penalty = float(np.mean(np.maximum(rot_angles - 2.6, 0.0)))
    else:
        max_right_arm_local_rot_angle = 0.0
        rot_extreme_penalty = 0.0

    wrist_penetration = float(np.max(np.maximum(wrist[:, 2] - wall_z, 0.0)))
    hand_penetration = float(np.max(np.maximum(hand[:, 2] - wall_z, 0.0)))
    forearm_penetration = wall_penetration(posed, wall_z, meta)
    right_arm_naturalness = (
        3.0 * elbow_extreme_penalty
        + 1.5 * rot_extreme_penalty
        + max(0.0, max_ra_step - 0.09) * 8.0
        + max(0.0, max_ra_accel - 0.12) * 5.0
        + max(0.0, boundary_step - 0.10) * 6.0
    )

    stroke_line_mean_error = float(np.mean(line_errors)) if line_errors else 0.0
    stroke_line_max_error = float(np.max(line_errors)) if line_errors else 0.0
    stroke_path_ratio = float(np.max(path_ratios)) if path_ratios else 0.0
    stroke_monotonicity = float(np.mean(mono_scores)) if mono_scores else 0.0
    wall_contact_error = float(np.mean(wall_contact_errors)) if wall_contact_errors else 0.0
    wrist_clearance_error = float(np.mean(wrist_clearance_errors)) if wrist_clearance_errors else 0.0
    pointwise_keyframe_error = float(np.mean(pointwise_keyframe_errors)) if pointwise_keyframe_errors else 0.0
    pointwise_keyframe_max_error = float(np.max(pointwise_keyframe_errors)) if pointwise_keyframe_errors else 0.0
    x_range_coverage = float(np.mean(x_range_coverages)) if x_range_coverages else 0.0
    avg_x_progress = float(np.mean(list(row_x_progress.values()))) if row_x_progress else 0.0
    avg_x_coverage = float(np.mean(list(row_x_coverage.values()))) if row_x_coverage else 0.0
    min_x_progress = float(np.min(list(row_x_progress.values()))) if row_x_progress else 0.0
    dead_stroke_count = int(sum(1 for value in row_x_progress.values() if value < 0.35))
    y_offset_error = float(np.mean(y_offset_errors)) if y_offset_errors else 0.0
    row_y_ranges = [metrics["y_range"] for metrics in row_y_metrics.values()]
    row_y_stds = [metrics["y_std"] for metrics in row_y_metrics.values()]
    row_y_endpoint_deltas = [metrics["y_endpoint_delta"] for metrics in row_y_metrics.values()]
    row_y_slopes = [metrics["y_slope"] for metrics in row_y_metrics.values()]
    row_y_curvatures = [metrics["y_curvature"] for metrics in row_y_metrics.values()]
    row_y_range_mean = float(np.mean(row_y_ranges)) if row_y_ranges else 0.0
    row_y_range_max = float(np.max(row_y_ranges)) if row_y_ranges else 0.0
    row_y_std_mean = float(np.mean(row_y_stds)) if row_y_stds else 0.0
    row_y_std_max = float(np.max(row_y_stds)) if row_y_stds else 0.0
    row_y_endpoint_delta_mean = float(np.mean(row_y_endpoint_deltas)) if row_y_endpoint_deltas else 0.0
    row_y_endpoint_delta_max = float(np.max(row_y_endpoint_deltas)) if row_y_endpoint_deltas else 0.0
    row_y_slope_mean = float(np.mean(row_y_slopes)) if row_y_slopes else 0.0
    row_y_slope_max = float(np.max(row_y_slopes)) if row_y_slopes else 0.0
    row_y_curvature_mean = float(np.mean(row_y_curvatures)) if row_y_curvatures else 0.0
    row_y_curvature_max = float(np.max(row_y_curvatures)) if row_y_curvatures else 0.0
    left_arm_noise = max_la_step + 0.5 * max_la_accel

    natural_score = (
        right_arm_naturalness
        + max(0.0, max_ra_step - 0.08) * 12.0
        + max(0.0, max_la_step - 0.08) * 8.0
        + max(0.0, boundary_step - 0.10) * 8.0
        + forearm_penetration * 20.0
    )
    turn_score = tail_root_yaw_delta / 25.0 + tail_root_yaw_path / 65.0 + tail_root_displacement / 0.20
    stroke_score = (
        stroke_line_mean_error / 0.045
        + stroke_line_max_error / 0.09
        + pointwise_keyframe_error / 0.06
        + max(0.0, 0.75 - x_range_coverage) * 2.5
        + y_offset_error / 0.055
        + max(0.0, stroke_path_ratio - 1.35) * 2.0
        + (1.0 - stroke_monotonicity) * 2.0
        + wall_contact_error / 0.05
        + wrist_clearance_error / 0.10
    )
    finish_score = max(0.0, 0.12 - final_lower_than_stroke) * 6.0 + final_hand_distance / 0.35
    composite_score = 4.0 * natural_score + 3.0 * turn_score + 2.0 * stroke_score + finish_score
    brush_likeness_penalty = (
        8.0 * max(0.0, 0.75 - avg_x_progress)
        + 6.0 * max(0.0, 0.70 - avg_x_coverage)
        + 5.0 * float(dead_stroke_count)
        + 4.0 * max(0.0, 0.50 - min_x_progress)
    )
    stroke_path_ratio_penalty = max(0.0, stroke_path_ratio - 1.35)
    brush_motion_score = (
        brush_likeness_penalty
        + 3.0 * stroke_line_mean_error
        + 1.0 * stroke_line_max_error
        + 0.03 * tail_root_yaw_path
        + 0.02 * abs(tail_root_yaw_delta)
        + 0.5 * stroke_path_ratio_penalty
        + 5.0 * forearm_penetration
        + 2.0 * right_arm_naturalness
    )
    flatness_penalty = (
        8.0 * max(0.0, row_y_range_mean / 0.040 - 1.0)
        + 6.0 * max(0.0, row_y_range_max / 0.055 - 1.0)
        + 6.0 * max(0.0, row_y_endpoint_delta_mean / 0.025 - 1.0)
        + 5.0 * max(0.0, row_y_curvature_mean / 0.025 - 1.0)
    )
    flat_brush_score = (
        flatness_penalty
        + 0.7 * brush_likeness_penalty
        + 3.0 * stroke_line_mean_error
        + 1.0 * stroke_line_max_error
        + 0.03 * tail_root_yaw_path
        + 0.02 * abs(tail_root_yaw_delta)
        + 5.0 * forearm_penetration
        + 2.0 * right_arm_naturalness
    )

    return {
        "motion_path": str(path),
        "active_hand": active_hand_name(meta),
        "motion_name": path.name,
        "run_name": meta.get("_run_name")
        or (path.parent.parent.name if path.parent.name.startswith("sample_") else path.parent.name),
        "sample_id": path.parent.name if path.parent.name.startswith("sample_") else path.stem,
        "filtered": bool(filtered),
        "variant": meta.get("variant", ""),
        "heading_mode": meta.get("heading_mode", ""),
        "frame_plan": meta.get("frame_plan", ""),
        "cfg_weight": json.dumps(meta.get("cfg_weight", "")),
        "num_denoising_steps": meta.get("num_denoising_steps", ""),
        "stroke_line_mean_error": stroke_line_mean_error,
        "stroke_line_max_error": stroke_line_max_error,
        "pointwise_keyframe_error": pointwise_keyframe_error,
        "pointwise_keyframe_max_error": pointwise_keyframe_max_error,
        "x_range_coverage": x_range_coverage,
        "row1_x_progress": row_x_progress.get(1, 0.0),
        "row2_x_progress": row_x_progress.get(2, 0.0),
        "row3_x_progress": row_x_progress.get(3, 0.0),
        "row1_x_coverage": row_x_coverage.get(1, 0.0),
        "row2_x_coverage": row_x_coverage.get(2, 0.0),
        "row3_x_coverage": row_x_coverage.get(3, 0.0),
        "avg_x_progress": avg_x_progress,
        "avg_x_coverage": avg_x_coverage,
        "min_x_progress": min_x_progress,
        "dead_stroke_count": dead_stroke_count,
        "row1_y_range": row_y_metrics.get(1, {}).get("y_range", 0.0),
        "row2_y_range": row_y_metrics.get(2, {}).get("y_range", 0.0),
        "row3_y_range": row_y_metrics.get(3, {}).get("y_range", 0.0),
        "row_y_range_mean": row_y_range_mean,
        "row_y_range_max": row_y_range_max,
        "row1_y_std": row_y_metrics.get(1, {}).get("y_std", 0.0),
        "row2_y_std": row_y_metrics.get(2, {}).get("y_std", 0.0),
        "row3_y_std": row_y_metrics.get(3, {}).get("y_std", 0.0),
        "row_y_std_mean": row_y_std_mean,
        "row_y_std_max": row_y_std_max,
        "row1_y_endpoint_delta": row_y_metrics.get(1, {}).get("y_endpoint_delta", 0.0),
        "row2_y_endpoint_delta": row_y_metrics.get(2, {}).get("y_endpoint_delta", 0.0),
        "row3_y_endpoint_delta": row_y_metrics.get(3, {}).get("y_endpoint_delta", 0.0),
        "row_y_endpoint_delta_mean": row_y_endpoint_delta_mean,
        "row_y_endpoint_delta_max": row_y_endpoint_delta_max,
        "row1_y_slope": row_y_metrics.get(1, {}).get("y_slope", 0.0),
        "row2_y_slope": row_y_metrics.get(2, {}).get("y_slope", 0.0),
        "row3_y_slope": row_y_metrics.get(3, {}).get("y_slope", 0.0),
        "row_y_slope_mean": row_y_slope_mean,
        "row_y_slope_max": row_y_slope_max,
        "row1_y_curvature": row_y_metrics.get(1, {}).get("y_curvature", 0.0),
        "row2_y_curvature": row_y_metrics.get(2, {}).get("y_curvature", 0.0),
        "row3_y_curvature": row_y_metrics.get(3, {}).get("y_curvature", 0.0),
        "row_y_curvature_mean": row_y_curvature_mean,
        "row_y_curvature_max": row_y_curvature_max,
        "y_offset_error": y_offset_error,
        "z_contact_error": wall_contact_error,
        "stroke_path_ratio": stroke_path_ratio,
        "stroke_monotonicity": stroke_monotonicity,
        "wall_contact_error": wall_contact_error,
        "wrist_clearance_error": wrist_clearance_error,
        "right_arm_naturalness": right_arm_naturalness,
        "max_right_arm_joint_velocity": max_ra_step,
        "right_arm_joint_velocity_peak": max_ra_step,
        "p95_right_arm_joint_velocity": p95_ra_step,
        "max_right_arm_joint_acceleration": max_ra_accel,
        "right_arm_joint_acceleration_peak": max_ra_accel,
        "p95_right_arm_joint_acceleration": p95_ra_accel,
        "max_right_hand_velocity": max_hand_step,
        "max_right_hand_acceleration": max_hand_accel,
        "transition_boundary_step": boundary_step,
        "boundary_step": boundary_step,
        "transition_boundary_accel": boundary_accel,
        "boundary_accel": boundary_accel,
        "root_yaw_source": yaw_source,
        "tail_root_yaw_delta": tail_root_yaw_delta,
        "tail_root_yaw_path": tail_root_yaw_path,
        "tail_body_yaw_delta": tail_body_yaw_delta,
        "tail_body_yaw_path": tail_body_yaw_path,
        "tail_root_displacement": tail_root_displacement,
        "final_right_hand_height": final_hand_height,
        "final_right_hand_lower_than_stroke_mean": final_lower_than_stroke,
        "final_right_hand_distance_to_neutral": final_hand_distance,
        "wall_penetration": forearm_penetration,
        "right_hand_wall_penetration": hand_penetration,
        "right_wrist_wall_penetration": wrist_penetration,
        "left_arm_noise": left_arm_noise,
        "max_left_arm_velocity": max_la_step,
        "max_left_arm_acceleration": max_la_accel,
        "right_elbow_min_angle_deg": float(np.min(elbow)),
        "right_elbow_max_angle_deg": float(np.max(elbow)),
        "max_right_arm_local_rot_angle_rad": max_right_arm_local_rot_angle,
        "preemphasis": meta.get("preemphasis", ""),
        "x_scale": meta.get("x_scale", ""),
        "y_offset": meta.get("y_offset", ""),
        "z_offset": meta.get("z_offset", ""),
        "z_wrist": meta.get("z_wrist", ""),
        "x_offset": meta.get("x_offset", ""),
        "prompt_variant": meta.get("prompt_variant", ""),
        "brush_likeness_penalty": brush_likeness_penalty,
        "brush_motion_score": brush_motion_score,
        "flatness_penalty": flatness_penalty,
        "flat_brush_score": flat_brush_score,
        "natural_score": natural_score,
        "turn_score": turn_score,
        "stroke_score": stroke_score,
        "finish_score": finish_score,
        "composite_score": composite_score,
    }


def prompt_boundaries(meta: dict) -> list[int]:
    if meta.get("boundaries"):
        return [int(x) for x in meta["boundaries"]]
    total = 0
    boundaries = []
    for count in meta.get("num_frames", [])[:-1]:
        total += int(count)
        boundaries.append(total)
    return boundaries


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    half = size // 2
    xs = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / max(sigma, 1e-6)) ** 2)
    return kernel / np.sum(kernel)


def convolve_time(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    half = len(kernel) // 2
    padded = np.pad(values, [(half, half)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    out = np.zeros_like(values, dtype=np.float64)
    for offset, weight in enumerate(kernel):
        out += weight * padded[offset : offset + len(values)]
    return out


def transition_mask(length: int, boundaries: list[int], radius: int) -> np.ndarray:
    mask = np.zeros((length,), dtype=np.float64)
    for boundary in boundaries:
        start = max(0, boundary - radius)
        end = min(length - 1, boundary + radius)
        for frame in range(start, end + 1):
            distance = abs(frame - boundary)
            value = 0.5 * (1.0 + np.cos(np.pi * distance / (radius + 1)))
            mask[frame] = max(mask[frame], value)
    return mask


def right_arm_hold_frames(meta: dict, length: int) -> set[int]:
    frames = set(target_frames(meta))
    for row in row_intervals(meta):
        start = max(0, int(row["start_frame"]))
        end = min(length - 1, int(row["end_frame"]))
        frames.update(range(start, end + 1))
    for window in meta.get("right_arm_filter_hold_windows", []) or []:
        if len(window) != 2:
            continue
        start = max(0, int(window[0]))
        end = min(length - 1, int(window[1]))
        frames.update(range(start, end + 1))
    return frames


def normalize_heading(heading: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(heading, axis=-1, keepdims=True)
    return heading / np.maximum(norms, 1e-8)


def constraint_aware_transition_filter(
    motion_path: Path,
    output_path: Path,
    meta: dict,
    radius: int,
    kernel_size: int,
    sigma: float,
    strength: float,
    passes: int,
    constrained_right_arm_strength: float,
) -> None:
    data = np.load(motion_path, allow_pickle=True)
    arrays = {key: data[key] for key in data.files}
    posed = arrays["posed_joints"].astype(np.float64)
    length = len(posed)
    boundaries = prompt_boundaries(meta)
    mask = transition_mask(length, boundaries, radius)
    kernel = gaussian_kernel(kernel_size, sigma)
    constrained_frames = right_arm_hold_frames(meta, length)
    active_arm = active_indices(meta)["arm"]

    for key in SMOOTH_KEYS:
        if key not in arrays:
            continue
        original = arrays[key].astype(np.float64, copy=False)
        filtered = original.copy()
        for _ in range(max(1, passes)):
            filtered = convolve_time(filtered, kernel)
        if key == "posed_joints":
            edit = np.broadcast_to(
                np.clip(mask * strength, 0.0, 1.0).reshape(length, 1, 1),
                original.shape,
            ).copy()
            for frame in constrained_frames:
                if 0 <= frame < length:
                    edit[frame, active_arm, :] *= constrained_right_arm_strength
        else:
            edit = np.clip(mask * strength, 0.0, 1.0).reshape((-1,) + (1,) * (original.ndim - 1))
        smoothed = original * (1.0 - edit) + filtered * edit
        if key == "global_root_heading":
            smoothed = normalize_heading(smoothed)
        arrays[key] = smoothed.astype(data[key].dtype, copy=False)

    arrays["constraint_aware_transition_filter_meta"] = np.asarray(
        json.dumps(
            {
                "source_motion": str(motion_path),
                "boundaries": boundaries,
                "radius": radius,
                "kernel_size": int(kernel_size if kernel_size % 2 == 1 else kernel_size + 1),
                "sigma": sigma,
                "strength": strength,
                "passes": passes,
                "constrained_right_arm_strength": constrained_right_arm_strength,
                "active_hand": active_hand_name(meta),
                "constrained_frames": sorted(constrained_frames),
            }
        ),
        dtype=object,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)


def discover_samples(root: Path) -> list[tuple[Path, Path, dict]]:
    samples = []
    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        prompts_path = run_dir / "prompts.json"
        if not prompts_path.exists():
            continue
        meta = load_json(metrics_path)
        prompts = load_json(prompts_path)
        meta = {**prompts, **meta}
        for motion_path in sorted(run_dir.glob("sample_*/motion.npz")):
            samples.append((motion_path, run_dir, meta))
    return samples


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_tag(value: object) -> str:
    try:
        return f"{float(value):.3f}".replace("-", "m").replace(".", "p")
    except Exception:
        return safe_name(value)


def passes_flat_brush_filter(row: dict, tail_yaw_limit: float) -> bool:
    return (
        float(row.get("avg_x_progress", 0.0)) >= 0.65
        and float(row.get("avg_x_coverage", 0.0)) >= 0.65
        and int(float(row.get("dead_stroke_count", 0))) == 0
        and float(row.get("row_y_range_mean", 999.0)) <= 0.040
        and float(row.get("row_y_range_max", 999.0)) <= 0.055
        and float(row.get("row_y_endpoint_delta_mean", 999.0)) <= 0.025
        and float(row.get("row_y_endpoint_delta_max", 999.0)) <= 0.040
        and float(row.get("row_y_curvature_mean", 999.0)) <= 0.025
        and float(row.get("row_y_curvature_max", 999.0)) <= 0.040
        and float(row.get("stroke_line_mean_error", 999.0)) <= 0.060
        and float(row.get("stroke_line_max_error", 999.0)) <= 0.110
        and float(row.get("wall_penetration", 999.0)) <= 0.010
        and float(row.get("right_arm_naturalness", 999.0)) == 0.0
        and float(row.get("tail_root_yaw_path", 999.0)) <= tail_yaw_limit
    )


def flat_brush_rank_rows(rows: list[dict]) -> tuple[list[dict], str]:
    strict = [row for row in rows if passes_flat_brush_filter(row, 20.0)]
    if strict:
        return sorted(strict, key=lambda row: float(row["flat_brush_score"])), "strict_tail20"
    relaxed = [row for row in rows if passes_flat_brush_filter(row, 30.0)]
    if relaxed:
        return sorted(relaxed, key=lambda row: float(row["flat_brush_score"])), "relaxed_tail30"
    return sorted(rows, key=lambda row: float(row["flat_brush_score"])), "score_only_no_pass"


def copy_top(rows: list[dict], target_dir: Path, ranking_label: str, score_key: str, limit: int = 10) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for old_file in target_dir.glob("*.npz"):
        try:
            old_file.unlink()
        except PermissionError:
            pass
    for rank, row in enumerate(rows[:limit], start=1):
        src = Path(row["motion_path"])
        variant = row.get("variant") or "unknown"
        heading = row.get("heading_mode") or "unknown"
        pre = row.get("preemphasis") or "pre?"
        z_wrist = score_tag(row.get("z_wrist") or "?")
        cfg = safe_name(row.get("cfg_weight") or "cfg?")
        sample = safe_name(row.get("sample_id") or src.stem)
        filtered = "transition_filtered" if row.get("filtered") in (True, "True", "true", 1, "1") else "raw"
        name = (
            f"r{rank:02d}_{ranking_label}_{pre}_{sample}_{filtered}_"
            f"orig{score_tag(row.get('composite_score', 0.0))}_"
            f"br{score_tag(row.get('brush_motion_score', 0.0))}_"
            f"fl{score_tag(row.get('flat_brush_score', 0.0))}_"
            f"ln{score_tag(row.get('stroke_line_mean_error', 0.0))}-{score_tag(row.get('stroke_line_max_error', 0.0))}_"
            f"xp{score_tag(row.get('avg_x_progress', 0.0))}_xc{score_tag(row.get('avg_x_coverage', 0.0))}_"
            f"yr{score_tag(row.get('row_y_range_mean', 0.0))}-{score_tag(row.get('row_y_range_max', 0.0))}_"
            f"yd{score_tag(row.get('row_y_endpoint_delta_mean', 0.0))}-{score_tag(row.get('row_y_endpoint_delta_max', 0.0))}_"
            f"yc{score_tag(row.get('row_y_curvature_mean', 0.0))}-{score_tag(row.get('row_y_curvature_max', 0.0))}_"
            f"d{row.get('dead_stroke_count', 0)}_yw{score_tag(row.get('tail_root_yaw_delta', 0.0))}-{score_tag(row.get('tail_root_yaw_path', 0.0))}.npz"
        )
        shutil.copy2(src, target_dir / safe_name(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--make_filtered", action="store_true")
    parser.add_argument("--filtered_dir_name", default="transition_filtered")
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--kernel_size", type=int, default=13)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--constrained_right_arm_strength", type=float, default=0.05)
    parser.add_argument("--ranking_root", type=Path)
    args = parser.parse_args()

    rows = []
    samples = discover_samples(args.root)
    for motion_path, run_dir, meta in samples:
        run_meta = {**meta, "_run_name": run_dir.name}
        rows.append(score_motion(motion_path, run_meta, filtered=False))
        if args.make_filtered:
            sample_name = motion_path.parent.name
            filtered_path = run_dir / args.filtered_dir_name / f"{sample_name}_transition_filtered.npz"
            constraint_aware_transition_filter(
                motion_path,
                filtered_path,
                run_meta,
                radius=args.radius,
                kernel_size=args.kernel_size,
                sigma=args.sigma,
                strength=args.strength,
                passes=args.passes,
                constrained_right_arm_strength=args.constrained_right_arm_strength,
            )
            rows.append(score_motion(filtered_path, run_meta, filtered=True))

    main_rows = [row for row in rows if row.get("variant") in ("endpoint_wrist", "endpoint_only")]
    no_turn_rows = sorted(main_rows, key=lambda row: float(row["composite_score"]))
    brush_rows = sorted(main_rows, key=lambda row: float(row["brush_motion_score"]))
    flat_rows, flat_filter_tier = flat_brush_rank_rows(main_rows)
    for rank, row in enumerate(no_turn_rows, start=1):
        row["best_no_turn_rank"] = rank
    for rank, row in enumerate(brush_rows, start=1):
        row["best_brush_motion_rank"] = rank
    for rank, row in enumerate(flat_rows, start=1):
        row["best_flat_brush_motion_rank"] = rank
        row["best_flat_brush_filter_tier"] = flat_filter_tier

    for row in rows:
        row.setdefault("best_no_turn_rank", "")
        row.setdefault("best_brush_motion_rank", "")
        row.setdefault("best_flat_brush_motion_rank", "")
        row.setdefault("best_flat_brush_filter_tier", flat_filter_tier)

    rows = sorted(rows, key=lambda row: float(row.get("flat_brush_score", row.get("brush_motion_score", row["composite_score"]))))
    summary_path = args.summary or (args.root / "summary.csv")
    write_summary(summary_path, rows)

    ranking_root = args.ranking_root or args.root
    filtered_no_turn = [row for row in no_turn_rows if row["filtered"]]
    filtered_brush = [row for row in brush_rows if row["filtered"]]
    filtered_flat = [row for row in flat_rows if row["filtered"]]
    copy_top(filtered_no_turn, ranking_root / "best_no_turn_top10", "best_no_turn", "composite_score")
    copy_top(filtered_brush, ranking_root / "best_brush_motion_top10", "best_brush_motion", "brush_motion_score")
    copy_top(filtered_flat, ranking_root / "best_flat_brush_motion_top10", "best_flat_brush_motion", "flat_brush_score")
    if ranking_root.resolve() == args.root.resolve():
        raw_rows = [row for row in no_turn_rows if not row["filtered"]]
        filtered_rows = [row for row in no_turn_rows if row["filtered"]]
        copy_top(raw_rows, args.root / "top_10", "best_no_turn_raw", "composite_score")
        copy_top(filtered_rows, args.root / "top_10_filtered", "best_no_turn_filtered", "composite_score")
    print(summary_path)
    if rows:
        print(json.dumps(rows[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
