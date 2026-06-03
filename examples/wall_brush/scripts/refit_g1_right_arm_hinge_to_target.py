#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kimodo.exports.motion_io import complete_motion_dict, save_kimodo_npz
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton.definitions import G1Skeleton34


RIGHT_ARM = [26, 27, 28, 29, 30, 31, 32, 33]
RIGHT_ARM_PHYSICAL = [26, 27, 28, 29, 30, 31, 32]
RIGHT_WRIST = [30, 31, 32]
RIGHT_HAND = 33


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_constraint_points(recipe_path: Path) -> list[dict[str, Any]]:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    candidate = recipe.get("candidate", recipe)
    points = []
    for constraint in candidate.get("constraints", []):
        if constraint.get("end_effector") != "right-hand":
            continue
        if "brush_stroke" not in str(constraint.get("role", "")) and "row_" not in str(constraint.get("label", "")):
            continue
        points.append(
            {
                "label": constraint.get("label", f"frame_{constraint.get('frame')}"),
                "frame": int(constraint["frame"]),
                "point": [float(value) for value in constraint.get("position", [])],
            }
        )
    points.sort(key=lambda item: item["frame"])
    return points


def build_edit_mask(
    source_posed: np.ndarray,
    target_posed: np.ndarray,
    threshold: float,
    pad: int,
    taper_frames: int,
) -> tuple[np.ndarray, int, int]:
    hand_delta = np.linalg.norm(target_posed[:, RIGHT_HAND] - source_posed[:, RIGHT_HAND], axis=-1)
    active = np.where(hand_delta > threshold)[0]
    if active.size == 0:
        mask = np.zeros(source_posed.shape[0], dtype=np.float32)
        return mask, 0, 0
    start = max(0, int(active.min()) - pad)
    end = min(source_posed.shape[0] - 1, int(active.max()) + pad)
    mask = np.zeros(source_posed.shape[0], dtype=np.float32)
    mask[start : end + 1] = 1.0
    taper_frames = max(0, int(taper_frames))
    if taper_frames > 0:
        ramp = min(taper_frames, max(0, end - start))
        for offset in range(ramp):
            phase = float(offset + 1) / float(ramp + 1)
            weight = 0.5 - 0.5 * np.cos(np.pi * phase)
            mask[start + offset] = min(mask[start + offset], weight)
            mask[end - offset] = min(mask[end - offset], weight)
    return mask, start, end


def extract_hinge_angles(
    converter: MujocoQposConverter,
    local_rot_mats: torch.Tensor,
) -> torch.Tensor:
    kimodo_indices = converter._mujoco_indices_to_kimodo_indices.to(device=local_rot_mats.device)
    rot_offsets = converter._rot_offsets_f2q.to(device=local_rot_mats.device, dtype=local_rot_mats.dtype)
    axis_f2q = converter._mujoco_joint_axis_values_f2q_space.to(
        device=local_rot_mats.device,
        dtype=local_rot_mats.dtype,
    )
    hinge_local = local_rot_mats[:, kimodo_indices.long()]
    hinge_f2q = rot_offsets[kimodo_indices.long()][None] @ hinge_local
    return converter._local_rots_to_joint_dofs_axis_angle(hinge_f2q, axis_f2q)


def reconstruct_with_opt_hinges(
    converter: MujocoQposConverter,
    source_local: torch.Tensor,
    all_angles: torch.Tensor,
    opt_kimodo_joints: list[int],
) -> torch.Tensor:
    projected = converter._joint_dofs_to_local_rot_mats(
        all_angles.unsqueeze(0),
        source_local.unsqueeze(0),
        source_local.device,
        source_local.dtype,
        use_relative=False,
    ).squeeze(0)
    out = source_local.clone()
    out[:, opt_kimodo_joints] = projected[:, opt_kimodo_joints]
    return out


def hinge_indices_for_joints(converter: MujocoQposConverter, joints: list[int]) -> list[int]:
    indices = []
    for joint in joints:
        one_based = int(converter._kimodo_indices_to_mujoco_indices[joint].item())
        if one_based <= 0:
            raise ValueError(f"Kimodo joint {joint} is not a physical G1 hinge joint.")
        indices.append(one_based - 1)
    return indices


def line_distance(hand: np.ndarray, constraints: list[dict[str, Any]]) -> dict[str, float]:
    if len(constraints) < 2:
        return {"mean_m": 0.0, "max_m": 0.0}
    start_frame = constraints[0]["frame"]
    end_frame = constraints[-1]["frame"]
    start = np.asarray(constraints[0]["point"], dtype=np.float64)
    end = np.asarray(constraints[-1]["point"], dtype=np.float64)
    segment = hand[start_frame : end_frame + 1]
    direction = end - start
    denom = float(direction @ direction)
    if denom < 1e-12:
        distances = np.linalg.norm(segment - start[None], axis=-1)
    else:
        t = np.clip(((segment - start[None]) @ direction) / denom, 0.0, 1.0)
        projection = start[None] + t[:, None] * direction[None]
        distances = np.linalg.norm(segment - projection, axis=-1)
    return {"mean_m": float(distances.mean()), "max_m": float(distances.max())}


def stroke_progress_metrics(hand: np.ndarray, constraints: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    if len(constraints) < 2:
        return {
            "progress_axis": "unknown",
            "progress_direction": [0.0, 0.0, 0.0],
            "progress_start_m": 0.0,
            "progress_end_m": 0.0,
            "progress_span_m": 0.0,
            "progress_backstep_count": 0,
            "progress_backstep_total_m": 0.0,
        }
    target_start = np.asarray(constraints[0]["point"], dtype=np.float64)
    target_end = np.asarray(constraints[-1]["point"], dtype=np.float64)
    stroke = hand[start : end + 1].astype(np.float64)
    direction = target_end - target_start
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        progress = np.zeros((stroke.shape[0],), dtype=np.float64)
        unit = np.zeros((3,), dtype=np.float64)
    else:
        unit = direction / norm
        progress = (stroke - target_start[None]) @ unit
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


def summarize(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    output: dict[str, np.ndarray],
    constraints: list[dict[str, Any]],
    mask: np.ndarray,
    start: int,
    end: int,
    history: list[dict[str, float]],
    params: dict[str, Any],
) -> dict[str, Any]:
    source_posed = source["posed_joints"].astype(np.float64)
    target_posed = target["posed_joints"].astype(np.float64)
    out_posed = output["posed_joints"].astype(np.float64)
    active = np.where(mask > 0.5)[0]
    if active.size == 0:
        active = np.arange(out_posed.shape[0])
    hand = out_posed[:, RIGHT_HAND]
    target_hand = target_posed[:, RIGHT_HAND]
    hand_target_error = np.linalg.norm(hand[active] - target_hand[active], axis=-1)
    right_arm_delta = np.linalg.norm(out_posed[:, RIGHT_ARM] - source_posed[:, RIGHT_ARM], axis=-1)

    key_error = {}
    if constraints:
        frames = [item["frame"] for item in constraints]
        points = np.asarray([item["point"] for item in constraints], dtype=np.float64)
        values = np.linalg.norm(hand[frames] - points, axis=-1)
        key_error = {
            "mean_m": float(values.mean()),
            "max_m": float(values.max()),
            "per_point_m": {item["label"]: float(values[idx]) for idx, item in enumerate(constraints)},
        }

    stroke = hand[start : end + 1]
    steps = np.linalg.norm(np.diff(stroke, axis=0), axis=-1)
    dx = np.diff(stroke[:, 0])
    return {
        "method": "G1_RIGHT_ARM_HINGE_DOF_TARGET_REFIT",
        "params": params,
        "edit_start_frame": int(start),
        "edit_end_frame": int(end),
        "constraint_key_error": key_error,
        "line_distance": line_distance(hand, constraints),
        "target_hand_error": {
            "mean_m": float(hand_target_error.mean()),
            "max_m": float(hand_target_error.max()),
        },
        "stroke_hand_speed": {
            "mean_step_m": float(steps.mean()) if steps.size else 0.0,
            "std_step_m": float(steps.std()) if steps.size else 0.0,
            "cv": float(steps.std() / max(steps.mean(), 1e-8)) if steps.size else 0.0,
            "max_step_m": float(steps.max()) if steps.size else 0.0,
            "x_backstep_count": int(np.sum(dx < -1e-5)),
            "x_backstep_total_m": float(np.maximum(-dx, 0.0).sum()),
            **stroke_progress_metrics(hand, constraints, start, end),
        },
        "max_right_arm_joint_delta_m": float(right_arm_delta.max()),
        "mean_right_arm_joint_delta_m": float(right_arm_delta.mean()),
        "loss_tail": history[-12:],
        "note": "The optimized joints are reconstructed from G1 physical hinge DOFs only; this prevents arbitrary SO(3) shoulder/elbow/wrist rotations.",
    }


def refit(
    source_motion: Path,
    target_motion: Path,
    recipe: Path,
    output: Path,
    report: Path,
    device: str,
    steps: int,
    lr: float,
    pad: int,
    threshold: float,
    taper_frames: int,
    fps: float,
    opt_joints: list[int],
    target_loss_weight: float,
    hand_loss_weight: float,
    pose_prior_weight: float,
    angle_prior_weight: float,
    angle_vel_weight: float,
    angle_acc_weight: float,
    hand_acc_weight: float,
) -> None:
    source_npz = np.load(source_motion, allow_pickle=False)
    target_npz = np.load(target_motion, allow_pickle=False)
    source = {key: np.asarray(source_npz[key]) for key in source_npz.files}
    target = {key: np.asarray(target_npz[key]) for key in target_npz.files}
    mask_np, start, end = build_edit_mask(source["posed_joints"], target["posed_joints"], threshold, pad, taper_frames)
    if not np.any(mask_np):
        raise ValueError("Target motion does not differ from source motion; no refit window found.")

    import kimodo

    kimodo_root = Path(inspect.getfile(kimodo)).resolve().parent
    skeleton = G1Skeleton34(folder=str(kimodo_root / "assets" / "skeletons" / "g1skel34"), load=True).to(device)
    converter = MujocoQposConverter(skeleton)

    source_local = torch.tensor(source["local_rot_mats"], device=device, dtype=torch.float32)
    source_root = torch.tensor(source["root_positions"], device=device, dtype=torch.float32)
    source_posed = torch.tensor(source["posed_joints"], device=device, dtype=torch.float32)
    target_posed = torch.tensor(target["posed_joints"], device=device, dtype=torch.float32)
    mask = torch.tensor(mask_np, device=device, dtype=torch.float32)
    active = mask > 0.5

    opt_hinges = hinge_indices_for_joints(converter, opt_joints)
    source_angles_all = extract_hinge_angles(converter, source_local).detach()
    rest = converter._rest_dofs_axis_angle.to(device=device, dtype=torch.float32)
    lo = converter._joint_limits_min.to(device=device, dtype=torch.float32) + rest
    hi = converter._joint_limits_max.to(device=device, dtype=torch.float32) + rest
    base_angles_all = source_angles_all.clone()
    base_angles_all[:, opt_hinges] = torch.clamp(base_angles_all[:, opt_hinges], lo[opt_hinges], hi[opt_hinges])

    delta = torch.zeros((source_local.shape[0], len(opt_hinges)), device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    history: list[dict[str, float]] = []

    for step in range(steps):
        opt.zero_grad()
        all_angles = base_angles_all.clone()
        proposed = base_angles_all[:, opt_hinges] + delta * mask[:, None]
        proposed = torch.max(torch.min(proposed, hi[opt_hinges][None]), lo[opt_hinges][None])
        all_angles[:, opt_hinges] = proposed
        local = reconstruct_with_opt_hinges(converter, source_local, all_angles, opt_joints)
        _global, posed, _ = skeleton.fk(local, source_root)

        arm_loss = (posed[active][:, RIGHT_ARM] - target_posed[active][:, RIGHT_ARM]).square().mean()
        hand_loss = (posed[active, RIGHT_HAND] - target_posed[active, RIGHT_HAND]).square().mean()
        pose_prior = (posed[active][:, RIGHT_ARM] - source_posed[active][:, RIGHT_ARM]).square().mean()
        active_delta = delta * mask[:, None]
        angle_prior = active_delta.square().mean()
        angle_vel = (active_delta[1:] - active_delta[:-1]).square().mean()
        angle_acc = (active_delta[2:] - 2.0 * active_delta[1:-1] + active_delta[:-2]).square().mean()
        hand_acc = posed[2:, RIGHT_HAND] - 2.0 * posed[1:-1, RIGHT_HAND] + posed[:-2, RIGHT_HAND]
        hand_acc_mask = ((mask[2:] + mask[1:-1] + mask[:-2]) / 3.0).clamp_min(1e-8)
        hand_acc_loss = (hand_acc.square().sum(dim=-1) * hand_acc_mask).sum() / hand_acc_mask.sum()
        loss = (
            target_loss_weight * arm_loss
            + hand_loss_weight * hand_loss
            + pose_prior_weight * pose_prior
            + angle_prior_weight * angle_prior
            + angle_vel_weight * angle_vel
            + angle_acc_weight * angle_acc
            + hand_acc_weight * hand_acc_loss
        )
        loss.backward()
        opt.step()

        with torch.no_grad():
            delta.mul_(mask[:, None])
            clamped = torch.max(
                torch.min(base_angles_all[:, opt_hinges] + delta, hi[opt_hinges][None]),
                lo[opt_hinges][None],
            )
            delta.copy_(clamped - base_angles_all[:, opt_hinges])

        if step % 25 == 0 or step == steps - 1:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "arm": float(arm_loss.detach().cpu()),
                    "hand": float(hand_loss.detach().cpu()),
                    "pose_prior": float(pose_prior.detach().cpu()),
                    "angle_prior": float(angle_prior.detach().cpu()),
                    "angle_vel": float(angle_vel.detach().cpu()),
                    "angle_acc": float(angle_acc.detach().cpu()),
                    "hand_acc": float(hand_acc_loss.detach().cpu()),
                }
            )

    with torch.no_grad():
        all_angles = base_angles_all.clone()
        all_angles[:, opt_hinges] = torch.max(
            torch.min(base_angles_all[:, opt_hinges] + delta * mask[:, None], hi[opt_hinges][None]),
            lo[opt_hinges][None],
        )
        final_local = reconstruct_with_opt_hinges(converter, source_local, all_angles, opt_joints)
        out_tensors = complete_motion_dict(final_local, source_root, skeleton, fps=fps)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_kimodo_npz(str(output), out_tensors)
    out_np = {
        key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        for key, value in out_tensors.items()
    }
    constraints = load_constraint_points(recipe)
    params = {
        "source_motion": str(source_motion),
        "target_motion": str(target_motion),
        "opt_joints": opt_joints,
        "opt_hinge_indices": opt_hinges,
        "steps": steps,
        "lr": lr,
        "pad": pad,
        "threshold": threshold,
        "taper_frames": taper_frames,
        "fps": fps,
        "target_loss_weight": target_loss_weight,
        "hand_loss_weight": hand_loss_weight,
        "pose_prior_weight": pose_prior_weight,
        "angle_prior_weight": angle_prior_weight,
        "angle_vel_weight": angle_vel_weight,
        "angle_acc_weight": angle_acc_weight,
        "hand_acc_weight": hand_acc_weight,
    }
    summary = summarize(source, target, out_np, constraints, mask_np, start, end, history, params)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "key": summary["constraint_key_error"],
                "line": summary["line_distance"],
                "target_hand_error": summary["target_hand_error"],
                "speed": summary["stroke_hand_speed"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit a right-arm target using only legal G1 hinge DoFs.")
    parser.add_argument("--source_motion", required=True, type=Path)
    parser.add_argument("--target_motion", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.018)
    parser.add_argument("--pad", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--taper_frames", type=int, default=3)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--opt_joints", default="26,27,28,29,30,31,32")
    parser.add_argument("--target_loss_weight", type=float, default=700.0)
    parser.add_argument("--hand_loss_weight", type=float, default=9000.0)
    parser.add_argument("--pose_prior_weight", type=float, default=12.0)
    parser.add_argument("--angle_prior_weight", type=float, default=18.0)
    parser.add_argument("--angle_vel_weight", type=float, default=90.0)
    parser.add_argument("--angle_acc_weight", type=float, default=140.0)
    parser.add_argument("--hand_acc_weight", type=float, default=2200.0)
    args = parser.parse_args()
    refit(
        source_motion=args.source_motion,
        target_motion=args.target_motion,
        recipe=args.recipe,
        output=args.output,
        report=args.report,
        device=args.device,
        steps=args.steps,
        lr=args.lr,
        pad=args.pad,
        threshold=args.threshold,
        taper_frames=args.taper_frames,
        fps=args.fps,
        opt_joints=parse_int_list(args.opt_joints),
        target_loss_weight=args.target_loss_weight,
        hand_loss_weight=args.hand_loss_weight,
        pose_prior_weight=args.pose_prior_weight,
        angle_prior_weight=args.angle_prior_weight,
        angle_vel_weight=args.angle_vel_weight,
        angle_acc_weight=args.angle_acc_weight,
        hand_acc_weight=args.hand_acc_weight,
    )


if __name__ == "__main__":
    main()
