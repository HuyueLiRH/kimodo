# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable wall-brushing demo presets and native end-effector constraints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import torch

from kimodo.constraints import RightHandConstraintSet


WALL_BRUSH_ONE_ROW_PRESET: dict[str, Any] = {
    "name": "wall_brush_one_row_demo_native_right_hand_raw",
    "description": (
        "A reproducible one-row wall-brushing raw KIMODO/G1 generation preset. "
        "It uses the same native RightHand End-Effectors constraint route as the demo UI."
    ),
    "model": "kimodo-g1-rp",
    "constraint_route": "demo_native_right_hand",
    "seed": 7023,
    "num_samples": 1,
    "segments": [30, 42, 30],
    "num_denoising_steps": 200,
    "cfg_type": "separated",
    "cfg_weight": [2.4, 4.0],
    "num_transition_frames": 3,
    "post_processing": False,
    "prompts": [
        (
            "A person stands balanced in place in front of a small wall patch, keeps the left arm relaxed by the side, "
            "and moves the right hand directly toward the left edge of the wall patch."
        ),
        (
            "A person keeps the left arm relaxed by the side while the right hand brushes one short straight "
            "horizontal stroke from left to right on a wall."
        ),
        (
            "A person keeps the left arm relaxed by the side, lowers the right hand directly to a neutral resting "
            "position beside the right thigh, and stops."
        ),
    ],
    "right_hand_targets": [
        {
            "label": "approach_1",
            "frame": 10,
            "point": [-0.1745, 0.72, 0.295],
            "true_point": [-0.155, 0.89, 0.295],
            "wrist_point": None,
            "use_wrist": False,
        },
        {
            "label": "approach_2",
            "frame": 20,
            "point": [-0.1308, 0.738, 0.312],
            "true_point": [-0.132, 0.908, 0.312],
            "wrist_point": None,
            "use_wrist": False,
        },
        {
            "label": "approach_final",
            "frame": 29,
            "point": [-0.108, 0.75, 0.32],
            "true_point": [-0.12, 0.92, 0.32],
            "wrist_point": None,
            "use_wrist": False,
        },
        {
            "label": "row_1_point_1",
            "frame": 36,
            "point": [-0.108, 0.75, 0.32],
            "true_point": [-0.12, 0.92, 0.32],
            "wrist_point": [-0.108, 0.735, 0.24],
            "use_wrist": True,
        },
        {
            "label": "row_1_point_2",
            "frame": 42,
            "point": [-0.0168, 0.75, 0.32],
            "true_point": [-0.072, 0.92, 0.32],
            "wrist_point": None,
            "use_wrist": False,
        },
        {
            "label": "row_1_point_3",
            "frame": 48,
            "point": [0.0744, 0.75, 0.32],
            "true_point": [-0.024, 0.92, 0.32],
            "wrist_point": None,
            "use_wrist": False,
        },
        {
            "label": "row_1_point_4",
            "frame": 54,
            "point": [0.1656, 0.75, 0.32],
            "true_point": [0.024, 0.92, 0.32],
            "wrist_point": None,
            "use_wrist": False,
        },
        {
            "label": "row_1_point_5",
            "frame": 60,
            "point": [0.2568, 0.75, 0.32],
            "true_point": [0.072, 0.92, 0.32],
            "wrist_point": None,
            "use_wrist": False,
        },
        {
            "label": "row_1_point_6",
            "frame": 66,
            "point": [0.348, 0.75, 0.32],
            "true_point": [0.12, 0.92, 0.32],
            "wrist_point": [0.348, 0.735, 0.24],
            "use_wrist": True,
        },
        {
            "label": "return_final",
            "frame": 101,
            "point": [-0.14633791, 0.86423099, 0.29977581],
            "true_point": [-0.14633791, 0.86423099, 0.29977581],
            "wrist_point": None,
            "use_wrist": False,
        },
    ],
}


def make_demo_default_pose(skeleton) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the demo UI's default G1 pose as ``(global_rots, global_positions)``."""
    local_rots = skeleton.rest_pose_local_rot.clone()
    root_positions = torch.zeros((1, 3), device=local_rots.device, dtype=local_rots.dtype)
    global_rots, global_positions, _ = skeleton.fk(local_rots.unsqueeze(0), root_positions)
    global_rots = global_rots[0]
    global_positions = global_positions[0]
    global_positions = global_positions.clone()
    global_positions[:, 1] -= global_positions[:, 1].min()
    return global_rots, global_positions


def _right_hand_position_indices(skeleton) -> list[int]:
    _, position_joint_names = skeleton.expand_joint_names(["RightHand"])
    return [skeleton.bone_index[joint_name] for joint_name in position_joint_names]


def _target_tensor(target: dict[str, Any], key: str, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(target[key], dtype=dtype, device=device)


def build_demo_native_right_hand_constraint(
    skeleton,
    right_hand_targets: Iterable[dict[str, Any]] | None = None,
) -> RightHandConstraintSet:
    """Build the same native RightHand End-Effectors constraint used by the demo UI."""
    targets = list(right_hand_targets or WALL_BRUSH_ONE_ROW_PRESET["right_hand_targets"])
    if not targets:
        raise ValueError("At least one right-hand target is required.")

    default_rots, default_positions = make_demo_default_pose(skeleton)
    dtype = default_positions.dtype
    device = default_positions.device
    right_hand_indices = _right_hand_position_indices(skeleton)
    if not right_hand_indices:
        raise ValueError("The skeleton did not expand RightHand to any position joints.")

    wrist_index = right_hand_indices[0]
    endpoint_index = right_hand_indices[-1]
    frame_indices: list[int] = []
    positions = []
    rotations = []

    for target in targets:
        point = _target_tensor(target, "point", dtype=dtype, device=device)
        frame_indices.append(int(target["frame"]))

        frame_positions = default_positions.clone()
        delta = point - default_positions[endpoint_index]
        for joint_index in right_hand_indices:
            frame_positions[joint_index] = default_positions[joint_index] + delta
        frame_positions[endpoint_index] = point

        wrist_point = target.get("wrist_point")
        if target.get("use_wrist", wrist_point is not None) and wrist_point is not None:
            frame_positions[wrist_index] = torch.tensor(wrist_point, dtype=dtype, device=device)

        positions.append(frame_positions)
        rotations.append(default_rots.clone())

    return RightHandConstraintSet(
        skeleton,
        frame_indices=torch.tensor(frame_indices, dtype=torch.long),
        global_joints_positions=torch.stack(positions, dim=0),
        global_joints_rots=torch.stack(rotations, dim=0),
        smooth_root_2d=None,
    )


def build_wall_brush_generation_kwargs(
    model,
    preset: dict[str, Any] | None = None,
    *,
    num_samples: int | None = None,
) -> dict[str, Any]:
    """Return raw model kwargs for the one-row wall-brushing demo preset."""
    preset = deepcopy(preset or WALL_BRUSH_ONE_ROW_PRESET)
    samples = int(num_samples if num_samples is not None else preset["num_samples"])
    prompts = list(preset["prompts"])
    segments = list(preset["segments"])
    return {
        "prompts": prompts,
        "generation_prompt": prompts,
        "segments": segments,
        "constraint_lst": [
            build_demo_native_right_hand_constraint(model.skeleton, preset["right_hand_targets"]),
        ],
        "num_denoising_steps": int(preset["num_denoising_steps"]),
        "num_samples": samples,
        "multi_prompt": True,
        "num_transition_frames": int(preset["num_transition_frames"]),
        "post_processing": bool(preset["post_processing"]),
        "return_numpy": True,
        "cfg_type": preset["cfg_type"],
        "cfg_weight": list(preset["cfg_weight"]),
    }


def wall_brush_preset_metadata(preset: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a JSON-friendly copy of the raw one-row wall-brushing preset."""
    return deepcopy(preset or WALL_BRUSH_ONE_ROW_PRESET)
