# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Web-equivalent raw generation helpers for scripted KIMODO experiments."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from kimodo.constraints import (
    LeftFootConstraintSet,
    LeftHandConstraintSet,
    RightFootConstraintSet,
    RightHandConstraintSet,
)
from kimodo.sanitize import sanitize_texts


@dataclass(frozen=True)
class EndEffectorTarget:
    frame: int
    point: list[float]
    label: str | None = None
    true_point: list[float] | None = None
    wrist_point: list[float] | None = None
    use_wrist: bool = False


@dataclass(frozen=True)
class EndEffectorSpec:
    type: str
    targets: list[EndEffectorTarget]


@dataclass(frozen=True)
class WebEquivalentTaskSpec:
    prompts: list[str]
    segments: list[int]
    end_effector: EndEffectorSpec
    model: str = "kimodo-g1-rp"
    seed: int = 7023
    num_samples: int = 1
    diffusion_steps: int = 200
    cfg_type: str = "separated"
    cfg_weight: list[float] = field(default_factory=lambda: [2.4, 4.0])
    num_transition_frames: int = 3
    post_processing: bool = False


END_EFFECTOR_TYPES = {
    "left-hand": LeftHandConstraintSet,
    "right-hand": RightHandConstraintSet,
    "left-foot": LeftFootConstraintSet,
    "right-foot": RightFootConstraintSet,
}


class CachedDemoTextEncoder:
    """Text encoder shim that reuses demo embedding cache entries without loading the LLM."""

    def __init__(
        self,
        model_names: Iterable[str],
        cache_root: str | Path = "/root/.cache/kimodo_demo/embeddings",
        encoder_id: str = "LLM2VecEncoder",
    ) -> None:
        self.model_names = list(model_names)
        self.cache_root = Path(cache_root)
        self.encoder_id = encoder_id
        self.device: str | torch.device = "cpu"
        self.dtype = torch.float32

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(self, texts: str | list[str]):
        if isinstance(texts, str):
            texts = [texts]
        texts = sanitize_texts(list(texts))
        arrays = []
        lengths = []
        for text in texts:
            entry_path = self._find_entry(text)
            if entry_path is None:
                raise FileNotFoundError(f"No cached demo text embedding found for prompt: {text!r}")
            array = np.load(entry_path)
            arrays.append(array)
            lengths.append(int(array.shape[0]))

        max_len = max(lengths)
        feat_dim = int(arrays[0].shape[-1])
        padded = np.zeros((len(arrays), max_len, feat_dim), dtype=arrays[0].dtype)
        for index, array in enumerate(arrays):
            padded[index, : array.shape[0]] = array
        return torch.from_numpy(padded).to(device=self.device, dtype=self.dtype), lengths

    def _find_entry(self, text: str) -> Path | None:
        for model_name in self.model_names:
            key = hashlib.sha256(f"{model_name}|{self.encoder_id}|{text}".encode("utf-8")).hexdigest()
            candidate = self.cache_root / model_name / f"{key}.npy"
            if candidate.exists():
                return candidate
        return None


def load_task_spec(path: str | Path) -> WebEquivalentTaskSpec:
    return load_task_spec_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_task_spec_dict(data: dict[str, Any]) -> WebEquivalentTaskSpec:
    prompts = list(data.get("prompts") or data.get("texts") or [])
    segments = [int(frame_count) for frame_count in data.get("segments") or data.get("num_frames") or []]
    if not prompts:
        raise ValueError("Task spec must include a non-empty prompts or texts list.")
    if len(prompts) != len(segments):
        raise ValueError("Task spec prompts and segments must have the same length.")

    effector_data = dict(data.get("end_effector") or {})
    effector_type = str(effector_data.get("type") or data.get("active_end_effector") or "").strip()
    target_data = effector_data.get("targets") or data.get("targets") or []
    if effector_type not in END_EFFECTOR_TYPES:
        supported = ", ".join(sorted(END_EFFECTOR_TYPES))
        raise ValueError(f"Unsupported end effector type {effector_type!r}. Supported: {supported}.")
    if not target_data:
        raise ValueError("Task spec must include at least one end-effector target.")

    targets = [_load_target(target) for target in target_data]
    cfg_data = dict(data.get("cfg") or {})
    cfg_weight = cfg_data.get("weight")
    if cfg_weight is None and "cfg_weight" in data:
        cfg_weight = data["cfg_weight"]
    if cfg_weight is None:
        cfg_weight = [2.4, 4.0]

    return WebEquivalentTaskSpec(
        model=str(data.get("model", "kimodo-g1-rp")),
        seed=int(data.get("seed", 7023)),
        num_samples=int(data.get("num_samples", 1)),
        prompts=prompts,
        segments=segments,
        diffusion_steps=int(data.get("diffusion_steps", data.get("num_denoising_steps", 200))),
        cfg_type=str(cfg_data.get("type", data.get("cfg_type", "separated"))),
        cfg_weight=[float(weight) for weight in cfg_weight],
        num_transition_frames=int(data.get("num_transition_frames", 3)),
        post_processing=bool(data.get("post_processing", data.get("postprocess", False))),
        end_effector=EndEffectorSpec(type=effector_type, targets=targets),
    )


def _load_target(data: dict[str, Any]) -> EndEffectorTarget:
    wrist_point = data.get("wrist_point")
    return EndEffectorTarget(
        frame=int(data["frame"]),
        point=[float(value) for value in data["point"]],
        label=data.get("label") or data.get("name"),
        true_point=data.get("true_point"),
        wrist_point=wrist_point,
        use_wrist=bool(data.get("use_wrist", wrist_point is not None)),
    )


def build_generation_kwargs(model, spec: WebEquivalentTaskSpec) -> dict[str, Any]:
    return {
        "prompts": list(spec.prompts),
        "generation_prompt": list(spec.prompts),
        "segments": list(spec.segments),
        "constraint_lst": [build_end_effector_constraint(model.skeleton, spec.end_effector)],
        "num_denoising_steps": int(spec.diffusion_steps),
        "num_samples": int(spec.num_samples),
        "multi_prompt": True,
        "num_transition_frames": int(spec.num_transition_frames),
        "post_processing": bool(spec.post_processing),
        "return_numpy": True,
        "cfg_type": spec.cfg_type,
        "cfg_weight": list(spec.cfg_weight),
    }


def build_end_effector_constraint(skeleton, end_effector: EndEffectorSpec):
    cls = END_EFFECTOR_TYPES[end_effector.type]
    return _build_translated_end_effector_constraint(skeleton, cls, end_effector.targets)


def make_demo_default_pose(skeleton) -> tuple[torch.Tensor, torch.Tensor]:
    local_rots = skeleton.rest_pose_local_rot.clone()
    root_positions = torch.zeros((1, 3), device=local_rots.device, dtype=local_rots.dtype)
    global_rots, global_positions, _ = skeleton.fk(local_rots.unsqueeze(0), root_positions)
    global_rots = global_rots[0]
    global_positions = global_positions[0].clone()
    global_positions[:, 1] -= global_positions[:, 1].min()
    return global_rots, global_positions


def _build_translated_end_effector_constraint(skeleton, cls, targets: Iterable[EndEffectorTarget]):
    targets = list(targets)
    if not targets:
        raise ValueError("At least one end-effector target is required.")

    default_rots, default_positions = make_demo_default_pose(skeleton)
    _, position_joint_names = skeleton.expand_joint_names(cls.joint_names)
    position_indices = [skeleton.bone_index[joint_name] for joint_name in position_joint_names]
    if not position_indices:
        raise ValueError(f"The skeleton did not expand {cls.joint_names!r} to any position joints.")

    helper_index = position_indices[0]
    endpoint_index = position_indices[-1]
    frame_indices: list[int] = []
    positions = []
    rotations = []
    for target in targets:
        point = torch.tensor(target.point, dtype=default_positions.dtype, device=default_positions.device)
        frame_indices.append(int(target.frame))

        frame_positions = default_positions.clone()
        delta = point - default_positions[endpoint_index]
        for joint_index in position_indices:
            frame_positions[joint_index] = default_positions[joint_index] + delta
        frame_positions[endpoint_index] = point

        if target.use_wrist and target.wrist_point is not None:
            frame_positions[helper_index] = torch.tensor(
                target.wrist_point,
                dtype=default_positions.dtype,
                device=default_positions.device,
            )

        positions.append(frame_positions)
        rotations.append(default_rots.clone())

    return cls(
        skeleton,
        frame_indices=torch.tensor(frame_indices, dtype=torch.long),
        global_joints_positions=torch.stack(positions, dim=0),
        global_joints_rots=torch.stack(rotations, dim=0),
        smooth_root_2d=None,
    )
