#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ.setdefault("HUGGINGFACE_CACHE_DIR", "/root/autodl-tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LOCAL_CACHE", "True")
os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
os.environ.setdefault("TEXT_ENCODER_MODE", "local")

from kimodo import load_model
from kimodo.constraints import RightHandConstraintSet, Root2DConstraintSet
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.model.registry import resolve_model_name
from kimodo.sanitize import sanitize_texts
from kimodo.tools import seed_everything


PROMPTS = [
    "A person stands still close to a wall and raises the right hand toward a small patch on the wall, preparing to wipe the wall with the right hand.",
    "A person stands still facing a wall with the right hand touching a small wall patch, sliding the right hand in one short straight horizontal wiping stroke from left to right.",
    "A person stands still facing a wall with the right hand touching the same small wall patch, sliding the right hand in one short straight horizontal wiping stroke from right to left slightly lower on the wall.",
    "A person stands still facing a wall with the right hand touching the same small wall patch, sliding the right hand in one short straight horizontal wiping stroke from left to right slightly lower on the wall.",
    "A person stands still facing a wall with the right hand near the wall, gently lowering the right hand down beside the right thigh and relaxing the arm.",
]

FULL_WIDTH_PROMPTS = [
    "A person stands still close to a wall and raises the right hand toward a small patch on the wall, preparing to wipe the wall with the right hand.",
    "A person stands still facing a wall with the right hand touching the left edge of a small wall patch, sliding the right hand across the full width of the patch in one short straight horizontal wiping stroke from left to right.",
    "A person stands still facing a wall with the right hand touching the right edge of the same small wall patch, sliding the right hand across the full width of the patch in one short straight horizontal wiping stroke from right to left slightly lower on the wall.",
    "A person stands still facing a wall with the right hand touching the left edge of the same small wall patch, sliding the right hand across the full width of the patch in one short straight horizontal wiping stroke from left to right slightly lower on the wall.",
    "A person stands still facing a wall with the right hand near the wall, gently lowering the right hand down beside the right thigh and relaxing the arm.",
]

LEVEL_FULL_WIDTH_PROMPTS = [
    "A person stands still close to a wall and raises the right hand toward a small patch on the wall, preparing to wipe the wall with the right hand.",
    "A person stands still facing a wall with the right hand touching the left edge of a small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from left to right.",
    "A person stands still facing a wall with the right hand touching the right edge of the same small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from right to left slightly lower on the wall.",
    "A person stands still facing a wall with the right hand touching the left edge of the same small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from left to right slightly lower on the wall.",
    "A person stands still facing a wall with the right hand near the wall, gently lowering the right hand down beside the right thigh and relaxing the arm.",
]

CFG_CANDIDATES = [
    {"cfg_weight": [2.2, 3.6], "num_denoising_steps": 200},
    {"cfg_weight": [2.2, 4.0], "num_denoising_steps": 200},
    {"cfg_weight": [2.4, 4.0], "num_denoising_steps": 200},
    {"cfg_weight": [2.4, 4.4], "num_denoising_steps": 200},
]

PREEMPHASIS_PRESETS = {
    "flat_A": {"x_scale": 1.7, "x_offset": 0.04, "base_y_offset": -0.07, "z_offset": 0.03, "z_wrist": 0.24, "use_y_closed_loop": True},
    "flat_B": {"x_scale": 1.9, "x_offset": 0.04, "base_y_offset": -0.07, "z_offset": 0.03, "z_wrist": 0.24, "use_y_closed_loop": True},
    "flat_C": {"x_scale": 1.9, "x_offset": 0.06, "base_y_offset": -0.08, "z_offset": 0.04, "z_wrist": 0.24, "use_y_closed_loop": True},
    "flat_D": {"x_scale": 1.7, "x_offset": 0.05, "base_y_offset": -0.06, "z_offset": 0.04, "z_wrist": 0.26, "use_y_closed_loop": True},
}


class CachedOnlyTextEncoder:
    def __init__(
        self,
        model_name: str,
        encoder_id: str = "LLM2VecEncoder",
        base_dir: str = "/root/.cache/kimodo_demo/embeddings",
    ):
        self.model_name = model_name
        self.encoder_id = encoder_id
        self.base_dir = Path(base_dir)
        self.device = "cpu"
        self.dtype = torch.float32

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def _key(self, text: str) -> str:
        import hashlib

        return hashlib.sha256(f"{self.model_name}|{self.encoder_id}|{text}".encode("utf-8")).hexdigest()

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = sanitize_texts(list(texts))

        arrays = []
        lengths = []
        missing = []
        for text in texts:
            path = self.base_dir / self.model_name / f"{self._key(text)}.npy"
            if not path.exists():
                missing.append(text)
                continue
            arr = np.load(path)
            arrays.append(arr)
            lengths.append(int(arr.shape[0]))

        if missing:
            raise FileNotFoundError(
                "Missing cached text embeddings. Run remote_cache_kimodo_prompts.py first. "
                f"Missing: {missing}"
            )

        max_len = max(lengths)
        feat_dim = arrays[0].shape[-1]
        padded = np.zeros((len(arrays), max_len, feat_dim), dtype=arrays[0].dtype)
        for i, arr in enumerate(arrays):
            padded[i, : arr.shape[0]] = arr
        return torch.from_numpy(padded).to(device=self.device, dtype=self.dtype), lengths


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def tag_float(value: float) -> str:
    text = f"{float(value):.2f}".replace("-", "m").replace(".", "p")
    return text.rstrip("0").rstrip("p") if "p" in text else text


def make_base_pose(skeleton, frame_count: int, root_y: float, device: str):
    joint_count = len(skeleton.bone_index)
    local_rots = torch.eye(3, device=device).repeat(frame_count, joint_count, 1, 1)
    root = torch.tensor([[0.0, root_y, 0.0]], device=device).repeat(frame_count, 1)
    global_rots, global_pos, _ = skeleton.fk(local_rots, root)
    return global_rots, global_pos


def idx(skeleton, name: str) -> int:
    return skeleton.bone_index[name]


def hand_token(active_hand: str) -> str:
    return "LeftHand" if str(active_hand).lower() == "left" else "RightHand"


def hand_pos_indices(skeleton, active_hand: str = "right"):
    _, pos_names = skeleton.expand_joint_names([hand_token(active_hand)])
    return [idx(skeleton, name) for name in pos_names], pos_names


def right_hand_pos_indices(skeleton):
    return hand_pos_indices(skeleton, "right")


def arm_indices(skeleton, active_hand: str = "right"):
    prefix = "left" if str(active_hand).lower() == "left" else "right"
    names = [
        f"{prefix}_shoulder_pitch_skel",
        f"{prefix}_shoulder_roll_skel",
        f"{prefix}_shoulder_yaw_skel",
        f"{prefix}_elbow_skel",
        f"{prefix}_wrist_roll_skel",
        f"{prefix}_wrist_pitch_skel",
        f"{prefix}_wrist_yaw_skel",
        f"{prefix}_hand_roll_skel",
    ]
    return [idx(skeleton, name) for name in names if name in skeleton.bone_index]


def right_arm_indices(skeleton):
    return arm_indices(skeleton, "right")


class RootHeightConstraintSet:
    name = "root-height"

    def __init__(self, frame_indices: torch.Tensor, root_y_pos: torch.Tensor):
        self.frame_indices = frame_indices
        self.root_y_pos = root_y_pos

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        data_dict["root_y_pos"].append(self.root_y_pos)
        index_dict["root_y_pos"].append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "RootHeightConstraintSet":
        mask = (self.frame_indices >= start) & (self.frame_indices < end)
        data_mask = mask.to(self.root_y_pos.device) if self.root_y_pos.device != mask.device else mask
        return RootHeightConstraintSet(
            frame_indices=self.frame_indices[mask] - start,
            root_y_pos=self.root_y_pos[data_mask],
        )

    def get_save_info(self) -> dict:
        return {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "root_y_pos": self.root_y_pos,
        }

    def to(self, device=None, dtype=None):
        if device is not None:
            self.root_y_pos = self.root_y_pos.to(device=device)
        if dtype is not None:
            self.root_y_pos = self.root_y_pos.to(dtype=dtype)
        return self


class RightHandWallContactPositionOnlyConstraintSet:
    """Right-hand contact constraints that write only global joint positions.

    This intentionally does not write global rotations or local rotations. It lets
    Kimodo solve shoulder/elbow/wrist pose during denoising instead of applying a
    post-generation right-arm IK pass.
    """

    name = "right-hand-wall-contact-position-only"

    def __init__(
        self,
        skeleton,
        frame_indices: torch.Tensor,
        endpoint_positions: torch.Tensor,
        wrist_positions: torch.Tensor | None,
        wrist_mask: torch.Tensor,
        smooth_root_2d: torch.Tensor,
        active_hand: str = "right",
    ):
        self.skeleton = skeleton
        self.active_hand = str(active_hand).lower()
        self.frame_indices = frame_indices
        self.endpoint_positions = endpoint_positions
        self.wrist_positions = wrist_positions
        self.wrist_mask = wrist_mask.bool()
        self.smooth_root_2d = smooth_root_2d
        pos_indices, pos_names = hand_pos_indices(skeleton, self.active_hand)
        self.wrist_index = int(pos_indices[0])
        self.endpoint_index = int(pos_indices[-1])
        self.pos_names = pos_names

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        indices = []
        points = []
        for local_i, frame in enumerate(self.frame_indices.tolist()):
            indices.append([int(frame), self.endpoint_index])
            points.append(self.endpoint_positions[local_i])
            if self.wrist_positions is not None and bool(self.wrist_mask[local_i]):
                indices.append([int(frame), self.wrist_index])
                points.append(self.wrist_positions[local_i])
        if points:
            index_dict["global_joints_positions"].append(
                torch.tensor(indices, dtype=torch.long, device=self.frame_indices.device)
            )
            data_dict["global_joints_positions"].append(torch.stack(points, dim=0))

        data_dict["smooth_root_2d"].append(self.smooth_root_2d)
        index_dict["smooth_root_2d"].append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "RightHandWallContactPositionOnlyConstraintSet":
        mask = (self.frame_indices >= start) & (self.frame_indices < end)
        data_mask = mask.to(self.endpoint_positions.device) if self.endpoint_positions.device != mask.device else mask
        wrist_positions = None if self.wrist_positions is None else self.wrist_positions[data_mask]
        wrist_mask = self.wrist_mask[data_mask]
        smooth_root_mask = mask.to(self.smooth_root_2d.device) if self.smooth_root_2d.device != mask.device else mask
        return RightHandWallContactPositionOnlyConstraintSet(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.endpoint_positions[data_mask],
            wrist_positions,
            wrist_mask,
            self.smooth_root_2d[smooth_root_mask],
            self.active_hand,
        )

    def get_save_info(self) -> dict:
        return {
            "type": self.name,
            "active_hand": self.active_hand,
            "frame_indices": self.frame_indices,
            "endpoint_positions": self.endpoint_positions,
            "wrist_positions": self.wrist_positions if self.wrist_positions is not None else [],
            "wrist_mask": self.wrist_mask,
            "smooth_root_2d": self.smooth_root_2d,
            "endpoint_index": self.endpoint_index,
            "wrist_index": self.wrist_index,
            "pos_names": self.pos_names,
        }

    def to(self, device=None, dtype=None):
        if device is not None:
            self.endpoint_positions = self.endpoint_positions.to(device=device)
            if self.wrist_positions is not None:
                self.wrist_positions = self.wrist_positions.to(device=device)
            self.smooth_root_2d = self.smooth_root_2d.to(device=device)
        if dtype is not None:
            self.endpoint_positions = self.endpoint_positions.to(dtype=dtype)
            if self.wrist_positions is not None:
                self.wrist_positions = self.wrist_positions.to(dtype=dtype)
            self.smooth_root_2d = self.smooth_root_2d.to(dtype=dtype)
        return self


def frame_plan(name: str) -> dict:
    if name == "189":
        return {
            "segments": [30, 36, 36, 36, 51],
            "boundaries": [30, 66, 102, 138],
            "approach": [(20, [-0.18, 0.88, 0.26], [-0.18, 0.86, 0.18]), (29, [-0.12, 0.92, 0.32], [-0.12, 0.90, 0.20])],
            "rows": [
                (1, [34, 41, 48, 55, 62], [[-0.12, 0.92, 0.32], [-0.06, 0.92, 0.32], [0.0, 0.92, 0.32], [0.06, 0.92, 0.32], [0.12, 0.92, 0.32]]),
                (2, [70, 77, 84, 91, 98], [[0.12, 0.89, 0.32], [0.06, 0.89, 0.32], [0.0, 0.89, 0.32], [-0.06, 0.89, 0.32], [-0.12, 0.89, 0.32]]),
                (3, [106, 113, 120, 127, 134], [[-0.12, 0.86, 0.32], [-0.06, 0.86, 0.32], [0.0, 0.86, 0.32], [0.06, 0.86, 0.32], [0.12, 0.86, 0.32]]),
            ],
            "return_frames": [164, 188],
            "stroke_end": [0.12, 0.86, 0.32],
        }
    if name == "180":
        return {
            "segments": [30, 36, 36, 36, 42],
            "boundaries": [30, 66, 102, 138],
            "approach": [(20, [-0.18, 0.88, 0.26], [-0.18, 0.86, 0.18]), (29, [-0.12, 0.92, 0.32], [-0.12, 0.90, 0.20])],
            "rows": frame_plan("189")["rows"],
            "return_frames": [158, 179],
            "stroke_end": [0.12, 0.86, 0.32],
        }
    if name == "210":
        return {
            "segments": [36, 42, 42, 42, 48],
            "boundaries": [36, 78, 120, 162],
            "approach": [(26, [-0.18, 0.88, 0.26], [-0.18, 0.86, 0.18]), (35, [-0.12, 0.92, 0.32], [-0.12, 0.90, 0.20])],
            "rows": [
                (1, [46, 52, 58, 64, 70, 76], [[-0.12, 0.92, 0.32], [-0.072, 0.92, 0.32], [-0.024, 0.92, 0.32], [0.024, 0.92, 0.32], [0.072, 0.92, 0.32], [0.12, 0.92, 0.32]]),
                (2, [88, 94, 100, 106, 112, 118], [[0.12, 0.89, 0.32], [0.072, 0.89, 0.32], [0.024, 0.89, 0.32], [-0.024, 0.89, 0.32], [-0.072, 0.89, 0.32], [-0.12, 0.89, 0.32]]),
                (3, [130, 136, 142, 148, 154, 160], [[-0.12, 0.86, 0.32], [-0.072, 0.86, 0.32], [-0.024, 0.86, 0.32], [0.024, 0.86, 0.32], [0.072, 0.86, 0.32], [0.12, 0.86, 0.32]]),
            ],
            "return_frames": [186, 209],
            "stroke_end": [0.12, 0.86, 0.32],
        }
    raise ValueError(f"Unknown frame plan: {name}")


def plan_from_task_spec(path: str, fallback_plan: dict | None = None) -> tuple[dict, list[str] | None, dict]:
    spec_path = Path(path)
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = data.get("task_spec", data)
    base = dict(fallback_plan or frame_plan(str(spec.get("frame_plan", "210"))))
    rows = []
    for row in spec.get("stroke_rows", []):
        rows.append(
            (
                int(row["row"]),
                [int(frame) for frame in row["frames"]],
                [[float(v) for v in point] for point in row["points"]],
            )
        )
    if rows:
        base["rows"] = rows
        base["stroke_end"] = rows[-1][2][-1]
    if "segments" in spec:
        base["segments"] = [int(v) for v in spec["segments"]]
    if "boundaries" in spec:
        base["boundaries"] = [int(v) for v in spec["boundaries"]]
    if "approach" in spec:
        base["approach"] = [
            (int(item["frame"]), [float(v) for v in item["endpoint"]], [float(v) for v in item.get("wrist", item["endpoint"])])
            for item in spec["approach"]
        ]
    elif rows:
        first = rows[0][2][0]
        base["approach"] = [
            (26, [first[0] - 0.06, first[1] - 0.04, first[2] - 0.06], [first[0] - 0.06, first[1] - 0.06, first[2] - 0.14]),
            (35, first, [first[0], first[1] - 0.02, first[2] - 0.12]),
        ]
    if "return_frames" in spec:
        base["return_frames"] = [int(v) for v in spec["return_frames"]]
    elif "total_frames" in spec:
        base["return_frames"] = [max(0, int(spec["total_frames"]) - 24), int(spec["total_frames"]) - 1]
    prompts = spec.get("prompts")
    overrides = {
        "task_spec_path": str(spec_path),
        "task_text": spec.get("task_text", data.get("task_text", "")),
        "task_type": spec.get("task_type", data.get("task_type", "")),
        "active_hand": str(spec.get("active_hand", spec.get("constraint_route", {}).get("active_hand", "right"))).lower(),
    }
    if "wall_z" in spec:
        overrides["wall_z"] = float(spec["wall_z"])
    elif spec.get("constraint_route", {}).get("z_contact") is not None:
        overrides["wall_z"] = float(spec["constraint_route"]["z_contact"])
    return base, prompts, overrides


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_y_corrections(reference_motion: str | None, plan: dict, active_hand: str = "right") -> dict[tuple[int, int], float]:
    if not reference_motion:
        return {}
    path = Path(reference_motion)
    if not path.exists():
        raise FileNotFoundError(f"y reference motion not found: {path}")
    data = np.load(path, allow_pickle=True)
    posed = data["posed_joints"].astype(np.float64)
    hand = posed[:, 25 if str(active_hand).lower() == "left" else 33]
    corrections: dict[tuple[int, int], float] = {}
    for row, frames, points in plan["rows"]:
        for col, (frame, point) in enumerate(zip(frames, points), start=1):
            frame = min(max(0, int(frame)), len(hand) - 1)
            desired_y = float(point[1])
            actual_y = float(hand[frame, 1])
            error_y = actual_y - desired_y
            corrections[(int(row), int(col))] = clamp(-0.7 * error_y, -0.06, 0.06)
    return corrections


def preemphasize_point(point: list[float], x_scale: float, x_offset: float, y_offset: float, z_offset: float) -> list[float]:
    x, y, z = [float(v) for v in point]
    return [float(x_offset) + x_scale * x, y + float(y_offset), z + float(z_offset)]


def build_targets(
    plan: dict,
    final_hand: np.ndarray,
    z_wrist: float,
    x_scale: float,
    x_offset: float,
    base_y_offset: float,
    z_offset: float,
    y_corrections: dict[tuple[int, int], float] | None,
    include_return_target: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    endpoint_targets = []
    contact_targets = []
    # Keep total endpoint constraints at 20: 18 stroke endpoints, one approach
    # endpoint, and one final return endpoint.
    frame, endpoint, _wrist = plan["approach"][-1]
    endpoint_for_constraint = preemphasize_point(endpoint, x_scale, x_offset, base_y_offset, z_offset)
    item = {
        "name": "approach_final",
        "phase": "approach",
        "row": 0,
        "col": 1,
        "frame": int(frame),
        "point": [float(x) for x in endpoint_for_constraint],
        "true_point": [float(x) for x in endpoint],
        "wrist_point": None,
        "use_wrist": False,
    }
    endpoint_targets.append(item)
    contact_targets.append(item)

    row_specs = []
    y_corrections = y_corrections or {}
    for row, frames, points in plan["rows"]:
        row_specs.append(
            {
                "row": int(row),
                "start_frame": int(frames[0]),
                "end_frame": int(frames[-1]),
                "start_point": [float(x) for x in points[0]],
                "end_point": [float(x) for x in points[-1]],
            }
        )
        for col, (frame, point) in enumerate(zip(frames, points), start=1):
            true_endpoint = [float(x) for x in point]
            y_offset = base_y_offset + float(y_corrections.get((int(row), int(col)), 0.0))
            endpoint = preemphasize_point(true_endpoint, x_scale, x_offset, y_offset, z_offset)
            wrist = [endpoint[0], endpoint[1] - 0.015, float(z_wrist)]
            item = {
                "name": f"row_{row}_point_{col}",
                "phase": "stroke",
                "row": int(row),
                "col": int(col),
                "frame": int(frame),
                "point": endpoint,
                "true_point": true_endpoint,
                "wrist_point": wrist,
                "y_correction": float(y_corrections.get((int(row), int(col)), 0.0)),
                "base_y_offset": float(base_y_offset),
                "use_wrist": col in (1, len(points)),
            }
            endpoint_targets.append(item)
            contact_targets.append(item)

    if include_return_target:
        endpoint_targets.append(
            {
                "name": "return_final",
                "phase": "return",
                "row": 4,
                "col": 0,
                "frame": int(plan["return_frames"][1]),
                "point": [float(x) for x in final_hand.astype(np.float32).tolist()],
                "true_point": [float(x) for x in final_hand.astype(np.float32).tolist()],
                "wrist_point": None,
                "use_wrist": False,
            }
        )

    return endpoint_targets, contact_targets, row_specs


def build_position_only_constraint(model, targets: list[dict], root_y: float, include_wrist: bool, active_hand: str = "right"):
    device = model.device if hasattr(model, "device") else "cuda:0"
    skeleton = model.skeleton
    frame_indices = torch.tensor([target["frame"] for target in targets], dtype=torch.long)
    endpoint_positions = torch.tensor([target["point"] for target in targets], dtype=torch.float32, device=device)
    wrist_positions = torch.zeros_like(endpoint_positions)
    wrist_mask = []
    for i, target in enumerate(targets):
        use_wrist = bool(include_wrist and target.get("use_wrist") and target.get("wrist_point") is not None)
        wrist_mask.append(use_wrist)
        if use_wrist:
            wrist_positions[i] = torch.tensor(target["wrist_point"], dtype=torch.float32, device=device)
    smooth_root_2d = torch.zeros((len(targets), 2), dtype=torch.float32, device=device)
    return RightHandWallContactPositionOnlyConstraintSet(
        skeleton,
        frame_indices=frame_indices,
        endpoint_positions=endpoint_positions,
        wrist_positions=wrist_positions,
        wrist_mask=torch.tensor(wrist_mask, dtype=torch.bool, device=device),
        smooth_root_2d=smooth_root_2d,
        active_hand=active_hand,
    )


def build_current_right_hand_constraint(model, targets: list[dict], root_y: float, include_wrist: bool, active_hand: str = "right"):
    if str(active_hand).lower() != "right":
        raise ValueError("current_right_hand baseline uses Kimodo's built-in RightHandConstraintSet and cannot drive the left hand.")
    device = model.device if hasattr(model, "device") else "cuda:0"
    skeleton = model.skeleton
    # KIMODO's built-in RightHandConstraintSet mixes index tensors with CPU-created
    # cartesian products internally, so keep its frame indices on CPU.
    frame_indices = torch.tensor([target["frame"] for target in targets], dtype=torch.long)
    global_rots, global_pos = make_base_pose(skeleton, len(targets), root_y, device)
    pos_indices, _ = right_hand_pos_indices(skeleton)
    for local_i, target in enumerate(targets):
        endpoint = torch.tensor(target["point"], device=device, dtype=global_pos.dtype)
        global_pos[local_i, pos_indices[-1]] = endpoint
        if include_wrist and target.get("use_wrist") and target.get("wrist_point") is not None:
            global_pos[local_i, pos_indices[0]] = torch.tensor(target["wrist_point"], device=device, dtype=global_pos.dtype)
    return RightHandConstraintSet(
        skeleton,
        frame_indices=frame_indices,
        global_joints_positions=global_pos,
        global_joints_rots=global_rots,
        smooth_root_2d=torch.zeros((len(targets), 2), dtype=torch.float32, device=device),
    )


def build_constraints(
    model,
    targets: list[dict],
    total_frames: int,
    root_y: float,
    variant: str,
    heading_mode: str,
    plan: dict,
    active_hand: str = "right",
):
    device = model.device if hasattr(model, "device") else "cuda:0"
    skeleton = model.skeleton
    constraints = []

    if variant == "current_right_hand":
        constraints.append(build_current_right_hand_constraint(model, targets, root_y, include_wrist=True, active_hand=active_hand))
    elif variant == "endpoint_only":
        constraints.append(build_position_only_constraint(model, targets, root_y, include_wrist=False, active_hand=active_hand))
    elif variant == "endpoint_wrist":
        constraints.append(build_position_only_constraint(model, targets, root_y, include_wrist=True, active_hand=active_hand))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    root_frames = sorted(set([0, *plan["boundaries"], total_frames - 1]))
    root_frames_t = torch.tensor(root_frames, dtype=torch.long)
    constraints.append(
        Root2DConstraintSet(
            skeleton,
            frame_indices=root_frames_t,
            smooth_root_2d=torch.zeros((len(root_frames), 2), dtype=torch.float32, device=device),
        )
    )
    constraints.append(
        RootHeightConstraintSet(
            frame_indices=root_frames_t,
            root_y_pos=torch.full((len(root_frames),), float(root_y), dtype=torch.float32, device=device),
        )
    )

    if heading_mode != "none":
        if heading_mode == "sparse":
            heading_frames = [0, *plan["boundaries"], plan["return_frames"][0], total_frames - 1]
        elif heading_mode == "return_only":
            heading_frames = [plan["boundaries"][-1], plan["return_frames"][0], total_frames - 1]
        else:
            raise ValueError(f"Unknown heading_mode: {heading_mode}")
        heading_frames = sorted(set(int(x) for x in heading_frames))
        heading_tensor = torch.tensor(heading_frames, dtype=torch.long)
        # Kimodo's zero first heading is [cos(0), sin(0)]. This is a generation-time
        # sparse condition, not a post-generation root lock.
        heading = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device).repeat(len(heading_frames), 1)
        constraints.append(
            Root2DConstraintSet(
                skeleton,
                frame_indices=heading_tensor,
                smooth_root_2d=torch.zeros((len(heading_frames), 2), dtype=torch.float32, device=device),
                global_root_heading=heading,
            )
        )

    return constraints


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


def validate(output: dict, skeleton, targets: list[dict], row_specs: list[dict], wall_z: float, active_hand: str = "right") -> dict:
    posed = output["posed_joints"]
    endpoint_idx = hand_pos_indices(skeleton, active_hand)[0][-1]
    results = []
    for sample_i in range(posed.shape[0]):
        sample = posed[sample_i]
        hand = sample[:, endpoint_idx]
        key_errors = [
            float(np.linalg.norm(hand[int(target["frame"])] - np.asarray(target.get("true_point", target["point"]), dtype=np.float32)))
            for target in targets
        ]
        row_errors = []
        row_ratios = []
        for row in row_specs:
            start_f = int(row["start_frame"])
            end_f = int(row["end_frame"])
            start = np.asarray(row["start_point"], dtype=np.float32)
            end = np.asarray(row["end_point"], dtype=np.float32)
            pts = hand[start_f : end_f + 1]
            row_errors.append(float(np.max(line_distance(pts, start, end))))
            row_ratios.append(path_len(pts) / max(float(np.linalg.norm(end - start)), 1e-8))
        stroke_frames = np.concatenate([np.arange(int(row["start_frame"]), int(row["end_frame"]) + 1) for row in row_specs])
        results.append(
            {
                "sample": int(sample_i),
                "max_keyframe_error_m": float(np.max(key_errors)),
                "mean_keyframe_error_m": float(np.mean(key_errors)),
                "max_row_line_error_m": float(np.max(row_errors)),
                "mean_row_line_error_m": float(np.mean(row_errors)),
                "max_path_length_ratio": float(np.max(row_ratios)),
                "mean_wall_contact_error_m": float(np.mean(np.abs(hand[stroke_frames, 2] - wall_z))),
            }
        )
    best = min(
        results,
        key=lambda x: (
            x["max_keyframe_error_m"] / 0.08
            + x["max_row_line_error_m"] / 0.09
            + max(0.0, x["max_path_length_ratio"] - 1.35) * 3.0
            + x["mean_wall_contact_error_m"] / 0.04
        ),
    )
    return {
        "results": results,
        "best_sample": int(best["sample"]),
        "best": best,
    }


def numpy_single(output: dict, sample_i: int) -> dict:
    single = {}
    n_samples = int(output["posed_joints"].shape[0])
    for key, value in output.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == n_samples:
            single[key] = value[sample_i]
        else:
            single[key] = value
    return single


def save_run(run_dir: Path, output: dict, model, metrics: dict, constraints, prompts_payload: dict) -> None:
    ensure_dir(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "prompts.json").write_text(json.dumps(prompts_payload, indent=2), encoding="utf-8")
    (run_dir / "constraints.json").write_text(
        json.dumps([to_jsonable(c.get_save_info()) for c in constraints], indent=2),
        encoding="utf-8",
    )
    n_samples = int(output["posed_joints"].shape[0])
    for sample_i in range(n_samples):
        sample_dir = run_dir / f"sample_{sample_i:02d}"
        ensure_dir(sample_dir)
        save_kimodo_npz(sample_dir / "motion.npz", numpy_single(output, sample_i))


def run_generation(args) -> None:
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolved_for_cache = resolve_model_name(args.model, "Kimodo")
    print(f"device={device}", flush=True)
    print(f"loading_model={args.model}", flush=True)
    print(f"cache_model={resolved_for_cache}", flush=True)
    model, resolved_model = load_model(
        args.model,
        device=device,
        default_family="Kimodo",
        return_resolved_name=True,
        text_encoder=CachedOnlyTextEncoder(resolved_for_cache),
    )
    print(f"resolved_model={resolved_model}", flush=True)
    print(f"skeleton={model.skeleton.name}", flush=True)
    print(f"fps={model.fps}", flush=True)

    plan = frame_plan(args.frame_plan)
    task_spec_prompts = None
    task_spec_overrides = {}
    if args.task_spec:
        plan, task_spec_prompts, task_spec_overrides = plan_from_task_spec(args.task_spec, fallback_plan=plan)
        if "wall_z" in task_spec_overrides:
            args.z_contact = float(task_spec_overrides["wall_z"])
        if "active_hand" in task_spec_overrides:
            args.active_hand = str(task_spec_overrides["active_hand"]).lower()
    segments = plan["segments"]
    total_frames = sum(segments)
    base_global_rots, base_global_pos = make_base_pose(model.skeleton, 1, args.root_y, device)
    endpoint_idx = hand_pos_indices(model.skeleton, args.active_hand)[0][-1]
    final_hand = base_global_pos[0, endpoint_idx].detach().cpu().numpy().astype(np.float32)
    y_corrections = compute_y_corrections(args.y_reference_motion, plan, args.active_hand)

    cfg_candidates = CFG_CANDIDATES
    if args.cfg_constraint_values:
        if args.cfg_text is None or args.diffusion_steps is None:
            raise ValueError("--cfg_constraint_values requires --cfg_text and --diffusion_steps")
        cfg_candidates = [
            {"cfg_weight": [args.cfg_text, cfg_constraint], "num_denoising_steps": args.diffusion_steps}
            for cfg_constraint in args.cfg_constraint_values
        ]
    elif args.cfg_text is not None and args.cfg_constraint is not None and args.diffusion_steps is not None:
        cfg_candidates = [{"cfg_weight": [args.cfg_text, args.cfg_constraint], "num_denoising_steps": args.diffusion_steps}]

    preset_names = list(PREEMPHASIS_PRESETS.keys()) if args.preemphasis_preset == "all" else [args.preemphasis_preset]
    prompt_sets = {
        "base": PROMPTS,
        "full_width": FULL_WIDTH_PROMPTS,
        "level_full_width": LEVEL_FULL_WIDTH_PROMPTS,
    }
    if args.active_hand == "left" and not task_spec_prompts:
        prompt_sets = {
            key: [text.replace("right hand", "left hand").replace("right thigh", "left thigh") for text in value]
            for key, value in prompt_sets.items()
        }
    if task_spec_prompts:
        prompt_sets[args.prompt_variant] = task_spec_prompts
    prompt_variant_names = list(prompt_sets.keys()) if args.prompt_variant == "both" else [args.prompt_variant]

    remote_summary = {
        "model": args.model,
        "resolved_model": resolved_model,
        "variant": args.variant,
        "heading_mode": args.heading_mode,
        "frame_plan": args.frame_plan,
        "segments": segments,
        "boundaries": plan["boundaries"],
        "wall_z": args.z_contact,
        "root_y": args.root_y,
        "active_hand": args.active_hand,
        "neutral_final_active_hand_side_pos": final_hand.tolist(),
        "neutral_final_right_hand_side_pos": final_hand.tolist(),
        "preemphasis_presets": {name: PREEMPHASIS_PRESETS[name] for name in preset_names},
        "prompt_variants": prompt_variant_names,
        "y_reference_motion": args.y_reference_motion,
        "task_spec": args.task_spec,
        "task_spec_overrides": task_spec_overrides,
        "y_corrections": {f"row{row}_col{col}": value for (row, col), value in y_corrections.items()},
        "conditions": [],
        "runs": [],
    }

    run_counter = 0
    for prompt_variant in prompt_variant_names:
        prompts = prompt_sets[prompt_variant]
        for preset_name in preset_names:
            pre = PREEMPHASIS_PRESETS[preset_name]
            z_wrist = float(args.z_wrist_override) if args.z_wrist_override is not None else float(pre["z_wrist"])
            targets, contact_targets, row_specs = build_targets(
                plan,
                final_hand,
                z_wrist=z_wrist,
                x_scale=pre["x_scale"],
                x_offset=pre["x_offset"],
                base_y_offset=pre["base_y_offset"],
                z_offset=pre["z_offset"],
                y_corrections=y_corrections if pre.get("use_y_closed_loop", False) and not args.disable_y_closed_loop else {},
                include_return_target=not args.disable_return_constraint,
            )
            if args.return_final_only:
                targets = [target for target in targets if target["phase"] != "return" or target["name"] == "return_final"]
            condition = {
                "preemphasis": preset_name,
                "x_scale": pre["x_scale"],
                "x_offset": pre["x_offset"],
                "base_y_offset": pre["base_y_offset"],
                "y_offset": pre["base_y_offset"],
                "z_offset": pre["z_offset"],
                "z_wrist": z_wrist,
                "use_y_closed_loop": bool(pre.get("use_y_closed_loop", False) and not args.disable_y_closed_loop),
                "prompt_variant": prompt_variant,
                "target_points": targets,
                "contact_targets": contact_targets,
                "row_specs": row_specs,
            }
            remote_summary["conditions"].append(condition)
            condition_tag = (
                f"{prompt_variant}_pre{preset_name}_x{tag_float(pre['x_scale'])}_xo{tag_float(pre['x_offset'])}_"
                f"y{tag_float(pre['base_y_offset'])}_z{tag_float(pre['z_offset'])}_zw{tag_float(z_wrist)}"
            )

            for cfg_i, cfg in enumerate(cfg_candidates):
                run_name = (
                    f"{args.variant}_{args.heading_mode}_plan{args.frame_plan}_"
                    f"{condition_tag}_cfg{cfg_i}_"
                    f"t{cfg['cfg_weight'][0]:.1f}_c{cfg['cfg_weight'][1]:.1f}_"
                    f"steps{cfg['num_denoising_steps']}"
                ).replace(".", "p")
                run_dir = out_dir / run_name
                constraints = build_constraints(
                    model,
                    targets,
                    total_frames=total_frames,
                    root_y=args.root_y,
                    variant=args.variant,
                    heading_mode=args.heading_mode,
                    plan=plan,
                    active_hand=args.active_hand,
                )
                seed_everything(args.seed if args.same_seed_for_cfg else args.seed + run_counter * 1000)
                run_counter += 1
                print(f"running={run_name}", flush=True)
                output = model(
                    prompts,
                    segments,
                    constraint_lst=constraints,
                    num_denoising_steps=int(cfg["num_denoising_steps"]),
                    num_samples=args.num_samples,
                    multi_prompt=True,
                    num_transition_frames=args.num_transition_frames,
                    post_processing=False,
                    return_numpy=True,
                    cfg_type="separated",
                    cfg_weight=cfg["cfg_weight"],
                )
                metrics = validate(output, model.skeleton, targets, row_specs, args.z_contact, args.active_hand)
                metrics.update(
                    {
                        "active_hand": args.active_hand,
                        "run_name": run_name,
                        "cfg_weight": cfg["cfg_weight"],
                        "num_denoising_steps": int(cfg["num_denoising_steps"]),
                        "variant": args.variant,
                        "heading_mode": args.heading_mode,
                        "frame_plan": args.frame_plan,
                        "segments": segments,
                        "boundaries": plan["boundaries"],
                        "prompts": prompts,
                        "target_points": targets,
                        "contact_targets": contact_targets,
                        "row_specs": row_specs,
                        "wall_z": args.z_contact,
                        "z_wrist": z_wrist,
                        "preemphasis": preset_name,
                        "x_scale": pre["x_scale"],
                        "x_offset": pre["x_offset"],
                        "base_y_offset": pre["base_y_offset"],
                        "y_offset": pre["base_y_offset"],
                        "z_offset": pre["z_offset"],
                        "use_y_closed_loop": bool(pre.get("use_y_closed_loop", False) and not args.disable_y_closed_loop),
                        "y_reference_motion": args.y_reference_motion,
                        "task_spec": args.task_spec,
                        "task_spec_overrides": task_spec_overrides,
                        "y_corrections": {f"row{row}_col{col}": value for (row, col), value in y_corrections.items()},
                        "prompt_variant": prompt_variant,
                        "neutral_final_active_hand_side_pos": final_hand.tolist(),
                        "neutral_final_right_hand_side_pos": final_hand.tolist(),
                        "no_ik": True,
                        "post_processing": False,
                    }
                )
                prompts_payload = {
                    "active_hand": args.active_hand,
                    "texts": prompts,
                    "num_frames": segments,
                    "generation_prompt": prompts,
                    "generation_num_frames": segments,
                    "multi_prompt": True,
                    "variant": args.variant,
                    "heading_mode": args.heading_mode,
                    "frame_plan": args.frame_plan,
                    "boundaries": plan["boundaries"],
                    "cfg_weight": cfg["cfg_weight"],
                    "num_denoising_steps": int(cfg["num_denoising_steps"]),
                    "target_points": targets,
                    "contact_targets": contact_targets,
                    "row_specs": row_specs,
                    "wall_z": args.z_contact,
                    "z_wrist": z_wrist,
                    "preemphasis": preset_name,
                    "x_scale": pre["x_scale"],
                    "x_offset": pre["x_offset"],
                    "base_y_offset": pre["base_y_offset"],
                    "y_offset": pre["base_y_offset"],
                    "z_offset": pre["z_offset"],
                    "use_y_closed_loop": bool(pre.get("use_y_closed_loop", False) and not args.disable_y_closed_loop),
                    "y_reference_motion": args.y_reference_motion,
                    "task_spec": args.task_spec,
                    "task_spec_overrides": task_spec_overrides,
                    "y_corrections": {f"row{row}_col{col}": value for (row, col), value in y_corrections.items()},
                    "prompt_variant": prompt_variant,
                    "neutral_final_active_hand_side_pos": final_hand.tolist(),
                    "neutral_final_right_hand_side_pos": final_hand.tolist(),
                    "no_ik": True,
                }
                save_run(run_dir, output, model, metrics, constraints, prompts_payload)
                remote_summary["runs"].append(
                    {
                        "run_name": run_name,
                        "path": str(run_dir),
                        "best_sample": metrics["best_sample"],
                        "best": metrics["best"],
                        "preemphasis": preset_name,
                        "z_wrist": z_wrist,
                        "prompt_variant": prompt_variant,
                    }
                )
                print(json.dumps(remote_summary["runs"][-1], indent=2), flush=True)

    (out_dir / "remote_summary.json").write_text(json.dumps(remote_summary, indent=2), encoding="utf-8")
    print(json.dumps(remote_summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Kimodo-G1-RP-v1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--frame_plan", choices=["189", "180", "210"], default="210")
    parser.add_argument("--variant", choices=["endpoint_wrist", "endpoint_only", "current_right_hand"], default="endpoint_wrist")
    parser.add_argument("--heading_mode", choices=["none", "sparse", "return_only"], default="none")
    parser.add_argument("--root_y", type=float, default=0.78)
    parser.add_argument("--z_contact", type=float, default=0.32)
    parser.add_argument("--z_wrist_override", type=float)
    parser.add_argument("--active_hand", choices=["right", "left"], default="right")
    parser.add_argument("--preemphasis_preset", choices=["all", *PREEMPHASIS_PRESETS.keys()], default="all")
    parser.add_argument("--prompt_variant", choices=["both", "base", "full_width", "level_full_width"], default="level_full_width")
    parser.add_argument("--num_transition_frames", type=int, default=5)
    parser.add_argument("--return_final_only", action="store_true")
    parser.add_argument("--disable_return_constraint", action="store_true")
    parser.add_argument("--y_reference_motion")
    parser.add_argument("--disable_y_closed_loop", action="store_true")
    parser.add_argument("--task_spec")
    parser.add_argument("--cfg_text", type=float)
    parser.add_argument("--cfg_constraint", type=float)
    parser.add_argument("--cfg_constraint_values", nargs="*", type=float)
    parser.add_argument("--diffusion_steps", type=int)
    parser.add_argument(
        "--same_seed_for_cfg",
        action="store_true",
        help="Reuse --seed for every cfg candidate so constraint-strength ablations are not confounded by sample seed.",
    )
    args = parser.parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
