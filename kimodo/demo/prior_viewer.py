# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local review-mode viewer for task-spec prior runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

CONSTRAINT_MARKER_RADIUS = 0.012
CONSTRAINT_LABEL_OFFSET = np.array([0.0, 0.035, 0.0], dtype=np.float64)
TARGET_LINE_WIDTH = 2.0


@dataclass(frozen=True)
class PriorMotionVariant:
    name: str
    kind: str
    motion_path: Path


@dataclass(frozen=True)
class PriorViewerCandidate:
    name: str
    status: str
    notes: str
    variants: list[PriorMotionVariant]
    prompt_segments: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PriorReviewFolder:
    run_folder: Path
    task_name: str
    model: str
    manifest: dict[str, Any]
    task: dict[str, Any]
    metrics: dict[str, Any]
    review: dict[str, Any]
    candidates: dict[str, PriorViewerCandidate]

    @property
    def candidate_names(self) -> list[str]:
        return list(self.candidates)


class PriorViewerApp:
    """Small local Viser app for reviewing already-generated prior motions."""

    def __init__(
        self,
        review: PriorReviewFolder,
        *,
        host: str = "127.0.0.1",
        port: int = 7861,
        server_factory: Callable[..., Any] | None = None,
    ):
        self.review = review
        self.host = host
        self.port = int(port)
        self.url = f"http://{host}:{self.port}/"
        self._client_state: dict[int, dict[str, Any]] = {}

        if server_factory is None:
            import viser

            server_factory = viser.ViserServer
        self.server = server_factory(
            host=host,
            port=self.port,
            label="Kimodo Prior Viewer",
            enable_camera_keyboard_controls=False,
        )
        if hasattr(self.server, "scene"):
            self.server.scene.world_axes.visible = False
            self.server.scene.set_up_direction("+y")
        self.server.on_client_connect(self._on_client_connect)

    def run(self) -> None:
        print(f"Kimodo prior viewer: {self.url}")
        try:
            while True:
                self._tick()
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            print("Kimodo prior viewer stopped.")

    def _tick(self) -> None:
        for state in list(self._client_state.values()):
            motion = state.get("motion")
            if not state.get("playing") or motion is None:
                continue
            now = time.time()
            if now - state.get("last_frame_time", 0.0) < 1.0 / state.get("fps", 30.0):
                continue
            state["last_frame_time"] = now
            next_frame = (state["frame"] + 1) % motion.length
            self._set_frame(state, next_frame, update_timeline=True)

    def _on_client_connect(self, client: Any) -> None:
        if not hasattr(client, "gui"):
            return
        self._setup_scene(client)
        first_candidate = self.review.candidate_names[0]
        state = {
            "client": client,
            "candidate": first_candidate,
            "variant": self.review.candidates[first_candidate].variants[0].name,
            "motion": None,
            "overlays": [],
            "frame": 0,
            "playing": False,
            "fps": 30.0,
            "last_frame_time": 0.0,
            "show_constraints": True,
            "show_constraint_labels": False,
            "show_targets": True,
        }
        self._client_state[client.client_id] = state
        self._create_gui(client, state)
        self._load_selected_motion(state)

    def _setup_scene(self, client: Any) -> None:
        import viser

        client.scene.world_axes.visible = False
        client.scene.set_up_direction("+y")
        client.camera.position = np.array([2.7, 1.9, 7.7], dtype=np.float64)
        client.camera.look_at = np.array([0.0, 0.7, 0.0], dtype=np.float64)
        client.camera.up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        client.camera.fov = np.deg2rad(45.0)
        client.scene.add_grid(
            "/prior/grid",
            width=12.0,
            height=12.0,
            wxyz=viser.transforms.SO3.from_x_radians(-np.pi / 2.0).wxyz,
            position=(0.0, 0.0001, 0.0),
            fade_distance=36.0,
            infinite_grid=True,
        )
        if hasattr(client, "timeline"):
            client.timeline.set_visible(True)
            client.timeline.set_defaults(
                default_text="prior motion",
                default_duration=29,
                min_duration=1,
                max_duration=1000,
                default_num_frames_zoom=180,
                max_frames_zoom=1000,
                fps=30.0,
            )
            client.timeline.set_current_frame(0)

    def _create_gui(self, client: Any, state: dict[str, Any]) -> None:
        statuses = self.review.review.get("review_statuses", ["needs_review"])
        with client.gui.add_folder("Prior Review", expand_by_default=True):
            candidate_dropdown = client.gui.add_dropdown(
                "Candidate",
                options=self.review.candidate_names,
                initial_value=state["candidate"],
            )
            variant_dropdown = client.gui.add_dropdown(
                "Variant",
                options=[variant.name for variant in self.review.candidates[state["candidate"]].variants],
                initial_value=state["variant"],
            )
            status_dropdown = client.gui.add_dropdown(
                "Review Status",
                options=statuses,
                initial_value=self.review.candidates[state["candidate"]].status,
            )
            notes_text = client.gui.add_text("Notes", initial_value=self.review.candidates[state["candidate"]].notes)
            save_button = client.gui.add_button("Save Review")
            info_markdown = client.gui.add_markdown(content=self._candidate_markdown(state["candidate"]))

        with client.gui.add_folder("Playback", expand_by_default=True):
            play_button = client.gui.add_button("Play / Pause")
            prev_button = client.gui.add_button("Previous Frame")
            next_button = client.gui.add_button("Next Frame")

        with client.gui.add_folder("Visualize", expand_by_default=True):
            mesh_checkbox = client.gui.add_checkbox("Show G1 Mesh", initial_value=True)
            skeleton_checkbox = client.gui.add_checkbox("Show Skeleton", initial_value=False)
            constraints_checkbox = client.gui.add_checkbox("Show Constraint Points", initial_value=True)
            constraint_labels_checkbox = client.gui.add_checkbox("Show Constraint Labels", initial_value=False)
            target_checkbox = client.gui.add_checkbox("Show Target Lines", initial_value=True)

        state["gui"] = {
            "candidate": candidate_dropdown,
            "variant": variant_dropdown,
            "status": status_dropdown,
            "notes": notes_text,
            "info": info_markdown,
            "mesh": mesh_checkbox,
            "skeleton": skeleton_checkbox,
            "constraints": constraints_checkbox,
            "constraint_labels": constraint_labels_checkbox,
            "targets": target_checkbox,
        }

        @candidate_dropdown.on_update
        def _candidate_changed(_event: Any) -> None:
            state["candidate"] = candidate_dropdown.value
            variants = [variant.name for variant in self.review.candidates[state["candidate"]].variants]
            variant_dropdown.options = variants
            variant_dropdown.value = variants[0]
            state["variant"] = variants[0]
            candidate = self.review.candidates[state["candidate"]]
            status_dropdown.value = candidate.status if candidate.status in statuses else statuses[0]
            notes_text.value = candidate.notes
            info_markdown.content = self._candidate_markdown(state["candidate"])
            self._load_selected_motion(state)

        @variant_dropdown.on_update
        def _variant_changed(_event: Any) -> None:
            state["variant"] = variant_dropdown.value
            self._load_selected_motion(state)

        @save_button.on_click
        def _save_review(_event: Any) -> None:
            save_review_decision(self.review.run_folder, state["candidate"], status_dropdown.value, notes_text.value)
            self.review = load_prior_review_folder(self.review.run_folder)
            info_markdown.content = self._candidate_markdown(state["candidate"])
            if hasattr(client, "add_notification"):
                client.add_notification(
                    "Review saved",
                    f"{state['candidate']} -> {status_dropdown.value}",
                    auto_close_seconds=2.0,
                )

        @play_button.on_click
        def _toggle_play(_event: Any) -> None:
            state["playing"] = not state["playing"]

        @prev_button.on_click
        def _prev(_event: Any) -> None:
            motion = state.get("motion")
            if motion is not None:
                self._set_frame(state, max(0, state["frame"] - 1), update_timeline=True)

        @next_button.on_click
        def _next(_event: Any) -> None:
            motion = state.get("motion")
            if motion is not None:
                self._set_frame(state, min(motion.length - 1, state["frame"] + 1), update_timeline=True)

        @mesh_checkbox.on_update
        def _mesh_changed(_event: Any) -> None:
            motion = state.get("motion")
            if motion is not None:
                motion.character.set_skinned_mesh_visibility(mesh_checkbox.value)

        @skeleton_checkbox.on_update
        def _skeleton_changed(_event: Any) -> None:
            motion = state.get("motion")
            if motion is not None:
                motion.character.set_skeleton_visibility(skeleton_checkbox.value)

        @constraints_checkbox.on_update
        def _constraints_changed(_event: Any) -> None:
            state["show_constraints"] = constraints_checkbox.value
            self._set_overlay_visibility(state)

        @constraint_labels_checkbox.on_update
        def _constraint_labels_changed(_event: Any) -> None:
            state["show_constraint_labels"] = constraint_labels_checkbox.value
            self._set_overlay_visibility(state)

        @target_checkbox.on_update
        def _targets_changed(_event: Any) -> None:
            state["show_targets"] = target_checkbox.value
            self._set_overlay_visibility(state)

        if hasattr(client, "timeline"):
            @client.timeline.on_frame_change
            def _timeline_frame_changed(frame_idx: int) -> None:
                self._set_frame(state, frame_idx, update_timeline=False)

    def _load_selected_motion(self, state: dict[str, Any]) -> None:
        client = state["client"]
        self._clear_state_scene(state)
        candidate = self.review.candidates[state["candidate"]]
        variant = next(item for item in candidate.variants if item.name == state["variant"])
        joints_pos, joints_rot, foot_contacts = _load_motion_tensors(variant.motion_path)

        from kimodo.skeleton.registry import build_skeleton
        from kimodo.viz.scene import Character
        from kimodo.viz.playback import CharacterMotion

        skeleton = build_skeleton(int(joints_pos.shape[1]))
        character = Character(
            "prior_candidate",
            client,
            skeleton,
            create_skeleton_mesh=True,
            create_skinned_mesh=True,
            visible_skeleton=state["gui"]["skeleton"].value,
            visible_skinned_mesh=state["gui"]["mesh"].value,
            mesh_mode="g1_stl" if int(joints_pos.shape[1]) == 34 else None,
        )
        motion = CharacterMotion(character, joints_pos, joints_rot, foot_contacts)
        state["motion"] = motion
        state["frame"] = 0
        state["fps"] = 30.0
        self._setup_timeline(state, motion.length)
        self._add_overlays(state, candidate)
        self._set_frame(state, 0, update_timeline=True)

    def _setup_timeline(self, state: dict[str, Any], frame_count: int) -> None:
        client = state["client"]
        if not hasattr(client, "timeline"):
            return
        try:
            client.timeline.clear_prompts()
            client.timeline.clear_keyframes()
            client.timeline.clear_intervals()
            client.timeline.set_zoom_settings(
                default_num_frames_zoom=max(60, frame_count),
                max_frames_zoom=max(1000, frame_count),
            )
        except Exception:
            pass
        candidate = self.review.candidates[state["candidate"]]
        for segment in candidate.prompt_segments:
            client.timeline.add_prompt(
                segment.get("text", ""),
                int(segment.get("start_frame", 0)),
                int(segment.get("end_frame", 0)),
            )
        client.timeline.set_current_frame(0)

    def _add_overlays(self, state: dict[str, Any], candidate: PriorViewerCandidate) -> None:
        client = state["client"]
        overlays = []
        visible_constraints = state.get("show_constraints", True)
        visible_constraint_labels = visible_constraints and state.get("show_constraint_labels", False)
        for constraint in candidate.constraints:
            if not constraint.get("show_in_review", True):
                continue
            position = np.asarray(constraint["position"], dtype=np.float64)
            marker = _add_constraint_marker(
                client,
                f"/prior_constraints/{constraint['label']}/marker",
                position,
                visible_constraints,
            )
            overlays.append(("constraint", marker))
            label = client.scene.add_label(
                name=f"/prior_constraints/{constraint['label']}_label",
                text=f"{constraint['label']} @ {constraint['frame']}",
                position=position + CONSTRAINT_LABEL_OFFSET,
                font_size_mode="screen",
                font_screen_scale=0.45,
                anchor="bottom-center",
            )
            label.visible = visible_constraint_labels
            overlays.append(("constraint_label", label))

        line_points = _target_line_segments(candidate.constraints)
        if line_points:
            line = client.scene.add_line_segments(
                name="/prior_targets/brush_lines",
                points=np.asarray(line_points, dtype=np.float64),
                colors=(102, 217, 255),
                line_width=TARGET_LINE_WIDTH,
            )
            line.visible = state.get("show_targets", True)
            overlays.append(("target", line))
        state["overlays"] = overlays

    def _set_overlay_visibility(self, state: dict[str, Any]) -> None:
        for kind, handle in state.get("overlays", []):
            visible = _overlay_visible(kind, state)
            if hasattr(handle, "set_visible"):
                handle.set_visible(visible)
            elif hasattr(handle, "visible"):
                handle.visible = visible

    def _clear_state_scene(self, state: dict[str, Any]) -> None:
        motion = state.get("motion")
        if motion is not None:
            motion.clear()
            state["motion"] = None
        client = state.get("client")
        for _kind, handle in state.get("overlays", []):
            if hasattr(handle, "clear"):
                handle.clear()
            elif client is not None and hasattr(handle, "name"):
                client.scene.remove_by_name(handle.name)
        state["overlays"] = []

    def _set_frame(self, state: dict[str, Any], frame_idx: int, *, update_timeline: bool) -> None:
        motion = state.get("motion")
        if motion is None:
            return
        frame_idx = max(0, min(int(frame_idx), motion.length - 1))
        state["frame"] = frame_idx
        motion.set_frame(frame_idx)
        if update_timeline and hasattr(state["client"], "timeline"):
            state["client"].timeline.set_current_frame(frame_idx)

    def _candidate_markdown(self, candidate_name: str) -> str:
        candidate = self.review.candidates[candidate_name]
        return "\n\n".join(
            [
                f"**Task:** `{self.review.task_name}`",
                f"**Model:** `{self.review.model}`",
                _prompt_markdown(candidate.prompt_segments),
                _constraint_markdown(candidate.constraints),
                _metrics_markdown(candidate.metrics),
            ]
        )


def launch_prior_viewer(
    run_folder: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7861,
    server_factory: Callable[..., Any] | None = None,
) -> PriorViewerApp:
    review = load_prior_review_folder(run_folder)
    return PriorViewerApp(review, host=host, port=port, server_factory=server_factory)


def load_prior_review_folder(run_folder: str | Path) -> PriorReviewFolder:
    """Load a prior run folder for local review without loading any generation model."""
    run_folder = Path(run_folder).expanduser().resolve()
    required = {
        "manifest": run_folder / "manifest.json",
        "task": run_folder / "task.json",
        "metrics": run_folder / "metrics.json",
        "review": run_folder / "review.json",
    }
    missing = [path.name for path in required.values() if not path.exists()]
    if missing:
        raise ValueError(f"Prior review folder is missing required file(s): {', '.join(sorted(missing))}")

    manifest = _read_json(required["manifest"])
    task = _read_json(required["task"])
    metrics = _read_json(required["metrics"])
    review = _read_json(required["review"])

    candidates: dict[str, PriorViewerCandidate] = {}
    metric_candidates = metrics.get("candidates", {})
    review_candidates = review.get("candidates", {})
    for name, candidate_data in manifest.get("candidates", {}).items():
        variants = _motion_variants(run_folder, candidate_data)
        review_data = review_candidates.get(name, {})
        candidates[name] = PriorViewerCandidate(
            name=name,
            status=review_data.get("status", "needs_review"),
            notes=review_data.get("notes", ""),
            variants=variants,
            prompt_segments=list(candidate_data.get("prompt_segments", [])),
            constraints=list(candidate_data.get("constraints", [])),
            metrics=dict(metric_candidates.get(name, {})),
        )

    if not candidates:
        raise ValueError("Prior review folder has no candidates in manifest.json")

    return PriorReviewFolder(
        run_folder=run_folder,
        task_name=manifest.get("task_name", task.get("task_name", run_folder.name)),
        model=manifest.get("model", task.get("model", "")),
        manifest=manifest,
        task=task,
        metrics=metrics,
        review=review,
        candidates=candidates,
    )


def save_review_decision(run_folder: str | Path, candidate_name: str, status: str, notes: str) -> dict[str, Any]:
    """Persist a human review decision back to ``review.json``."""
    run_folder = Path(run_folder).expanduser().resolve()
    review_path = run_folder / "review.json"
    review = _read_json(review_path)
    candidates = review.setdefault("candidates", {})
    if candidate_name not in candidates:
        raise ValueError(f"Unknown review candidate: {candidate_name}")

    allowed = review.get("review_statuses")
    if allowed and status not in allowed:
        raise ValueError(f"Unknown review status '{status}'. Expected one of: {allowed}")

    candidate = candidates[candidate_name]
    candidate["status"] = status
    candidate["notes"] = notes
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    return candidate


def _load_motion_tensors(motion_path: Path):
    import torch

    from kimodo.exports.motion_io import load_kimodo_npz

    data = load_kimodo_npz(str(motion_path))
    joints_pos = _coerce_motion_array(data["posed_joints"], dims=3)
    if "global_rot_mats" in data:
        joints_rot = _coerce_motion_array(data["global_rot_mats"], dims=4)
    else:
        joints_rot = np.broadcast_to(np.eye(3, dtype=np.float32), joints_pos.shape[:2] + (3, 3)).copy()
    foot_contacts = data.get("foot_contacts")
    if foot_contacts is not None:
        foot_contacts = _coerce_motion_array(foot_contacts, dims=2)
        foot_contacts = torch.as_tensor(foot_contacts)
    return (
        torch.as_tensor(joints_pos, dtype=torch.float32),
        torch.as_tensor(joints_rot, dtype=torch.float32),
        foot_contacts,
    )


def _add_constraint_marker(client: Any, name: str, position: np.ndarray, visible: bool):
    import trimesh

    marker = trimesh.creation.icosphere(subdivisions=2, radius=CONSTRAINT_MARKER_RADIUS)
    return client.scene.add_mesh_simple(
        name=name,
        vertices=marker.vertices,
        faces=marker.faces,
        position=position,
        color=(255, 80, 96),
        visible=visible,
    )


def _overlay_visible(kind: str, state: dict[str, Any]) -> bool:
    if kind == "constraint":
        return state.get("show_constraints", True)
    if kind == "constraint_label":
        return state.get("show_constraints", True) and state.get("show_constraint_labels", False)
    return state.get("show_targets", True)


def _coerce_motion_array(array: np.ndarray, *, dims: int) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == dims + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != dims:
        raise ValueError(f"Expected motion array with {dims} dimensions, got shape {array.shape}")
    return array


def _target_line_segments(constraints: list[dict[str, Any]]) -> list[list[list[float]]]:
    stroke = [
        constraint
        for constraint in constraints
        if "row_" in constraint.get("label", "") or "brush_stroke" in str(constraint.get("role", ""))
    ]
    if len(stroke) < 2:
        return []
    start = stroke[0].get("true_point") or stroke[0]["position"]
    end = stroke[-1].get("true_point") or stroke[-1]["position"]
    return [[start, end]]


def _prompt_markdown(prompt_segments: list[dict[str, Any]]) -> str:
    rows = ["**Prompt Segments**"]
    for segment in prompt_segments:
        rows.append(
            f"- `{segment.get('label', 'segment')}` [{segment.get('start_frame')}-{segment.get('end_frame')}]: "
            f"{segment.get('text', '')}"
        )
    return "\n".join(rows)


def _constraint_markdown(constraints: list[dict[str, Any]]) -> str:
    rows = ["**Constraint Points**"]
    for constraint in constraints:
        rows.append(
            f"- `{constraint.get('label')}` frame `{constraint.get('frame')}` "
            f"{constraint.get('end_effector')}: `{constraint.get('position')}`"
        )
    return "\n".join(rows)


def _metrics_markdown(metrics: dict[str, Any]) -> str:
    keys = ["constraint_error", "start_jump", "root_drift", "extra_motion_after_task"]
    rows = ["**Metrics**"]
    for key in keys:
        rows.append(f"- `{key}`: `{metrics.get(key)}`")
    return "\n".join(rows)


def _motion_variants(run_folder: Path, candidate_data: dict[str, Any]) -> list[PriorMotionVariant]:
    variants = [
        PriorMotionVariant(
            name="raw",
            kind="raw",
            motion_path=_resolve_run_path(run_folder, candidate_data["raw_motion"]),
        )
    ]
    for name, treatment in candidate_data.get("postprocessed", {}).items():
        variants.append(
            PriorMotionVariant(
                name=name,
                kind="postprocessed",
                motion_path=_resolve_run_path(run_folder, treatment["output_motion"]),
            )
        )
    return variants


def _resolve_run_path(run_folder: Path, rel_path: str) -> Path:
    path = (run_folder / rel_path).resolve()
    if not path.exists():
        raise ValueError(f"Motion file does not exist: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
