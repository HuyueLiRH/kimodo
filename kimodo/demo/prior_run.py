# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-spec driven prior-run helpers for building motion experiments."""

from __future__ import annotations

import html
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kimodo.demo.web_equivalent import (
    END_EFFECTOR_TYPES,
    EndEffectorSpec,
    EndEffectorTarget,
    build_end_effector_constraint,
)
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.tools import seed_everything


SUPPORTED_FIRST_STAGE_MODEL = "kimodo-g1-rp"
REVIEW_STATUSES = [
    "needs_review",
    "raw_accepted",
    "postprocessed_accepted",
    "needs_postprocess",
    "needs_regeneration",
    "rejected",
]

G1_BONE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (0, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
    (17, 26),
    (26, 27),
    (27, 28),
    (28, 29),
    (29, 30),
    (30, 31),
    (31, 32),
    (32, 33),
]


@dataclass(frozen=True)
class PromptSegment:
    label: str
    text: str
    start_frame: int
    end_frame: int
    segment_source: str = "hand_authored"

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class DeclaredConstraintPoint:
    label: str
    end_effector: str
    frame: int
    position: list[float]
    coordinate_frame: str = "world"
    used_for_generation: bool = True
    show_in_review: bool = True
    used_for_postprocess: bool = False
    role: str | None = None
    wrist_point: list[float] | None = None
    true_point: list[float] | None = None
    use_wrist: bool = False

    def to_end_effector_target(self) -> EndEffectorTarget:
        return EndEffectorTarget(
            frame=self.frame,
            point=list(self.position),
            label=self.label,
            true_point=self.true_point,
            wrist_point=self.wrist_point,
            use_wrist=self.use_wrist,
        )


@dataclass(frozen=True)
class PostprocessingTreatment:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PriorCandidateSpec:
    name: str
    recorded_seed: int
    prompt_segments: list[PromptSegment]
    constraints: list[DeclaredConstraintPoint]
    prompt_strategy: str = "single_prompt"
    cfg_type: str = "separated"
    cfg_weight: list[float] = field(default_factory=lambda: [2.4, 4.0])
    diffusion_steps: int = 200
    num_transition_frames: int = 3
    postprocessing: list[PostprocessingTreatment] = field(default_factory=list)

    @property
    def prompts(self) -> list[str]:
        return [segment.text for segment in self.prompt_segments]

    @property
    def segments(self) -> list[int]:
        return [segment.frame_count for segment in self.prompt_segments]


@dataclass(frozen=True)
class PriorTaskSpec:
    task_name: str
    model: str
    duration_frames: int
    candidates: list[PriorCandidateSpec]


def load_prior_task_spec(path: str | Path) -> PriorTaskSpec:
    return load_prior_task_spec_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_prior_task_spec_dict(data: dict[str, Any]) -> PriorTaskSpec:
    model = str(data.get("model") or "").strip()
    if model != SUPPORTED_FIRST_STAGE_MODEL:
        raise ValueError(f"First-stage prior runs require model {SUPPORTED_FIRST_STAGE_MODEL!r}; got {model!r}.")

    task_name = str(data.get("task_name") or data.get("name") or "").strip()
    if not task_name:
        raise ValueError("Executable task spec requires task_name.")

    duration_frames = _duration_frames(data)
    candidate_data = data.get("candidates")
    if not isinstance(candidate_data, list) or not candidate_data:
        raise ValueError("Executable task spec requires a non-empty candidates list.")

    names: set[str] = set()
    candidates = []
    for index, candidate in enumerate(candidate_data):
        parsed = _load_candidate(candidate, duration_frames, index=index, defaults=data)
        if parsed.name in names:
            raise ValueError(f"Duplicate candidate name {parsed.name!r}.")
        names.add(parsed.name)
        candidates.append(parsed)

    return PriorTaskSpec(
        task_name=task_name,
        model=model,
        duration_frames=duration_frames,
        candidates=candidates,
    )


def run_prior_with_model(
    model,
    spec: PriorTaskSpec,
    output_dir: str | Path,
    *,
    task_source: dict[str, Any] | None = None,
    save_csv: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if task_source is not None:
        (output_dir / "task.json").write_text(json.dumps(task_source, indent=2), encoding="utf-8")
    else:
        (output_dir / "task.json").write_text(json.dumps(_task_spec_to_json(spec), indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "first_stage_prior_run",
        "task_name": spec.task_name,
        "model": spec.model,
        "candidates": {},
    }
    metrics: dict[str, Any] = {"candidates": {}}
    review: dict[str, Any] = {
        "schema_version": 1,
        "review_statuses": list(REVIEW_STATUSES),
        "candidates": {},
    }

    for candidate in spec.candidates:
        raw_dir = output_dir / "raw" / candidate.name
        raw_dir.mkdir(parents=True, exist_ok=True)
        recipe = _candidate_recipe(spec, candidate)
        (raw_dir / "recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")

        seed_everything(candidate.recorded_seed)
        kwargs = build_candidate_generation_kwargs(model, candidate)
        prompts = kwargs.pop("prompts")
        segments = kwargs.pop("segments")
        output = model(prompts, segments, **kwargs)
        sample = _single_sample(output, 0, int(output["posed_joints"].shape[0]))
        save_kimodo_npz(str(raw_dir / "motion.npz"), sample)
        if save_csv:
            _save_csv_if_available(model, output, raw_dir)

        candidate_metrics = compute_candidate_metrics(model, sample, candidate)
        metrics["candidates"][candidate.name] = candidate_metrics
        (raw_dir / "metrics.json").write_text(json.dumps(candidate_metrics, indent=2), encoding="utf-8")

        manifest_candidate = {
            "name": candidate.name,
            "recorded_seed": candidate.recorded_seed,
            "prompt_strategy": candidate.prompt_strategy,
            "prompt_segments": [asdict(segment) for segment in candidate.prompt_segments],
            "constraints": [asdict(constraint) for constraint in candidate.constraints],
            "raw_motion": _rel(output_dir, raw_dir / "motion.npz"),
            "raw_recipe": _rel(output_dir, raw_dir / "recipe.json"),
            "metrics": f"metrics.json#candidates/{candidate.name}",
            "review": f"review.json#candidates/{candidate.name}",
            "postprocessed": {},
        }

        for treatment in candidate.postprocessing:
            post_dir = output_dir / "postprocessed" / candidate.name / treatment.name
            post_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_dir / "motion.npz", post_dir / "motion.npz")
            treatment_data = {
                "name": treatment.name,
                "params": treatment.params,
                "source_raw_motion": _rel(output_dir, raw_dir / "motion.npz"),
                "output_motion": _rel(output_dir, post_dir / "motion.npz"),
            }
            (post_dir / "treatment.json").write_text(json.dumps(treatment_data, indent=2), encoding="utf-8")
            manifest_candidate["postprocessed"][treatment.name] = treatment_data

        manifest["candidates"][candidate.name] = manifest_candidate
        review["candidates"][candidate.name] = {
            "status": "needs_review",
            "admission_blockers": [],
            "refinement_debt": [],
            "notes": "",
        }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    write_review_gallery(output_dir, manifest, metrics, review)
    return output_dir


def build_candidate_generation_kwargs(model, candidate: PriorCandidateSpec) -> dict[str, Any]:
    constraints = []
    for end_effector, points in _generation_constraint_groups(candidate).items():
        targets = [point.to_end_effector_target() for point in points]
        constraints.append(build_end_effector_constraint(model.skeleton, EndEffectorSpec(type=end_effector, targets=targets)))
    return {
        "prompts": candidate.prompts,
        "segments": candidate.segments,
        "constraint_lst": constraints,
        "num_denoising_steps": int(candidate.diffusion_steps),
        "num_samples": 1,
        "multi_prompt": len(candidate.prompt_segments) > 1,
        "num_transition_frames": int(candidate.num_transition_frames),
        "post_processing": False,
        "return_numpy": True,
        "cfg_type": candidate.cfg_type,
        "cfg_weight": list(candidate.cfg_weight),
    }


def compute_candidate_metrics(model, motion: dict[str, Any], candidate: PriorCandidateSpec) -> dict[str, Any]:
    posed_joints = np.asarray(motion.get("posed_joints", []))
    root_positions = np.asarray(motion.get("root_positions", []))
    metrics: dict[str, Any] = {
        "frame_count": int(posed_joints.shape[0]) if posed_joints.ndim >= 1 else 0,
        "start_jump": None,
        "root_drift": None,
        "body_rotation_change": None,
        "constraint_error": None,
        "extra_motion_after_task": None,
        "target_points": _metric_target_points(candidate),
        "row_specs": _metric_row_specs(candidate),
    }
    if root_positions.ndim == 2 and len(root_positions) > 1:
        metrics["start_jump"] = float(np.linalg.norm(root_positions[1] - root_positions[0]))
        metrics["root_drift"] = float(np.linalg.norm(root_positions[-1] - root_positions[0]))
        last_constraint_frame = max(point.frame for point in candidate.constraints)
        if last_constraint_frame + 1 < len(root_positions):
            metrics["extra_motion_after_task"] = float(
                np.linalg.norm(root_positions[-1] - root_positions[last_constraint_frame])
            )

    heading = np.asarray(motion.get("global_root_heading", []))
    if heading.ndim == 2 and len(heading) > 1:
        metrics["body_rotation_change"] = float(np.linalg.norm(heading[-1] - heading[0]))

    errors = []
    if posed_joints.ndim == 3:
        for end_effector, points in _generation_constraint_groups(candidate).items():
            endpoint_index = _endpoint_joint_index(model.skeleton, end_effector)
            if endpoint_index is None:
                continue
            for point in points:
                if 0 <= point.frame < len(posed_joints):
                    actual = posed_joints[point.frame, endpoint_index]
                    errors.append(float(np.linalg.norm(actual - np.asarray(point.position, dtype=float))))
    if errors:
        metrics["constraint_error"] = {
            "mean": float(np.mean(errors)),
            "max": float(np.max(errors)),
            "count": len(errors),
        }
    return metrics


def _metric_target_points(candidate: PriorCandidateSpec) -> list[dict[str, Any]]:
    targets = []
    for index, point in enumerate(candidate.constraints):
        targets.append(
            {
                "name": point.label,
                "phase": point.role or "",
                "frame": point.frame,
                "point": list(point.position),
                "true_point": point.true_point,
                "wrist_point": point.wrist_point,
                "use_wrist": point.use_wrist,
                "row": 1 if "row" in point.label or "brush" in (point.role or "") else None,
                "col": index,
            }
        )
    return targets


def _metric_row_specs(candidate: PriorCandidateSpec) -> list[dict[str, Any]]:
    stroke_points = [
        point
        for point in candidate.constraints
        if "row_" in point.label or "brush_stroke" in (point.role or "")
    ]
    if len(stroke_points) < 2:
        return []
    start = stroke_points[0].true_point or stroke_points[0].position
    end = stroke_points[-1].true_point or stroke_points[-1].position
    return [
        {
            "row": 1,
            "start_point": list(start),
            "end_point": list(end),
        }
    ]


def write_review_gallery(
    output_dir: str | Path,
    manifest: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    review: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    manifest = manifest or json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = metrics or json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    review = review or json.loads((output_dir / "review.json").read_text(encoding="utf-8"))
    motion_payload = []
    rows = []
    for name, candidate in manifest["candidates"].items():
        motion = _gallery_motion_payload(output_dir, name, candidate)
        motion_payload.append(motion)
        segments = "".join(
            "<li><strong>{}</strong> [{}-{}]: {} <em>{}</em></li>".format(
                html.escape(segment["label"]),
                segment["start_frame"],
                segment["end_frame"],
                html.escape(segment["text"]),
                html.escape(segment.get("segment_source", "")),
            )
            for segment in candidate["prompt_segments"]
        )
        constraints = "".join(
            "<li><strong>{}</strong> {} frame {} {} {}</li>".format(
                html.escape(constraint["label"]),
                html.escape(constraint["end_effector"]),
                constraint["frame"],
                html.escape(constraint["coordinate_frame"]),
                html.escape(str(constraint["position"])),
            )
            for constraint in candidate["constraints"]
            if constraint.get("show_in_review", True)
        )
        rows.append(
            """
            <section class="candidate">
              <h2>{name}</h2>
              <p>Status: <code>{status}</code></p>
              <p>Raw motion: <code>{raw_motion}</code></p>
              <h3>Prompt Segments</h3>
              <ul>{segments}</ul>
              <h3>Constraint Points</h3>
              <ul>{constraints}</ul>
              <h3>Metrics</h3>
              <pre>{metrics}</pre>
            </section>
            """.format(
                name=html.escape(name),
                status=html.escape(review["candidates"][name]["status"]),
                raw_motion=html.escape(candidate["raw_motion"]),
                segments=segments,
                constraints=constraints,
                metrics=html.escape(json.dumps(metrics["candidates"].get(name, {}), indent=2)),
            )
        )
    html_doc = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title} prior review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    .viewer {{ display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; align-items: start; }}
    canvas {{ width: 100%; min-height: 520px; background: #0f1720; border-radius: 8px; }}
    .controls {{ display: grid; gap: 10px; }}
    select, input, button {{ font: inherit; padding: 7px 9px; }}
    .candidate {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    h2 {{ margin-top: 0; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>{title} Review</h1>
  <div class="viewer">
    <canvas id="motionCanvas" width="960" height="640"></canvas>
    <div class="controls">
      <label>Candidate <select id="candidateSelect"></select></label>
      <button id="playButton">Play</button>
      <label>Frame <span id="frameLabel">0</span><input id="frameSlider" type="range" min="0" value="0"></label>
      <div id="candidateInfo"></div>
    </div>
  </div>
  {rows}
  <script>
    const DATA = {payload};
    const canvas = document.getElementById("motionCanvas");
    const ctx = canvas.getContext("2d");
    const select = document.getElementById("candidateSelect");
    const slider = document.getElementById("frameSlider");
    const playButton = document.getElementById("playButton");
    const frameLabel = document.getElementById("frameLabel");
    const info = document.getElementById("candidateInfo");
    let motionIndex = 0;
    let frame = 0;
    let playing = false;

    function current() {{ return DATA.motions[motionIndex]; }}
    function project(point, bounds) {{
      const scale = Math.min(canvas.width, canvas.height) * 0.68 / Math.max(bounds.radius, 0.2);
      const x = point[0] - bounds.center[0];
      const y = point[1] - bounds.center[1];
      const z = point[2] - bounds.center[2];
      return [
        canvas.width * 0.5 + (x + z * 0.38) * scale,
        canvas.height * 0.62 - y * scale + z * 0.08 * scale,
      ];
    }}
    function drawPoint(point, bounds, radius, color) {{
      const p = project(point, bounds);
      ctx.beginPath();
      ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }}
    function drawLabel(point, bounds, text, color) {{
      const p = project(point, bounds);
      ctx.fillStyle = color;
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText(text, p[0] + 8, p[1] - 8);
    }}
    function drawLine(a, b, bounds, color, width) {{
      const pa = project(a, bounds);
      const pb = project(b, bounds);
      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
      ctx.lineWidth = width;
      ctx.strokeStyle = color;
      ctx.stroke();
    }}
    function draw() {{
      const m = current();
      const pose = m.frames[frame];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      m.rowSpecs.forEach(row => drawLine(row.start_point, row.end_point, m.bounds, "#66d9ff", 3));
      m.constraints.forEach(target => {{
        const near = Math.abs(frame - target.frame) <= 2;
        if (target.true_point) drawLine(target.true_point, target.position, m.bounds, near ? "#ffd27a" : "#8a7045", near ? 2 : 1);
        if (target.wrist_point) drawLine(target.position, target.wrist_point, m.bounds, near ? "#d8a8ff" : "#7d5aa0", near ? 2 : 1);
        drawPoint(target.position, m.bounds, near ? 7 : 4.5, near ? "#ff4f64" : "#cf4052");
        if (target.true_point) drawPoint(target.true_point, m.bounds, near ? 5 : 3.5, "#46b9d9");
        if (target.wrist_point) drawPoint(target.wrist_point, m.bounds, near ? 5 : 3.5, "#9e78ca");
        if (near) drawLabel(target.position, m.bounds, target.label, "#ffb7c1");
      }});
      m.edges.forEach(edge => drawLine(pose[edge[0]], pose[edge[1]], m.bounds, "#86a9ff", 3));
      pose.forEach(point => drawPoint(point, m.bounds, 3.2, "#d7e2ff"));
      frameLabel.textContent = `${{frame}} / ${{m.frames.length - 1}}`;
      slider.max = String(m.frames.length - 1);
      slider.value = String(frame);
    }}
    function renderInfo() {{
      const m = current();
      info.innerHTML = `
        <h2>${{m.name}}</h2>
        <p>Status: <code>${{m.status}}</code></p>
        <p>Raw: <code>${{m.rawMotion}}</code></p>
        <h3>Segments</h3>
        <ul>${{m.promptSegments.map(s => `<li><strong>${{s.label}}</strong> [${{s.start_frame}}-${{s.end_frame}}]: ${{s.text}}</li>`).join("")}}</ul>
      `;
    }}
    DATA.motions.forEach((m, i) => {{
      const option = document.createElement("option");
      option.value = String(i);
      option.textContent = m.name;
      select.appendChild(option);
    }});
    select.addEventListener("change", () => {{
      motionIndex = Number(select.value);
      frame = 0;
      renderInfo();
      draw();
    }});
    slider.addEventListener("input", () => {{ frame = Number(slider.value); draw(); }});
    playButton.addEventListener("click", () => {{
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
    }});
    function tick() {{
      if (playing) {{
        frame = (frame + 1) % current().frames.length;
        draw();
      }}
      setTimeout(tick, 1000 / 30);
    }}
    if (DATA.motions.length) {{
      renderInfo();
      draw();
      tick();
    }}
  </script>
</body>
</html>
""".format(
        title=html.escape(manifest["task_name"]),
        rows="\n".join(rows),
        payload=json.dumps({"motions": motion_payload}, ensure_ascii=False),
    )
    gallery_path = output_dir / "gallery.html"
    gallery_path.write_text(html_doc, encoding="utf-8")
    return gallery_path


def _gallery_motion_payload(output_dir: Path, name: str, candidate: dict[str, Any]) -> dict[str, Any]:
    motion_path = output_dir / candidate["raw_motion"]
    frames = []
    bounds = {"center": [0, 0, 0], "radius": 1}
    edges = []
    if motion_path.exists():
        with np.load(motion_path, allow_pickle=True) as data:
            if "posed_joints" in data:
                posed = data["posed_joints"].astype(float)
                frames = posed.round(5).tolist()
                flat = posed.reshape(-1, 3)
                extras = []
                for constraint in candidate["constraints"]:
                    extras.append(constraint["position"])
                    if constraint.get("true_point"):
                        extras.append(constraint["true_point"])
                    if constraint.get("wrist_point"):
                        extras.append(constraint["wrist_point"])
                if extras:
                    flat = np.concatenate([flat, np.asarray(extras, dtype=float).reshape(-1, 3)], axis=0)
                mins = flat.min(axis=0)
                maxs = flat.max(axis=0)
                center = ((mins + maxs) * 0.5).tolist()
                radius = float(np.linalg.norm(maxs - mins) * 0.5)
                bounds = {"center": center, "radius": max(radius, 0.2)}
                joint_count = int(posed.shape[1])
                edges = [[a, b] for a, b in G1_BONE_EDGES if a < joint_count and b < joint_count]
    return {
        "name": name,
        "status": "needs_review",
        "rawMotion": candidate["raw_motion"],
        "promptSegments": candidate["prompt_segments"],
        "constraints": candidate["constraints"],
        "rowSpecs": _gallery_row_specs(candidate["constraints"]),
        "frames": frames,
        "bounds": bounds,
        "edges": edges,
    }


def _gallery_row_specs(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stroke_points = [
        point
        for point in constraints
        if "row_" in point.get("label", "") or "brush_stroke" in str(point.get("role", ""))
    ]
    if len(stroke_points) < 2:
        return []
    return [
        {
            "row": 1,
            "start_point": stroke_points[0].get("true_point") or stroke_points[0]["position"],
            "end_point": stroke_points[-1].get("true_point") or stroke_points[-1]["position"],
        }
    ]


def sync_local_review_folder(source: str | Path, destination: str | Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    allowed_suffixes = {".npz", ".json", ".html", ".css", ".js"}
    for path in source.rglob("*"):
        if path.is_dir() or path.suffix not in allowed_suffixes:
            continue
        rel = path.relative_to(source)
        out = destination / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
    return destination


def write_recipe_note_draft(run_folder: str | Path, candidate_name: str, output_root: str | Path) -> tuple[Path, Path]:
    run_folder = Path(run_folder)
    output_root = Path(output_root)
    manifest = json.loads((run_folder / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_folder / "metrics.json").read_text(encoding="utf-8"))
    review = json.loads((run_folder / "review.json").read_text(encoding="utf-8"))
    candidate = manifest["candidates"][candidate_name]
    raw_recipe = json.loads((run_folder / candidate["raw_recipe"]).read_text(encoding="utf-8"))

    recipe_dir = output_root / "examples" / "prior_recipes"
    note_dir = output_root / "docs" / "prior_recipes"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    note_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipe_dir / f"{candidate_name}.json"
    note_path = note_dir / f"{candidate_name}.md"
    recipe_path.write_text(json.dumps(raw_recipe, indent=2), encoding="utf-8")

    review_record = review["candidates"].get(candidate_name, {})
    note = [
        f"# {candidate_name}",
        "",
        f"- Task: `{manifest['task_name']}`",
        f"- Model: `{manifest['model']}`",
        f"- Review status: `{review_record.get('status', 'needs_review')}`",
        f"- Raw motion: `{candidate['raw_motion']}`",
        f"- Gallery: `{run_folder / 'gallery.html'}`",
        "",
        "## Prompt Segments",
        "",
    ]
    for segment in candidate["prompt_segments"]:
        note.append(f"- `{segment['label']}` frames {segment['start_frame']}-{segment['end_frame']}: {segment['text']}")
    note.extend(["", "## Constraint Points", ""])
    for constraint in candidate["constraints"]:
        note.append(
            f"- `{constraint['label']}` `{constraint['end_effector']}` frame {constraint['frame']} "
            f"{constraint['position']} in `{constraint['coordinate_frame']}`"
        )
    note.extend(
        [
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(metrics["candidates"].get(candidate_name, {}), indent=2),
            "```",
            "",
            "## Review Notes",
            "",
            review_record.get("notes", ""),
        ]
    )
    note_path.write_text("\n".join(note), encoding="utf-8")
    return recipe_path, note_path


def _duration_frames(data: dict[str, Any]) -> int:
    raw = data.get("duration_frames") or data.get("num_frames") or data.get("duration")
    if raw is None:
        raise ValueError("Executable task spec requires duration_frames.")
    duration = int(raw)
    if duration <= 0:
        raise ValueError("duration_frames must be positive.")
    return duration


def _load_candidate(data: dict[str, Any], duration_frames: int, *, index: int, defaults: dict[str, Any]) -> PriorCandidateSpec:
    name = str(data.get("name") or "").strip()
    if not name:
        raise ValueError(f"Candidate at index {index} requires a candidate name.")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", name):
        raise ValueError(f"Invalid candidate name {name!r}.")

    seed = data.get("seed", defaults.get("seed"))
    if seed is None:
        raise ValueError(f"Candidate {name!r} requires a recorded seed.")

    prompt_segments = _load_prompt_segments(data, duration_frames, name)
    constraints = _load_constraints(data, duration_frames, name, defaults=defaults)
    if not any(point.used_for_generation for point in constraints):
        raise ValueError(f"Candidate {name!r} requires at least one generation constraint.")
    generation_effectors = {point.end_effector for point in constraints if point.used_for_generation}
    if len(generation_effectors) > 1:
        raise ValueError(f"Candidate {name!r} currently supports one generation end effector type per candidate.")

    cfg_data = dict(data.get("cfg") or defaults.get("cfg") or {})
    cfg_weight = data.get("cfg_weight", defaults.get("cfg_weight", cfg_data.get("weight", [2.4, 4.0])))
    return PriorCandidateSpec(
        name=name,
        recorded_seed=int(seed),
        prompt_strategy=str(data.get("prompt_strategy") or _infer_prompt_strategy(prompt_segments)),
        prompt_segments=prompt_segments,
        constraints=constraints,
        cfg_type=str(data.get("cfg_type", defaults.get("cfg_type", cfg_data.get("type", "separated")))),
        cfg_weight=[float(value) for value in cfg_weight],
        diffusion_steps=int(data.get("diffusion_steps", defaults.get("diffusion_steps", defaults.get("num_denoising_steps", 200)))),
        num_transition_frames=int(data.get("num_transition_frames", defaults.get("num_transition_frames", 3))),
        postprocessing=_load_postprocessing(data.get("postprocessing", defaults.get("postprocessing"))),
    )


def _load_prompt_segments(data: dict[str, Any], duration_frames: int, candidate_name: str) -> list[PromptSegment]:
    if data.get("prompt_segments"):
        segments = []
        for index, segment in enumerate(data["prompt_segments"]):
            label = str(segment.get("label") or f"segment_{index}").strip()
            text = str(segment.get("text") or segment.get("prompt") or "").strip()
            if not text:
                raise ValueError(f"Candidate {candidate_name!r} prompt segment {label!r} requires text.")
            start = int(segment.get("start_frame", segment.get("start", 0)))
            end = int(segment.get("end_frame", segment.get("end", start)))
            _validate_frame_range(start, end, duration_frames, f"Candidate {candidate_name!r} prompt segment {label!r}")
            segments.append(
                PromptSegment(
                    label=label,
                    text=text,
                    start_frame=start,
                    end_frame=end,
                    segment_source=str(segment.get("segment_source") or segment.get("source") or "hand_authored"),
                )
            )
        return segments

    prompts = list(data.get("prompts") or data.get("texts") or [])
    segments = [int(value) for value in data.get("segments") or data.get("num_frames") or []]
    if not prompts:
        raise ValueError(f"Candidate {candidate_name!r} requires prompt_segments or prompts.")
    if len(prompts) != len(segments):
        raise ValueError(f"Candidate {candidate_name!r} prompts and segments must have the same length.")
    out = []
    start = 0
    for index, (prompt, frame_count) in enumerate(zip(prompts, segments)):
        if frame_count <= 0:
            raise ValueError(f"Candidate {candidate_name!r} segment frame counts must be positive.")
        end = start + frame_count - 1
        _validate_frame_range(start, end, max(duration_frames, end + 1), f"Candidate {candidate_name!r} segment_{index}")
        out.append(
            PromptSegment(
                label=f"segment_{index}",
                text=str(prompt),
                start_frame=start,
                end_frame=end,
                segment_source="explicit_segments",
            )
        )
        start = end + 1
    return out


def _load_constraints(
    data: dict[str, Any],
    duration_frames: int,
    candidate_name: str,
    *,
    defaults: dict[str, Any] | None = None,
) -> list[DeclaredConstraintPoint]:
    defaults = defaults or {}
    raw_constraints = data.get("constraints", defaults.get("constraints"))
    if raw_constraints is None:
        end_effector = data.get("end_effector") or defaults.get("end_effector") or {}
        effector_type = end_effector.get("type") or data.get("active_end_effector")
        if not effector_type:
            effector_type = defaults.get("active_end_effector")
        targets = end_effector.get("targets") or data.get("targets") or defaults.get("targets") or []
        raw_constraints = [
            {
                "label": target.get("label") or target.get("name") or f"target_{index}",
                "end_effector": effector_type,
                "frame": target.get("frame"),
                "position": target.get("point") or target.get("position"),
                "coordinate_frame": target.get("coordinate_frame", "world"),
                "used_for_generation": target.get("used_for_generation", True),
                "show_in_review": target.get("show_in_review", True),
                "used_for_postprocess": target.get("used_for_postprocess", False),
                "role": target.get("role"),
                "wrist_point": target.get("wrist_point"),
                "true_point": target.get("true_point"),
                "use_wrist": target.get("use_wrist", target.get("wrist_point") is not None),
            }
            for index, target in enumerate(targets)
        ]
    if not isinstance(raw_constraints, list) or not raw_constraints:
        raise ValueError(f"Candidate {candidate_name!r} requires declared constraint points.")

    constraints = []
    for index, constraint in enumerate(raw_constraints):
        label = str(constraint.get("label") or constraint.get("name") or f"constraint_{index}").strip()
        end_effector = str(constraint.get("end_effector") or constraint.get("type") or "").strip()
        if end_effector not in END_EFFECTOR_TYPES:
            raise ValueError(f"Candidate {candidate_name!r} constraint {label!r} has unsupported end effector.")
        frame = int(constraint["frame"])
        if frame < 0 or frame >= duration_frames:
            raise ValueError(f"Candidate {candidate_name!r} constraint {label!r} frame is outside duration.")
        position = constraint.get("position") or constraint.get("point")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"Candidate {candidate_name!r} constraint {label!r} requires a 3D position.")
        coordinate_frame = str(constraint.get("coordinate_frame") or "world")
        if not coordinate_frame:
            raise ValueError(f"Candidate {candidate_name!r} constraint {label!r} requires a coordinate frame.")
        wrist_point = constraint.get("wrist_point")
        constraints.append(
            DeclaredConstraintPoint(
                label=label,
                end_effector=end_effector,
                frame=frame,
                position=[float(value) for value in position],
                coordinate_frame=coordinate_frame,
                used_for_generation=bool(constraint.get("used_for_generation", True)),
                show_in_review=bool(constraint.get("show_in_review", True)),
                used_for_postprocess=bool(constraint.get("used_for_postprocess", False)),
                role=constraint.get("role"),
                wrist_point=[float(value) for value in wrist_point] if wrist_point is not None else None,
                true_point=constraint.get("true_point"),
                use_wrist=bool(constraint.get("use_wrist", wrist_point is not None)),
            )
        )
    return constraints


def _load_postprocessing(data: Any) -> list[PostprocessingTreatment]:
    if not data:
        return []
    if data is True:
        raise ValueError("postprocessing must name an executable treatment; boolean true is ambiguous.")
    if isinstance(data, dict):
        if not data.get("enabled", False):
            return []
        treatments = data.get("treatments")
        if treatments is None:
            name = data.get("treatment") or data.get("name")
            if not name:
                raise ValueError("Enabled postprocessing requires a treatment name.")
            treatments = [{"name": name, "params": data.get("params", {})}]
    elif isinstance(data, list):
        treatments = data
    else:
        raise ValueError("postprocessing must be a dict or list.")

    out = []
    for treatment in treatments:
        name = str(treatment.get("name") or treatment.get("treatment") or "").strip()
        if not name:
            raise ValueError("Postprocessing treatment requires a name.")
        out.append(PostprocessingTreatment(name=name, params=dict(treatment.get("params") or {})))
    return out


def _validate_frame_range(start: int, end: int, duration_frames: int, context: str) -> None:
    if start < 0 or end < start or end >= duration_frames:
        raise ValueError(f"{context} has invalid frame range {start}-{end}.")


def _infer_prompt_strategy(prompt_segments: list[PromptSegment]) -> str:
    return "composed_action_prompt" if len(prompt_segments) > 1 else "single_prompt"


def _generation_constraint_groups(candidate: PriorCandidateSpec) -> dict[str, list[DeclaredConstraintPoint]]:
    groups: dict[str, list[DeclaredConstraintPoint]] = {}
    for point in candidate.constraints:
        if point.used_for_generation:
            groups.setdefault(point.end_effector, []).append(point)
    return groups


def _endpoint_joint_index(skeleton, end_effector: str) -> int | None:
    cls = END_EFFECTOR_TYPES[end_effector]
    _, position_joint_names = skeleton.expand_joint_names(cls.joint_names)
    if not position_joint_names:
        return None
    return skeleton.bone_index[position_joint_names[-1]]


def _candidate_recipe(spec: PriorTaskSpec, candidate: PriorCandidateSpec) -> dict[str, Any]:
    return {
        "task_name": spec.task_name,
        "model": spec.model,
        "duration_frames": spec.duration_frames,
        "candidate": {
            "name": candidate.name,
            "recorded_seed": candidate.recorded_seed,
            "prompt_strategy": candidate.prompt_strategy,
            "prompt_segments": [asdict(segment) for segment in candidate.prompt_segments],
            "constraints": [asdict(constraint) for constraint in candidate.constraints],
            "cfg_type": candidate.cfg_type,
            "cfg_weight": candidate.cfg_weight,
            "diffusion_steps": candidate.diffusion_steps,
            "num_transition_frames": candidate.num_transition_frames,
            "postprocessing": [asdict(treatment) for treatment in candidate.postprocessing],
        },
    }


def _task_spec_to_json(spec: PriorTaskSpec) -> dict[str, Any]:
    return {
        "task_name": spec.task_name,
        "model": spec.model,
        "duration_frames": spec.duration_frames,
        "candidates": [_candidate_recipe(spec, candidate)["candidate"] for candidate in spec.candidates],
    }


def _single_sample(output: dict[str, Any], index: int, n_samples: int) -> dict[str, Any]:
    return {
        key: (value[index] if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == n_samples else value)
        for key, value in output.items()
    }


def _save_csv_if_available(model, output: dict[str, Any], output_dir: Path) -> None:
    try:
        from kimodo.exports.mujoco import MujocoQposConverter
    except Exception as exc:
        print(f"Skipping CSV export because MuJoCo converter is unavailable: {exc}")
        return
    converter = MujocoQposConverter(model.skeleton)
    qpos = converter.dict_to_qpos(output, getattr(model, "device", "cpu"))
    converter.save_csv(qpos, str(output_dir / "motion.csv"))


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
