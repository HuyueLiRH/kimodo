#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


G1_BONES_WITH_PARENTS = [
    ("pelvis_skel", None),
    ("left_hip_pitch_skel", "pelvis_skel"),
    ("left_hip_roll_skel", "left_hip_pitch_skel"),
    ("left_hip_yaw_skel", "left_hip_roll_skel"),
    ("left_knee_skel", "left_hip_yaw_skel"),
    ("left_ankle_pitch_skel", "left_knee_skel"),
    ("left_ankle_roll_skel", "left_ankle_pitch_skel"),
    ("left_toe_base", "left_ankle_roll_skel"),
    ("right_hip_pitch_skel", "pelvis_skel"),
    ("right_hip_roll_skel", "right_hip_pitch_skel"),
    ("right_hip_yaw_skel", "right_hip_roll_skel"),
    ("right_knee_skel", "right_hip_yaw_skel"),
    ("right_ankle_pitch_skel", "right_knee_skel"),
    ("right_ankle_roll_skel", "right_ankle_pitch_skel"),
    ("right_toe_base", "right_ankle_roll_skel"),
    ("waist_yaw_skel", "pelvis_skel"),
    ("waist_roll_skel", "waist_yaw_skel"),
    ("waist_pitch_skel", "waist_roll_skel"),
    ("left_shoulder_pitch_skel", "waist_pitch_skel"),
    ("left_shoulder_roll_skel", "left_shoulder_pitch_skel"),
    ("left_shoulder_yaw_skel", "left_shoulder_roll_skel"),
    ("left_elbow_skel", "left_shoulder_yaw_skel"),
    ("left_wrist_roll_skel", "left_elbow_skel"),
    ("left_wrist_pitch_skel", "left_wrist_roll_skel"),
    ("left_wrist_yaw_skel", "left_wrist_pitch_skel"),
    ("left_hand_roll_skel", "left_wrist_yaw_skel"),
    ("right_shoulder_pitch_skel", "waist_pitch_skel"),
    ("right_shoulder_roll_skel", "right_shoulder_pitch_skel"),
    ("right_shoulder_yaw_skel", "right_shoulder_roll_skel"),
    ("right_elbow_skel", "right_shoulder_yaw_skel"),
    ("right_wrist_roll_skel", "right_elbow_skel"),
    ("right_wrist_pitch_skel", "right_wrist_roll_skel"),
    ("right_wrist_yaw_skel", "right_wrist_pitch_skel"),
    ("right_hand_roll_skel", "right_wrist_yaw_skel"),
]


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kimodo Motion Gallery</title>
<style>
  :root {
    color-scheme: dark;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #101217;
    color: #f3f5f9;
  }
  body {
    margin: 0;
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr auto;
  }
  header, footer {
    padding: 12px 16px;
    background: #171b23;
    border-bottom: 1px solid #2b3240;
  }
  footer {
    border-top: 1px solid #2b3240;
    border-bottom: 0;
    color: #aeb7c5;
    font-size: 13px;
  }
  h1 {
    margin: 0;
    font-size: 17px;
    font-weight: 650;
    letter-spacing: 0;
  }
  main {
    min-height: 0;
    display: grid;
    grid-template-columns: 1fr 360px;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    background: #0d1016;
    cursor: grab;
  }
  canvas:active {
    cursor: grabbing;
  }
  aside {
    overflow: auto;
    padding: 14px;
    background: #171b23;
    border-left: 1px solid #2b3240;
  }
  label {
    display: block;
    margin: 10px 0 5px;
    color: #b7c0ce;
    font-size: 13px;
  }
  select, button, a.button, input[type=text] {
    border: 1px solid #3c4657;
    background: #232a36;
    color: #f3f5f9;
    padding: 8px 10px;
    border-radius: 6px;
    font: inherit;
  }
  a.button {
    display: block;
    text-align: center;
    text-decoration: none;
    box-sizing: border-box;
  }
  select, input[type=text] {
    width: 100%;
    box-sizing: border-box;
  }
  button:hover, a.button:hover {
    background: #2d3544;
  }
  input[type=range] {
    width: 100%;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 10px 0;
  }
  .row > button {
    flex: 1;
  }
  .panel {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #2b3240;
    font-size: 13px;
    line-height: 1.5;
  }
  .label {
    color: #93b7ff;
    font-weight: 650;
  }
  .muted {
    color: #aeb7c5;
  }
  .item {
    padding: 7px 0;
    border-bottom: 1px solid #272f3c;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
  }
  .chip {
    border: 1px solid #394354;
    border-radius: 999px;
    padding: 3px 8px;
    color: #c5cfdd;
  }
  .hint {
    margin-top: 6px;
    color: #8f9caf;
    font-size: 12px;
    line-height: 1.4;
  }
  @media (max-width: 860px) {
    main {
      grid-template-columns: 1fr;
      grid-template-rows: 1fr auto;
    }
    aside {
      max-height: 46vh;
      border-left: 0;
      border-top: 1px solid #2b3240;
    }
  }
</style>
</head>
<body>
<header><h1>Kimodo Motion Gallery</h1></header>
<main>
  <canvas id="view"></canvas>
  <aside>
    <label for="categorySelect">Category</label>
    <select id="categorySelect"></select>
    <label for="searchInput">Search</label>
    <input id="searchInput" type="text" placeholder="best_no_turn, endpoint+wrist, sample_15...">
    <div class="hint" id="filterHint"></div>
    <label for="motionSelect">Motion</label>
    <select id="motionSelect"></select>
    <div class="row">
      <button id="play">Play</button>
      <button id="reset">Reset View</button>
    </div>
    <div class="row">
      <a id="downloadMotion" class="button" download>Download NPZ</a>
      <button id="copyPath">Copy Path</button>
    </div>
    <div class="hint" id="downloadHint"></div>
    <div class="row"><span id="frameLabel">Frame 0 / 0</span></div>
    <input id="slider" type="range" min="0" max="0" value="0">
    <div class="panel" id="summary"></div>
    <div class="panel" id="details"></div>
    <div class="panel muted">
      Drag to rotate. Scroll to zoom. Orange is right hand, teal is left hand.
      Wall overlay: red = endpoint constraint, cyan = true wall target, purple = wrist constraint.
    </div>
  </aside>
</main>
<footer id="footer"></footer>
<script>
const DATA = __DATA__;
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const categorySelect = document.getElementById("categorySelect");
const searchInput = document.getElementById("searchInput");
const filterHint = document.getElementById("filterHint");
const select = document.getElementById("motionSelect");
const playBtn = document.getElementById("play");
const resetBtn = document.getElementById("reset");
const downloadLink = document.getElementById("downloadMotion");
const copyPathBtn = document.getElementById("copyPath");
const downloadHint = document.getElementById("downloadHint");
const slider = document.getElementById("slider");
const frameLabel = document.getElementById("frameLabel");
const summary = document.getElementById("summary");
const details = document.getElementById("details");
const footer = document.getElementById("footer");

let motionIndex = 0;
let frame = 0;
let playing = false;
let lastTime = 0;
let yaw = -0.72;
let pitch = -0.18;
let zoom = 1.0;
let dragging = false;
let dragStart = [0, 0];
let angleStart = [yaw, pitch];
let filteredMotionIndices = [];

function current() {
  return DATA.motions[motionIndex];
}

function fmt(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "n/a";
  return Number(v).toFixed(digits);
}

function pointText(p) {
  return `[${p.map(v => Number(v).toFixed(3)).join(", ")}]`;
}

function resetCamera() {
  yaw = -0.72;
  pitch = -0.18;
  zoom = 1.0;
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function rotatePoint(p) {
  const m = current();
  const x = p[0] - m.center[0];
  const y = p[1] - m.center[1];
  const z = p[2] - m.center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y1 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1;
  return [x1, y1, z2];
}

function project(p) {
  const rect = canvas.getBoundingClientRect();
  const r = rotatePoint(p);
  const base = Math.min(rect.width, rect.height) * 0.44 / Math.max(current().radius, 0.2);
  const scale = base * zoom;
  return [rect.width * 0.5 + r[0] * scale, rect.height * 0.56 - r[1] * scale, scale];
}

function drawPoint(p, radius, color, stroke = null) {
  const q = project(p);
  ctx.beginPath();
  ctx.arc(q[0], q[1], radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  if (stroke) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }
}

function drawLabel(p, text, color) {
  const q = project(p);
  ctx.fillStyle = color;
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillText(text, q[0] + 9, q[1] - 9);
}

function drawPolyline(points, color, width) {
  if (!points || points.length === 0) return;
  ctx.beginPath();
  points.forEach((p, i) => {
    const q = project(p);
    if (i === 0) ctx.moveTo(q[0], q[1]);
    else ctx.lineTo(q[0], q[1]);
  });
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawAxes() {
  const m = current();
  const c = m.center;
  const r = Math.max(m.radius * 0.35, 0.08);
  drawPolyline([c, [c[0] + r, c[1], c[2]]], "#ff6b6b", 2);
  drawPolyline([c, [c[0], c[1] + r, c[2]]], "#6bff9a", 2);
  drawPolyline([c, [c[0], c[1], c[2] + r]], "#6ba9ff", 2);
}

function endpointList(m) {
  if (m.overlay.type === "tile") return [
    { index: DATA.leftHandIndex, color: "#70e0d6", label: "left hand" },
    { index: DATA.rightHandIndex, color: "#ffb15f", label: "right hand" }
  ];
  if (m.tags && m.tags.includes("left-hand")) {
    return [{ index: DATA.leftHandIndex, color: "#70e0d6", label: "left hand" }];
  }
  return [{ index: DATA.rightHandIndex, color: "#ffb15f", label: "right hand" }];
}

function drawOverlay(m) {
  const o = m.overlay;
  if (o.type === "hammer") {
    o.targets.forEach((target, i) => {
      const near = Math.abs(frame - target.frame) <= 3;
      drawPoint(target.point, near ? 8 : 6, near ? "#ff4f64" : "#cf4052", near ? "#ffe3e8" : null);
      drawLabel(target.point, `${i + 1}`, "#ffb7c1");
    });
  } else if (o.type === "wall") {
    o.rows.forEach((row, i) => {
      drawPolyline([row.start_point, row.end_point], i % 2 ? "#66d9ff" : "#75f0a7", 3);
    });
    o.targets.forEach((target, i) => {
      const near = Math.abs(frame - target.frame) <= 2;
      if (target.truePoint) {
        drawPoint(target.truePoint, near ? 5.5 : 3.5, near ? "#69e2ff" : "#46b9d9");
        drawPolyline([target.truePoint, target.point], near ? "#ffd27a" : "#8a7045", near ? 2 : 1);
      }
      if (target.wristPoint) {
        drawPolyline([target.point, target.wristPoint], near ? "#d8a8ff" : "#7d5aa0", near ? 2 : 1);
        drawPoint(target.wristPoint, near ? 5.5 : 3.5, near ? "#d8a8ff" : "#9e78ca");
      }
      drawPoint(target.point, near ? 6.5 : 4.2, near ? "#ff4f64" : "#cf4052");
      if (near) drawLabel(target.point, `${target.name || i + 1}`, "#ffb7c1");
    });
  } else if (o.type === "tile") {
    o.placements.forEach((placement, i) => {
      const near = Math.abs(frame - placement.frame) <= 3;
      drawPolyline([placement.left, placement.right], near ? "#ff6b7b" : "#c94758", near ? 4 : 2);
      drawPoint(placement.left, near ? 6 : 4, "#cf4052");
      drawPoint(placement.right, near ? 6 : 4, "#cf4052");
      if (i === 0) drawLabel(placement.left, "tile", "#ffb7c1");
    });
  }
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const m = current();
  const pose = m.frames[frame];
  drawAxes();
  drawOverlay(m);

  endpointList(m).forEach(ep => {
    const trail = m.frames.slice(0, frame + 1).map(f => f[ep.index]);
    drawPolyline(trail, ep.color, 3);
  });

  DATA.edges.forEach(([a, b]) => {
    const pa = project(pose[a]);
    const pb = project(pose[b]);
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.lineWidth = 4;
    const endpointEdge = endpointList(m).some(ep => ep.index === a || ep.index === b);
    ctx.strokeStyle = endpointEdge ? "#ffd28f" : "#86a9ff";
    ctx.stroke();
  });

  pose.forEach((p, i) => {
    const isEndpoint = endpointList(m).some(ep => ep.index === i);
    drawPoint(p, isEndpoint ? 6 : 3.5, isEndpoint ? "#ffd28f" : "#d7e2ff");
  });

  endpointList(m).forEach(ep => drawLabel(pose[ep.index], ep.label, ep.color));
  frameLabel.textContent = `Frame ${frame} / ${m.frames.length - 1}`;
  slider.value = String(frame);
}

function renderSidebar() {
  const m = current();
  slider.max = String(m.frames.length - 1);
  slider.value = String(frame);
  downloadLink.href = m.fileUrl || "#";
  if (m.fileName) downloadLink.setAttribute("download", m.fileName);
  downloadHint.textContent = m.absolutePath || m.path;
  footer.textContent = `${DATA.motions.length} motion files embedded. Source root: ${DATA.sourceRoot}`;
  const metrics = m.metrics || {};
  const chips = [
    m.category || "uncategorized",
    m.overlay.type,
    `${m.frames.length} frames`,
    `${m.jointCount} joints`,
    metrics.passed === true ? "passed" : metrics.passed === false ? "not passed" : "no pass flag"
  ];
  summary.innerHTML = `
    <div><span class="label">Motion:</span> ${m.label}</div>
    <div><span class="label">File:</span> ${m.path}</div>
    <div><span class="label">Category:</span> ${m.category || "n/a"}</div>
    <div><span class="label">Sample:</span> sample_${String(m.sampleIndex).padStart(2, "0")}</div>
    <div><span class="label">Best sample:</span> ${metrics.best_sample ?? "n/a"}</div>
    <div><span class="label">Iteration:</span> ${metrics.iteration ?? "n/a"}</div>
    <div class="chips">${chips.map(c => `<span class="chip">${c}</span>`).join("")}</div>
  `;
  details.innerHTML = detailHtml(m);
}

function detailHtml(m) {
  const o = m.overlay;
  const metrics = m.metrics || {};
  if (o.type === "hammer") {
    return `
      <div><span class="label">Max exact:</span> ${fmt(metrics.best_max_exact_error_m)} m</div>
      <div><span class="label">Max window:</span> ${fmt(metrics.best_max_window_error_m)} m</div>
      <div><span class="label">Max hand step:</span> ${fmt(metrics.best_max_hand_step_m)} m/frame</div>
      <div><span class="label">Max hand accel:</span> ${fmt(metrics.best_max_right_hand_accel_m_per_frame2)} m/frame^2</div>
      <div><span class="label">Max left-arm step:</span> ${fmt(metrics.best_max_left_arm_step_m)} m/frame</div>
      ${o.targets.map(t => `<div class="item"><span class="label">${t.name}</span> frame ${t.frame}<br><span class="muted">${pointText(t.point)}</span><br>error ${fmt(t.error_m)} m</div>`).join("")}
    `;
  }
  if (o.type === "wall") {
    return `
      <div><span class="label">Max keyframe:</span> ${fmt(metrics.best_max_keyframe_error_m)} m</div>
      <div><span class="label">Max plane:</span> ${fmt(metrics.best_max_plane_error_m)} m</div>
      <div><span class="label">Max row-line:</span> ${fmt(metrics.best_max_row_line_error_m)} m</div>
      <div><span class="label">Max wall penetration:</span> ${fmt(metrics.best_max_wall_penetration_m)} m</div>
      <div><span class="label">Max right-arm penetration:</span> ${fmt(metrics.best_max_right_arm_wall_penetration_m)} m</div>
      <div><span class="label">Max hand step:</span> ${fmt(metrics.best_max_hand_step_m)} m/frame</div>
      <div><span class="label">Max all-joint step:</span> ${fmt(metrics.best_max_all_joint_step_m)} m/frame</div>
      ${o.rows.map(r => `<div class="item"><span class="label">row ${r.row}</span><br>${pointText(r.start_point)} -> ${pointText(r.end_point)}</div>`).join("")}
      ${o.targets.map(t => `<div class="item"><span class="label">${t.name || "target"}</span> frame ${t.frame}<br>constraint ${pointText(t.point)}${t.truePoint ? `<br>true wall ${pointText(t.truePoint)}` : ""}${t.wristPoint ? `<br>wrist ${pointText(t.wristPoint)}` : ""}</div>`).join("")}
    `;
  }
  if (o.type === "tile") {
    return `
      <div><span class="label">Max hand:</span> ${fmt(metrics.best_max_hand_error_m)} m</div>
      <div><span class="label">Max center:</span> ${fmt(metrics.best_max_center_error_m)} m</div>
      <div><span class="label">Max plane:</span> ${fmt(metrics.best_max_plane_error_m)} m</div>
      <div><span class="label">Max width:</span> ${fmt(metrics.best_max_width_error_m)} m</div>
      <div><span class="label">Max hold drift:</span> ${fmt(metrics.best_max_hold_center_drift_m)} m</div>
      ${o.placements.map(p => `<div class="item"><span class="label">${p.name}</span> frame ${p.frame}<br>L ${pointText(p.left)}<br>R ${pointText(p.right)}<br>center error ${fmt(p.center_error_m)} m</div>`).join("")}
    `;
  }
  return `<div class="muted">No recognized task metrics found for this motion.</div>`;
}

function selectMotion(index) {
  motionIndex = index;
  frame = 0;
  playing = false;
  playBtn.textContent = "Play";
  resetCamera();
  renderSidebar();
  draw();
}

function searchableText(m) {
  return [
    m.label,
    m.path,
    m.category,
    ...(m.tags || [])
  ].join(" ").toLowerCase();
}

function buildCategories() {
  const counts = new Map();
  DATA.motions.forEach(m => counts.set(m.category || "Uncategorized", (counts.get(m.category || "Uncategorized") || 0) + 1));
  const categories = Array.from(counts.keys()).sort((a, b) => {
    const pa = DATA.categoryOrder?.[a] ?? 999;
    const pb = DATA.categoryOrder?.[b] ?? 999;
    if (pa !== pb) return pa - pb;
    return a.localeCompare(b);
  });
  categorySelect.innerHTML = "";
  const all = document.createElement("option");
  all.value = "__all__";
  all.textContent = `All motions (${DATA.motions.length})`;
  categorySelect.appendChild(all);
  categories.forEach(category => {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = `${category} (${counts.get(category)})`;
    categorySelect.appendChild(option);
  });
}

function refreshMotionList(preferredIndex = null) {
  const category = categorySelect.value;
  const query = searchInput.value.trim().toLowerCase();
  filteredMotionIndices = DATA.motions
    .map((m, i) => [m, i])
    .filter(([m]) => category === "__all__" || (m.category || "Uncategorized") === category)
    .filter(([m]) => !query || searchableText(m).includes(query))
    .sort((a, b) => {
      const pa = a[0].priority ?? 1000;
      const pb = b[0].priority ?? 1000;
      if (pa !== pb) return pa - pb;
      return a[0].label.localeCompare(b[0].label);
    })
    .map(([, i]) => i);
  select.innerHTML = "";
  filteredMotionIndices.forEach(i => {
    const m = DATA.motions[i];
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = m.label;
    select.appendChild(option);
  });
  filterHint.textContent = `${filteredMotionIndices.length} shown. Default: Category "Wall Brush v36 Soft Full-Body Projection Selected", search "DEFAULT".`;
  if (!filteredMotionIndices.length) {
    summary.innerHTML = "No motion files match the current filter.";
    details.innerHTML = "";
    return;
  }
  const next = filteredMotionIndices.includes(preferredIndex) ? preferredIndex : filteredMotionIndices[0];
  select.value = String(next);
  selectMotion(next);
}

select.addEventListener("change", () => selectMotion(Number(select.value)));
categorySelect.addEventListener("change", () => refreshMotionList());
searchInput.addEventListener("input", () => refreshMotionList());
playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
});
resetBtn.addEventListener("click", () => {
  resetCamera();
  draw();
});
async function copyCurrentPath() {
  const m = current();
  const text = m.absolutePath || m.path;
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.focus();
    area.select();
    document.execCommand("copy");
    document.body.removeChild(area);
  }
  copyPathBtn.textContent = "Copied";
  window.setTimeout(() => {
    copyPathBtn.textContent = "Copy Path";
  }, 900);
}
copyPathBtn.addEventListener("click", copyCurrentPath);
slider.addEventListener("input", () => {
  frame = Number(slider.value);
  draw();
});
canvas.addEventListener("mousedown", e => {
  dragging = true;
  dragStart = [e.clientX, e.clientY];
  angleStart = [yaw, pitch];
});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  yaw = angleStart[0] + (e.clientX - dragStart[0]) * 0.008;
  pitch = Math.max(-1.25, Math.min(1.25, angleStart[1] + (e.clientY - dragStart[1]) * 0.008));
  draw();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  zoom = Math.max(0.35, Math.min(4.0, zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
  draw();
}, { passive: false });
window.addEventListener("resize", resize);

function tick(t) {
  if (playing && t - lastTime > 1000 / 30) {
    const m = current();
    frame = (frame + 1) % m.frames.length;
    lastTime = t;
    draw();
  }
  requestAnimationFrame(tick);
}

if (DATA.motions.length === 0) {
  summary.innerHTML = "No motion files were embedded.";
} else {
  buildCategories();
  const preferredIndex = DATA.motions.findIndex(m => m.preferred);
  if (preferredIndex >= 0) {
    categorySelect.value = DATA.motions[preferredIndex].category || "__all__";
  }
  refreshMotionList(preferredIndex >= 0 ? preferredIndex : null);
}
resize();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def sample_index_from_name(path: Path) -> int:
    match = re.search(r"sample_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return 0


RANKING_DIR_NAMES = {
    "top_10",
    "top_10_filtered",
    "best_no_turn_top10",
    "best_brush_motion_top10",
    "best_flat_brush_motion_top10",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_metadata_path(path: Path) -> Path | None:
    direct_dirs = [path.parent]
    if path.parent.parent != path.parent:
        direct_dirs.append(path.parent.parent)
    for directory in direct_dirs:
        metrics_path = directory / "metrics.json"
        if metrics_path.exists():
            return metrics_path

    ranking_roots = [parent.parent for parent in path.parents if parent.name in RANKING_DIR_NAMES]
    for root in ranking_roots:
        metrics_paths = sorted(root.rglob("metrics.json"))
        if len(metrics_paths) == 1:
            return metrics_paths[0]
    return None


def merge_prompt_metadata(metrics_path: Path) -> dict:
    metrics = load_json(metrics_path)
    prompts_path = metrics_path.parent / "prompts.json"
    if not prompts_path.exists():
        return metrics
    prompts = load_json(prompts_path)
    return {**prompts, **metrics}


def load_metrics(path: Path) -> dict:
    metrics_path = nearest_metadata_path(path)
    if metrics_path is None:
        return {}
    metrics = merge_prompt_metadata(metrics_path)

    eval_candidates = [path.with_name(f"{path.stem}_eval.json")]
    if path.stem.endswith("_motion"):
        eval_candidates.append(path.with_name(f"{path.stem[:-7]}_eval.json"))

    eval_path = next((candidate for candidate in eval_candidates if candidate.exists()), None)
    if eval_path is None:
        return metrics

    post_eval = json.loads(eval_path.read_text(encoding="utf-8"))
    metrics = dict(metrics)
    metrics["postprocess_eval"] = post_eval
    metrics["passed"] = post_eval.get("passed", metrics.get("passed"))
    metrics["best_max_exact_error_m"] = post_eval.get(
        "max_exact_error_m",
        metrics.get("best_max_exact_error_m"),
    )
    metrics["best_max_window_error_m"] = post_eval.get(
        "max_window_error_m",
        metrics.get("best_max_window_error_m"),
    )
    metrics["best_max_root_horizontal_drift_m"] = post_eval.get(
        "max_root_horizontal_drift_m",
        metrics.get("best_max_root_horizontal_drift_m"),
    )
    metrics["best_max_root_vertical_drift_m"] = post_eval.get(
        "max_root_vertical_drift_m",
        metrics.get("best_max_root_vertical_drift_m"),
    )
    metrics["best_max_hand_step_m"] = post_eval.get(
        "max_right_hand_step_m",
        metrics.get("best_max_hand_step_m"),
    )
    metrics["best_max_keyframe_error_m"] = post_eval.get(
        "max_keyframe_error_m",
        metrics.get("best_max_keyframe_error_m"),
    )
    metrics["best_max_plane_error_m"] = post_eval.get(
        "max_plane_error_m",
        metrics.get("best_max_plane_error_m"),
    )
    metrics["best_max_row_line_error_m"] = post_eval.get(
        "max_row_line_error_m",
        metrics.get("best_max_row_line_error_m"),
    )
    metrics["best_max_wall_penetration_m"] = post_eval.get(
        "max_wall_penetration_m",
        metrics.get("best_max_wall_penetration_m"),
    )
    metrics["best_max_right_arm_wall_penetration_m"] = post_eval.get(
        "max_right_arm_wall_penetration_m",
        metrics.get("best_max_right_arm_wall_penetration_m"),
    )
    metrics["best_max_path_length_ratio"] = post_eval.get(
        "max_path_length_ratio",
        metrics.get("best_max_path_length_ratio"),
    )
    metrics["best_max_all_joint_step_m"] = post_eval.get(
        "max_all_joint_step_m",
        metrics.get("best_max_all_joint_step_m"),
    )
    metrics["best_max_left_arm_step_m"] = post_eval.get(
        "max_left_arm_step_m",
        metrics.get("best_max_left_arm_step_m"),
    )
    metrics["best_max_left_hand_step_m"] = post_eval.get(
        "max_left_hand_step_m",
        metrics.get("best_max_left_hand_step_m"),
    )
    metrics["best_max_right_hand_accel_m_per_frame2"] = post_eval.get(
        "max_right_hand_accel_m_per_frame2",
        metrics.get("best_max_right_hand_accel_m_per_frame2"),
    )
    if post_eval.get("hits"):
        metrics["results"] = [{"sample": sample_index_from_name(path), "hits": post_eval["hits"]}]
    return metrics


def find_sample_result(metrics: dict, sample_index: int) -> dict:
    for result in metrics.get("results", []):
        if int(result.get("sample", -1)) == sample_index:
            return result
    best = metrics.get("best_sample")
    for result in metrics.get("results", []):
        if int(result.get("sample", -1)) == int(best or -1):
            return result
    return {}


def build_overlay(metrics: dict, sample_result: dict) -> dict:
    if "hit_frames" in metrics:
        hits_by_name = {hit["name"]: hit for hit in sample_result.get("hits", [])}
        targets = []
        if metrics.get("hit_specs"):
            hit_items = sorted(metrics.get("hit_specs", []), key=lambda item: int(item.get("frame", 0)))
        else:
            hit_items = [
                {"name": f"hit_{i}", "point_name": f"point_{i}", "frame": metrics.get("hit_frames", {}).get(f"hit_{i}", 0)}
                for i in range(1, 4)
            ]
        for item in hit_items:
            name = item.get("name", "")
            point_name = item.get("point_name") or metrics.get("hit_to_point", {}).get(name, "")
            hit = hits_by_name.get(name, {})
            targets.append(
                {
                    "name": name,
                    "point": metrics.get("target_points", {}).get(point_name, [0, 0, 0]),
                    "frame": int(metrics.get("hit_frames", {}).get(name, item.get("frame", 0))),
                    "error_m": hit.get("exact_error_m"),
                }
            )
        return {"type": "hammer", "targets": targets}

    if "row_specs" in metrics:
        targets = [
            {
                "name": target.get("name", ""),
                "phase": target.get("phase", ""),
                "frame": int(target.get("frame", 0)),
                "point": target.get("point", [0, 0, 0]),
                "truePoint": target.get("true_point"),
                "wristPoint": target.get("wrist_point") if target.get("use_wrist") else None,
                "useWrist": bool(target.get("use_wrist")),
                "row": target.get("row"),
                "col": target.get("col"),
            }
            for target in metrics.get("target_points", [])
        ]
        return {"type": "wall", "targets": targets, "rows": metrics.get("row_specs", [])}

    if "tile_spec" in metrics:
        placements_by_name = {
            placement.get("name"): placement
            for placement in sample_result.get("placements", [])
        }
        placements = []
        for target in metrics.get("target_points", []):
            if not target.get("validate"):
                continue
            name = target.get("name", "")
            pair = target.get("pair", {})
            actual = placements_by_name.get(name, {})
            placements.append(
                {
                    "name": name,
                    "frame": int(target.get("frame", 0)),
                    "left": pair.get("left", [0, 0, 0]),
                    "right": pair.get("right", [0, 0, 0]),
                    "center_error_m": actual.get("center_error_m"),
                }
            )
        return {"type": "tile", "placements": placements}

    return {"type": "unknown"}


def overlay_points(overlay: dict) -> list[list[float]]:
    points = []
    if overlay["type"] == "hammer":
        points.extend(target["point"] for target in overlay["targets"])
    elif overlay["type"] == "wall":
        points.extend(target["point"] for target in overlay["targets"])
        points.extend(target["truePoint"] for target in overlay["targets"] if target.get("truePoint"))
        points.extend(target["wristPoint"] for target in overlay["targets"] if target.get("wristPoint"))
        for row in overlay["rows"]:
            points.append(row["start_point"])
            points.append(row["end_point"])
    elif overlay["type"] == "tile":
        for placement in overlay["placements"]:
            points.append(placement["left"])
            points.append(placement["right"])
    return points


def motion_bounds(frames: np.ndarray, overlay: dict) -> tuple[list[float], float]:
    flat = frames.reshape(-1, 3)
    extras = np.asarray(overlay_points(overlay), dtype=float)
    if extras.size:
        flat = np.concatenate([flat, extras.reshape(-1, 3)], axis=0)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = ((mins + maxs) * 0.5).tolist()
    radius = float(np.linalg.norm(maxs - mins) * 0.5)
    return center, max(radius, 0.2)


def summarize_metrics(metrics: dict) -> dict:
    keys = [
        "passed",
        "best_sample",
        "iteration",
        "best_max_exact_error_m",
        "best_max_window_error_m",
        "best_max_keyframe_error_m",
        "best_max_plane_error_m",
        "best_max_row_line_error_m",
        "best_max_wall_penetration_m",
        "best_max_right_arm_wall_penetration_m",
        "best_max_hand_step_m",
        "best_max_right_hand_accel_m_per_frame2",
        "best_max_left_arm_step_m",
        "best_max_left_hand_step_m",
        "best_max_all_joint_step_m",
        "best_max_path_length_ratio",
        "best_max_hand_error_m",
        "best_max_center_error_m",
        "best_max_width_error_m",
        "best_max_hold_center_drift_m",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def decode_tag_number(value: str) -> str:
    return value.replace("m", "-").replace("p", ".")


def prompt_ablation_variant_name(motion_path: Path) -> str | None:
    parts = list(motion_path.parts)
    for index, part in enumerate(parts[:-1]):
        if part.startswith("wall_brush_prompt_ablation") and index + 1 < len(parts):
            return parts[index + 1]
    return None


def seed_context_name(motion_path: Path) -> str | None:
    for part in motion_path.parts:
        if re.fullmatch(r"seed_\d+", part):
            return part
    return None


def label_from_run_name(run_name: str, sample_name: str, prefix: str, context: str | None = None) -> str | None:
    variant = None
    for candidate in ("current_right_hand", "endpoint_wrist", "endpoint_only"):
        if run_name.startswith(candidate):
            variant = candidate
            break
    cfg_index_match = re.search(r"_cfg(\d+)_", run_name)
    cfg_match = re.search(r"_t([0-9mp]+)_c([0-9mp]+)_steps(\d+)", run_name)
    if variant and cfg_match:
        cfg_index = cfg_index_match.group(1) if cfg_index_match else "?"
        cfg_t = decode_tag_number(cfg_match.group(1))
        cfg_c = decode_tag_number(cfg_match.group(2))
        steps = cfg_match.group(3)
        context_text = f"{context} | " if context else ""
        return f"{prefix} {context_text}{variant} | cfg#{cfg_index} C={cfg_c} | T={cfg_t} | {steps} steps | {sample_name}"
    return None


def compact_run_label(motion_path: Path) -> str | None:
    context_parts = [part for part in (prompt_ablation_variant_name(motion_path), seed_context_name(motion_path)) if part]
    context = " ".join(context_parts) if context_parts else None
    if motion_path.parent.name == "transition_filtered" and motion_path.stem.endswith("_transition_filtered"):
        run_name = motion_path.parent.parent.name
        sample_name = motion_path.stem.removesuffix("_transition_filtered")
        parsed = label_from_run_name(run_name, sample_name, "FILTERED", context)
        if parsed:
            return parsed
        if run_name:
            return f"FILTERED {run_name} / {sample_name}"
        return None

    if not motion_path.parent.name.startswith("sample_"):
        return None
    run_name = motion_path.parent.parent.name
    parsed = label_from_run_name(run_name, motion_path.parent.name, "RAW", context)
    if parsed:
        return parsed
    if run_name:
        return f"{run_name} / {motion_path.parent.name}"
    return None


def classify_motion(motion_path: Path, overlay: dict) -> dict:
    parts = set(motion_path.parts)
    path_text = str(motion_path).replace("\\", "/")
    stem = motion_path.stem
    parent = motion_path.parent.name
    run_text = f"{parent} {stem} {path_text}".lower()
    tags = []
    priority = 500
    preferred = False

    if "wall_brush_generalization_left_mirror" in parts:
        category = "Wall Brush Generalization - Left-Hand Mirror Runs"
        priority = -250
        tags.extend(["wall-brush", "left-hand", "mirror", "generalization", "soft-full-body-projection"])
        if "gallery_selected" in parts:
            category = "Wall Brush Generalization - Left-Hand Mirror Selected"
            priority = -260
            preferred = True
            tags.extend(["selected", "easy-find", "recommended"])
        for variant_tag in (
            "baseline_center",
            "left_lower_close",
            "right_upper_mid",
            "wide_center_mid",
            "small_right_close",
        ):
            if variant_tag in run_text:
                tags.append(variant_tag.replace("_", "-"))
    elif "wall_brush_left_hand_from_scratch" in parts:
        category = "Wall Brush Left-Hand From Scratch Runs"
        priority = -255
        tags.extend(["wall-brush", "left-hand", "from-scratch", "soft-full-body-projection"])
        if "gallery_selected" in parts:
            category = "Wall Brush Left-Hand From Scratch Selected"
            priority = -265
            preferred = True
            tags.extend(["selected", "easy-find", "recommended"])
    elif "wall_brush_generalization" in parts:
        category = "Wall Brush Generalization Runs"
        priority = -230
        tags.extend(["wall-brush", "generalization", "from-scratch", "soft-full-body-projection"])
        if "gallery_selected" in parts:
            category = "Wall Brush Generalization Selected"
            priority = -240
            preferred = True
            tags.extend(["selected", "easy-find", "recommended"])
        if "baseline_center" in run_text:
            tags.append("baseline-center")
        if "left_lower_close" in run_text:
            tags.append("left-lower-close")
        if "right_upper_mid" in run_text:
            tags.append("right-upper-mid")
        if "wide_center_mid" in run_text:
            tags.append("wide-center-mid")
        if "small_right_close" in run_text:
            tags.append("small-right-close")
    elif "wall_brush_g1_v36_soft_full_body_projection" in parts:
        tags.extend(["v36", "soft-full-body-projection", "flat-brush", "projection", "not-no-ik"])
        if "gallery_selected" in parts:
            category = "Wall Brush v36 Soft Full-Body Projection Selected"
            priority = -170
            tags.extend(["selected", "easy-find"])
            if "default" in run_text or ("sfbp_cont" in run_text and "lam5" in run_text):
                preferred = True
                priority = -210
                tags.extend(["default", "recommended", "current-default", "sfbp-cont", "temporal-continuity-filter"])
            if "best_natural" in run_text or "seed_c_final" in run_text:
                preferred = True
                tags.append("recommended")
        elif "best_wall_safe_current" in parts:
            category = "Wall Brush v36 - Best Wall Safe"
            priority = -160
            tags.append("best-wall-safe")
        elif "best_flat_brush_current" in parts:
            category = "Wall Brush v36 - Best Flat Brush"
            priority = -155
            tags.append("best-flat-brush")
        elif "best_natural_flat_brush_current" in parts:
            category = "Wall Brush v36 - Best Natural Flat Brush"
            priority = -150
            tags.append("best-natural-flat-brush")
        elif "seeds" in parts:
            category = "Wall Brush v36 - Seed Comparison"
            priority = -145
            tags.append("seed-comparison")
        elif "best_wall_safe" in parts or "best_flat_brush" in parts or "best_natural_flat_brush" in parts:
            category = "Wall Brush v36 Archive Rankings"
            priority = -20
            tags.append("archive-ranking")
        else:
            category = "Wall Brush v36 Soft Projection Runs"
            priority = -18
        if "seed_a" in run_text:
            tags.append("seed-a")
        if "seed_b" in run_text:
            tags.append("seed-b")
        if "seed_c" in run_text:
            tags.append("seed-c")
        if "stage1" in run_text:
            tags.append("stage1-wall-safe")
        if "stage2" in run_text:
            tags.append("stage2-flat")
        if "final" in run_text:
            tags.append("final")
        if "edge_plus" in run_text:
            tags.extend(["edge-plus-filter", "smoothed-boundaries"])
        elif "edge_filtered" in run_text or "edge-filtered" in run_text:
            tags.extend(["edge-filter", "smoothed-boundaries"])
        if "continuity" in run_text:
            tags.extend(["temporal-continuity-filter", "smoothest"])
            if "best_smooth" in run_text:
                preferred = True
    elif "wall_brush_g1_v35_dno_lite" in parts:
        tags.extend(["v35", "dno-lite", "noise-search", "flat-brush", "multi-prompt"])
        if "gallery_selected" in parts:
            category = "Wall Brush v35 DNO-lite Selected"
            priority = -120
            tags.extend(["selected", "easy-find"])
            if "default" in run_text or "seed_c_v35_best_natural_flat" in run_text:
                tags.extend(["former-default", "recommended", "seed-c"])
            if "best_natural_flat" in run_text or "best_flat" in run_text:
                preferred = True
                tags.append("recommended")
        elif "best_natural_flat_brush" in parts:
            category = "Wall Brush v35 - Best Natural Flat Brush"
            priority = -110
            tags.append("best-natural-flat-brush")
        elif "best_flat_brush" in parts:
            category = "Wall Brush v35 - Best Flat Brush"
            priority = -105
            tags.append("best-flat-brush")
        elif "best_no_penetration_flat_brush" in parts:
            category = "Wall Brush v35 - Best No Penetration"
            priority = -100
            tags.append("best-no-penetration")
        elif "seeds" in parts:
            category = "Wall Brush v35 - Seed Comparison"
            priority = -95
            tags.append("seed-comparison")
        else:
            category = "Wall Brush v35 DNO-lite Runs"
            priority = -25
        if "seed_a" in run_text:
            tags.append("seed-a")
        if "seed_b" in run_text:
            tags.append("seed-b")
        if "soft_full_body" in run_text:
            tags.append("soft-full-body-projection")
    elif "wall_brush_g1_mp_constraint_v34_flat_brush" in parts:
        tags.extend(["v34", "flat-brush", "no-ik", "multi-prompt", "native-constraint"])
        if "gallery_selected" in parts:
            category = "Wall Brush v34 Flat Brush Selected"
            priority = -80
            tags.extend(["selected", "easy-find"])
            if "best_flat" in run_text or "v34_best_flat" in run_text:
                preferred = True
                priority = -90
                tags.extend(["recommended", "best_flat_brush"])
        elif "best_flat_brush_motion_top10" in parts:
            category = "Wall Brush v34 - Best Flat Brush Motion"
            priority = -75
            tags.extend(["best-flat-brush-motion", "reranked"])
        elif "best_brush_motion_top10" in parts:
            category = "Wall Brush v34 - Best Brush Motion"
            priority = -70
            tags.extend(["best-brush-motion", "reranked"])
        elif "best_no_turn_top10" in parts:
            category = "Wall Brush v34 - Best No Turn"
            priority = -65
            tags.extend(["best-no-turn", "reranked"])
        else:
            category = "Wall Brush v34 Flat Brush Runs"
            priority = -15
        if "level_full_width" in run_text:
            tags.append("level-full-width-prompt")
        for preset in ("flat_a", "flat_b", "flat_c", "flat_d"):
            if preset in run_text or f"pre{preset}" in run_text:
                tags.append(preset.replace("_", "-"))
    elif "wall_brush_g1_mp_constraint_v33_brush_motion" in parts:
        tags.extend(["v33", "no-ik", "multi-prompt", "native-constraint", "brush-motion"])
        if "gallery_selected" in parts:
            category = "Wall Brush v33 Brush Motion Selected"
            priority = -45
            tags.extend(["selected", "easy-find"])
            if "best_brush_motion" in run_text:
                preferred = True
                priority = -60
                tags.extend(["recommended", "best_brush_motion"])
        elif "best_no_turn_top10" in parts:
            category = "Wall Brush v33 Brush Motion - Best No Turn"
            priority = -35
            tags.extend(["best-no-turn", "reranked"])
        elif "best_brush_motion_top10" in parts:
            category = "Wall Brush v33 Brush Motion - Best Brush Motion"
            priority = -30
            tags.extend(["best-brush-motion", "reranked"])
        else:
            category = "Wall Brush v33 Brush Motion Runs"
            priority = 25
        if "full_width" in run_text:
            tags.append("full-width-prompt")
        for preset in ("v33_a", "v33_b", "v33_c", "v33_d", "v33_e", "v33_f", "v33_g", "v33_h"):
            if preset in run_text:
                tags.append(preset.replace("_", "-"))
        if "v33_e" in run_text or "v33_f" in run_text or "v33_g" in run_text or "v33_h" in run_text:
            tags.append("safe-z")
    elif "wall_brush_g1_mp_constraint_v33_rerank" in parts:
        tags.extend(["v33", "rerank", "no-ik", "multi-prompt", "native-constraint"])
        if "best_no_turn_top10" in parts:
            category = "Wall Brush v33 Rerank - Best No Turn"
            priority = -25
            tags.append("best-no-turn")
        elif "best_brush_motion_top10" in parts:
            category = "Wall Brush v33 Rerank - Best Brush Motion"
            priority = -20
            tags.append("best-brush-motion")
        else:
            category = "Wall Brush v33 Rerank"
            priority = 20
    elif "wall_brush_g1_mp_constraint_v32_210_delayed_preemphasis" in parts:
        tags.extend(["v32", "no-ik", "multi-prompt", "native-constraint", "delayed", "preemphasis"])
        if "gallery_selected" in parts:
            category = "Wall Brush v32 Selected"
            priority = 0
            tags.extend(["selected", "easy-find"])
            if "best_line_mean" in run_text and "transition_filtered" in run_text:
                preferred = True
                priority = -20
                tags.extend(["recommended", "best_line_mean", "v32_preferred"])
        elif "top_10_filtered" in parts:
            category = "Wall Brush v32 Top 10 Filtered"
            priority = 10
        elif "top_10" in parts:
            category = "Wall Brush v32 Top 10 Raw"
            priority = 15
        else:
            category = "Wall Brush v32 Delayed Preemphasis"
            priority = 45
        if "preb" in run_text:
            tags.append("preemphasis-b")
        if "prec" in run_text:
            tags.append("preemphasis-c")
    elif "gallery_selected" in parts:
        category = "Wall Brush v31 Selected"
        priority = 0
        tags.extend(["selected", "v31", "easy-find"])
        if "best_no_turn" in run_text and "transition_filtered" in run_text and "without_sparse_heading" in run_text:
            preferred = True
            priority = -10
            tags.extend(["recommended", "best_no_turn", "current_preferred"])
    elif "wall_brush_g1_mp_constraint_native_v31" in parts:
        tags.extend(["v31", "no-ik", "multi-prompt", "native-constraint"])
        if "top_10_filtered" in parts:
            category = "Wall Brush v31 Top 10 Filtered"
            priority = 20
        elif "top_10" in parts:
            category = "Wall Brush v31 Top 10 Raw"
            priority = 25
        elif "current_right_hand" in run_text:
            category = "Wall Brush v31 Rotation Constraint Baseline"
            tags.append("rotation-baseline")
            priority = 60
        elif "endpoint_only" in run_text:
            category = "Wall Brush v31 Endpoint Only"
            tags.append("endpoint-only")
            priority = 55
        elif "return_only" in run_text:
            category = "Wall Brush v31 Endpoint+Wrist Return-Only Heading"
            tags.extend(["endpoint+wrist", "return-only-heading"])
            priority = 35
        elif "sparse" in run_text:
            category = "Wall Brush v31 Endpoint+Wrist Sparse Heading"
            tags.extend(["endpoint+wrist", "sparse-heading"])
            priority = 40
        else:
            category = "Wall Brush v31 Endpoint+Wrist No Heading"
            tags.extend(["endpoint+wrist", "no-heading"])
            priority = 30
    elif "wall_brush_prompt_ablation" in run_text:
        tags.extend(["wall-brush", "prompt-ablation", "raw-kimodo"])
        if "transition_filtered" in run_text:
            category = "Wall Brush Prompt Ablation - Transition Filtered"
            priority = -63
        else:
            category = "Wall Brush Prompt Ablation - Raw"
            priority = -62
    elif "wall_brush_constraint_strictness" in run_text:
        tags.extend(["wall-brush", "constraint-strictness", "raw-kimodo"])
        if "transition_filtered" in run_text:
            category = "Wall Brush Constraint Strictness - Transition Filtered"
            priority = -55
        else:
            category = "Wall Brush Constraint Strictness - Raw"
            priority = -50
    elif "wall_brush" in run_text:
        category = "Wall Brush Older Runs"
        tags.append("wall-brush")
        priority = 120
    elif overlay["type"] == "hammer" or "hammer" in run_text:
        category = "Hammer / Nail"
        tags.append("hammer")
        priority = 150
    elif overlay["type"] == "tile" or "tile" in run_text:
        category = "Tile"
        tags.append("tile")
        priority = 160
    else:
        category = "Other Motions"
        priority = 900

    if "transition_filtered" in run_text:
        tags.append("transition-filtered")
    if "raw" in run_text:
        tags.append("raw")
    if "line_ik" in run_text:
        tags.append("line-ik")
    if "no_ik" in run_text or "no-ik" in run_text:
        tags.append("no-ik")

    if "gallery_selected" in parts:
        label = stem
    else:
        label = compact_run_label(motion_path) or f"{parent} / {stem}"

    return {
        "category": category,
        "tags": sorted(set(tags)),
        "priority": priority,
        "preferred": preferred,
        "label": label,
    }


def is_transition_filtered_motion(motion_path: Path) -> bool:
    return motion_path.parent.name == "transition_filtered" and motion_path.name.endswith("_transition_filtered.npz")


def should_include_motion(
    motion_path: Path,
    include_all: bool,
    include_rankings: bool,
    direct_generated_only: bool,
) -> bool:
    if direct_generated_only:
        return motion_path.name.endswith("motion.npz") or is_transition_filtered_motion(motion_path)
    if include_all:
        extra_npz_dirs = {"gallery_selected", "top_10", "top_10_filtered"}
        return (
            motion_path.name.endswith("motion.npz")
            or is_transition_filtered_motion(motion_path)
            or any(parent.name in extra_npz_dirs for parent in motion_path.parents)
        )
    selected_dirs = {
        "gallery_selected",
        "seeds",
        "best_wall_safe_current",
        "best_flat_brush_current",
        "best_natural_flat_brush_current",
    }
    current_dirs = {"best_wall_safe_current", "best_flat_brush_current", "best_natural_flat_brush_current"}
    if any(parent.name in current_dirs for parent in motion_path.parents):
        # Older generated ranking names were very long and accumulated when Windows
        # could not remove locked files. Keep the refreshed short-name copies only.
        if motion_path.name.startswith("new_") or len(motion_path.name) > 180:
            return False
    if include_rankings:
        selected_dirs.update(
            {
                "top_10",
                "top_10_filtered",
                "best_no_turn_top10",
                "best_brush_motion_top10",
                "best_flat_brush_motion_top10",
            }
        )
    return any(parent.name in selected_dirs for parent in motion_path.parents)


def build_payload(
    logs_dir: Path,
    include_all: bool = False,
    include_rankings: bool = False,
    direct_generated_only: bool = False,
    path_contains: list[str] | None = None,
) -> dict:
    joints = [name for name, _ in G1_BONES_WITH_PARENTS]
    index = {name: i for i, name in enumerate(joints)}
    edges = [
        [index[parent], index[name]]
        for name, parent in G1_BONES_WITH_PARENTS
        if parent is not None
    ]
    motions = []
    for motion_path in sorted(logs_dir.rglob("*.npz")):
        if path_contains and not any(token in str(motion_path) for token in path_contains):
            continue
        if not should_include_motion(motion_path, include_all, include_rankings, direct_generated_only):
            continue
        data = np.load(motion_path, allow_pickle=True)
        if "posed_joints" not in data:
            continue
        posed = data["posed_joints"].astype(float)
        if posed.ndim != 3 or posed.shape[2] != 3:
            continue
        sample_index = sample_index_from_name(motion_path)
        metrics = load_metrics(motion_path)
        sample_result = find_sample_result(metrics, sample_index)
        overlay = build_overlay(metrics, sample_result)
        center, radius = motion_bounds(posed, overlay)
        rel_path = motion_path.relative_to(logs_dir.parent)
        classification = classify_motion(motion_path, overlay)
        motions.append(
            {
                "label": classification["label"],
                "path": str(rel_path).replace("\\", "/"),
                "absolutePath": str(motion_path.resolve()),
                "fileUrl": motion_path.resolve().as_uri(),
                "fileName": motion_path.name,
                "category": classification["category"],
                "tags": classification["tags"],
                "priority": classification["priority"],
                "preferred": classification["preferred"],
                "sampleIndex": sample_index,
                "jointCount": int(posed.shape[1]),
                "frames": posed.round(5).tolist(),
                "center": center,
                "radius": radius,
                "overlay": overlay,
                "metrics": summarize_metrics(metrics),
            }
        )
    motions.sort(key=lambda motion: (motion["priority"], motion["category"], motion["label"]))
    return {
        "sourceRoot": str(logs_dir.resolve()),
        "joints": joints,
        "edges": edges,
        "leftHandIndex": index["left_hand_roll_skel"],
        "rightHandIndex": index["right_hand_roll_skel"],
        "categoryOrder": {
            "Wall Brush Left-Hand From Scratch Selected": -265,
            "Wall Brush Left-Hand From Scratch Runs": -255,
            "Wall Brush Generalization - Left-Hand Mirror Selected": -260,
            "Wall Brush Generalization - Left-Hand Mirror Runs": -250,
            "Wall Brush Generalization Selected": -240,
            "Wall Brush Generalization Runs": -230,
            "Wall Brush v36 Soft Full-Body Projection Selected": -210,
            "Wall Brush v36 - Best Wall Safe": -160,
            "Wall Brush v36 - Best Flat Brush": -155,
            "Wall Brush v36 - Best Natural Flat Brush": -150,
            "Wall Brush v36 - Seed Comparison": -145,
            "Wall Brush v36 Archive Rankings": -20,
            "Wall Brush v36 Soft Projection Runs": -18,
            "Wall Brush v35 DNO-lite Selected": -120,
            "Wall Brush v35 - Best Natural Flat Brush": -110,
            "Wall Brush v35 - Best Flat Brush": -105,
            "Wall Brush v35 - Best No Penetration": -100,
            "Wall Brush v35 - Seed Comparison": -95,
            "Wall Brush v35 DNO-lite Runs": -25,
            "Wall Brush v34 Flat Brush Selected": -80,
            "Wall Brush v34 - Best Flat Brush Motion": -75,
            "Wall Brush v34 - Best Brush Motion": -70,
            "Wall Brush v34 - Best No Turn": -65,
            "Wall Brush v34 Flat Brush Runs": -15,
            "Wall Brush v33 Brush Motion Selected": -60,
            "Wall Brush v33 Brush Motion - Best No Turn": -50,
            "Wall Brush v33 Brush Motion - Best Brush Motion": -45,
            "Wall Brush v33 Rerank - Best No Turn": -40,
            "Wall Brush v33 Rerank - Best Brush Motion": -35,
            "Wall Brush v33 Rerank": -30,
            "Wall Brush v32 Selected": -20,
            "Wall Brush v32 Top 10 Filtered": -10,
            "Wall Brush v32 Top 10 Raw": -5,
            "Wall Brush v32 Delayed Preemphasis": 5,
            "Wall Brush v33 Brush Motion Runs": 6,
            "Wall Brush v31 Selected": 0,
            "Wall Brush v31 Top 10 Filtered": 10,
            "Wall Brush v31 Top 10 Raw": 20,
            "Wall Brush v31 Endpoint+Wrist No Heading": 30,
            "Wall Brush v31 Endpoint+Wrist Return-Only Heading": 40,
            "Wall Brush v31 Endpoint+Wrist Sparse Heading": 50,
            "Wall Brush v31 Endpoint Only": 60,
            "Wall Brush v31 Rotation Constraint Baseline": 70,
            "Wall Brush Prompt Ablation - Transition Filtered": -63,
            "Wall Brush Prompt Ablation - Raw": -62,
            "Wall Brush Constraint Strictness - Transition Filtered": -55,
            "Wall Brush Constraint Strictness - Raw": -50,
            "Wall Brush Older Runs": 100,
            "Hammer / Nail": 150,
            "Tile": 160,
            "Other Motions": 900,
        },
        "motions": motions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--output", default="logs/kimodo_motion_gallery.html")
    parser.add_argument(
        "--include_all",
        action="store_true",
        help="Embed every motion.npz under logs. This can create a very large, slow HTML file.",
    )
    parser.add_argument(
        "--include_rankings",
        action="store_true",
        help="Also embed historical top10/ranking folders. Larger than the default selected-only gallery.",
    )
    parser.add_argument(
        "--direct_generated_only",
        action="store_true",
        help="Embed only direct sample motion.npz files and their transition_filtered counterparts, excluding ranking copies.",
    )
    parser.add_argument(
        "--path_contains",
        action="append",
        default=[],
        help="Only include motions whose path contains this substring. Can be repeated.",
    )
    parser.add_argument(
        "--category_prefix",
        default="",
        help="Keep only motions whose computed category starts with this prefix, e.g. 'Wall Brush v36'.",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        logs_dir,
        include_all=args.include_all,
        include_rankings=args.include_rankings,
        direct_generated_only=args.direct_generated_only,
        path_contains=args.path_contains,
    )
    if args.category_prefix:
        payload = dict(payload)
        payload["motions"] = [
            motion
            for motion in payload["motions"]
            if str(motion.get("category", "")).startswith(args.category_prefix)
        ]
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    output.write_text(html, encoding="utf-8")
    print(f"motions={len(payload['motions'])}")
    print(output.resolve())


if __name__ == "__main__":
    main()
