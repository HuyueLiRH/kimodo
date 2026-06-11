# One-Row Wall Brush Direct-From-Raw Pipeline

Date accepted: 2026-05-26

This note records the current accepted prior-generation and postprocess route for one-row G1-RP wall-brushing priors.

## Status

- Task: one-row wall brushing
- Model: `kimodo-g1-rp`
- Raw generation path: KIMODO demo/web-equivalent generation
- Raw prompt: `A person stands still and slides the right palm flat on the outside surface of a wall from left to right.`
- Constraint frames: `[36, 51, 66]`
- Final variant: `right_arm_g1_direct_from_raw_line_default_return`
- Review status: stable-range 108 batch generated and accepted as the current default experiment route

This replaces the previous staged FABRIK/lightlock/intermediate G1 line-refit/default-return route as the default cleanup method. The previous route remains useful only as historical comparison.

## Executable Assets

- Batch target file: `examples/wall_brush/task_specs/one_row_wall_brush_108_direct_from_raw_batch.json`
- Postprocess treatment: `examples/wall_brush/postprocess_treatments/direct_from_raw_g1_hinge_default_return.json`
- Remote batch runner: `examples/wall_brush/scripts/remote_wall_brush_direct_from_raw_batch_runner.py`
- Local postprocess runner: `examples/wall_brush/scripts/direct_wall_brush_g1_from_raw.py`
- G1 hinge refit helper: `examples/wall_brush/scripts/refit_g1_right_arm_hinge_to_target.py`
- Web-equivalent raw generator: `kimodo/scripts/web_equivalent_generate.py`
- Local review viewer: `kimodo/scripts/prior_viewer.py`

## Raw Generation

Raw motions are generated from the same style of task spec used by the demo UI. The batch runner writes one task spec per candidate and calls:

```bash
python -m kimodo.scripts.web_equivalent_generate \
  --task raw/<candidate>/task_spec.json \
  --output _generation/<candidate>/raw_generation \
  --device cuda:0 \
  --use-demo-embedding-cache \
  --embedding-cache-root /root/.cache/kimodo_demo/embeddings \
  --embedding-cache-model-name kimodo-g1-rp \
  --embedding-cache-model-name Kimodo-G1-RP-v1 \
  --no-csv
```

The generated `sample_00/motion.npz` is copied into:

```text
raw/<candidate>/motion.npz
```

Raw generation stays unpostprocessed:

```text
post_processing: false
num_samples: 1
num_denoising_steps: 200
cfg_type: separated
cfg_weight: [2.4, 4.0]
num_transition_frames: 3
```

## Constraint Geometry

Each candidate uses three generation-time and postprocess-time right-hand constraints:

- `row_1_start` at frame `36`
- `row_1_mid` at frame `51`
- `row_1_end` at frame `66`

The true brush point lies on the wall surface. The KIMODO right-hand endpoint target is shifted toward the robot body by the brush length:

```text
constraint_z = wall_z - 0.17
```

This models the brush length and avoids the earlier invalid height-offset calibration.

Stable-range 108 grid:

```text
height_y = [0.84, 0.90, 0.96, 1.02]
center_x = [-0.12, 0.00, 0.12]
width_x = [0.18, 0.24, 0.30]
wall_z = [0.42, 0.45, 0.48]
```

## Direct-From-Raw Postprocess

The direct postprocess skips the old multi-step staged route.

Current route:

```text
raw motion + recipe
  -> right_arm_g1_direct_from_raw_line_default_return_target
  -> one G1 right-arm hinge refit
  -> right_arm_g1_direct_from_raw_line_default_return
```

Target construction:

1. Read the right-hand brush constraints from `recipe.json`.
2. Linearly interpolate the right-hand target between frame `36`, `51`, and `66`.
3. Build a body-relative G1 default right-arm return from frame `67` to the final frame using `smootherstep`.
4. Start the return hand target from the final brush constraint to avoid a hard switch at frame `66`.
5. Keep all non-right-arm source motion from the raw prior as the natural full-body reference.

The final refit optimizes only legal physical G1 right-arm hinge joints:

```text
[26, 27, 28, 29, 30, 31, 32]
```

The endpoint/helper joint `33` is not directly optimized.

Default refit parameters:

```text
steps: 1600
lr: 0.012
pad: 8
threshold: 0.0001
taper_frames: 0
target_loss_weight: 520.0
hand_loss_weight: 17000.0
pose_prior_weight: 8.0
angle_prior_weight: 10.0
angle_vel_weight: 180.0
angle_acc_weight: 340.0
hand_acc_weight: 5200.0
```

## Validation Snapshot

Stable-range 108 batch:

| Metric | Value |
| --- | ---: |
| Completed candidates | `108 / 108` |
| Normal reach cases | `84` |
| Extreme-x reach cases | `24` |
| Mean max constraint error | `0.0043546306 m` |
| Worst max constraint error | `0.0074151507 m` |
| Mean max line distance | `0.0039349349 m` |
| Mean stroke speed CV | `0.0894181033` |
| Max x-backstep total | `0.0 m` |

The accepted visual-review expectation is to inspect `raw` against `right_arm_g1_direct_from_raw_line_default_return` in the local prior viewer.

Newer summaries also report direction-independent stroke progress metrics:

```text
progress_axis
progress_direction
progress_backstep_count
progress_backstep_total_m
```

For this horizontal batch, `progress_axis` should be `x`. Keep `x_backstep_*` only as a backward-compatible horizontal field.

## Local Review

```bash
python -m kimodo.scripts.prior_viewer \
  --run-folder outputs/wall_brush_generalization_108_direct_from_raw \
  --host 127.0.0.1 \
  --port 7861
```

The viewer reads `review.json`; the current default variant should be:

```text
right_arm_g1_direct_from_raw_line_default_return
```
