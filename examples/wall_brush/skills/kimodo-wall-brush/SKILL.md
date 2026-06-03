---
name: kimodo-wall-brush
description: Generate and review one-row KIMODO/G1-RP wall-brushing building-motion priors with web-equivalent raw generation and direct-from-raw G1 right-arm hinge postprocess.
---

# Kimodo Wall Brush

## Default Route

Use this route for the current building-motion prior experiments:

1. Generate raw KIMODO/G1-RP motion with the demo/web-equivalent interface.
2. Use one single prompt and three right-hand constraint points for the one-row stroke.
3. Build a direct line-stroke plus body-default-return target from the raw motion.
4. Run one G1 right-arm hinge refit from raw to target.
5. Review `raw` against `right_arm_g1_direct_from_raw_line_default_return`.

Do not use the previous staged FABRIK/lightlock/intermediate G1 line-refit/default-return route as the default. Keep it only for historical comparison.

## Generation Defaults

```text
model: kimodo-g1-rp
seed: 7023
num_samples: 1
prompt: A person stands still and slides the right palm flat on the outside surface of a wall from left to right.
duration_frames: 102
segments: [102]
constraint_frames: [36, 51, 66]
cfg_type: separated
cfg_weight: [2.4, 4.0]
num_denoising_steps: 200
num_transition_frames: 3
post_processing: false
brush_offset_m: 0.17
```

The right-hand constraint z coordinate is:

```text
constraint_z = wall_z - brush_offset_m
```

This approximates brush length and avoids placing the hand endpoint directly on the wall surface.

## Current 108 Batch

Use:

```text
examples/wall_brush/task_specs/one_row_wall_brush_108_direct_from_raw_batch.json
```

Grid:

```text
height_y = [0.84, 0.90, 0.96, 1.02]
center_x = [-0.12, 0.00, 0.12]
width_x = [0.18, 0.24, 0.30]
wall_z = [0.42, 0.45, 0.48]
```

## Remote Execution

Run:

```bash
/root/miniconda3/bin/python examples/wall_brush/scripts/remote_wall_brush_direct_from_raw_batch_runner.py \
  --batch examples/wall_brush/task_specs/one_row_wall_brush_108_direct_from_raw_batch.json \
  --remote_root /root/autodl-tmp/KIMODO/work/prior_runs/wall_brush_generalization_108_direct_from_raw \
  --kimodo_repo /root/autodl-tmp/KIMODO/work/kimodo \
  --python /root/miniconda3/bin/python \
  --device cuda:0 \
  --resume
```

The runner calls `kimodo.scripts.web_equivalent_generate`, so the result should match the web UI generation style better than the old scripted route.

## Postprocess

Use:

```text
examples/wall_brush/scripts/direct_wall_brush_g1_from_raw.py
```

Final variant:

```text
right_arm_g1_direct_from_raw_line_default_return
```

Direct target variant:

```text
right_arm_g1_direct_from_raw_line_default_return_target
```

The postprocess optimizes only physical G1 right-arm hinge joints:

```text
[26, 27, 28, 29, 30, 31, 32]
```

Do not directly optimize endpoint/helper joint `33`.

## Review

Launch the viewer:

```bash
python -m kimodo.scripts.prior_viewer --run-folder outputs/wall_brush_generalization_108_direct_from_raw --port 7861
```

The viewer should default to `review.current_best_variant`, which should be:

```text
right_arm_g1_direct_from_raw_line_default_return
```

## Acceptance Baseline

Accepted stable-range 108 batch:

```text
completed: 108/108
constraint_max_mean_m: 0.004354630561123724
constraint_max_worst_m: 0.007415150686916008
line_max_mean_m: 0.003934934896011222
stroke_x_backstep_total_max_m: 0.0
```

The most important human review criteria are:

- right hand moves directly to the stroke start without a high extra raise
- right hand brushes one simple horizontal row without repeated scribbling
- left hand does not become the task hand
- right arm stays plausible under G1 mesh visualization
- after the stroke, the right arm returns naturally toward the default/debug pose
