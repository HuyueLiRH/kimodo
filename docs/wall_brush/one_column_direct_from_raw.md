# One-Column Wall Brush Direct-From-Raw Pipeline

Date planned: 2026-06-02

This note records the first vertical bottom-to-top wall-brushing task spec for G1-RP building-motion prior experiments.

## Status

- Task: one-column wall brushing
- Direction: bottom-to-top
- Model: `kimodo-g1-rp`
- Raw generation path: KIMODO demo/web-equivalent generation
- Raw prompt: `A person stands still and slides the right palm flat on the outside surface of a wall from bottom to top.`
- Constraint frames: `[36, 51, 66]`
- Planned candidates: `27`
- Final variant: `right_arm_g1_direct_from_raw_line_default_return`
- Review status: task spec prepared; remote generation and visual acceptance still pending

This task is a vertical-stroke sibling of one-row wall brushing. It does not replace the accepted horizontal one-row route.

## Executable Assets

- Batch target file: `examples/wall_brush/task_specs/one_column_wall_brush_27_direct_from_raw_batch.json`
- Postprocess treatment: `examples/wall_brush/postprocess_treatments/direct_from_raw_g1_hinge_default_return.json`
- Remote batch runner: `examples/wall_brush/scripts/remote_wall_brush_direct_from_raw_batch_runner.py`
- Local postprocess runner: `examples/wall_brush/scripts/direct_wall_brush_g1_from_raw.py`
- G1 hinge refit helper: `examples/wall_brush/scripts/refit_g1_right_arm_hinge_to_target.py`
- Web-equivalent raw generator: `kimodo/scripts/web_equivalent_generate.py`
- Local review viewer: `kimodo/scripts/prior_viewer.py`

## Raw Generation

Raw motions use the same web-equivalent generation path as the accepted horizontal wall-brush route:

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

- `column_1_start` at frame `36`
- `column_1_mid` at frame `51`
- `column_1_end` at frame `66`

The true brush point lies on the wall surface. The KIMODO right-hand endpoint target is shifted toward the robot body by the brush length:

```text
constraint_z = wall_z - 0.17
```

First 27-candidate grid:

```text
center_y = [0.90]
center_x = [-0.12, 0.00, 0.12]
stroke_height_y = [0.18, 0.24, 0.30]
wall_z = [0.42, 0.45, 0.48]
stroke_axis = y
stroke_direction = bottom_to_top
```

Constraint construction:

```text
start = [center_x, center_y - stroke_height_y / 2, wall_z - 0.17]
mid   = [center_x, center_y,                       wall_z - 0.17]
end   = [center_x, center_y + stroke_height_y / 2, wall_z - 0.17]
```

## Postprocess

The vertical task uses the same direct-from-raw postprocess route:

```text
raw motion + recipe
  -> right_arm_g1_direct_from_raw_line_default_return_target
  -> one G1 right-arm hinge refit
  -> right_arm_g1_direct_from_raw_line_default_return
```

The target line is built from the declared constraint direction, so the same target builder supports horizontal and vertical strokes.

Metrics should use direction-independent stroke progress:

```text
progress_axis
progress_direction
progress_backstep_count
progress_backstep_total_m
```

The older `x_backstep_*` fields remain as horizontal-compatibility fields, but should not be used as the main quality signal for vertical strokes.

## Remote Run

```bash
/root/miniconda3/bin/python examples/wall_brush/scripts/remote_wall_brush_direct_from_raw_batch_runner.py \
  --batch examples/wall_brush/task_specs/one_column_wall_brush_27_direct_from_raw_batch.json \
  --remote_root /root/autodl-tmp/KIMODO/work/prior_runs/wall_brush_one_column_27_direct_from_raw \
  --kimodo_repo /root/autodl-tmp/KIMODO/work/kimodo \
  --python /root/miniconda3/bin/python \
  --device cuda:0 \
  --embedding-cache-root /root/.cache/kimodo_demo/embeddings \
  --resume \
  --archive /root/autodl-tmp/KIMODO/work/prior_runs/wall_brush_one_column_27_direct_from_raw.tgz
```

## Local Review

```bash
python -m kimodo.scripts.prior_viewer \
  --run-folder outputs/wall_brush_one_column_27_direct_from_raw \
  --host 127.0.0.1 \
  --port 7861
```

The expected review comparison is `raw` versus:

```text
right_arm_g1_direct_from_raw_line_default_return
```
