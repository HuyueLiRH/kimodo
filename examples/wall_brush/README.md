# Wall Brush Direct-From-Raw Generation

This example package records the current default KIMODO/G1-RP single-stroke wall-brushing prior workflow used in the HuyueLiRH building-motion experiments.

The default pipeline is now:

1. Generate a semantically natural raw motion through the KIMODO demo/web-equivalent generation path.
2. Build a direct target from the raw motion: a straight right-hand line stroke through the declared brush constraints, followed by a body-relative default right-arm return. The line may be horizontal or vertical.
3. Refit once from raw to that target using legal G1 right-arm hinge DoFs.
4. Review `raw` against `right_arm_g1_direct_from_raw_line_default_return` in the local prior viewer.

This replaces the previous staged FABRIK/lightlock/intermediate G1 line-refit/default-return route as the default wall-brush cleanup method.

## Contents

- `scripts/direct_wall_brush_g1_from_raw.py`: local direct-from-raw postprocess for an existing raw run folder.
- `scripts/remote_wall_brush_direct_from_raw_batch_runner.py`: remote batch runner that performs web-equivalent raw generation and direct postprocess from one batch target file.
- `scripts/refit_g1_right_arm_hinge_to_target.py`: G1 hinge-space refit helper used by the direct postprocess.
- `postprocess_treatments/direct_from_raw_g1_hinge_default_return.json`: accepted postprocess treatment definition.
- `task_specs/one_row_wall_brush_108_direct_from_raw_batch.json`: stable-range 108 horizontal one-row target batch.
- `task_specs/one_column_wall_brush_27_direct_from_raw_batch.json`: first 27-candidate vertical bottom-to-top one-column target batch.
- `skills/kimodo-wall-brush`: default Codex skill and JSON pipeline metadata.

Historical raw prompt ablation assets remain under `task_specs/top3_raw_pipelines` and `skills/kimodo-wall-brush-raw-generation`.

## Remote Batch Run

Run from a KIMODO environment where the G1-RP model and demo embedding cache already exist:

```bash
/root/miniconda3/bin/python examples/wall_brush/scripts/remote_wall_brush_direct_from_raw_batch_runner.py \
  --batch examples/wall_brush/task_specs/one_row_wall_brush_108_direct_from_raw_batch.json \
  --remote_root /root/autodl-tmp/KIMODO/work/prior_runs/wall_brush_generalization_108_direct_from_raw \
  --kimodo_repo /root/autodl-tmp/KIMODO/work/kimodo \
  --python /root/miniconda3/bin/python \
  --device cuda:0 \
  --embedding-cache-root /root/.cache/kimodo_demo/embeddings \
  --resume \
  --archive /root/autodl-tmp/KIMODO/work/prior_runs/wall_brush_generalization_108_direct_from_raw.tgz
```

For the first vertical bottom-to-top batch, replace `--batch` and `--remote_root` with:

```text
--batch examples/wall_brush/task_specs/one_column_wall_brush_27_direct_from_raw_batch.json
--remote_root /root/autodl-tmp/KIMODO/work/prior_runs/wall_brush_one_column_27_direct_from_raw
```

The raw generation stage calls:

```text
kimodo.scripts.web_equivalent_generate
```

For a workstation account without writable `/root`, point the runner to that
machine's copied demo embedding cache, for example:

```text
--embedding-cache-root /project/huyue/cache/kimodo_demo/embeddings
```

The final review variant is:

```text
right_arm_g1_direct_from_raw_line_default_return
```

## Local Postprocess For Existing Raw Runs

For an existing run folder with `raw/<candidate>/motion.npz` and `raw/<candidate>/recipe.json`:

```bash
PYTHONPATH=examples/wall_brush/scripts:. python examples/wall_brush/scripts/direct_wall_brush_g1_from_raw.py \
  --run-root outputs/wall_brush_generalization_108 \
  --device cpu \
  --steps 1600 \
  --resume
```

Expected output files:

- `postprocessed/<candidate>/right_arm_g1_direct_from_raw_line_default_return_target/motion.npz`
- `postprocessed/<candidate>/right_arm_g1_direct_from_raw_line_default_return/motion.npz`
- `direct_from_raw_summary.csv`
- `direct_from_raw_summary.json`
- updated `manifest.json`, `metrics.json`, and `review.json`

## Accepted Validation Snapshot

Stable-range 108 batch, reviewed locally on 2026-05-26:

- generated/postprocessed candidates: `108 / 108`
- normal reach cases: `84`
- extreme-x reach cases: `24`
- direct constraint max error mean: `0.0043546306 m`
- direct constraint max error worst: `0.0074151507 m`
- direct line max distance mean: `0.0039349349 m`
- direct stroke speed CV mean: `0.0894181033`
- direct stroke x-backstep total: `0.0 m`

The vertical `one_column_wall_brush_27_direct_from_raw_batch.json` spec is prepared for the next generation run but has not yet been visually accepted as a stable default.

Worst recorded constraint case:

```text
y0p84_xm0p12_w0p30_z0p48
constraint max: 0.0074151507 m
```

## Generation Defaults

```text
model: kimodo-g1-rp
seed: 7023
num_samples: 1
prompt: A person stands still and slides the right palm flat on the outside surface of a wall from left to right.
duration_frames: 102
constraint_frames: [36, 51, 66]
cfg_text: 2.4
cfg_constraint: 4.0
diffusion_steps: 200
num_transition_frames: 3
post_processing: false
brush_offset_m: 0.17
```

See `docs/wall_brush/one_row_direct_from_raw.md` for the full rationale.
