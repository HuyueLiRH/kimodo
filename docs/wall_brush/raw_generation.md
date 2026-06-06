# Wall Brush Raw Generation

This document records the currently successful KIMODO/G1 raw wall-brush generation method.

## Goal

Generate a natural base motion that roughly resembles wall brushing:

- body remains mostly facing the wall
- root drift is small
- left arm stays quiet
- right hand performs horizontal brush-like strokes
- no large return-turn or unnatural endpoint snap

Precise wall contact is left to later optimization. The raw generation stage should prioritize natural motion and avoid strange extra body actions.

## Validated Pipelines

Use the frozen task specs in:

```text
examples/wall_brush/task_specs/top3_raw_pipelines/
```

Pipelines:

- `outside_surface`: best quiet-body base motion
- `left_arm_relaxed`: best brush-like coverage candidate
- `seed_upright_style`: third reproducible style baseline

Each task spec contains five temporal prompts and the wall-brush layout used to build endpoint/wrist/root constraints.

## Raw-Only Rule

For reproducibility and visual inspection, the raw pipeline must not use:

- KIMODO postprocessing
- transition-aware smoothing
- IK
- V36 soft full-body projection
- DNO/noise search
- many-candidate seed selection

Use one generated motion per `{pipeline, seed}` pair.

## Command

Run from `examples/wall_brush`:

```bash
/root/miniconda3/bin/python scripts/run_wall_brush_top3_seed_robustness_raw.py \
  --output_root outputs/wall_brush_prompt_ablation_top3_seed_raw \
  --seeds 7023 8023 9023 \
  --execute
```

For one pipeline and one seed:

```bash
/root/miniconda3/bin/python scripts/run_wall_brush_top3_seed_robustness_raw.py \
  --output_root outputs/wall_brush_single_raw \
  --variants outside_surface \
  --seeds 10023 \
  --execute
```

## Verification

The generated gallery should contain labels such as:

```text
RAW outside_surface seed_7023 | endpoint_wrist | cfg#0 C=4.0 | T=2.4 | 200 steps | sample_00
```

It should not contain `FILTERED` or `transition_filtered`.

## Observed Direction

Prompt wording is the strongest lever for this task. Increasing constraint CFG alone made motions stiffer and less natural in strictness experiments. Keep `cfg_constraint=4.0` as the baseline unless there is a specific reason to retune.

## Accepted One-Row Postprocessed Prior

The current accepted one-row wall-brush prior collection is documented in:

```text
docs/wall_brush/one_row_g1_hinge_default_return.md
```

That collection uses the accepted single prompt plus a 27-position constraint
grid, then applies the G1 hinge-space default-return treatment:

```text
examples/wall_brush/postprocess_treatments/g1_hinge_default_return_no_taper.json
```

The raw motions remain preserved. The accepted outputs are postprocessed
building-motion priors with explicit raw-to-postprocessed lineage.
