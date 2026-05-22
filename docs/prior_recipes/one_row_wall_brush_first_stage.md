# One-Row Wall Brush First-Stage Recipe

This recipe is the first-stage wall-brushing Candidate Prior Set for G1-RP building-motion experiments.

It uses the web-equivalent constrained generation path with three named prompt candidates:

- `outside_surface`
- `left_arm_relaxed`
- `seed_upright_style`

Each candidate uses native multi-prompt generation for one continuous action:

1. approach the wall patch
2. brush one horizontal row
3. lower the hand and stop naturally

The declared right-hand constraint points are passed to KIMODO generation by default and should also be shown in review. Postprocessing is intentionally off by default; any postprocessed version should be recorded as a derived motion with raw-to-postprocessed lineage.

Wall-contact geometry is explicit in the recipe. The first-stage raw recipe uses only three sparse wall-contact anchors: start, middle, and end of the desired brush row. `true_point` is the wall surface / brush-tip point at `z=0.45`. `position` is the right-hand endpoint passed to KIMODO; it keeps the same horizontal coordinate and height as `true_point`, then shifts `0.17 m` toward the body along `-Z` to account for brush length. This avoids the earlier invalid calibration where the offset was accidentally applied to height and too many dense points overconstrained generation.

Run on the remote server with:

```bash
kimodo_prior_run \
  --task examples/prior_recipes/one_row_wall_brush_first_stage.json \
  --output /root/autodl-tmp/KIMODO/work/prior_runs/one_row_wall_brush_first_stage \
  --use-demo-embedding-cache \
  --no-csv
```

After generation, sync the lightweight review artifacts, including `.npz` motions, manifest, metrics, review state, recipes, and gallery, to a local review folder before visual inspection.

The first review goal is Semantic Task Success: clean one-row brushing behavior with no start jump, turn-around, unrelated arm waving, excessive body drift, or unrelated motion after the task is complete. Small contact or line-tracking offsets are Contact Precision Debt, not automatic rejection.

## Current Accepted Postprocessed Candidate

The current accepted postprocessed pipeline for the baseline single-row sample is documented in
[`one_row_wall_brush_g1_hinge_default_return.md`](one_row_wall_brush_g1_hinge_default_return.md).

This is not raw KIMODO output. It is a postprocessed prior candidate that keeps raw-to-postprocessed
lineage and should be treated as the current best wall-brushing cleanup method for the baseline sample:

- source raw recipe: `outside_surface`-style single prompt with three right-hand constraints
- accepted variant: `right_arm_g1_hinge_line_refit_smooth_body_default_return_no_taper`
- key decision: optimize only legal G1 right-arm hinge DoFs, then return the right arm to a body-relative frame-0 default posture
- current status: accepted for this baseline sample and applied to the 27-position wall-brush grid; right-shifted wide strokes remain the main visual-review edge cases
