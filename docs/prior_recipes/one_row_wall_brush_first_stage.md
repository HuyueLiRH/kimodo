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
