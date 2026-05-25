# One-Row Wall Brush G1 Hinge Default Return

Date accepted: 2026-05-25

This note records the current accepted postprocessed prior pipeline for the
one-row G1-RP wall-brushing prior collection.

## Status

- Task: one-row wall brushing
- Model: `kimodo-g1-rp`
- Raw prompt: `A person stands still and slides the right palm flat on the outside surface of a wall from left to right.`
- Candidate set: 27 position variants across height, horizontal center, and stroke width
- Accepted postprocessed variant: `right_arm_g1_hinge_line_refit_smooth_body_default_return_no_taper`
- Review decision: all 27 variants accepted as postprocessed building-motion priors after local visual review

The accepted motions are postprocessed priors, not raw KIMODO output and not
final deployed robot policies. The raw motions and the postprocessed outputs
must remain linked by raw-to-postprocessed lineage.

## Executable Assets

- Recipe: `examples/wall_brush/task_specs/one_row_wall_brush_27_g1_hinge_default_return.json`
- Treatment: `examples/wall_brush/postprocess_treatments/g1_hinge_default_return_no_taper.json`
- Runner: `examples/wall_brush/scripts/apply_wall_brush_g1_hinge_default_return_27.py`
- Hinge refit helper: `examples/wall_brush/scripts/refit_g1_right_arm_hinge_to_target.py`
- Review record: `docs/wall_brush/reviews/wall_brush_generalization_27_smooth_return_20260521.json`

The treatment expects an existing prior run folder containing raw motions and
the intermediate light-lock/uniform-line postprocess candidates. It then writes
the G1 hinge-space line refit, body-relative default-return target, final
no-taper result, and summary metrics.

Example:

```bash
PYTHONPATH=. python examples/wall_brush/scripts/apply_wall_brush_g1_hinge_default_return_27.py \
  --run_root outputs/wall_brush_generalization_27_smooth_return_20260521 \
  --resume
```

## Raw Generation Recipe

The raw KIMODO/G1-RP motions use a single accepted prompt segment:

```text
A person stands still and slides the right palm flat on the outside surface of a wall from left to right.
```

Generation settings:

- seed: `7023`
- `cfg_text`: `2.4`
- `cfg_constraint`: `4.0`
- diffusion steps: `200`
- KIMODO built-in postprocessing: disabled
- duration: `102` frames

Each candidate uses three right-hand constraint points:

- `row_1_start` at frame `36`
- `row_1_mid` at frame `51`
- `row_1_end` at frame `66`

The true wall / brush-tip point is on wall surface `z=0.45`. The KIMODO
right-hand endpoint constraint is shifted toward the robot body by
`brush_length_m=0.17`, so `constraint_z=0.28`. This represents the brush length
and avoids the earlier invalid height-offset calibration.

The accepted 27-variant grid is:

- `height_y`: `[0.86, 0.92, 0.98]`
- `center_x`: `[-0.08, 0.0, 0.08]`
- `width_x`: `[0.18, 0.24, 0.3]`

## Accepted Postprocess

The final treatment is a G1 hinge-space postprocess:

1. Start from the semantically clean raw/base wall-brush motion.
2. Preserve the useful wrist/hand orientation with the light-lock smooth-return source.
3. Build a straight right-hand line target over frames `36`, `51`, and `66`.
4. Refit only legal physical G1 right-arm hinge DoFs:
   `[26, 27, 28, 29, 30, 31, 32]`.
5. Do not optimize helper/end-effector joint `33`.
6. Return the right arm to the body-relative frame-0 G1 debug/default posture
   over frames `67-101`.
7. Disable final edit-mask taper so frame `101` reaches the return target.

This avoids the earlier problem where arbitrary skeleton refits made the
endpoint path straighter by creating physically implausible right-arm rotations.

## Review Record

Local review folder:

```text
logs/wall_brush_generalization_27_smooth_return_20260521
```

The local run-level `review.json` was updated on 2026-05-25:

- all 27 candidates: `postprocessed_accepted`
- accepted variant: `right_arm_g1_hinge_line_refit_smooth_body_default_return_no_taper`
- admission blockers: none recorded

The right-shifted wide-stroke cases remain flagged as reach/naturalness edge
cases for future multi-row scaling:

- `y0p86_x0p08_w0p30`
- `y0p92_x0p08_w0p30`
- `y0p98_x0p08_w0p30`

## Generalization Metrics

Recorded from the 27-position batch:

| Metric | Value |
| --- | ---: |
| Generated postprocessed motions | `27 / 27` |
| Constraint max error <= 0.005 m | `11 / 27` |
| Constraint max error <= 0.010 m | `21 / 27` |
| Constraint max error <= 0.015 m | `24 / 27` |
| Constraint max error <= 0.020 m | `26 / 27` |
| Worst stroke line max distance | `0.00843156773477459 m` |
| Worst final hand distance to body-relative default target | `0.0029571345075964928 m` |
| Worst return max step | `0.034470152109861374 m/frame` |

## Next Work

Do not expand directly to new tasks before preserving this first-stage result.
The next experiment step is multi-row wall brushing, using this one-row recipe
and treatment as the baseline prior-generation and cleanup method.
