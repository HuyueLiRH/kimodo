# Add web-equivalent native End-Effectors wall-brush raw generation script

> Issue draft created because GitHub Issues are currently disabled for `HuyueLiRH/kimodo`.

## What to build

Add a reproducible raw KIMODO/G1 wall-brushing generation path that matches the KIMODO web UI's native End-Effectors behavior for the current one-row wall-brush task.

This slice should preserve the verified script-side route from the migration workspace: generate directly from KIMODO with `post_processing=False`, no transition filter, no IK, and no sample reranking. The important behavior is that the script constructs right-hand End-Effectors constraints the same way the web UI does: start from the G1 demo/default pose, translate the right-hand end-effector chain to each target point, optionally set wrist points, and pass the full pose into KIMODO's native `RightHandConstraintSet`.

The goal is effect-level parity with the web UI: direct raw generation should follow the red constraint points at roughly the same accuracy as the web UI-generated motion. Exact equality with the old saved `test1.npz` should not be required because that file did not preserve seed/timeline metadata and had a different frame count.

## Acceptance criteria

- [ ] Add a named generation route for the wall-brush task equivalent to `demo_native_right_hand` / web-native End-Effectors construction.
- [ ] Add or document a one-row wall-brush preset using the validated `left_arm_relaxed` prompt, `bias_compensated_flatB` targets, seed `7023`, `cfg_text=2.4`, `cfg_constraint=4.0`, `diffusion_steps=200`, and `num_transition_frames=3`.
- [ ] Ensure the generation path is raw KIMODO output only: no postprocess, no transition-aware smooth filter, no IK, no projection/optimization, and no candidate selection/reranking.
- [ ] Include a repeatable command or script for running the one-row wall-brush case from the repo.
- [ ] Save enough metadata with each generated motion to compare with web outputs later: prompts, segment lengths, seed, CFG weights, diffusion steps, constraint target points, active hand, and constraint route name.
- [ ] Add a simple verification artifact or instructions that report frame count and right-hand constraint-point errors. The expected mean red-constraint error should be on the order of a few centimeters, comparable to the web UI result observed in the migration workspace.
- [ ] Document that old web-saved motions without seed/timeline metadata are not strict deterministic baselines; strict web-vs-script equality requires regenerating from the web preset with the same seed, same prompt durations, and same constraints.

## Blocked by

None - can start immediately.

## Notes from migration workspace

- Verified raw-only gallery: `logs/wall_brush_one_row_native_constraint_raw/one_row_native_constraint_raw_gallery.html`.
- Web-vs-script comparison gallery: `logs/web_vs_script_same_task_compare/web_vs_script_same_task_gallery.html`.
- Observed web `test1.npz`: 178 frames, no saved seed metadata.
- Current scripted preset: 102 frames unless explicitly probing longer timelines.
- Both web and script-side native routes can follow the red constraint points with mean error around 2 cm; old `test1.npz` is not a deterministic equality baseline because its timeline/generation metadata is incomplete.
