# Wall Brush Raw Generation

This example package preserves the reproducible KIMODO/G1 wall-brushing raw-generation setup used in the HuyueLiRH experiments.

The current validated raw pipelines are:

- `outside_surface`: quiet body motion and low extra turning
- `left_arm_relaxed`: stronger brush-like coverage with relaxed left arm prompts
- `seed_upright_style`: upright/leaning style baseline with visible brushing intent

These pipelines are intentionally **raw KIMODO generation**: no transition smoothing, no postprocessing, no IK, no V36 projection, and no many-seed candidate selection.

## Contents

- `scripts/run_wall_brush_top3_seed_robustness_raw.py`: run the three validated prompt+constraint pipelines across requested seeds.
- `scripts/remote_wall_brush_multiprompt_constraint_v34_flat.py`: KIMODO generation-time constraint runner used by the raw pipelines.
- `scripts/create_kimodo_motion_gallery.py`: local HTML gallery builder for visual inspection.
- `task_specs/top3_raw_pipelines/*/task_spec.json`: frozen prompt+constraint task specs for the three successful pipelines.
- `skills/kimodo-wall-brush-raw-generation`: Codex skill for reproducing and continuing the raw-generation workflow.
- `skills/kimodo-wall-brush`: broader historical wall-brush pipeline skill and defaults.

## Run

From this directory on a KIMODO environment:

```bash
/root/miniconda3/bin/python scripts/run_wall_brush_top3_seed_robustness_raw.py \
  --output_root outputs/wall_brush_prompt_ablation_top3_seed_raw \
  --seeds 7023 8023 9023 \
  --execute
```

Expected output:

```text
outputs/wall_brush_prompt_ablation_top3_seed_raw/top3_seed_robustness_raw_gallery.html
outputs/wall_brush_prompt_ablation_top3_seed_raw/top3_seed_robustness_raw_summary.csv
```

Verify that the generated gallery contains only `RAW` labels and no `FILTERED` or `transition_filtered` motions.

## Generation Defaults

```text
variant: endpoint_wrist
cfg_text: 2.4
cfg_constraint: 4.0
diffusion_steps: 200
num_samples: 1
frame_plan: 210
preemphasis_preset: flat_B
heading_mode: none
prompt_variant: level_full_width
num_transition_frames: 5
active_hand: right
disable_y_closed_loop: true
disable_return_constraint: true
post_processing: false
```

See `docs/wall_brush/raw_generation.md` for more detail.
