# Prompts And Parameters

## Exact Multi-Prompts

### outside_surface

1. A person stands still and raises the right hand toward the outside surface of a small wall patch.
2. A person stands still and slides the right palm flat on the outside surface of a wall from left to right.
3. A person stands still and slides the right palm flat on the outside surface of a wall from right to left.
4. A person stands still and keeps the right hand on the outside surface while making three straight horizontal brush strokes.
5. A person stands still and lowers the right hand away from the outside surface of the wall.

### left_arm_relaxed

1. A person stands balanced in place in front of a small wall patch, with the left arm relaxed by the side and the right hand ready to brush.
2. A person keeps the left arm relaxed by the side while the right hand brushes one straight horizontal stroke from left to right on a wall.
3. A person keeps the left arm relaxed by the side while the right hand brushes one straight horizontal stroke from right to left on a wall.
4. A person keeps the left arm relaxed by the side while the right hand brushes one straight horizontal stroke from left to right on a wall.
5. A person relaxes the left arm by the side and smoothly lowers the right hand after brushing.

### seed_upright_style

1. A person is standing in an upright stance in front of a wall.
2. A person standing in an upright stance leans forward and moves the right hand across the wall from left to right.
3. A person leaning forward moves the right hand across the wall from right to left.
4. A person leaning forward moves the right hand across the wall from left to right.
5. A person leaning forward lowers the right hand from the wall.

## Constraint And Scene Parameters

Reuse the original `task_spec.json` files whenever possible. They preserve both prompts and the wall-brush target layout.

Generation defaults:

```text
model: Kimodo-G1-RP-v1
cache_model: kimodo-g1-rp
skeleton: g1skel34
active_hand: right
frame_plan: 210
segments: 36,42,42,42,48
boundaries: 36,78,120,162
variant: endpoint_wrist
heading_mode: none
preemphasis_preset: flat_B
prompt_variant: level_full_width
num_transition_frames: 5
disable_y_closed_loop: true
disable_return_constraint: true
cfg_text: 2.4
cfg_constraint: 4.0
diffusion_steps: 200
num_samples: 1
post_processing: false
```

Wall/task layout:

```text
center_x: 0.0
top_y: 0.92
row_gap: 0.03
width: 0.24
z_contact: 0.32
z_wrist: 0.24
points_per_row: 6
x_scale: 1.9
x_offset: 0.04
y_offset: -0.07
z_offset: 0.03
```

Constraints used during KIMODO generation:

- Right-hand endpoint trajectory through the wall-brush keyframes.
- Sparse wrist points at the row endpoints to encourage palm/hand orientation.
- Sparse root 2D and root-height constraints.
- No final neutral-hand endpoint in this saved pipeline.

Validated raw seed robustness run:

```text
seeds: 7023, 8023, 9023
gallery: logs/wall_brush_prompt_ablation_top3_seed_raw/top3_seed_robustness_raw_gallery.html
summary: logs/wall_brush_prompt_ablation_top3_seed_raw/top3_seed_robustness_raw_summary.csv
```
