---
name: kimodo-wall-brush
description: Generate and validate precise Kimodo G1 row-by-row wall brushing, wiping, painting, or surface-coverage motions using multi-prompt timing, native position-only endpoint/wrist constraints, sparse root constraints, transition smoothing, and optional right-arm line IK for exact rows. Use when the task requires vertical-plane coverage, natural no-IK motion, straight row trajectories, AutoDL Kimodo generation, and numerical pass/fail validation.
---

# Kimodo Wall Brush

## Purpose

Use this skill to create precise row-by-row right-hand motion across a vertical plane, such as brushing, wiping, or painting. Treat success as numerical trajectory accuracy over row segments, not visual plausibility.

For "each row almost straight, otherwise calm" requests, the preferred natural-coordinated pattern is two stage:

1. Use Kimodo to generate a natural in-place brushing base motion from sparse constraints.
2. Preserve Kimodo's body and left-arm motion as the natural full-body prior, while applying right-hand/right-arm correction only where row precision is required.

Do not interpret "left hand and body do not participate" as "freeze them." For the current target style, the left hand and body may move naturally for balance and coordination, but they should not look like they are doing the brushing task, crossing the wall, or producing sudden jumps.

## Required Setup

- Use AutoDL through `scripts\autodl_remote.py`; never print `.env`.
- Use model `Kimodo-G1-RP-v1`.
- Use endpoint joint `right_hand_roll_skel` for validation.
- Use cached LLM2Vec embeddings through `CachedOnlyTextEncoder`.
- Prefer `cfg_type=separated`, `cfg_text=1.3`, `cfg_constraint=9.0`.
- Use `post_processing=False`; prior tests showed post-processing can damage G1 endpoint accuracy.
- Keep root near origin with `Root2DConstraintSet`.
- Add root height constraints when possible; in-place wall tasks should not be solved by large pelvis drift.
- For the natural-coordinated style, prefer the original Kimodo body/left-arm prior plus `scripts\postprocess_wall_brush_line_ik.py` on the right-arm chain. This matches the visual style of `sample_13_line_ik_motion`: full body remains coordinated, while the right hand achieves near-exact rows.
- Use `scripts\postprocess_wall_brush_taskline_calm.py` only as a strict stability/control reference when the user explicitly wants non-task joints removed from the reference. It freezes or heavily calms non-task joints and does not match the desired natural-coordinated style.
- If row-to-row gap frames should leave the wall instead of connecting directly along the wall, add a bookend/gap smoothing pass after line IK or use a natural variant of task-line correction that does not freeze the body.

## Task Design

Define a vertical wall plane and three horizontal rows. Keep geometry close, low, and reachable. Avoid high or far walls; they cause strange full-body compensation and large root drift.

Known-good G1 geometry:

- wall plane `z = 0.32`
- row heights `y = 0.94`, `0.90`, `0.86`
- horizontal sweep `x = -0.12` to `0.12`
- root height target `root_y = 0.78`

Use nearby variants before widening or raising the patch. If the target is too high/far, the robot may reach by bending the torso, shifting root, or making circular hand motion around targets.

Use serpentine direction:

1. row 1: left to right
2. row 2: right to left
3. row 3: left to right

Use five target points per row. More points can over-constrain the sampler; fewer points can weaken line coverage.

Use action-specific prompt phases. Prefer "straight horizontal stroke" over "paint" or "wipe" as the main control phrase.

If the user cares about a natural final hand return and explicitly does not want return IK, prefer the current 5-phase return design. It removes the extra idle/rest prompt that previously encouraged a final body turn:

1. `A person stands still facing a wall and raises the right hand toward a small wall patch.`
2. `A person stands still facing the wall and slides the right hand in a straight horizontal stroke from left to right on the wall.`
3. `A person stands still facing the wall and slides the right hand in a straight horizontal stroke from right to left on the wall.`
4. `A person stands still facing the wall and slides the right hand in a straight horizontal stroke from left to right on the wall.`
5. `A person remains facing the wall and gently brings the right hand down beside the right thigh.`

Use durations `[0.8, 1.2, 1.2, 1.2, 1.2]`. The final return needs enough time to lower the hand; a short `0.45-0.8s` return often reduces yaw but leaves the hand too high. Do not add a sixth "stands/rests with arms at sides" phase unless a new experiment shows it does not reintroduce turning.

Older six-phase row/return prompts are still useful as a baseline:

1. `A person stands still and raises the right hand toward a small wall patch.`
2. `A person stands still and slides the right hand in a straight horizontal stroke from left to right on a wall.`
3. `A person stands still and slides the right hand in a straight horizontal stroke from right to left on a wall.`
4. `A person stands still and slides the right hand in a straight horizontal stroke from left to right on a wall.`
5. `A person stands still and makes three straight horizontal brush strokes on a small wall patch.`
6. `A person stands still and lowers the right hand after the straight brush strokes.`

Use durations `[0.8, 1.2, 1.2, 1.2, 1.2, 0.7]` for this older baseline.

Avoid relying on vague prompts such as `paint a wall` or `wipe a wall`; they can produce local looping/contact behavior around target points.

Avoid overusing phrases like `outside surface` in every phase. In testing, this eliminated wall penetration but made Kimodo keep the hand too far outside the wall, producing large plane and keyframe errors. The better prompt-level behavior is still the `straight horizontal stroke ... on a wall` phrasing, then handle non-penetration with constraints and validation.

Prompt-only attempts to explicitly tell Kimodo that the left arm or body is passive have not improved this task. Tested variants such as `left arm relaxed`, `left arm passive`, `non-working left hand`, `small natural torso and shoulder motion`, and `only the right hand` generally made the generated base motion less accurate, more loopy, or more jumpy. Adding motion-quality adjectives such as `calmly` and `smooth` also did not improve non-row smoothness in testing. Keep the prompt short and action-centric; let the learned full-body prior supply natural coordination.

When the final segment turns the body instead of simply lowering the right hand, use positive facing-wall return language rather than a negative instruction such as `do not turn`. The best prompt-only improvement tested so far is `gentle_return_5`: keep the original straight row wording, remove the final idle/rest phase, and explicitly say the person remains facing the wall while gently bringing the right hand down beside the right thigh. Select samples by tail behavior after generation, not only by Kimodo's original keyframe score, because line IK will later fix the row geometry.

SEED/BONES prompt-style notes:

- SEED timeline annotations are phase descriptions, not command-like constraints. They use simple present-tense phrases such as `A person begins to clean the floor with a mop`, `A person is cleaning the floor with a mop`, and `A person stops cleaning the floor and stands upright while holding the mop in their right hand`.
- For idle/non-task body behavior, SEED uses natural state descriptions such as `arms swaying at the sides`, `arms hanging naturally at the sides`, `standing in an upright stance`, and `standing still`. Prefer this style over `passive`, `non-working`, `only`, or explicit suppression.
- SEED's `wipe` examples mostly describe wiping shoes by rubbing them against the floor. This is a warning that `wipe` can bias the model toward foot/floor contact rather than a vertical wall task.
- The public SEED timeline search did not show a strong wall-painting/brush-wall cluster; for wall brushing, keep the geometric phrase `straight horizontal stroke ... on a wall` and use constraints for the missing spatial precision.
- Candidate SEED-style bookend text for future prompt tests: `A person stands in an upright stance in front of a wall`, row phases as `A person moves the right hand across a wall in a horizontal stroke`, and finish as `A person lowers their arms to the sides and stands in an upright stance`.
- Tested SEED-style prompts did not beat the original `straight` prompt. They are useful for understanding language style, but for this task the row action still needs the original `slides the right hand in a straight horizontal stroke...` wording.

## Multi-prompt + Native Constraint Route

Use this route when the user wants a natural G1 wall-brush motion and explicitly does not want right-arm line IK, return IK, FABRIK, post-generation root/torso locks, or a single-prompt replacement. Single-prompt wall brushing is not the main route because it is too semantically loose: the robot often just scribbles somewhere near the wall and the hand posture is less task-like. Multi-prompt remains the preferred generator interface because each phase separately anchors approach, row 1, row 2, row 3, and return.

Current no-IK 189-frame prompt plan:

1. frames `0-29`: `A person stands still facing a wall and raises the right hand toward a small patch on the wall, preparing to wipe the wall with the right hand.`
2. frames `30-65`: `A person stands still facing a wall with the right hand on a small wall patch, sliding the right hand in one short straight horizontal wiping stroke from left to right.`
3. frames `66-101`: `A person stands still facing a wall with the right hand on the same small wall patch, sliding the right hand in one short straight horizontal wiping stroke from right to left slightly lower on the wall.`
4. frames `102-137`: `A person stands still facing a wall with the right hand on the same small wall patch, sliding the right hand in one short straight horizontal wiping stroke from left to right slightly lower on the wall.`
5. frames `138-188`: `A person stands still facing a wall with the right hand near the wall, gently lowering the right hand down beside the right thigh and relaxing the arm.`

Do not add negative turn language such as `do not turn`, `no turning`, or `without turning`. Keep the positive phrase `stands still facing a wall` in every segment. Do not end with `both arms resting at the sides`; it can trigger a generic rest-pose template and weaken the right-hand return.

Native position-only constraint design:

- Use `scripts\remote_wall_brush_multiprompt_constraint_native.py`.
- Use `RightHandWallContactPositionOnlyConstraintSet`.
- Write only `global_joints_positions`; do not write `global_joints_rots`, hand/wrist rotation, local rotations, or any post-generation right-arm chain edits.
- For each stroke keyframe, constrain two positions: `right_hand_endpoint` on the wall and `right_wrist` outside the wall. For the current wall, use endpoint `z_contact=0.32` and wrist `z_wrist=0.20`.
- Generate wrist points as `[endpoint_x, endpoint_y - 0.02, 0.20]`. This gives the sampler a soft palm-facing-wall cue without hard-locking wrist rotation.
- Return is sparse endpoint-only: use the neutral/base standing right-hand endpoint as `final_right_hand_side_pos`, plus at most one intermediate off-wall point. Do not constrain return wrist rotation or run return IK.

Use separated CFG around the tested range:

```powershell
python scripts\autodl_remote.py exec /root/miniconda3/bin/python /root/autodl-tmp/kimodo_scripts/remote_wall_brush_multiprompt_constraint_native.py `
  --model Kimodo-G1-RP-v1 `
  --output_dir /root/autodl-tmp/wall_brush_runs/g1_mp_constraint_native_v31_189_endpoint_wrist_no_heading `
  --seed 31 `
  --num_samples 16 `
  --frame_plan 189 `
  --variant endpoint_wrist `
  --heading_mode none `
  --num_transition_frames 5
```

The script tests `[1.8,2.4]@150`, `[2.0,2.8]@150`, `[2.2,3.0]@200`, and `[2.4,3.2]@200` by default. Text weight cannot be too low or the phase semantics loosen; constraint weight cannot be too high or the arm becomes stiff.

Use `scripts\score_wall_brush_mp_constraint_native.py` for scoring and constraint-aware transition filtering:

```powershell
python scripts\score_wall_brush_mp_constraint_native.py `
  --root logs\wall_brush_g1_mp_constraint_native_v31 `
  --make_filtered `
  --summary logs\wall_brush_g1_mp_constraint_native_v31\summary.csv `
  --radius 10 `
  --kernel_size 13 `
  --sigma 3.0 `
  --strength 0.85 `
  --passes 3
```

The filter is allowed because it is temporal smoothing around prompt boundaries, not IK. It should only act near `[30, 66, 102, 138]`, and it should reduce right-arm smoothing strength at constrained right-hand keyframes so the stroke constraints are not washed out.

Current v31 no-IK result:

- local run root: `logs\wall_brush_g1_mp_constraint_native_v31`
- summary: `logs\wall_brush_g1_mp_constraint_native_v31\summary.csv`
- selected gallery copies: `logs\wall_brush_g1_mp_constraint_native_v31\gallery_selected`
- current preferred no-IK native sample: `logs\wall_brush_g1_mp_constraint_native_v31\endpoint_wrist_none_plan189_cfg1_t2p0_c2p8_steps150\transition_filtered\sample_15_transition_filtered.npz`
- metrics for that sample: mean/max row-line error about `0.108/0.161 m`, path ratio about `1.64`, root yaw delta/path about `0.10/4.36 deg`, transition boundary max step about `0.034 m/frame`, wall penetration about `0.0077 m`
- interpretation: it is much more natural than line IK and avoids the final turn, but the row path is only approximate. This is acceptable only when natural motion and no-turn behavior outrank exact straight-line accuracy.

Sparse heading comparison:

- best line-like sparse-heading sample: `endpoint_wrist_sparse_plan189_cfg1_t2p0_c2p8_steps150\transition_filtered\sample_09_transition_filtered.npz`
- it improves row-line mean error to about `0.047 m` and path ratio to about `1.15`, but tail root yaw path rises to about `33 deg`
- use sparse heading only as an optional comparison; do not choose it if the visual return shows yaw sway or stiffness
- return-only sparse heading is also only a comparison. Its best balanced v31 sample is `endpoint_wrist_return_only_plan189_cfg3_t2p4_c3p2_steps200\transition_filtered\sample_01_transition_filtered.npz`: row-line mean/max about `0.073/0.113 m`, path ratio about `1.50`, tail yaw delta/path about `3.0/15.9 deg`, wall penetration about `0.0077 m`. It improves trajectory over the no-heading sample but still adds more yaw activity, so keep no-heading as the preferred no-turn route unless visual inspection favors return-only.

Baseline comparison:

- old line IK baseline: `logs\wall_brush_g1_v29\sample_04_line_ik_motion.npz`; trajectory is far more precise, but the right arm can look IK-pulled or twisted.
- current `RightHandConstraintSet` baseline: `current_right_hand_none_plan189_cfg0_t2p0_c2p8_steps150\transition_filtered\sample_08_transition_filtered.npz`; it reaches about `1 cm` mean line error because it constrains rotation too, but it is a rotation-constrained baseline, not the native position-only route.
- native endpoint-only baseline is weaker than endpoint+wrist for hand posture and wall contact. Prefer endpoint+wrist position-only when avoiding IK.

V32 delayed + pre-emphasis update:

- local run root: `logs\wall_brush_g1_mp_constraint_v32_210_delayed_preemphasis`
- script: `scripts\remote_wall_brush_multiprompt_constraint_v32.py`
- summary: `logs\wall_brush_g1_mp_constraint_v32_210_delayed_preemphasis\summary.csv`
- diagnostic for the old v31 preferred sample: `logs\wall_brush_g1_mp_constraint_v32\diagnostics_current_best.csv`
- current preferred v32 no-IK sample: `logs\wall_brush_g1_mp_constraint_v32_210_delayed_preemphasis\endpoint_wrist_none_plan210_preC_x2_ym0p09_z0p04_zw0p2_cfg2_t2p2_c3p6_steps200\transition_filtered\sample_03_transition_filtered.npz`
- selected gallery copies: `logs\wall_brush_g1_mp_constraint_v32_210_delayed_preemphasis\gallery_selected`

V32 design:

- keep multi-prompt; do not switch to single prompt
- use `210` frames with boundaries `[36, 78, 120, 162]`
- delay stroke constraints so the first stroke keyframe is at `boundary + 10` instead of `boundary + 4`
- use position-only endpoint constraints at all 15 stroke keyframes
- use wrist position constraints only at the first and last point of each row, not at every endpoint
- do not constrain wrist or hand rotation
- do not use line IK, FABRIK, return IK, no-turn return IK, or post-generation root/torso locks
- use transition filtering only around prompt boundaries, with low right-arm smoothing at constrained frames

V32 best metrics:

- stroke line mean/max error: about `0.043/0.087 m`
- pointwise keyframe mean/max error: about `0.071/0.127 m`
- x range coverage: about `0.49` of the requested `0.24 m` width
- path ratio: about `1.47`
- tail root yaw delta/path: about `0.89/6.33 deg`
- wall penetration: `0.0 m`
- right-arm naturalness penalty: `0.0`
- max right-arm velocity/acceleration: about `0.047/0.048 m/frame`

Interpretation: v32 is the first native no-IK route that beats the `0.05 m` mean row-line target while keeping the natural/no-turn style. It still does not fully cover `x = -0.12..0.12`; pre-emphasis improves line distance more than horizontal range. If the user wants fuller horizontal coverage while preserving natural motion, the next route is DNO-lite or diffusion guidance, not right-arm FABRIK.

If the no-IK native route is not visually acceptable after sample selection, the next research step is diffusion guidance or DNO-lite. Do not fall back to right-arm IK unless the user explicitly changes the task priority back to exact rows.

## Generation Recipe

1. Cache every prompt text with `scripts\remote_cache_kimodo_prompts.py` before the run.
2. Generate with `scripts\remote_wall_brush.py` or a close variant.
3. Use sparse structured right-hand constraints: three rows times five points. Do not increase to dense hand targets as a first response; Kimodo is more reliable with sparse end-effector constraints.
4. Use root 2D and root height constraints for in-place behavior.
5. Keep the wrist on the outside side of the wall when constructing `RightHandConstraintSet`. For the current G1 wall plane `z = 0.32`, outside is lower `z`, so use a wrist offset like `-0.12` from the endpoint.
6. Clamp calibrated constraint target z to the wall plane with `--constraint_z_max 0.32`. This prevents row-rigid calibration from "solving" an underreach by moving the target through the wall.
7. Validate constrained keyframes, wall plane error, row-line distance, path ratio, backtracking, root drift, and wall penetration. Use a penetration threshold around `0.003 m` before line IK.
8. Use row-rigid calibration, not pointwise calibration. Row-rigid moves a whole row together and preserves row collinearity better.
9. For the current preferred natural-coordinated behavior, apply right-arm line IK to a good `straight` prompt sample while preserving the Kimodo-generated body and left-arm motion:

   ```powershell
   python scripts\postprocess_wall_brush_line_ik.py `
     --motion logs\wall_brush_g1_v21\sample_13_motion.npz `
     --metrics logs\wall_brush_g1_v21\metrics.json `
     --output logs\wall_brush_g1_v21\sample_13_line_ik_motion.npz `
     --blend_frames 10 `
     --iterations 36
   ```

   Design details:

   - The right hand is forced onto the row lines during the brushing intervals.
   - Kimodo's natural root, torso, legs, and left arm are preserved instead of frozen.
   - The left hand and body are judged visually and numerically as passive coordination: they may move, but should not appear to paint the wall or make abrupt compensatory jumps.
   - This is the closest current match to the desired `sample_13_line_ik_motion` style.

10. Use task-line calm IK only as the strict non-task-joint reference, not as the default natural style:

   ```powershell
   python scripts\postprocess_wall_brush_taskline_calm.py `
     --motion logs\wall_brush_g1_v21\sample_13_motion.npz `
     --metrics logs\wall_brush_g1_v21\metrics.json `
     --output logs\wall_brush_g1_v21\sample_13_taskline_calm_v1_motion.npz `
     --body_mode hold_initial `
     --gap_retract_z 0.045 `
     --rest_point_json "[0.12,0.66,0.14]"
   ```

   Design details:

   - The row intervals are solved as exact straight endpoint paths between each row's first and last target.
   - Row-to-row gaps are off-wall arcs, not straight wall-contact lines. The retract profile uses zero-velocity endpoints so the hand does not pop at the row boundary.
   - `--body_mode hold_initial` freezes root, torso, legs, and left arm in the generated initial pose. The right shoulder is also held as the IK root; only the right-arm child joints solve the task.
   - Use this when preparing a minimal control reference, or when diagnosing whether unwanted motion comes from Kimodo's full-body prior versus the right-arm task path.

11. Validate the post-processed result with:

   ```powershell
   python scripts\evaluate_wall_brush_motion.py `
     --motion logs\wall_brush_g1_v21\sample_13_taskline_calm_v1_motion.npz `
     --metrics logs\wall_brush_g1_v21\metrics.json
   ```

   For the natural-coordinated mode, success requires near-zero row-line error, near-zero wall penetration, smooth right-hand motion during rows, and no obvious left-arm/body task participation. Do not require zero left-arm movement unless using the strict calm/frozen reference.

12. Older line-IK behavior is still useful when a direct on-wall connection between rows is desired. Use enough blend frames to smooth non-row transition frames:

   ```powershell
   python scripts\postprocess_wall_brush_line_ik.py `
     --motion logs\wall_brush_g1_v15\sample_06_motion.npz `
     --metrics logs\wall_brush_g1_v15\metrics.json `
     --output logs\wall_brush_g1_v15\sample_06_line_ik_smooth_v2_motion.npz `
     --blend_frames 10 `
     --iterations 36
   ```

   This line IK connects the end of one row to the start of the next row during non-row gap frames. Do not use it when row changes should leave the wall and should not look like extra brush strokes.

13. If the rows are accurate but the approach-to-wall or exit-after-wall segments are too fast, add the bookend smoothing pass to the older line-IK pipeline. This preserves the exact row lines, replaces the pre-contact hand path with a smooth reach to the first row, replaces the post-contact hand path with a smooth retract/return, and smooths body/shoulder anchor motion across task segments:

   ```powershell
   python scripts\postprocess_wall_brush_bookend_smooth.py `
     --motion logs\wall_brush_g1_v18\sample_13_line_ik_motion.npz `
     --metrics logs\wall_brush_g1_v18\metrics.json `
     --output logs\wall_brush_g1_v18\sample_13_line_ik_bookend_v10_motion.npz `
     --exit_retract_frames 18 `
     --exit_retract_y -0.02 `
     --exit_retract_z -0.12 `
     --rest_point_json "[0.12,0.66,0.14]"
   ```

   Use this pass when preparing trajectories for later whole-body-control tracking; the goal is to keep maximum per-frame joint displacement low in the task bookends, not only to satisfy sparse endpoint constraints.

   Do not use `--rest_mode final` when the desired ending is "put the hand back simply". The original Kimodo final pose can place the hand left/up/back from the wall patch, which creates a visible turn after the final row. Use an explicit same-side rest point instead.

14. Regenerate the gallery:

   ```powershell
   python scripts\create_kimodo_motion_gallery.py --logs_dir logs --output logs\kimodo_motion_gallery.html
   ```

15. If the robot jitters at multi-prompt phase switches, add a prompt-transition smoothing pass before the final line-IK pass. This is a local temporal low-pass filter around the cumulative prompt `num_frames` boundaries, not a global retiming pass. For the natural no-return-IK pipeline, smooth the generated motion first, then apply stroke-only line IK with `--blend_frames 0` so the return is not IK-constrained:

   ```powershell
   python scripts\postprocess_prompt_transition_smooth.py `
     --motion logs\wall_brush_g1_v37\sample_11_motion.npz `
     --prompts logs\wall_brush_g1_v37\prompts.json `
     --output logs\wall_brush_g1_v37\sample_11_transition_smooth_r10_motion.npz `
     --radius 10 `
     --kernel_size 11 `
     --sigma 2.5 `
     --strength 0.9 `
     --passes 2

   python scripts\postprocess_wall_brush_line_ik.py `
     --motion logs\wall_brush_g1_v37\sample_11_transition_smooth_r10_motion.npz `
     --metrics logs\wall_brush_g1_v37\metrics.json `
     --output logs\wall_brush_g1_v37\sample_11_smooth_r10_line_ik_b0_motion.npz `
     --blend_frames 0 `
     --iterations 36
   ```

   Current reference metrics for `sample_11_smooth_r10_line_ik_b0_motion.npz`: row/keyframe error about `1e-5m`, final tail yaw delta/path about `8.0/23.3deg`, final right-hand position about `[-0.0415, 0.7028, 0.1715]`, tail max right-hand step `0.032m/frame`, root tail drift `0.050m`. This resolves the visible final turn without applying return IK. A `--blend_frames 2` line-IK variant can reduce max all-joint step further (`~0.081m/frame`) but it touches the first two frames after each row; use it only if the user accepts that tiny row-exit blend.

16. Older transition smoothing for the six-phase line-IK pipeline:

   ```powershell
   python scripts\postprocess_prompt_transition_smooth.py `
     --motion logs\wall_brush_g1_v30\sample_15_line_ik_motion.npz `
     --prompts logs\wall_brush_g1_v30\prompts.json `
     --output logs\wall_brush_g1_v30\sample_15_transition_smooth_v3_motion.npz `
     --radius 18 `
     --kernel_size 17 `
     --sigma 4.0 `
     --strength 1.0 `
     --passes 4

   python scripts\postprocess_wall_brush_line_ik.py `
     --motion logs\wall_brush_g1_v30\sample_15_transition_smooth_v3_motion.npz `
     --metrics logs\wall_brush_g1_v30\metrics.json `
     --output logs\wall_brush_g1_v30\sample_15_transition_smooth_v3_line_ik_motion.npz `
     --blend_frames 10 `
     --iterations 36
   ```

   The second line-IK pass restores the right-hand row precision after the all-joint smoothing pass. This filter reduces switch-local jitter and makes the return less abrupt, but it is not a semantic no-turn constraint; if the generated tail contains a slow body turn, the filter can attenuate it but not fully remove it.

17. Treat the no-turn return pass as a diagnostic or hard-reference fallback, not the preferred natural solution. This pass uses the correct Kimodo ground plane convention (Y-up, XZ horizontal), locks the tail body's XZ lateral yaw to the final row frame, optionally holds root XZ in place, and solves only the right arm to place the hand back beside the torso:

   ```powershell
   python scripts\postprocess_wall_brush_no_turn_return.py `
     --motion logs\wall_brush_g1_v30\sample_15_transition_smooth_v3_line_ik_motion.npz `
     --metrics logs\wall_brush_g1_v30\metrics.json `
     --output logs\wall_brush_g1_v30\sample_15_no_turn_return_v4_motion.npz `
     --return_frames 42 `
     --root_hold_blend_frames 18 `
     --root_hold_strength 1.0 `
     --rest_reference_frame 0 `
     --iterations 96 `
     --right_arm_z_max 0.32
   ```

   Use this only when the completion criterion explicitly allows a constrained return. It is a constraint/postprocess solution, not a prompt solution, and it can make the final return look less natural. If the user says "I only want the right hand to retract; do not use IK for putting it back," use the 5-phase `gentle_return_5` plus transition smoothing and stroke-only line IK instead.

Note: line IK currently adjusts `posed_joints` for visual/trajectory precision. If the result must be deployed as G1 motor commands, add a follow-up retarget/refit step to synchronize rotations or qpos.

## Validation

Pass only if:

- max constrained hand keyframe error is `<= 0.05 m`
- max wall-plane error at constrained keyframes is `<= 0.05 m`
- max row-line distance during brushing intervals is `<= 0.06 m` for generated base motion
- max row-line distance is near zero after line IK for "almost straight row" requests
- max path length ratio is low enough to rule out circular motion, preferably `<= 1.25`
- root horizontal drift is acceptable for an in-place task, preferably around `0.11 m` or lower
- for natural right-hand-only tasks, left-arm/body motion is allowed, but should be passive, coordinated, and free of sudden peaks; inspect max and p95 left-arm step rather than requiring zero
- for strict calm/frozen references only, left arm max step should be effectively zero, preferably `<= 0.0001 m/frame`
- max right-arm wall penetration is near zero, preferably `<= 0.0005 m`

Report:

- best sample id
- max and mean keyframe error
- max wall-plane error
- max and mean row-line error
- max path length ratio and backtrack ratio
- root horizontal and vertical drift
- max right-hand step and acceleration
- max left-arm step
- per-row line and plane metrics

## Known Good Reference

Natural no-return-IK reference for "straight rows, then simply retract the right hand":

- remote: `/root/autodl-tmp/wall_brush_runs/g1_v37_gentle_return5_120_20260505_001`
- local base motion: `logs\wall_brush_g1_v37\sample_11_motion.npz`
- local preferred motion: `logs\wall_brush_g1_v37\sample_11_smooth_r10_line_ik_b0_motion.npz`
- optional lower max-step variant: `logs\wall_brush_g1_v37\sample_11_smooth_r10_line_ik_b2_motion.npz`
- prompt mode: `gentle_return_5`
- durations: `[0.8, 1.2, 1.2, 1.2, 1.2]`
- base tail metrics: final tail yaw delta/path `8.0/29.9deg`, final right hand `[-0.0415, 0.7028, 0.1715]`, tail max step `0.032m/frame`, root tail drift `0.054m`
- preferred postprocess: transition smoothing `radius=10`, `kernel_size=11`, `sigma=2.5`, `strength=0.9`, `passes=2`, then line IK with `blend_frames=0`
- preferred postprocess metrics: keyframe/row error about `1e-5m`, final tail yaw delta/path `8.0/23.3deg`, tail path ratio `1.87`, tail max step `0.032m/frame`, root tail drift `0.050m`, right-hand max step `0.071m/frame`
- important constraint: the return segment is not IK-constrained; line IK is applied to stroke frames only.

Strict straight-row reference:

- remote: `/root/autodl-tmp/wall_brush_runs/g1_v15_row_rigid_straight_pass_20260504_001`
- local base motion: `logs\wall_brush_g1_v15\sample_06_motion.npz`
- local line-IK motion: `logs\wall_brush_g1_v15\sample_06_line_ik_motion.npz`
- prompt mode: `straight`
- calibration mode: `row_rigid`
- best sample: `sample_06`
- base max keyframe error: `0.046585 m`
- base max wall-plane error: `0.034723 m`
- base max row-line error: `0.055035 m`
- base max path length ratio: `1.237201`
- base root horizontal drift: `0.107842 m`
- line-IK max keyframe error: approximately `0.00001 m`
- line-IK max row-line error: approximately `0.00001 m`
- line-IK root drift: unchanged from base motion
- temporal smooth line-IK motion: `logs\wall_brush_g1_v15\sample_06_line_ik_smooth_v2_motion.npz`
- temporal smooth line-IK row-to-row transition hand step: about `0.0086 m`
- temporal smooth line-IK row-to-row transition arm-joint step: about `0.0283 m`

Non-penetrating generation-stage reference:

- remote: `/root/autodl-tmp/wall_brush_runs/g1_v21_straight_z_clamp_20260505_001`
- local base motion: `logs\wall_brush_g1_v21\sample_13_motion.npz`
- prompt mode: `straight`
- wrist outside offset: `-0.12`
- constraint z clamp: `0.32`
- generated base max wall penetration: `0.0 m`
- generated base max right-arm wall penetration: `0.0 m`
- generated base max keyframe error: about `0.0688 m`
- conclusion: the generator can avoid penetration with the right constraint design, but still needs line IK/bookend smoothing for precision and stable deployment.

Natural-coordinated reference preferred by the user:

- remote: `/root/autodl-tmp/wall_brush_runs/g1_v21_straight_z_clamp_20260505_001`
- local base motion: `logs\wall_brush_g1_v21\sample_13_motion.npz`
- local natural line-IK motion: `logs\wall_brush_g1_v21\sample_13_line_ik_motion.npz`
- prompt mode: `straight`
- prompt lesson: do not add explicit left-arm/body role instructions; the short `straight horizontal stroke ... on a wall` prompt keeps the best Kimodo full-body prior
- max keyframe error after line IK: about `0.000007 m`
- max row-line error after line IK: about `0.000008 m`
- max wall penetration after line IK: about `0.000005 m`
- max right-arm wall penetration after line IK: about `0.0868 m`
- max right-hand step: about `0.0946 m/frame`
- max right-hand acceleration: about `0.0576 m/frame^2`
- interpretation: this is the closest current style match when the right hand performs the task and body/left arm remain natural rather than frozen, but it is not deployment-clean because the right arm can still pass through the wall even when the right-hand endpoint stays on the plane

Prompt-repaired final return reference:

- remote: `/root/autodl-tmp/wall_brush_runs/g1_v30_straight_facing_return_20260505_001`
- local base motion: `logs\wall_brush_g1_v30\sample_15_motion.npz`
- local final motion: `logs\wall_brush_g1_v30\sample_15_line_ik_motion.npz`
- prompt mode: `straight_facing_return`
- prompt lesson: keep row phases as `slides the right hand in a straight horizontal stroke...`, but use finish phases `A person remains facing the wall and lowers the right hand down beside the torso.` and `A person remains facing the wall with both arms resting at the sides.`
- max keyframe error after line IK: about `0.000379 m`
- max row-line error after line IK: about `0.000273 m`
- max right-arm wall penetration after line IK: about `0.0170 m`
- max right-hand step: about `0.0737 m/frame`
- max left-arm step: about `0.0626 m/frame`
- tail after final row: root yaw delta about `-5 deg`, root yaw path about `27.8 deg`, and right-hand tail path ratio about `2.40`
- interpretation: this is the best prompt-level fix so far for the unnecessary final turn. It is not a hardcoded return trajectory; the remaining tail path is still generated by Kimodo.

Prompt-transition smoothed reference:

- source motion: `logs\wall_brush_g1_v30\sample_15_line_ik_motion.npz`
- local filtered motion: `logs\wall_brush_g1_v30\sample_15_transition_smooth_v3_line_ik_motion.npz`
- postprocessor: `scripts\postprocess_prompt_transition_smooth.py`
- prompt boundary frames from `num_frames`: `[24, 60, 96, 132, 168]`
- smoothing parameters: `radius=18`, `kernel_size=17`, `sigma=4.0`, `strength=1.0`, `passes=4`
- max row-line error after the final line-IK pass: about `0.000134 m`
- max right-arm wall penetration: about `0.000001 m`
- max right-hand step: about `0.0432 m/frame`
- max all-joint step: about `0.0432 m/frame`
- boundary-window max joint step: reduced from about `0.0737 m/frame` to `0.0432 m/frame`
- boundary-window max joint acceleration: reduced from about `0.1178 m/frame^2` to about `0.0300 m/frame^2`
- tail yaw path after final row: reduced from about `27.8 deg` to about `15.4 deg`
- interpretation: this is the current best smoothed visual trajectory when the user complains about prompt-switch jitter. It reduces but does not semantically forbid a slow final turn.

No-turn final return reference:

- source motion: `logs\wall_brush_g1_v30\sample_15_transition_smooth_v3_line_ik_motion.npz`
- local final motion: `logs\wall_brush_g1_v30\sample_15_no_turn_return_v4_motion.npz`
- postprocessor: `scripts\postprocess_wall_brush_no_turn_return.py`
- important correction: measure body turn on the XZ ground plane, not XY. Earlier yaw numbers based on XY under-reported the visual turn.
- tail start frame: `127`
- tail XZ yaw delta: about `67.7 deg -> 0.0 deg`
- tail XZ yaw path: about `103.3 deg -> 0.0 deg`
- right-hand tail path ratio: about `1.81 -> 1.02`
- right-hand tail max step: about `0.0371 m/frame -> 0.0184 m/frame`
- root tail displacement: about `0.1266 m -> 0.0140 m`
- max row-line error: about `0.000134 m`
- max right-arm wall penetration: about `0.000001 m`
- max right-hand step over whole motion: about `0.0432 m/frame`
- interpretation: this is the first version that actually removes the final body turn while keeping straight rows and smooth right-hand return. For deployment, it still needs rotation/qpos refit because the local postprocessors edit `posed_joints`.

Prompt-only ablation results, all using the same geometry and right-hand constraints before line IK:

- `coordinated_right`: phrases like `small natural torso and shoulder motion` made body/root motion too active and the generated hand path more loopy.
- `balanced_right`: phrases like `left arm relaxed by the side` weakened right-hand task compliance.
- `working_right`: phrases like `working right hand` and `non-working left hand` increased non-task motion and did not improve precision.
- `straight_right_only`: adding `only the right hand` was closest, but still produced worse non-row right-hand jumps than the original `straight` prompt.
- `smooth_straight`: adding `calmly`, `smooth`, and `one smooth straight horizontal stroke` did not make the generated reference smoother; after line IK, non-row max right-hand step was about `0.164 m/frame` and max right-hand acceleration was about `0.153 m/frame^2`.
- `seed_wall_return`: full SEED-style row wording plus natural arms-at-sides finish reduced right-arm wall penetration compared with the original line-IK reference, but weakened base right-hand compliance. After line IK, max right-hand step was about `0.113 m/frame`, max left-arm step about `0.100 m/frame`, and max right-arm wall penetration about `0.045 m`.
- `seed_wall_lean`: SEED-style forward-lean wall wording made the post-IK right-arm penetration nearly zero, but introduced a large non-row right-hand jump. After line IK, max right-hand step was about `0.187 m/frame` and max left-arm step about `0.148 m/frame`.
- `straight_seed_return`: hybrid prompt with original row phrases and SEED-style finish came closest to the original reference, but still did not improve it. After line IK, max right-hand step was about `0.097 m/frame`, max right-hand acceleration about `0.109 m/frame^2`, max left-arm step about `0.090 m/frame`, and max right-arm wall penetration about `0.092 m`.
- `straight_facing_return`: original straight row wording plus `remains facing the wall` return text reduced the unnecessary final turn. The best tail-selected sample was `logs\wall_brush_g1_v30\sample_15_line_ik_motion.npz`: root yaw delta after the final row about `-5 deg`, tail right-hand path ratio about `2.40`, max right-hand step about `0.074 m/frame`, and max right-arm wall penetration about `0.017 m`.
- `planted_facing_return`: adding `keeps both feet planted` and `torso facing the wall` did not improve the return; it often weakened row compliance or left the hand high. Use it only as a diagnostic, not the default.
- conclusion: for final-turn problems, prefer `straight_facing_return` plus tail-based sample selection. For row precision, still use right-arm line IK while preserving Kimodo's natural full-body prior.

Bookend-smoothed reference for accurate rows plus smooth approach/exit:

- local source motion: `logs\wall_brush_g1_v18\sample_13_line_ik_motion.npz`
- local final motion: `logs\wall_brush_g1_v18\sample_13_line_ik_bookend_v10_motion.npz`
- max keyframe error: about `0.000009 m`
- max row-line error: about `0.000009 m`
- pre-contact all-joint max step: about `0.0246 m/frame`
- post-contact all-joint max step: about `0.0114 m/frame`
- whole-motion all-joint max step: about `0.0317 m/frame`
- ending rest point: `[0.12, 0.66, 0.14]`, chosen to lower the hand on the same side instead of turning back to the original generated final pose

Non-penetrating final reference:

- local base motion: `logs\wall_brush_g1_v21\sample_13_motion.npz`
- local final motion: `logs\wall_brush_g1_v21\sample_13_line_ik_bookend_v1_motion.npz`
- max keyframe error: about `0.000010 m`
- max row-line error: about `0.000010 m`
- max wall penetration: about `0.000002 m`
- max right-arm wall penetration: about `0.000008 m`
- max right-hand step: about `0.0195 m/frame`
- max all-joint step: about `0.0300 m/frame`

Strict calm/frozen technical reference:

- local base motion: `logs\wall_brush_g1_v21\sample_13_motion.npz`
- local final motion: `logs\wall_brush_g1_v21\sample_13_taskline_calm_v1_motion.npz`
- postprocessor: `scripts\postprocess_wall_brush_taskline_calm.py`
- body mode: `hold_initial`
- max keyframe error: about `0.000010 m`
- max row-line error: about `0.000009 m`
- max wall penetration: about `0.000006 m`
- max right-arm wall penetration: about `0.000006 m`
- max path length ratio: about `0.99998`
- max right-hand step: about `0.0195 m/frame`
- max all-joint step: about `0.0195 m/frame`
- max left-arm step: `0.0 m/frame`
- interpretation: use this reference when diagnosing or preparing a minimal controller-friendly trajectory. Do not use it as the default answer when the user asks for natural, coordinated body motion.

Older loose reference, kept only as a regression baseline:

- remote: `/root/autodl-tmp/wall_brush_runs/g1_v9_resume_v8_iter2_low_gain_20260504_001`
- local: `logs\wall_brush_g1_v9`
- best sample: `sample_04`
- max keyframe error: `0.048616 m`
- max wall-plane error: `0.036529 m`
- max row-line error: `0.060595 m`

Do not treat the older reference as sufficient for "almost straight rows"; it used higher/farther targets and can show large full-body compensation.

## V33 No-IK Brush-Likeness Route

Use this route when the user explicitly rejects right-arm FABRIK, line IK, return IK, or root/torso hard locking. Keep the main route as multi-prompt plus Kimodo generation-time native constraints. Transition filtering is allowed because it is temporal smoothing, not a geometric IK correction.

Current script set:

- generator: `scripts\remote_wall_brush_multiprompt_constraint_v33.py`
- scorer/reranker: `scripts\score_wall_brush_mp_constraint_native.py`
- rank01/rank10 diagnostic: `scripts\compare_wall_brush_v33_rerank_samples.py`
- main output: `logs\wall_brush_g1_mp_constraint_v33_brush_motion`
- rerank-only output: `logs\wall_brush_g1_mp_constraint_v33_rerank`

Prompt structure:

- use 5 complete multi-prompt phases over 210 frames
- boundaries: `[36, 78, 120, 162]`
- stroke keyframes delayed after each boundary: row frames `[46,54,62,70,76]`, `[88,96,104,112,118]`, `[130,138,146,154,160]`
- keep the neutral positive prompt `stands still facing a wall`
- do not use negative phrases such as `do not turn`, `no turning`, or `without turning`
- the `full_width` prompt variant improves brush-likeness by saying the hand starts at the left/right edge and moves across the full width of the patch

Constraint structure:

- use position-only `right_hand_endpoint` constraints for all 15 stroke points
- use sparse position-only `right_wrist` constraints only at row endpoints, 6 stroke wrist points total
- do not constrain wrist/hand rotations
- do not run line IK, FABRIK, return IK, or right-arm chain edits after generation
- keep transition filtering boundary-local and constraint-aware

Brush-likeness reranking:

- old composite scores over-rewarded low line error and low yaw. This selected samples where the hand stayed near the target line but barely swept across the patch.
- add `row*_x_progress`, `row*_x_coverage`, `avg_x_progress`, `avg_x_coverage`, `min_x_progress`, and `dead_stroke_count`.
- a row is a dead stroke if `row_x_progress < 0.35`.
- keep two leaderboards: `best_no_turn` for calm/no-turn samples and `best_brush_motion` for visible horizontal brushing. They may be different samples.

Important v32 diagnostic:

- v32 automatic best: line mean about `0.043 m`, avg progress about `0.429`, avg coverage about `0.488`, dead strokes `1`, brush score about `10.36`. It scored well because it stayed near the desired line but did not move enough laterally.
- v32 visual rank10: line mean about `0.046 m`, avg progress about `0.651`, avg coverage about `0.681`, dead strokes `0`, brush score about `3.09`. It looks more like brushing because the right hand actually sweeps across the wall patch.

Current v33 findings:

- best brush-motion sample: `logs\wall_brush_g1_mp_constraint_v33_brush_motion\gallery_selected\V33_best_brush_motion_full_width_v33C_filtered_line0p042_prog0p803_cov0p857_pen0p025.npz`
- config: `full_width`, `v33_C`, `x_scale=1.9`, `x_offset=0.04`, `y_offset=-0.08`, `z_offset=0.03`, `z_wrist=0.24`, CFG `[2.2, 3.6]`, 200 denoising steps
- metrics: line mean about `0.042 m`, line max about `0.079 m`, avg progress about `0.803`, avg coverage about `0.857`, dead strokes `0`, tail yaw path about `8.7 deg`, naturalness penalty `0`
- limitation: wall penetration is about `0.025 m`, so it is not deployment-clean even though it is the best visible no-IK brushing motion.
- best no-turn sample: `logs\wall_brush_g1_mp_constraint_v33_brush_motion\gallery_selected\V33_best_no_turn_v33B_filtered_line0p028_prog0p436_dead2.npz`
- limitation: it has low line error and low yaw but two dead strokes, so do not treat it as the best brush-motion sample.
- safe-wall native candidate: `logs\wall_brush_g1_mp_constraint_v33_brush_motion\gallery_selected\V33_safe_wall_candidate_not_accepted_v33E_line0p082_prog0p656_pen0p003_yawpath40p3.npz`
- limitation: it keeps penetration near zero and has no dead strokes, but line error, line max, and yaw path miss the current acceptance thresholds.

Conclusion for no-IK native constraints:

- Multi-prompt plus native position-only endpoint/wrist constraints can produce natural, visible row-wise brushing without the line-IK arm twist.
- In the current v33 search it has not simultaneously achieved all strict acceptance targets: visible lateral coverage, line mean under `0.06 m`, line max under `0.11 m`, wall penetration under `0.01 m`, and yaw path under about `35 deg`.
- The next technical route should be DNO-lite or soft full-body projection with a strong pose prior, not a return to right-arm-only FABRIK/line IK.

## V34 Flat Brush Native Route

Use this route when the user likes the v32/v33 no-IK brush style but wants each row to be visibly flatter. Keep the same no-IK contract: no FABRIK, no line IK, no return IK, no post root/torso lock, and no right-arm chain editing after generation.

Current script set:

- generator: `scripts\remote_wall_brush_multiprompt_constraint_v34_flat.py`
- scorer/reranker: `scripts\score_wall_brush_mp_constraint_native.py`
- v33 flatness diagnostic: `scripts\diagnose_wall_brush_v34_flatness.py`
- output: `logs\wall_brush_g1_mp_constraint_v34_flat_brush`
- gallery: `logs\kimodo_motion_gallery.html`

V33 flatness diagnosis:

- v33 best brush-motion sample looks like brushing because avg progress is about `0.807` and avg coverage is about `0.860`.
- Its row height is not flat: row y-range mean is about `0.106 m`, row y-range max about `0.124 m`.
- v33 best no-turn is calmer and flatter, with row y-range mean about `0.049 m`, but brush motion is weak: avg progress about `0.436`, avg coverage about `0.520`.

V34 generation design:

- use 210 frames with boundaries `[36, 78, 120, 162]`
- use the `level_full_width` multi-prompt variant, explicitly saying the right hand moves `level across the full width` and `at the same height`
- use 18 endpoint constraints, 6 per row, with row frames `[46,52,58,64,70,76]`, `[88,94,100,106,112,118]`, and `[130,136,142,148,154,160]`
- keep wrist constraints sparse, only at row endpoints, with position-only wrist clearance and no rotation constraints
- use closed-loop y pre-emphasis from the v33 best brush sample: `correction_i = clamp(-0.7 * (actual_y_i - desired_y_i), -0.06, 0.06)`
- keep x compensation from the v33 brush route: `x_for_constraint = x_offset + x_scale * desired_x`
- use flat presets `flat_A` through `flat_D`; the best v34 candidates all came from `flat_B`
- use flatness-aware transition filtering: during stroke active windows, do not filter the right-arm chain

V34 scoring:

- keep the v33 `best_no_turn` and `best_brush_motion` leaderboards
- add `best_flat_brush_motion`
- add row flatness metrics: row y-range, row y-std, endpoint y-delta, y-slope, and y-curvature
- the hard flat-brush target is no dead strokes, avg progress/coverage at least `0.65`, row y-range mean at most `0.040 m`, row y-range max at most `0.055 m`, row y-curvature mean at most `0.025 m`, wall penetration at most `0.010 m`, and right-arm naturalness penalty `0`

Current v34 findings:

- best flat-brush candidate: `logs\wall_brush_g1_mp_constraint_v34_flat_brush\endpoint_wrist_none_plan210_level_full_width_preflat_B_x1p9_xo0p04_ym0p07_z0p03_zw0p24_cfg2_t2p4_c4p0_steps200\transition_filtered\sample_01_transition_filtered.npz`
- config: `flat_B`, `level_full_width`, `x_scale=1.9`, `x_offset=0.04`, `base_y_offset=-0.07`, `z_offset=0.03`, `z_wrist=0.24`, CFG `[2.4, 4.0]`, 200 steps
- metrics: line mean about `0.044 m`, line max about `0.127 m`, avg progress about `0.670`, avg coverage about `0.699`, dead strokes `0`, row y-range mean about `0.056 m`, row y-range max about `0.078 m`, row endpoint y-delta mean about `0.023 m`, row y-curvature mean about `0.040 m`, tail yaw path about `24.8 deg`, naturalness penalty `0`
- it improves flatness compared with v33 best brush, but it fails strict flat-brush acceptance because row y-range, row y-curvature, line max, and wall penetration remain too high.
- best v34 brush-motion sample: `logs\wall_brush_g1_mp_constraint_v34_flat_brush\endpoint_wrist_none_plan210_level_full_width_preflat_B_x1p9_xo0p04_ym0p07_z0p03_zw0p24_cfg0_t2p2_c3p6_steps200\transition_filtered\sample_13_transition_filtered.npz`
- limitation: it has strong brush-likeness, but row y-range mean is still about `0.104 m`, so it is not a flat brush.
- best v34 no-turn sample: `logs\wall_brush_g1_mp_constraint_v34_flat_brush\endpoint_wrist_none_plan210_level_full_width_preflat_B_x1p9_xo0p04_ym0p07_z0p03_zw0p24_cfg2_t2p4_c4p0_steps200\transition_filtered\sample_13_transition_filtered.npz`
- limitation: it has low yaw path, but it is not the best flat or best brush sample.

Conclusion for flat native constraints:

- V34 native constraints found a useful tradeoff sample, but did not satisfy the strict flat-brush acceptance criteria.
- The best v34 flat sample is better than v33 best brush on row height drift, but not flat enough and still has wall penetration.
- The next route should be `v35_dno_lite_flat_brush`: optimize diffusion guidance or initial noise with row y-deviation, y-slope, y-curvature, x progress, x coverage, wall contact, smoothness, root-heading stability, and distance-to-current-natural-motion losses.
- Do not respond to this failure by returning to right-arm-only FABRIK, line IK, or return IK.

## V35 DNO-Lite / Noise-Search Route

Use this section after V34 when native prompt/constraint tuning is no longer enough. The first V35 implementation uses score-based initial-noise search, not gradient DNO, because the public Kimodo API does not expose saved initial noise or DDIM inversion and the default denoising step is wrapped in `torch.inference_mode()`.

Former default natural-style choice:

- former default sample: `logs\wall_brush_g1_v36_soft_full_body_projection\seeds\seed_C_v35_best_natural_flat.npz`
- former gallery default copy: `logs\wall_brush_g1_v35_dno_lite\gallery_selected\DEFAULT_seed_C_v35_best_natural_flat_noise_search_natural_style.npz`
- keep this as a subjective natural-style baseline, but do not use it as the current default after the user selected the V36 continuity-filtered projection as better.

Version check:

- local checkout `kimodo_fork_work2` includes upstream multi-prompt transition fixes `c6c8ba7` / `191af10`
- AutoDL package is installed as `kimodo 1.0.0` from site-packages without a git SHA, but it contains the improved `num_transition_frames` transition handling in `kimodo/model/kimodo_model.py`
- details: `logs\wall_brush_g1_v35_dno_lite\version_check.txt`

Seed diagnostics:

- seed A: v32 rank10 subjective best, `logs\wall_brush_g1_mp_constraint_v32_210_delayed_preemphasis\top_10_filtered\rank10_NO_IK_multi-prompt_endpoint_wrist_none_B_zw0p24_transition_filtered_score18.95.npz`
- seed A metrics: line mean about `0.046 m`, line max about `0.084 m`, avg progress about `0.651`, avg coverage about `0.681`, row y-range mean about `0.088 m`, row y-range max about `0.138 m`, wall penetration about `0.054 m`, yaw path about `28.2 deg`
- seed B: v34 flat_B sample01, `logs\wall_brush_g1_mp_constraint_v34_flat_brush\endpoint_wrist_none_plan210_level_full_width_preflat_B_x1p9_xo0p04_ym0p07_z0p03_zw0p24_cfg2_t2p4_c4p0_steps200\transition_filtered\sample_01_transition_filtered.npz`
- seed B metrics: line mean about `0.044 m`, line max about `0.127 m`, avg progress about `0.670`, avg coverage about `0.699`, row y-range mean about `0.056 m`, row y-range max about `0.078 m`, wall penetration about `0.061 m`, yaw path about `24.8 deg`
- diagnostics CSV: `logs\wall_brush_g1_v35_dno_lite\seed_diagnostics.csv`

Noise-search implementation:

- remote script: `scripts\remote_wall_brush_v35_noise_search.py`
- local scorer: `scripts\score_wall_brush_v35_dno_lite.py`
- output: `logs\wall_brush_g1_v35_dno_lite`
- method: reconstruct the selected seed sample's initial-noise slice from the original run seed and sample index, then run population-based search around that noise
- parameters used: population `24`, iterations `6`, sigma start `0.08`, sigma decay `0.65`, top_k `6`
- this is generation-space optimization: every candidate is regenerated through Kimodo denoising and multi-prompt stitching
- it does not use right-arm FABRIK, line IK, return IK, right-arm chain edits, or post root/torso locking

Current V35 findings:

- best natural flat sample: `logs\wall_brush_g1_v35_dno_lite\seed_B_v34_flatB_sample01_best_center\transition_filtered\sample_00_transition_filtered.npz`
- method: `noise-search`
- seed: `seed_B_v34_flatB_sample01`
- metrics: line mean about `0.042 m`, line max about `0.112 m`, avg progress about `0.667`, avg coverage about `0.688`, dead strokes `0`, row y-range mean about `0.051 m`, row y-range max about `0.064 m`, row y-curvature mean about `0.035 m`, wall penetration about `0.059 m`, yaw path about `23.1 deg`, seed preservation about `0.032`, naturalness penalty `0`
- improvement over V34 best flat: flat score improved from about `10.67` to about `7.34`, row y-range mean improved from about `0.056 m` to about `0.051 m`, and line max improved from about `0.127 m` to about `0.112 m`
- limitation: it still fails strict acceptance because avg progress/coverage are slightly below `0.70`, row y-range and curvature are above target, wall penetration remains far above `0.010 m`, and yaw path is above `20 deg`

Conclusion for V35:

- V35 noise-search is a useful incremental improvement but not a final deployable flat-brush solution.
- Seed B is the better optimization base for flat brushing. Seed A preserves a more brush-like visual style but starts with too much y drift, especially on the lower row.
- The persistent wall penetration likely comes from the seed/config family and generation-time wall contact targets; pure noise search cannot reliably overcome that strong prior.
- If the user wants to continue beyond V35, the next route should be either true gradient DNO with a patched differentiable sampler and saved/inverted noise, or `SOFT_FULL_BODY_PROJECTION` with strong pose prior and explicit wall-penetration loss. Keep it clearly labeled; do not call it no-IK.

## V36 Soft Full-Body Projection Route

Use this route only after V35 when the user accepts that the next step is no longer `NO IK`. Label every output as `SOFT_FULL_BODY_PROJECTION`. This route is not right-arm FABRIK, not line IK, not return IK, and not a hard root/torso lock.

Current script set:

- remote optimizer: `scripts\remote_wall_brush_v36_soft_projection.py`
- local diagnostics: `scripts\diagnose_wall_brush_v36_penetration.py`
- local scorer/reranker: `scripts\score_wall_brush_v36_soft_projection.py`
- output: `logs\wall_brush_g1_v36_soft_full_body_projection`
- summary: `logs\wall_brush_g1_v36_soft_full_body_projection\summary.csv`
- gallery: `logs\kimodo_motion_gallery.html`

Seed selection:

- seed A: v32 rank10 subjective best, strong brush feel but loose flatness and wall safety.
- seed B: v34 flat_B sample01, flatter native-constraint seed but wall penetration and yaw remain.
- seed C: v35 best natural flat brush, best optimization base for V36.
- seed list: `logs\wall_brush_g1_v36_soft_full_body_projection\seed_list.txt`

Wall penetration diagnosis:

- diagnostics file: `logs\wall_brush_g1_v36_soft_full_body_projection\wall_penetration_diagnostics.csv`
- wall convention: wall plane `z = 0.32`; safe outside is `z <= 0.32`; penetration is `z > 0.32`
- for all three seeds, the dominant penetration source is the right-hand endpoint
- wrist, elbow, forearm capsule, and upper-arm capsule penetration are `0` in the initial diagnostics with capsule radius `0.015 m`

Projection design:

- optimize local rotations and small root variables through G1 forward kinematics, not global joint positions
- variables: small root XZ translation, small root yaw delta, torso joints, right shoulder joints, right elbow, right wrist, and right hand roll
- no left-arm optimization by default
- stage 1 wall safety: pull right hand/wrist/right-arm capsules out of the wall with strong pose prior
- stage 2 flat brush: add row y-deviation, y-slope, y-curvature, x progress, x coverage, x bounds, wall contact, smoothness, naturalness, and seed-preservation losses
- approach frames `20-45` are included only for wall safety because seed C had a pre-stroke hand penetration at frame `39`
- return frames use only a light root-yaw soft guidance target, clipped to `4 deg`, to reduce tail yaw without hard-locking root or changing the return with IK

Joint delta limits:

- root XZ max in final best: about `0.010 m`
- root yaw max in final best: about `2.85 deg`
- torso max in final best: about `3.48 deg`
- shoulder max in final best: about `5.75 deg`
- elbow max in final best: about `8.89 deg`
- wrist max in final best: about `8.00 deg`
- joint delta limit violation: `0`

Current V36 best:

- best natural flat sample: `logs\wall_brush_g1_v36_soft_full_body_projection\seed_C_final_smoothed_result\sample_00\motion.npz`
- gallery-selected copy: `logs\wall_brush_g1_v36_soft_full_body_projection\gallery_selected\v36_best_natural_flat_SOFT_FULL_BODY_PROJECTION_seed_C_final_smoothed_result_line0p006-0p016_xp0p866_xc0p866_yr0p001_yc0p000_pen0p001_cap0p000_yaw19p511_sp0p061.npz`
- method: `SOFT_FULL_BODY_PROJECTION`
- seed: seed C, the V35 best natural flat brush
- metrics: line mean about `0.0064 m`, line max about `0.0156 m`, avg progress about `0.866`, avg coverage about `0.866`, dead strokes `0`
- flatness: row y-range mean about `0.00057 m`, row y-range max about `0.00073 m`, row y-curvature mean about `0.00037 m`
- wall safety: whole-motion endpoint penetration about `0.0010 m`, forearm capsule penetration `0`, upper-arm capsule penetration `0`
- tail root yaw path: about `19.5 deg`
- right-arm naturalness penalty: `0`
- final hand distance to seed neutral: about `0.005 m`

Current default generation method:

- full reproducible pipeline spec: `skills\kimodo-wall-brush\default_wall_brush_pipeline.json`
- pipeline runner: `scripts\run_wall_brush_default_pipeline.py`
- default sample: `logs\wall_brush_g1_v36_soft_full_body_projection\seed_C_final_edge_plus_continuity_lam5p0_result\sample_00\motion.npz`
- gallery default copy: `logs\wall_brush_g1_v36_soft_full_body_projection\gallery_selected\DEFAULT_wall_brush_v36_SFBP_CONT_edge_plus_lam5_best_visual.npz`
- method label: `SOFT_FULL_BODY_PROJECTION_TEMPORAL_CONTINUITY_FILTERED`
- recipe: task text -> task recognition -> `task_spec.json` -> V34 multi-prompt text + endpoint/wrist position-only native constraints -> score/filter current run and select seed_B -> V35 noise-search around current seed_B to create current seed_C -> V36 soft full-body projection only on current seed_C -> `edge_plus` smoothing -> temporal continuity smoothing with `lambda=5.0` and keyframe weight `40.0`.
- do not reuse an old `seed_C`, old best npz, or old y-reference motion when the wall position/constraints/task spec changes.
- seed_B and seed_C are intermediate artifacts produced inside the current run. For a different but structurally similar wall-brush task, change the task text or wall/constraint parameters and rerun the same stages from scratch.
- the generic V36 research script can still run seed A/B/C for comparison, but the current from-scratch default pipeline must use V35's current-run seed_C and pass `--only_seed seed_C` to the V36 stage.
- print the full command sequence with:

  ```powershell
  python scripts\run_wall_brush_default_pipeline.py --task_text "一个人在刷墙" --write_commands logs\wall_brush_default_pipeline\commands.txt
  ```
- execute the full command sequence on the GPU/Kimodo environment with:

  ```powershell
  python scripts\run_wall_brush_default_pipeline.py --task_text "一个人在刷墙" --execute
  ```
- reproduce the projection stage with:

  ```powershell
  python scripts\remote_wall_brush_v36_soft_projection.py `
    --output_dir logs\wall_brush_g1_v36_soft_full_body_projection `
    --seed_a logs\wall_brush_g1_v36_soft_full_body_projection\seeds\seed_A_v32_rank10.npz `
    --seed_b logs\wall_brush_g1_v36_soft_full_body_projection\seeds\seed_B_v34_flatB_sample01.npz `
    --seed_c logs\wall_brush_g1_v36_soft_full_body_projection\seeds\seed_C_v35_best_natural_flat.npz `
    --only_seed seed_C
  ```
- reproduce the post-filter from the projected sample with:

  ```powershell
  python scripts\postprocess_wall_brush_v36_edge_filter.py `
    --source logs\wall_brush_g1_v36_soft_full_body_projection\seed_C_final_smoothed_result\sample_00\motion.npz `
    --output logs\wall_brush_g1_v36_soft_full_body_projection\seed_C_final_edge_plus_continuity_lam5p0_result\sample_00\motion.npz `
    --profile edge_plus `
    --strength 1.0 `
    --continuity_lambda 5.0 `
    --continuity_key_weight 40.0
  ```
- metrics: line mean/max about `0.0064 / 0.0149 m`, avg x progress/coverage about `0.811 / 0.811`, row y-range mean about `0.0021 m`, wall penetration `0`, max right-hand acceleration about `0.025 m/frame^2`, tail root yaw path about `12.1 deg`.
- use this as the current default when asked for the best wall-brush action, unless the user specifically asks for no-IK/native-only or for the older V35 subjective baseline.

Conclusion for V36:

- V36 is substantially closer to final usable than V35: it fixes wall penetration, flatness, line error, and brush coverage while preserving the seed's natural style.
- The main wall penetration source was the right-hand endpoint, not wrist, forearm, or upper arm.
- Seed C is the best base for the final projection because it starts closest to the desired flat/natural compromise.
- The final selected sample satisfies the strict numeric targets used in this round, except that body-yaw path can still look slightly active even though root-yaw path is under `20 deg`.
- If visual review still dislikes the return/body yaw, the next step should be a return-specific soft pose prior/yaw objective, not return IK and not a hard root lock.

### V36 Edge / Continuity Filter

Use this only after the `SOFT_FULL_BODY_PROJECTION` result is already geometrically good but the right hand jumps near stroke constraint boundaries. This is a temporal smoothing pass on the generated motion, not IK, not FABRIK, not line IK, and not return IK.

Script:

- `scripts\postprocess_wall_brush_v36_edge_filter.py`
- reproduce the smoothest version with `--profile edge_plus --continuity_lambda 5.0 --continuity_key_weight 40.0`

Profiles:

- `edge`: smooths only the transition windows `[28,46]`, `[76,88]`, `[118,130]`, and `[160,209]`.
- `edge_plus`: starts smoothing slightly before the stroke tails: `[28,46]`, `[70,88]`, `[112,130]`, and `[154,209]`. Prefer this when the hand jumps immediately before or after a stroke endpoint.

Current recommended edge-filtered sample:

- `logs\wall_brush_g1_v36_soft_full_body_projection\seed_C_final_edge_plus_filtered_result\sample_00\motion.npz`
- gallery copy: `logs\wall_brush_g1_v36_soft_full_body_projection\gallery_selected\v36_best_natural_EDGE_PLUS_filtered_SOFT_FULL_BODY_PROJECTION_line0p006_xp0p827_acc0p103_yaw12p1_pen0p000.npz`
- method label: `SOFT_FULL_BODY_PROJECTION_EDGE_FILTERED`
- effect: max right-hand step improves from about `0.088 m/frame` in the original V36 sample to about `0.054 m/frame`; tail root yaw path improves from about `19.5 deg` to about `12.1 deg`; wall penetration becomes `0`.
- tradeoff: avg x progress/coverage decreases from about `0.866` to about `0.827`, but it remains above the V36 acceptance target.

Current smoothest continuity-filtered sample:

- `logs\wall_brush_g1_v36_soft_full_body_projection\seed_C_final_edge_plus_continuity_lam5p0_result\sample_00\motion.npz`
- gallery copy: `logs\wall_brush_g1_v36_soft_full_body_projection\gallery_selected\v36_best_smooth_CONTINUITY_lam5_SOFT_FULL_BODY_PROJECTION_line0p006_xp0p811_acc0p025_yaw12p1_pen0p000.npz`
- method label: `SOFT_FULL_BODY_PROJECTION_TEMPORAL_CONTINUITY_FILTERED`
- effect: max right-hand acceleration improves from about `0.104 m/frame^2` to about `0.025 m/frame^2`; max right-hand step is about `0.049 m/frame`; wall penetration remains `0`.
- tradeoff: avg x progress/coverage decreases to about `0.811`. Use this version when visual smoothness matters more than preserving the exact widest stroke coverage.

Recommended selection rule:

- For the user's current default, use `DEFAULT_wall_brush_v36_SFBP_CONT_edge_plus_lam5_best_visual.npz`.
- For strict V36 geometry with slightly wider stroke coverage, use `edge_plus`.
- For the least jumpy V36 projection motion, use the `CONTINUITY_lam5` method. The default sample above is the selected `CONTINUITY_lam5` version.
- Keep the V35 seed C default copy only as a former natural-style baseline.
- Do not run a right-arm IK or return IK after these filters.

### Generalization Validation

Use this when checking whether the current wall-brush method works after changing wall/constraint geometry. The validation must run the full from-scratch pipeline for each variant; it must not reuse an old `seed_C` or old best npz as input.

Script:

- `scripts\run_wall_brush_generalization_batch.py`

Default variants:

- `baseline_center`: original center patch, `center_x=0.0`, `top_y=0.92`, `width=0.24`, `z_contact=0.32`.
- `left_lower_close`: left/lower/closer patch, `center_x=-0.055`, `top_y=0.88`, `width=0.20`, `z_contact=0.30`.
- `right_upper_mid`: right/higher patch, `center_x=0.055`, `top_y=0.96`, `width=0.22`, `z_contact=0.33`.
- `wide_center_mid`: wider central patch, `width=0.30`, `top_y=0.90`, `z_contact=0.33`.
- `small_right_close`: small low/right/close patch, `center_x=0.04`, `top_y=0.86`, `width=0.18`, `z_contact=0.28`.

Dry-run command:

```powershell
python scripts\run_wall_brush_generalization_batch.py
```

Full execution command on the GPU/Kimodo environment:

```powershell
python scripts\run_wall_brush_generalization_batch.py --execute
```

Outputs:

- `logs\wall_brush_generalization\summary.csv`
- `logs\wall_brush_generalization\manifest.json`
- `logs\wall_brush_generalization\gallery_selected`
- `logs\kimodo_motion_gallery_generalization.html`

Acceptance:

- Treat `passes_strict_v36_target=true` as a strong pass.
- Treat `passes_relaxed_generalization=true` as visually promising but requiring manual gallery review.
- If variants fail only when the patch is too high/far/wide, narrow the supported constraint envelope instead of changing the default method blindly.
