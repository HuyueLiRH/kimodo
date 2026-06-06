---
name: kimodo-wall-brush-raw-generation
description: Generate and verify reproducible raw KIMODO/G1 wall-brushing base motions from the three validated prompt+constraint pipelines. Use when the user asks to reproduce, rerun, compare seeds, preserve, inspect, or continue the wall-brush raw generation workflow without postprocessing, transition filtering, IK, V36 projection, or selecting from many generated candidates.
---

# KIMODO Wall Brush Raw Generation

## Core Rule

Generate direct KIMODO raw motions only unless the user explicitly asks for filtering or optimization. Do not run transition-aware smooth filtering, KIMODO postprocessing, IK, V36 soft full-body projection, DNO, or seed selection/reranking for the raw-generation pipeline.

The saved reproducible setup is:

- Three prompt pipelines: `outside_surface`, `left_arm_relaxed`, `seed_upright_style`
- Constraint route: `endpoint_wrist`
- Return constraint: disabled; do not add the final neutral-hand endpoint
- Guidance: `cfg_text=2.4`, `cfg_constraint=4.0`
- Sampling: `diffusion_steps=200`, `num_samples=1`
- Pose/scene setup: `frame_plan=210`, `heading_mode=none`, `preemphasis_preset=flat_B`, `prompt_variant=level_full_width`, `num_transition_frames=5`, `active_hand=right`, `disable_y_closed_loop`
- Output expectation: one raw motion per requested seed and pipeline

For exact prompts and constraint details, read [prompts-and-params.md](references/prompts-and-params.md).

## Workflow

1. Work from `/Users/huyue/Projects/codex_migration_required`.
2. Use the remote workspace through `scripts/autodl_remote.py`; the complete original task specs are on the remote at:
   `logs/wall_brush_prompt_ablation_v2_no_return_constraint/{outside_surface,left_arm_relaxed,seed_upright_style}/task_spec.json`.
3. Use `scripts/run_wall_brush_top3_seed_robustness_raw.py` for seed robustness or raw reruns. It reuses the original task specs and creates a direct raw gallery.
4. Pull back the generated HTML and CSV summary for local inspection.
5. Verify the HTML has only `RAW` labels and no `FILTERED`/`transition_filtered` motions.

## Remote Commands

Run the three saved pipelines for new seeds:

```bash
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/wall_brush_pipeline && /root/miniconda3/bin/python scripts/run_wall_brush_top3_seed_robustness_raw.py --output_root logs/wall_brush_prompt_ablation_top3_seed_raw --seeds 7023 8023 9023 --execute"
```

Run only one pipeline and one seed:

```bash
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/wall_brush_pipeline && /root/miniconda3/bin/python scripts/run_wall_brush_top3_seed_robustness_raw.py --output_root logs/wall_brush_prompt_ablation_top3_seed_raw_single --variants outside_surface --seeds 10023 --execute"
```

Pull artifacts:

```bash
python3 scripts/autodl_remote.py get /root/autodl-tmp/wall_brush_pipeline/logs/wall_brush_prompt_ablation_top3_seed_raw/top3_seed_robustness_raw_gallery.html logs/wall_brush_prompt_ablation_top3_seed_raw/top3_seed_robustness_raw_gallery.html
python3 scripts/autodl_remote.py get /root/autodl-tmp/wall_brush_pipeline/logs/wall_brush_prompt_ablation_top3_seed_raw/top3_seed_robustness_raw_summary.csv logs/wall_brush_prompt_ablation_top3_seed_raw/top3_seed_robustness_raw_summary.csv
```

## Verification

Check that a generated gallery is raw-only:

```bash
node - <<'NODE'
const fs=require('fs'), vm=require('vm');
const p='logs/wall_brush_prompt_ablation_top3_seed_raw/top3_seed_robustness_raw_gallery.html';
const h=fs.readFileSync(p,'utf8');
const data=JSON.parse(h.match(/const DATA = ([\s\S]*?);\nconst canvas/)[1]);
console.log('motions', data.motions.length);
console.log(data.motions.map(m=>m.label).join('\n'));
console.log('filtered?', data.motions.some(m=>m.label.includes('FILTERED') || m.path.includes('transition_filtered')));
new vm.Script(h.match(/<script>([\s\S]*)<\/script>/)[1]);
NODE
```

Expected: one `RAW ... seed_...` motion for each requested `{pipeline, seed}` pair and `filtered? false`.

## Notes

- Treat prompt quality as the main lever. Do not compensate for bad raw motion by raising `cfg_constraint` aggressively; previous strictness experiments showed high constraint CFG can make motion stiffer or stranger.
- Prefer `outside_surface` for quiet, low-extra-motion base motions.
- Prefer `left_arm_relaxed` when the user wants stronger brush-like coverage while accepting later wall-contact optimization.
- Use `seed_upright_style` as a third reproducible baseline with more visible brushing intent but potentially more body drift.
