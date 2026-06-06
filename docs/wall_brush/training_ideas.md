# Training Ideas For Higher Precision Wall Brush Generation

The next model-training direction should improve the raw KIMODO prior so it naturally produces quieter and more accurate wall-brushing motions before downstream optimization.

## Main Idea

Do not train only text-to-motion. Train with the same structure used at inference time:

```text
temporal multi-prompt + sampled endpoint/wrist/root constraints -> motion
```

This teaches the model to satisfy spatial constraints naturally instead of relying on high constraint CFG or postprocess corrections.

## Data Strategy

Use SOMA/BONES-style motion data to mine near-task actions:

- standing reach
- hand wiping/cleaning/sweeping
- horizontal hand motion
- touch or slide hand near a surface
- quiet lower-body upper-body gestures

Then relabel with timeline prompts:

- approach segment
- left-to-right stroke
- right-to-left stroke
- repeated horizontal stroke
- smooth return/lower hand segment

Include control details such as active hand, torso facing, left arm relaxed, small root drift, and horizontal palm motion.

## Constraint-Conditioned Fine-Tuning

For each motion, sample constraints from the actual motion:

- sparse right-hand endpoint keyframes
- sparse wrist keyframes
- root 2D/root-height frames
- variants with endpoint-only and endpoint+wrist constraints

Randomize constraint density during training so the model is robust to sparse and moderately dense constraints.

## Teacher Distillation

Use high-quality optimized motions as teacher targets:

1. generate natural raw KIMODO motion
2. optimize wall contact while preserving full-body naturalness
3. train the model to produce the optimized target directly from the same prompts and constraints

This can absorb part of the postprocess effect into raw generation.

## Preference Data

Use human visual labels:

- good
- usable after optimization
- unusable

Negative labels should capture:

- large tail/root yaw
- odd return turn
- left arm noise
- hand motion not brush-like
- body drift or stepping

Train a ranker/reward model first, then use it for data filtering or preference fine-tuning.

## Evaluation

Hold out prompts, seeds, and wall geometry. Track:

- endpoint constraint error
- wall plane/contact error
- row line error
- root yaw path
- root drift
- left-arm velocity/noise
- human visual rating

The target is not just lower endpoint error; the target is lower endpoint error while preserving a quiet, natural base motion.
