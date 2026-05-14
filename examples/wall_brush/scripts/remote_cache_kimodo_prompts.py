#!/usr/bin/env python
import argparse
import os

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ.setdefault("HUGGINGFACE_CACHE_DIR", "/root/autodl-tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LOCAL_CACHE", "True")

from kimodo.demo.embedding_cache import CachedTextEncoder
from kimodo.model import LLM2VecEncoder


DEFAULT_PROMPTS = [
    "A person inspects an object in front of him",
    "A person hammer a nail",
    "A person reaches forward with the right hand.",
    "A person wipes a wall from left to right with the right hand.",
    "A person wipes a wall from right to left with the right hand.",
    "A person paints a wall row by row with the right hand.",
    "A person brushes a vertical wall with the right hand.",
    "A person lowers the right hand after painting a wall.",
    "A person stands in place and reaches forward with the right hand.",
    "A person stands in place and wipes a small wall area from left to right with the right hand.",
    "A person stands in place and wipes a small wall area from right to left with the right hand.",
    "A person stands in place and paints a small wall patch row by row with the right hand.",
    "A person stands in place and lowers the right hand after painting a wall.",
    "A person stands still and raises the right hand toward a small wall patch.",
    "A person stands still and slides the right hand in a straight horizontal stroke from left to right on a wall.",
    "A person stands still and slides the right hand in a straight horizontal stroke from right to left on a wall.",
    "A person stands still and makes three straight horizontal brush strokes on a small wall patch.",
    "A person stands still and lowers the right hand after the straight brush strokes.",
    "A person stands still and raises only the right hand toward a small wall patch.",
    "A person stands still and slides only the right hand in a straight horizontal stroke from left to right on a wall.",
    "A person stands still and slides only the right hand in a straight horizontal stroke from right to left on a wall.",
    "A person stands still and slides only the right hand in a straight horizontal stroke from left to right on a wall.",
    "A person stands still and makes three straight horizontal brush strokes using only the right hand on a small wall patch.",
    "A person stands still and lowers only the right hand after the straight brush strokes.",
    "A person calmly stands still and raises the right hand toward a small wall patch.",
    "A person calmly slides the right hand in one smooth straight horizontal stroke from left to right on a wall.",
    "A person calmly slides the right hand in one smooth straight horizontal stroke from right to left on a wall.",
    "A person calmly slides the right hand in one smooth straight horizontal stroke from left to right on a wall.",
    "A person calmly makes three smooth straight horizontal brush strokes on a small wall patch.",
    "A person calmly lowers the right hand after the smooth straight brush strokes.",
    "A person standing in an upright stance raises the right hand toward a wall in front.",
    "A person moves the right hand across the wall from left to right in a horizontal stroke.",
    "A person moves the right hand across the wall from right to left in a horizontal stroke.",
    "A person moves the right hand across the wall from left to right in a horizontal stroke.",
    "A person lowers the right hand from the wall toward the side.",
    "A person stands upright with the arms hanging naturally at the sides.",
    "A person stands still facing a small wall patch and raises the right hand toward it.",
    "A person stands still facing the wall and slides the right hand in a straight horizontal stroke from left to right on the wall.",
    "A person stands still facing the wall and slides the right hand in a straight horizontal stroke from right to left on the wall.",
    "A person stands still facing the wall and slides the right hand in a straight horizontal stroke from left to right on the wall.",
    "A person remains facing the wall and lowers the right hand down beside the torso.",
    "A person remains facing the wall with both arms resting at the sides.",
    "A person keeps both feet planted, faces a small wall patch, and raises the right hand toward it.",
    "A person keeps both feet planted and faces the wall while sliding the right hand in a straight horizontal stroke from left to right on the wall.",
    "A person keeps both feet planted and faces the wall while sliding the right hand in a straight horizontal stroke from right to left on the wall.",
    "A person keeps both feet planted and faces the wall while sliding the right hand in a straight horizontal stroke from left to right on the wall.",
    "A person keeps both feet planted and the torso facing the wall while smoothly lowering the right hand beside the torso.",
    "A person keeps both feet planted, faces the wall, and rests both arms at the sides.",
    "A person is standing in an upright stance in front of a wall.",
    "A person standing in an upright stance leans forward and moves the right hand across the wall from left to right.",
    "A person leaning forward moves the right hand across the wall from right to left.",
    "A person leaning forward moves the right hand across the wall from left to right.",
    "A person leaning forward lowers the right hand from the wall.",
    "A person stands in an upright stance with the arms at the sides.",
    "A person stands still and raises the right hand toward the outside surface of a small wall patch.",
    "A person stands still and slides the right palm flat on the outside surface of a wall from left to right.",
    "A person stands still and slides the right palm flat on the outside surface of a wall from right to left.",
    "A person stands still and keeps the right hand on the outside surface while making three straight horizontal brush strokes.",
    "A person stands still and lowers the right hand away from the outside surface of the wall.",
    "A person stands balanced in place in front of a small wall patch, with the left arm relaxed by the side and the right hand ready to brush.",
    "A person keeps the left arm relaxed by the side while the right hand brushes one straight horizontal stroke from left to right on a wall.",
    "A person keeps the left arm relaxed by the side while the right hand brushes one straight horizontal stroke from right to left on a wall.",
    "A person keeps the left arm relaxed by the side while the right hand brushes one straight horizontal stroke from left to right on a wall.",
    "A person stands balanced and uses only the right hand to make three straight horizontal brush strokes on a small wall patch.",
    "A person relaxes the left arm by the side and smoothly lowers the right hand after brushing.",
    "A person stands balanced in front of a small wall patch, reaching forward with the right hand while the left arm stays relaxed as a passive counterbalance.",
    "A person uses small natural torso and shoulder motion while the right hand slides in a straight horizontal stroke from left to right on a wall, with the left arm passive.",
    "A person uses small natural torso and shoulder motion while the right hand slides in a straight horizontal stroke from right to left on a wall, with the left arm passive.",
    "A person uses small natural torso and shoulder motion while the right hand slides in a straight horizontal stroke from left to right on a wall, with the left arm passive.",
    "A person makes three controlled right-hand brush strokes on a small wall patch, with relaxed whole-body balance and a passive left arm.",
    "A person smoothly lowers the right hand after brushing, keeping a relaxed balanced stance and passive left arm.",
    "A person faces a small wall patch in a relaxed stance and prepares the right hand for brushing while the non-working left hand rests near the body.",
    "A person brushes the wall with the working right hand in a straight horizontal stroke from left to right, while the non-working left hand rests near the body.",
    "A person brushes the wall with the working right hand in a straight horizontal stroke from right to left, while the non-working left hand rests near the body.",
    "A person brushes the wall with the working right hand in a straight horizontal stroke from left to right, while the non-working left hand rests near the body.",
    "A person uses the working right hand for three straight horizontal brush strokes on a small wall patch while the left hand remains non-working.",
    "A person finishes brushing and smoothly lowers the working right hand while the left hand remains non-working.",
    "A person stands still facing a wall and raises the right hand toward a small wall patch.",
    "A person stands still facing the wall and lowers the right hand to the side.",
    "A person remains facing the wall and gently brings the right hand down beside the right thigh.",
    "A person stands still facing a wall, brushes three straight horizontal rows with the right hand, and lowers the right hand to the side.",
    "A person stands still facing a wall and raises the right hand toward a small patch on the wall, preparing to wipe the wall with the right hand.",
    "A person stands still facing a wall with the right hand on a small wall patch, sliding the right hand in one short straight horizontal wiping stroke from left to right.",
    "A person stands still facing a wall with the right hand on the same small wall patch, sliding the right hand in one short straight horizontal wiping stroke from right to left slightly lower on the wall.",
    "A person stands still facing a wall with the right hand on the same small wall patch, sliding the right hand in one short straight horizontal wiping stroke from left to right slightly lower on the wall.",
    "A person stands still facing a wall with the right hand near the wall, gently lowering the right hand down beside the right thigh and relaxing the arm.",
    "A person stands still close to a wall and raises the right hand toward a small patch on the wall, preparing to wipe the wall with the right hand.",
    "A person stands still facing a wall with the right hand touching a small wall patch, sliding the right hand in one short straight horizontal wiping stroke from left to right.",
    "A person stands still facing a wall with the right hand touching the same small wall patch, sliding the right hand in one short straight horizontal wiping stroke from right to left slightly lower on the wall.",
    "A person stands still facing a wall with the right hand touching the same small wall patch, sliding the right hand in one short straight horizontal wiping stroke from left to right slightly lower on the wall.",
    "A person stands still facing a wall with the right hand touching the left edge of a small wall patch, sliding the right hand across the full width of the patch in one short straight horizontal wiping stroke from left to right.",
    "A person stands still facing a wall with the right hand touching the right edge of the same small wall patch, sliding the right hand across the full width of the patch in one short straight horizontal wiping stroke from right to left slightly lower on the wall.",
    "A person stands still facing a wall with the right hand touching the left edge of the same small wall patch, sliding the right hand across the full width of the patch in one short straight horizontal wiping stroke from left to right slightly lower on the wall.",
    "A person stands still facing a wall with the right hand touching the left edge of a small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from left to right.",
    "A person stands still facing a wall with the right hand touching the right edge of the same small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from right to left slightly lower on the wall.",
    "A person stands still facing a wall with the right hand touching the left edge of the same small wall patch, sliding the right hand level across the full width of the patch at the same height in one short straight horizontal wiping stroke from left to right slightly lower on the wall.",
    "A person reaches down to pick up a tile with both hands.",
    "A person holds a tile with both hands.",
    "A person carries a tile toward a wall with both hands.",
    "A person places a tile on a wall with both hands.",
    "A person presses a tile against a wall with both hands.",
    "A person lowers both hands after placing the tile.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="kimodo-g1-rp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", action="append", default=[])
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    encoder = LLM2VecEncoder(
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        dtype="bfloat16",
        llm_dim=4096,
        device=args.device,
    )
    cached = CachedTextEncoder(encoder, model_name=args.model_name)
    tensor, lengths = cached(prompts)
    print(f"cached_prompt_count={len(prompts)}", flush=True)
    print(f"embedding_shape={tuple(tensor.shape)}", flush=True)
    for prompt, length in zip(prompts, lengths):
        print(f"length={length} prompt={prompt}", flush=True)


if __name__ == "__main__":
    main()
