import json
import tempfile
import unittest
import hashlib
from pathlib import Path

import numpy as np
import torch


class FakeG1Skeleton:
    root_idx = 0
    hip_joint_idx = (1, 2)
    bone_index = {
        "pelvis": 0,
        "right_hip": 1,
        "left_hip": 2,
        "right_wrist_yaw_skel": 3,
        "right_hand_roll_skel": 4,
    }

    def __init__(self):
        self.rest_pose_local_rot = torch.eye(3).repeat(len(self.bone_index), 1, 1)

    def expand_joint_names(self, joint_names):
        self.requested_joint_names = list(joint_names)
        return (
            ["right_wrist_yaw_skel", "right_hand_roll_skel"],
            ["right_wrist_yaw_skel", "right_hand_roll_skel"],
        )

    def fk(self, local_rots, root_positions):
        frame_count = local_rots.shape[0]
        global_rots = local_rots.clone()
        global_pos = torch.zeros((frame_count, len(self.bone_index), 3), dtype=local_rots.dtype)
        global_pos[:, 0] = root_positions
        global_pos[:, 1] = torch.tensor([0.10, 0.82, 0.00])
        global_pos[:, 2] = torch.tensor([-0.10, 0.82, 0.00])
        global_pos[:, 3] = torch.tensor([0.20, 1.00, 0.10])
        global_pos[:, 4] = torch.tensor([0.30, 1.05, 0.12])
        return global_rots, global_pos, global_pos - root_positions[:, None]


class WebEquivalentGenerationTests(unittest.TestCase):
    def test_load_task_spec_accepts_segments_and_right_hand_targets(self):
        from kimodo.demo.web_equivalent import load_task_spec

        spec_data = {
            "model": "kimodo-g1-rp",
            "seed": 7023,
            "num_samples": 1,
            "prompts": ["reach", "brush"],
            "segments": [30, 42],
            "diffusion_steps": 200,
            "cfg": {"type": "separated", "weight": [2.4, 4.0]},
            "num_transition_frames": 3,
            "post_processing": False,
            "end_effector": {
                "type": "right-hand",
                "targets": [
                    {"frame": 12, "point": [0.70, 1.20, 0.30]},
                    {"frame": 18, "point": [0.82, 1.25, 0.34], "wrist_point": [0.75, 1.18, 0.26]},
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "task.json"
            path.write_text(json.dumps(spec_data), encoding="utf-8")
            spec = load_task_spec(path)

        self.assertEqual(spec.model, "kimodo-g1-rp")
        self.assertEqual(spec.seed, 7023)
        self.assertEqual(spec.prompts, ["reach", "brush"])
        self.assertEqual(spec.segments, [30, 42])
        self.assertEqual(spec.cfg_type, "separated")
        self.assertEqual(spec.cfg_weight, [2.4, 4.0])
        self.assertEqual(spec.end_effector.type, "right-hand")
        self.assertEqual(len(spec.end_effector.targets), 2)

    def test_build_generation_kwargs_preserves_spatial_end_effector_targets(self):
        from kimodo.demo.web_equivalent import build_generation_kwargs, load_task_spec_dict

        class FakeModel:
            skeleton = FakeG1Skeleton()

        target = [0.82, 1.25, 0.34]
        wrist = [0.75, 1.18, 0.26]
        spec = load_task_spec_dict(
            {
                "prompts": ["reach", "brush"],
                "segments": [30, 42],
                "end_effector": {
                    "type": "right-hand",
                    "targets": [
                        {"frame": 18, "point": target, "wrist_point": wrist, "use_wrist": True},
                    ],
                },
            }
        )

        kwargs = build_generation_kwargs(FakeModel(), spec)
        constraint = kwargs["constraint_lst"][0]
        endpoint_index = FakeModel.skeleton.bone_index["right_hand_roll_skel"]
        wrist_index = FakeModel.skeleton.bone_index["right_wrist_yaw_skel"]

        self.assertEqual(kwargs["prompts"], ["reach", "brush"])
        self.assertEqual(kwargs["segments"], [30, 42])
        self.assertFalse(kwargs["post_processing"])
        self.assertEqual(constraint.name, "right-hand")
        self.assertEqual(constraint.frame_indices.tolist(), [18])
        torch.testing.assert_close(constraint.global_joints_positions[0, endpoint_index], torch.tensor(target))
        torch.testing.assert_close(constraint.global_joints_positions[0, wrist_index], torch.tensor(wrist))

    def test_cached_demo_text_encoder_reads_existing_prompt_embeddings(self):
        from kimodo.demo.web_equivalent import CachedDemoTextEncoder

        prompt = "A person brushes one short straight horizontal stroke from left to right on a wall."
        model_name = "kimodo-g1-rp"
        encoder_id = "LLM2VecEncoder"
        embedding = np.arange(12, dtype=np.float32).reshape(3, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            model_dir = cache_root / model_name
            model_dir.mkdir()
            key = hashlib.sha256(f"{model_name}|{encoder_id}|{prompt}".encode("utf-8")).hexdigest()
            np.save(model_dir / f"{key}.npy", embedding)

            encoder = CachedDemoTextEncoder([model_name], cache_root=cache_root, encoder_id=encoder_id)
            tensor, lengths = encoder([prompt])

        self.assertEqual(lengths, [3])
        self.assertEqual(tuple(tensor.shape), (1, 3, 4))
        torch.testing.assert_close(tensor[0], torch.from_numpy(embedding))

    def test_run_generation_with_model_writes_motion_and_reproducible_metadata(self):
        from kimodo.demo.web_equivalent import load_task_spec_dict
        from kimodo.scripts.web_equivalent_generate import run_generation_with_model

        class FakeModel:
            skeleton = FakeG1Skeleton()
            device = "cpu"

            def __init__(self):
                self.calls = []

            def __call__(self, prompts, segments, **kwargs):
                self.calls.append((prompts, segments, kwargs))
                frames = sum(segments)
                joints = len(self.skeleton.bone_index)
                eye = np.eye(3, dtype=np.float32)
                local_rot_mats = np.broadcast_to(eye, (1, frames, joints, 3, 3)).copy()
                return {
                    "local_rot_mats": local_rot_mats,
                    "global_rot_mats": local_rot_mats.copy(),
                    "posed_joints": np.zeros((1, frames, joints, 3), dtype=np.float32),
                    "root_positions": np.zeros((1, frames, 3), dtype=np.float32),
                    "smooth_root_pos": np.zeros((1, frames, 3), dtype=np.float32),
                    "foot_contacts": np.zeros((1, frames, 4), dtype=bool),
                    "global_root_heading": np.zeros((1, frames, 2), dtype=np.float32),
                }

        spec = load_task_spec_dict(
            {
                "model": "kimodo-g1-rp",
                "seed": 123,
                "prompts": ["reach", "brush"],
                "segments": [2, 3],
                "end_effector": {"type": "right-hand", "targets": [{"frame": 1, "point": [0.4, 1.1, 0.2]}]},
            }
        )
        model = FakeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = run_generation_with_model(model, spec, Path(tmpdir), save_csv=False)
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            with np.load(output_dir / "sample_00" / "motion.npz") as motion:
                self.assertEqual(motion["posed_joints"].shape, (5, len(model.skeleton.bone_index), 3))

        prompts, segments, kwargs = model.calls[0]
        self.assertEqual(prompts, ["reach", "brush"])
        self.assertEqual(segments, [2, 3])
        self.assertFalse(kwargs["post_processing"])
        self.assertEqual(kwargs["cfg_type"], "separated")
        self.assertEqual(len(kwargs["constraint_lst"]), 1)
        self.assertEqual(metadata["model"], "kimodo-g1-rp")
        self.assertEqual(metadata["seed"], 123)
        self.assertFalse(metadata["post_processing"])


if __name__ == "__main__":
    unittest.main()
