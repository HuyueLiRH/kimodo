import unittest
from pathlib import Path

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


class WallBrushWebEquivalentTests(unittest.TestCase):
    def test_legacy_wall_brush_entry_point_is_not_supported(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertNotIn("kimodo_wall_brush_one_row", pyproject)
        self.assertIn("kimodo_web_equiv_gen", pyproject)

    def test_preset_declares_raw_web_equivalent_one_row_generation(self):
        from kimodo.demo.wall_brush import WALL_BRUSH_ONE_ROW_PRESET

        self.assertEqual(WALL_BRUSH_ONE_ROW_PRESET["constraint_route"], "demo_native_right_hand")
        self.assertEqual(WALL_BRUSH_ONE_ROW_PRESET["seed"], 7023)
        self.assertEqual(WALL_BRUSH_ONE_ROW_PRESET["segments"], [30, 42, 30])
        self.assertEqual(WALL_BRUSH_ONE_ROW_PRESET["cfg_weight"], [2.4, 4.0])
        self.assertEqual(WALL_BRUSH_ONE_ROW_PRESET["num_denoising_steps"], 200)
        self.assertEqual(WALL_BRUSH_ONE_ROW_PRESET["num_transition_frames"], 3)
        self.assertFalse(WALL_BRUSH_ONE_ROW_PRESET["post_processing"])
        self.assertEqual(len(WALL_BRUSH_ONE_ROW_PRESET["right_hand_targets"]), 10)
        stroke_targets = [
            target for target in WALL_BRUSH_ONE_ROW_PRESET["right_hand_targets"] if target["label"].startswith("row_1")
        ]
        for target in stroke_targets:
            self.assertEqual(target["point"][0], target["true_point"][0])
            self.assertEqual(target["point"][1], target["true_point"][1])
            self.assertAlmostEqual(target["point"][2], target["true_point"][2] - 0.17)

    def test_build_generation_kwargs_are_raw_and_reproducible(self):
        from kimodo.demo.wall_brush import build_wall_brush_generation_kwargs

        class FakeModel:
            skeleton = FakeG1Skeleton()

        kwargs = build_wall_brush_generation_kwargs(FakeModel())

        self.assertEqual(kwargs["prompts"], kwargs["generation_prompt"])
        self.assertEqual(kwargs["segments"], [30, 42, 30])
        self.assertEqual(kwargs["cfg_weight"], [2.4, 4.0])
        self.assertEqual(kwargs["num_denoising_steps"], 200)
        self.assertEqual(kwargs["num_transition_frames"], 3)
        self.assertFalse(kwargs["post_processing"])
        self.assertEqual(kwargs["cfg_type"], "separated")
        self.assertEqual(len(kwargs["constraint_lst"]), 1)

    def test_right_hand_constraint_matches_web_ui_target_translation(self):
        from kimodo.demo.wall_brush import build_demo_native_right_hand_constraint

        skeleton = FakeG1Skeleton()
        targets = [
            {"frame": 5, "point": [0.80, 1.40, 0.30], "wrist_point": [0.72, 1.32, 0.22], "use_wrist": True},
            {"frame": 9, "point": [0.95, 1.45, 0.35], "wrist_point": None, "use_wrist": False},
        ]

        constraint = build_demo_native_right_hand_constraint(skeleton, targets)
        pos = constraint.global_joints_positions
        wrist_index = skeleton.bone_index["right_wrist_yaw_skel"]
        endpoint_index = skeleton.bone_index["right_hand_roll_skel"]

        self.assertEqual(constraint.name, "right-hand")
        self.assertEqual(constraint.frame_indices.tolist(), [5, 9])
        torch.testing.assert_close(pos[0, endpoint_index], torch.tensor(targets[0]["point"]))
        torch.testing.assert_close(pos[0, wrist_index], torch.tensor(targets[0]["wrist_point"]))

        default_endpoint = torch.tensor([0.30, 1.05, 0.12])
        default_wrist = torch.tensor([0.20, 1.00, 0.10])
        expected_delta = torch.tensor(targets[1]["point"]) - default_endpoint
        torch.testing.assert_close(pos[1, endpoint_index], torch.tensor(targets[1]["point"]))
        torch.testing.assert_close(pos[1, wrist_index], default_wrist + expected_delta)

    def test_wall_brush_preset_can_be_loaded_as_generic_web_equivalent_task(self):
        from kimodo.demo.wall_brush import wall_brush_preset_metadata
        from kimodo.demo.web_equivalent import load_task_spec_dict

        preset = wall_brush_preset_metadata()
        spec = load_task_spec_dict(
            {
                "model": preset["model"],
                "seed": preset["seed"],
                "num_samples": preset["num_samples"],
                "prompts": preset["prompts"],
                "segments": preset["segments"],
                "diffusion_steps": preset["num_denoising_steps"],
                "cfg_type": preset["cfg_type"],
                "cfg_weight": preset["cfg_weight"],
                "num_transition_frames": preset["num_transition_frames"],
                "post_processing": preset["post_processing"],
                "end_effector": {
                    "type": "right-hand",
                    "targets": preset["right_hand_targets"],
                },
            }
        )

        self.assertEqual(spec.model, "kimodo-g1-rp")
        self.assertEqual(spec.segments, [30, 42, 30])
        self.assertFalse(spec.post_processing)
        self.assertEqual(spec.end_effector.type, "right-hand")
        self.assertEqual(len(spec.end_effector.targets), 10)


if __name__ == "__main__":
    unittest.main()
