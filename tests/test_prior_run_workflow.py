import json
import tempfile
import unittest
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
        "left_wrist_yaw_skel": 3,
        "left_hand_roll_skel": 4,
        "right_wrist_yaw_skel": 5,
        "right_hand_roll_skel": 6,
    }

    def __init__(self):
        self.rest_pose_local_rot = torch.eye(3).repeat(len(self.bone_index), 1, 1)

    def expand_joint_names(self, joint_names):
        if "LeftHand" in joint_names:
            return (
                ["left_wrist_yaw_skel", "left_hand_roll_skel"],
                ["left_wrist_yaw_skel", "left_hand_roll_skel"],
            )
        if "RightHand" in joint_names:
            return (
                ["right_wrist_yaw_skel", "right_hand_roll_skel"],
                ["right_wrist_yaw_skel", "right_hand_roll_skel"],
            )
        return ([], [])

    def fk(self, local_rots, root_positions):
        frame_count = local_rots.shape[0]
        global_rots = local_rots.clone()
        global_pos = torch.zeros((frame_count, len(self.bone_index), 3), dtype=local_rots.dtype)
        global_pos[:, 0] = root_positions
        global_pos[:, 1] = torch.tensor([0.10, 0.82, 0.00])
        global_pos[:, 2] = torch.tensor([-0.10, 0.82, 0.00])
        global_pos[:, 3] = torch.tensor([-0.20, 1.00, 0.10])
        global_pos[:, 4] = torch.tensor([-0.30, 1.05, 0.12])
        global_pos[:, 5] = torch.tensor([0.20, 1.00, 0.10])
        global_pos[:, 6] = torch.tensor([0.30, 1.05, 0.12])
        return global_rots, global_pos, global_pos - root_positions[:, None]


class FakePriorModel:
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
        posed_joints = np.zeros((1, frames, joints, 3), dtype=np.float32)
        root_positions = np.zeros((1, frames, 3), dtype=np.float32)
        root_positions[0, :, 0] = np.linspace(0.0, 0.05, frames)
        posed_joints[0, :, self.skeleton.bone_index["left_hand_roll_skel"]] = np.array([0.4, 1.2, 0.3])
        return {
            "local_rot_mats": local_rot_mats,
            "global_rot_mats": local_rot_mats.copy(),
            "posed_joints": posed_joints,
            "root_positions": root_positions,
            "smooth_root_pos": root_positions.copy(),
            "foot_contacts": np.zeros((1, frames, 4), dtype=bool),
            "global_root_heading": np.zeros((1, frames, 2), dtype=np.float32),
        }


def one_row_task_spec(postprocess=False):
    candidate = {
        "name": "composed_left_hand_brush",
        "seed": 7023,
        "prompt_segments": [
            {
                "label": "approach",
                "text": "raise the left hand toward the wall",
                "start_frame": 0,
                "end_frame": 3,
                "segment_source": "ai_suggested",
            },
            {
                "label": "brush",
                "text": "brush one horizontal row on the wall",
                "start_frame": 4,
                "end_frame": 11,
                "segment_source": "ai_suggested",
            },
        ],
        "constraints": [
            {
                "label": "brush_line_start",
                "end_effector": "left-hand",
                "frame": 4,
                "position": [0.4, 1.2, 0.3],
                "coordinate_frame": "world",
                "used_for_generation": True,
                "show_in_review": True,
                "used_for_postprocess": True,
                "role": "brush_stroke_start",
            }
        ],
    }
    if postprocess:
        candidate["postprocessing"] = {
            "enabled": True,
            "treatments": [{"name": "copy", "params": {"reason": "lineage smoke"}}],
        }
    return {
        "task_name": "one_row_wall_brush",
        "model": "kimodo-g1-rp",
        "duration_frames": 12,
        "candidates": [
            candidate,
            {
                **candidate,
                "name": "constraint_focused_brush",
                "seed": 7024,
                "prompt_segments": [
                    {
                        "label": "brush",
                        "text": "move the left hand through the target wall brushing line",
                        "start_frame": 0,
                        "end_frame": 11,
                        "segment_source": "hand_authored",
                    }
                ],
            },
        ],
    }


class PriorRunWorkflowTests(unittest.TestCase):
    def test_prior_run_generates_named_candidates_manifest_review_metrics_and_gallery(self):
        from kimodo.demo.prior_run import load_prior_task_spec_dict, run_prior_with_model

        task = one_row_task_spec()
        spec = load_prior_task_spec_dict(task)
        model = FakePriorModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = run_prior_with_model(model, spec, tmpdir, task_source=task)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            review = json.loads((run_dir / "review.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            gallery = (run_dir / "gallery.html").read_text(encoding="utf-8")

            self.assertTrue((run_dir / "raw" / "composed_left_hand_brush" / "motion.npz").exists())
            raw_metrics = json.loads(
                (run_dir / "raw" / "composed_left_hand_brush" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertTrue((run_dir / "raw" / "constraint_focused_brush" / "recipe.json").exists())

        self.assertEqual(len(model.calls), 2)
        prompts, segments, kwargs = model.calls[0]
        self.assertEqual(prompts, ["raise the left hand toward the wall", "brush one horizontal row on the wall"])
        self.assertEqual(segments, [4, 8])
        self.assertTrue(kwargs["multi_prompt"])
        self.assertFalse(kwargs["post_processing"])
        self.assertEqual(kwargs["constraint_lst"][0].name, "left-hand")

        self.assertEqual(sorted(manifest["candidates"]), ["composed_left_hand_brush", "constraint_focused_brush"])
        candidate = manifest["candidates"]["composed_left_hand_brush"]
        self.assertEqual(candidate["recorded_seed"], 7023)
        self.assertEqual(candidate["prompt_segments"][0]["segment_source"], "ai_suggested")
        self.assertEqual(candidate["constraints"][0]["label"], "brush_line_start")
        self.assertEqual(review["candidates"]["composed_left_hand_brush"]["status"], "needs_review")
        self.assertEqual(metrics["candidates"]["composed_left_hand_brush"]["constraint_error"]["count"], 1)
        self.assertEqual(raw_metrics["target_points"][0]["name"], "brush_line_start")
        self.assertIn("composed_left_hand_brush", gallery)
        self.assertIn("motionCanvas", gallery)
        self.assertIn("const DATA", gallery)
        self.assertIn("brush_line_start", gallery)
        self.assertIn("brush one horizontal row", gallery)

    def test_postprocessing_is_default_off_and_records_raw_to_postprocessed_lineage_when_requested(self):
        from kimodo.demo.prior_run import load_prior_task_spec_dict, run_prior_with_model

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_run = run_prior_with_model(FakePriorModel(), load_prior_task_spec_dict(one_row_task_spec()), Path(tmpdir) / "raw")
            self.assertFalse((raw_run / "postprocessed").exists())

            post_run = run_prior_with_model(
                FakePriorModel(),
                load_prior_task_spec_dict(one_row_task_spec(postprocess=True)),
                Path(tmpdir) / "post",
            )
            manifest = json.loads((post_run / "manifest.json").read_text(encoding="utf-8"))

            treatment = manifest["candidates"]["composed_left_hand_brush"]["postprocessed"]["copy"]
            self.assertEqual(treatment["source_raw_motion"], "raw/composed_left_hand_brush/motion.npz")
            self.assertEqual(
                treatment["output_motion"],
                "postprocessed/composed_left_hand_brush/copy/motion.npz",
            )
            self.assertTrue((post_run / treatment["output_motion"]).exists())

    def test_local_sync_and_recipe_note_draft_preserve_review_artifacts(self):
        from kimodo.demo.prior_run import (
            load_prior_task_spec_dict,
            run_prior_with_model,
            sync_local_review_folder,
            write_recipe_note_draft,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = run_prior_with_model(FakePriorModel(), load_prior_task_spec_dict(one_row_task_spec()), root / "remote")
            review = json.loads((run_dir / "review.json").read_text(encoding="utf-8"))
            review["candidates"]["composed_left_hand_brush"]["status"] = "raw_accepted"
            review["candidates"]["composed_left_hand_brush"]["notes"] = "Clean one-row brushing semantics."
            (run_dir / "review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")

            local_dir = sync_local_review_folder(run_dir, root / "local")
            self.assertTrue((local_dir / "raw" / "composed_left_hand_brush" / "motion.npz").exists())
            self.assertTrue((local_dir / "manifest.json").exists())

            recipe_path, note_path = write_recipe_note_draft(local_dir, "composed_left_hand_brush", root / "repo")
            note = note_path.read_text(encoding="utf-8")
            self.assertTrue(recipe_path.exists())
            self.assertIn("raw_accepted", note)
            self.assertIn("Clean one-row brushing semantics.", note)
            self.assertIn("brush_line_start", note)


if __name__ == "__main__":
    unittest.main()
