from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERTICAL_BATCH = (
    REPO_ROOT / "examples" / "wall_brush" / "task_specs" / "one_column_wall_brush_27_direct_from_raw_batch.json"
)
SCRIPT_DIR = REPO_ROOT / "examples" / "wall_brush" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


class OneColumnWallBrushBatchTests(unittest.TestCase):
    def test_batch_declares_27_bottom_to_top_vertical_targets(self) -> None:
        batch = json.loads(VERTICAL_BATCH.read_text(encoding="utf-8"))

        self.assertEqual(batch["task_name"], "wall_brush_one_column_27_bottom_to_top_20260602")
        self.assertEqual(batch["model"], "kimodo-g1-rp")
        self.assertEqual(batch["seed"], 7023)
        self.assertEqual(batch["duration_frames"], 102)
        self.assertEqual(len(batch["targets"]), 27)
        self.assertEqual(batch["constraint_mode"], "three_point")
        self.assertEqual(batch["grid"]["constraint_frames"], [36, 51, 66])
        self.assertEqual(batch["grid"]["stroke_axis"], "y")
        self.assertEqual(batch["grid"]["stroke_direction"], "bottom_to_top")

        for target in batch["targets"]:
            constraints = target["constraints"]
            self.assertEqual([point["label"] for point in constraints], [
                "column_1_start",
                "column_1_mid",
                "column_1_end",
            ])
            self.assertEqual([point["frame"] for point in constraints], [36, 51, 66])

            positions = [point["position"] for point in constraints]
            true_points = [point["true_point"] for point in constraints]
            self.assertLess(positions[0][1], positions[1][1])
            self.assertLess(positions[1][1], positions[2][1])
            self.assertAlmostEqual(positions[0][0], positions[1][0])
            self.assertAlmostEqual(positions[1][0], positions[2][0])
            self.assertAlmostEqual(positions[0][2], positions[1][2])
            self.assertAlmostEqual(positions[1][2], positions[2][2])
            for position, true_point in zip(positions, true_points):
                self.assertAlmostEqual(position[0], true_point[0])
                self.assertAlmostEqual(position[1], true_point[1])
                self.assertAlmostEqual(position[2], true_point[2] - 0.17)
            self.assertEqual(target["task_spec"]["post_processing"], False)
            self.assertEqual(target["task_spec"]["end_effector"]["type"], "right-hand")
            self.assertEqual(len(target["task_spec"]["end_effector"]["targets"]), 3)

    def test_remote_summary_preserves_vertical_stroke_fields(self) -> None:
        from remote_wall_brush_direct_from_raw_batch_runner import DIRECT_VARIANT, write_artifacts

        batch = json.loads(VERTICAL_BATCH.read_text(encoding="utf-8"))
        target = batch["targets"][0]
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            write_artifacts(
                run_root,
                batch,
                [
                    {
                        "name": target["name"],
                        "recorded_seed": batch["seed"],
                        "prompt_strategy": "one_column_outside_surface_single_prompt_three_point",
                        "prompt_segments": batch["prompt_segments"],
                        "constraints": target["constraints"],
                        "grid_variant": target["grid_variant"],
                        "endpoint_info": target["endpoint_info"],
                        "paths": {
                            "task_spec": "raw/candidate/task_spec.json",
                            "recipe": "raw/candidate/recipe.json",
                            "line_metrics": "raw/candidate/line_metrics.json",
                            "raw_motion": "raw/candidate/motion.npz",
                            "target_motion": "postprocessed/candidate/target/motion.npz",
                            "target_report": "postprocessed/candidate/target/report.json",
                            "final_motion": "postprocessed/candidate/final/motion.npz",
                            "final_report": "postprocessed/candidate/final/report.json",
                        },
                        "metrics": {
                            "raw": {},
                            DIRECT_VARIANT: {
                                "constraint_error": {"max_m": 0.0},
                                "line_distance": {"max_m": 0.0},
                                "stroke_hand_speed": {
                                    "cv": 0.0,
                                    "progress_axis": "y",
                                    "progress_backstep_total_m": 0.0,
                                },
                            },
                        },
                    }
                ],
            )
            summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

        row = summary["rows"][0]
        self.assertEqual(row["center_y"], 0.9)
        self.assertEqual(row["stroke_height_y"], 0.18)
        self.assertEqual(row["stroke_axis"], "y")
        self.assertEqual(row["stroke_direction"], "bottom_to_top")
        self.assertIn("final_stroke_hand_speed_progress_backstep_total_m", row)

    def test_each_vertical_task_spec_loads_as_web_equivalent_generation_spec(self) -> None:
        from kimodo.demo.web_equivalent import load_task_spec_dict

        batch = json.loads(VERTICAL_BATCH.read_text(encoding="utf-8"))
        for target in batch["targets"]:
            with self.subTest(target=target["name"]):
                spec = load_task_spec_dict(target["task_spec"])
                self.assertEqual(spec.model, "kimodo-g1-rp")
                self.assertEqual(spec.segments, [102])
                self.assertEqual(spec.end_effector.type, "right-hand")
                self.assertEqual(len(spec.end_effector.targets), 3)
                self.assertFalse(spec.post_processing)


if __name__ == "__main__":
    unittest.main()
