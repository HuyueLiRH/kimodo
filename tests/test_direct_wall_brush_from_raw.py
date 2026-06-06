from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "examples" / "wall_brush" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

spec = importlib.util.spec_from_file_location(
    "direct_wall_brush_g1_from_raw",
    SCRIPT_DIR / "direct_wall_brush_g1_from_raw.py",
)
direct = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(direct)


class DirectWallBrushTargetTests(unittest.TestCase):
    def test_interpolated_stroke_targets_pass_constraint_points(self) -> None:
        constraints = [
            {"label": "row_1_start", "frame": 10, "point": [-0.12, 0.92, 0.28]},
            {"label": "row_1_mid", "frame": 20, "point": [0.0, 0.92, 0.28]},
            {"label": "row_1_end", "frame": 30, "point": [0.12, 0.92, 0.28]},
        ]

        targets = direct.interpolate_stroke_hand_targets(total_frames=40, constraints=constraints)

        self.assertTrue(np.allclose(targets[10], [-0.12, 0.92, 0.28]))
        self.assertTrue(np.allclose(targets[20], [0.0, 0.92, 0.28]))
        self.assertTrue(np.allclose(targets[30], [0.12, 0.92, 0.28]))
        self.assertTrue(np.allclose(targets[15], [-0.06, 0.92, 0.28]))
        self.assertTrue(np.allclose(targets[25], [0.06, 0.92, 0.28]))
        self.assertNotIn(9, targets)
        self.assertNotIn(31, targets)

    def test_metrics_report_stroke_progress_backstep_for_vertical_strokes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            motion_path = tmp_path / "motion.npz"
            recipe_path = tmp_path / "recipe.json"
            posed = np.zeros((6, 34, 3), dtype=np.float32)
            root = np.zeros((6, 3), dtype=np.float32)
            posed[1, direct.RIGHT_HAND] = [0.0, 0.00, 0.25]
            posed[2, direct.RIGHT_HAND] = [0.0, 0.20, 0.25]
            posed[3, direct.RIGHT_HAND] = [0.0, 0.15, 0.25]
            posed[4, direct.RIGHT_HAND] = [0.0, 0.30, 0.25]
            np.savez(motion_path, posed_joints=posed, root_positions=root)
            recipe_path.write_text(
                json.dumps(
                    {
                        "candidate": {
                            "constraints": [
                                {
                                    "label": "column_1_start",
                                    "end_effector": "right-hand",
                                    "frame": 1,
                                    "position": [0.0, 0.00, 0.25],
                                    "used_for_postprocess": True,
                                    "role": "brush_stroke_start",
                                },
                                {
                                    "label": "column_1_end",
                                    "end_effector": "right-hand",
                                    "frame": 4,
                                    "position": [0.0, 0.30, 0.25],
                                    "used_for_postprocess": True,
                                    "role": "brush_stroke_end",
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            metrics = direct.compute_basic_metrics(motion_path, recipe_path)

        self.assertEqual(metrics["stroke_hand_speed"]["progress_axis"], "y")
        self.assertEqual(metrics["stroke_hand_speed"]["progress_backstep_count"], 1)
        self.assertAlmostEqual(metrics["stroke_hand_speed"]["progress_backstep_total_m"], 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
