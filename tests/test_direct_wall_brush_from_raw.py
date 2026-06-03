from __future__ import annotations

import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
