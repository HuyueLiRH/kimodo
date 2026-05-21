import json
import tempfile
import unittest
from pathlib import Path
import tomllib

import numpy as np


def make_prior_run_folder(root: Path) -> Path:
    run_dir = root / "prior_review"
    raw_dir = run_dir / "raw" / "outside_surface"
    post_dir = run_dir / "postprocessed" / "outside_surface" / "copy"
    raw_dir.mkdir(parents=True)
    post_dir.mkdir(parents=True)

    frames = 6
    joints = 34
    motion = {
        "posed_joints": np.zeros((frames, joints, 3), dtype=np.float32),
        "global_rot_mats": np.broadcast_to(np.eye(3, dtype=np.float32), (frames, joints, 3, 3)).copy(),
        "local_rot_mats": np.broadcast_to(np.eye(3, dtype=np.float32), (frames, joints, 3, 3)).copy(),
        "root_positions": np.zeros((frames, 3), dtype=np.float32),
        "foot_contacts": np.zeros((frames, 4), dtype=bool),
    }
    np.savez(raw_dir / "motion.npz", **motion)
    np.savez(post_dir / "motion.npz", **motion)

    manifest = {
        "schema_version": 1,
        "run_kind": "first_stage_prior_run",
        "task_name": "one_row_wall_brush_first_stage",
        "model": "kimodo-g1-rp",
        "candidates": {
            "outside_surface": {
                "name": "outside_surface",
                "recorded_seed": 7023,
                "prompt_segments": [
                    {
                        "label": "brush",
                        "text": "brush one horizontal row",
                        "start_frame": 0,
                        "end_frame": 5,
                    }
                ],
                "constraints": [
                    {
                        "label": "brush_line_start",
                        "end_effector": "left-hand",
                        "frame": 0,
                        "position": [0.4, 1.2, 0.3],
                        "coordinate_frame": "world",
                        "used_for_generation": True,
                        "show_in_review": True,
                        "role": "brush_stroke_start",
                    }
                ],
                "raw_motion": "raw/outside_surface/motion.npz",
                "postprocessed": {
                    "copy": {
                        "name": "copy",
                        "source_raw_motion": "raw/outside_surface/motion.npz",
                        "output_motion": "postprocessed/outside_surface/copy/motion.npz",
                    }
                },
            }
        },
    }
    metrics = {
        "candidates": {
            "outside_surface": {
                "constraint_error": {"mean": 0.02, "max": 0.03, "count": 1},
                "start_jump": 0.01,
                "root_drift": 0.05,
                "extra_motion_after_task": 0.0,
            }
        }
    }
    review = {
        "schema_version": 1,
        "review_statuses": ["needs_review", "raw_accepted", "needs_regeneration", "rejected"],
        "candidates": {
            "outside_surface": {
                "status": "needs_review",
                "notes": "",
                "admission_blockers": ["keep"],
            }
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "task.json").write_text(json.dumps({"task_name": "one row"}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "review.json").write_text(json.dumps(review), encoding="utf-8")
    return run_dir


class PriorViewerTests(unittest.TestCase):
    def test_load_prior_review_folder_exposes_candidates_variants_and_metadata(self):
        from kimodo.demo.prior_viewer import load_prior_review_folder

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_prior_run_folder(Path(tmpdir))
            review = load_prior_review_folder(run_dir)

        self.assertEqual(review.task_name, "one_row_wall_brush_first_stage")
        self.assertEqual(review.model, "kimodo-g1-rp")
        self.assertEqual(review.candidate_names, ["outside_surface"])
        candidate = review.candidates["outside_surface"]
        self.assertEqual(candidate.status, "needs_review")
        self.assertEqual(candidate.variants[0].name, "raw")
        self.assertEqual(candidate.variants[1].name, "copy")
        self.assertEqual(candidate.prompt_segments[0]["text"], "brush one horizontal row")
        self.assertEqual(candidate.constraints[0]["label"], "brush_line_start")
        self.assertEqual(candidate.metrics["constraint_error"]["mean"], 0.02)

    def test_save_review_decision_updates_status_and_notes_without_dropping_existing_fields(self):
        from kimodo.demo.prior_viewer import save_review_decision

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_prior_run_folder(Path(tmpdir))
            save_review_decision(run_dir, "outside_surface", "raw_accepted", "good base motion")
            review = json.loads((run_dir / "review.json").read_text(encoding="utf-8"))

        candidate = review["candidates"]["outside_surface"]
        self.assertEqual(candidate["status"], "raw_accepted")
        self.assertEqual(candidate["notes"], "good base motion")
        self.assertEqual(candidate["admission_blockers"], ["keep"])

    def test_prior_viewer_cli_parses_local_run_folder_and_is_registered(self):
        from kimodo.scripts.prior_viewer import parse_args

        args = parse_args(["--run-folder", "logs/prior", "--host", "127.0.0.1", "--port", "7861"])

        self.assertEqual(args.run_folder, "logs/prior")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 7861)

        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(
            pyproject["project"]["scripts"]["kimodo_prior_viewer"],
            "kimodo.scripts.prior_viewer:main",
        )

    def test_launch_prior_viewer_registers_server_without_starting_generation(self):
        from kimodo.demo.prior_viewer import launch_prior_viewer

        class FakeServer:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.connect_callback = None
                FakeServer.instances.append(self)

            def on_client_connect(self, callback):
                self.connect_callback = callback

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_prior_run_folder(Path(tmpdir))
            viewer = launch_prior_viewer(
                run_dir,
                host="127.0.0.1",
                port=7861,
                server_factory=FakeServer,
            )

        self.assertEqual(viewer.url, "http://127.0.0.1:7861/")
        self.assertEqual(viewer.review.task_name, "one_row_wall_brush_first_stage")
        self.assertEqual(FakeServer.instances[0].kwargs["host"], "127.0.0.1")
        self.assertEqual(FakeServer.instances[0].kwargs["port"], 7861)
        self.assertIsNotNone(FakeServer.instances[0].connect_callback)

    def test_prior_viewer_uses_small_review_constraint_markers(self):
        content = Path("kimodo/demo/prior_viewer.py").read_text(encoding="utf-8")

        self.assertIn("CONSTRAINT_MARKER_RADIUS", content)
        self.assertIn("Show Constraint Labels", content)
        self.assertIn("show_constraint_labels", content)
        self.assertNotIn("WaypointMesh", content)

    def test_workspace_launch_script_uses_local_venv_and_prior_viewer_entrypoint(self):
        script = Path("scripts/start_prior_viewer.sh")

        self.assertTrue(script.exists())
        content = script.read_text(encoding="utf-8")
        self.assertIn(".venvs/kimodo-prior-viewer", content)
        self.assertIn("python -m venv", content)
        self.assertIn("Python 3.10+", content)
        self.assertIn("SKIP_MOTION_CORRECTION_IN_SETUP=1", content)
        self.assertIn("kimodo_prior_viewer", content)
        self.assertIn("--run-folder", content)


if __name__ == "__main__":
    unittest.main()
