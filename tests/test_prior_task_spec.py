import unittest


class PriorTaskSpecValidationTests(unittest.TestCase):
    def test_executable_task_spec_requires_g1_rp_named_candidates_and_declared_constraints(self):
        from kimodo.demo.prior_run import load_prior_task_spec_dict

        valid_spec = {
            "task_name": "one_row_wall_brush",
            "model": "kimodo-g1-rp",
            "duration_frames": 12,
            "candidates": [
                {
                    "name": "composed_left_hand_brush",
                    "seed": 7023,
                    "prompt_segments": [
                        {
                            "label": "approach",
                            "text": "raise the left hand toward the wall",
                            "start_frame": 0,
                            "end_frame": 3,
                            "segment_source": "hand_authored",
                        },
                        {
                            "label": "brush",
                            "text": "brush one horizontal row on the wall",
                            "start_frame": 4,
                            "end_frame": 11,
                            "segment_source": "hand_authored",
                        },
                    ],
                    "constraints": [
                        {
                            "label": "brush_line_start",
                            "end_effector": "left-hand",
                            "frame": 4,
                            "position": [0.35, 1.18, 0.32],
                            "coordinate_frame": "world",
                            "used_for_generation": True,
                            "show_in_review": True,
                            "used_for_postprocess": True,
                            "role": "brush_stroke_start",
                        }
                    ],
                }
            ],
        }

        spec = load_prior_task_spec_dict(valid_spec)

        self.assertEqual(spec.model, "kimodo-g1-rp")
        self.assertEqual(spec.candidates[0].name, "composed_left_hand_brush")
        self.assertEqual(spec.candidates[0].recorded_seed, 7023)
        self.assertEqual(spec.candidates[0].prompt_segments[1].segment_source, "hand_authored")
        self.assertTrue(spec.candidates[0].constraints[0].used_for_generation)

    def test_executable_task_spec_rejects_ambiguous_or_unsupported_inputs(self):
        from kimodo.demo.prior_run import load_prior_task_spec_dict

        base = {
            "task_name": "one_row_wall_brush",
            "model": "kimodo-g1-rp",
            "duration_frames": 12,
            "candidates": [
                {
                    "name": "candidate",
                    "seed": 1,
                    "prompts": ["brush one row"],
                    "segments": [12],
                    "constraints": [
                        {
                            "label": "brush_line_start",
                            "end_effector": "left-hand",
                            "frame": 4,
                            "position": [0.35, 1.18, 0.32],
                            "coordinate_frame": "world",
                            "used_for_generation": True,
                            "show_in_review": True,
                            "used_for_postprocess": True,
                            "role": "brush_stroke_start",
                        }
                    ],
                }
            ],
        }

        wrong_model = {**base, "model": "kimodo-soma-rp"}
        with self.assertRaisesRegex(ValueError, "kimodo-g1-rp"):
            load_prior_task_spec_dict(wrong_model)

        missing_name = {**base, "candidates": [{**base["candidates"][0], "name": ""}]}
        with self.assertRaisesRegex(ValueError, "candidate name"):
            load_prior_task_spec_dict(missing_name)

        missing_generation_constraint = {
            **base,
            "candidates": [
                {
                    **base["candidates"][0],
                    "constraints": [{**base["candidates"][0]["constraints"][0], "used_for_generation": False}],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "generation constraint"):
            load_prior_task_spec_dict(missing_generation_constraint)

    def test_repo_persisted_one_row_wall_brush_recipe_is_executable(self):
        from kimodo.demo.prior_run import load_prior_task_spec

        spec = load_prior_task_spec("examples/prior_recipes/one_row_wall_brush_first_stage.json")

        self.assertEqual(spec.task_name, "one_row_wall_brush_first_stage")
        self.assertEqual(spec.model, "kimodo-g1-rp")
        self.assertEqual([candidate.name for candidate in spec.candidates], [
            "outside_surface",
            "left_arm_relaxed",
            "seed_upright_style",
        ])
        self.assertEqual(spec.candidates[0].segments, [30, 42, 30])
        self.assertEqual(len(spec.candidates[0].constraints), 10)
        self.assertTrue(all(point.used_for_generation for point in spec.candidates[0].constraints))


if __name__ == "__main__":
    unittest.main()
