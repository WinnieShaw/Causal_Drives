"""CPU-only checks of formula equivalence, protocol identity and child dispatch."""
from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_mavis_model_scaling as engine
import run_mavis_complete_answers as complete
import run_mavis_support_stability as stability


class SupplementTests(unittest.TestCase):
    def test_formula_matches_probability_reference_and_reconstructs(self):
        q = [[math.log(x) for x in a] for a in ([.2, .4, .6], [.1, .5, .3])]
        joint = [[math.log(x) for x in a] for a in ([.7, .8, .9], [.3, .2, .4])]
        visual = [math.log(x) for x in [.4, .3, .2]]
        null = [math.log(x) for x in [.1, .2, .3]]
        metrics = stability.metrics_from_members(q, joint, visual, null, 2)
        for step in metrics["per_step"]:
            t = step["t"]
            denominator = math.prod(math.exp(x) for x in null[:t])
            q_probability = sum(math.prod(math.exp(x) for x in m[:t]) for m in q) / 2
            v_probability = math.prod(math.exp(x) for x in visual[:t])
            p_probability = sum(math.exp(m[t - 1]) for m in joint) / 2
            for name, numerator in (("QCD", q_probability), ("VCD", v_probability), ("PCD", p_probability)):
                self.assertAlmostEqual(step[f"{name}_t"], math.log2(numerator / denominator), places=12)
        # Very small probabilities must remain finite rather than be floored.
        tiny = stability.metrics_from_members([[-1000.] * 3] * 2, [[-1002.] * 3] * 2,
                                              [-1001.] * 3, [-1010.] * 3, 2)
        self.assertTrue(all(math.isfinite(x["PCD_t"]) for x in tiny["per_step"]))

    def test_manifest_is_metric_independent_nested_and_reproducible(self):
        source = [{"global_index": i, "question": f"Question {i}"} for i in range(500)]
        manifest = stability.make_manifest(source, 100)
        self.assertEqual(manifest, stability.make_manifest(source, 100))
        self.assertEqual(len(manifest["records"]), 100)
        for record in manifest["records"]:
            streams = record["support_indices_by_seed"]
            self.assertEqual(len({tuple(x) for x in streams.values()}), 3)
            for members in streams.values():
                self.assertEqual(members[0], record["global_index"])
                self.assertEqual(len(set(members)), 40)
                for k in stability.KS:
                    self.assertEqual(len(members[:k]), k)

    def test_child_dispatch_retains_long_cap_and_fresh_generation(self):
        importlib.reload(engine)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "generation_config.json").write_text('{"eos_token_id": [3, 4]}', encoding="utf-8")
            arguments = ["--launch", "--project_root", str(root), "--model_path", str(model),
                         "--dataset_path", str(root / "data.parquet"), "--model_size", "8b",
                         "--output_dir", str(root / "outputs/answer_source_validation/test"),
                         "--max_new_tokens", "256"]
            args = complete.parse_args(arguments)
            self.assertIsNone(args.fixed_source)
            with patch.dict(os.environ):
                complete.configure(args)
                self.assertEqual(engine.MAX_NEW_TOKENS, 256)
                self.assertEqual(Path(engine.__file__).name, "run_mavis_complete_answers.py")
                child = complete.parse_args(["--worker", *arguments[1:-2], "--worker_id", "7"])
                self.assertEqual(child.max_new_tokens, 256)
                self.assertIsNone(child.fixed_source)
            importlib.reload(engine)

    def test_source_token_alignment_rejects_partial_arrays(self):
        with self.assertRaises(ValueError):
            stability.metrics_from_members([[-1., -2.]], [[-1.]], [-1., -2.], [-2., -3.], 1)


if __name__ == "__main__":
    unittest.main()
