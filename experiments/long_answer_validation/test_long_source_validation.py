"""CPU engineering checks, never used as research measurements."""
import ast
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

# Source-tree development uses the existing helpers; the deployment archive puts
# these same modules next to this test without modifying an active old job.
base = Path(__file__).resolve().parent
experiments = base.parents[1]
for folder in (experiments / "who-drives-vlm/source", experiments / "source-response-validation/source"):
    if folder.exists():
        sys.path.append(str(folder))

from long_source_common import (HFAdapter, build_manifest, cumulative, logmean,
                               native_coordinates, prefix_conditions, windows, window_summary)


class LongSourceTests(unittest.TestCase):
    def test_native_equations_and_pcd_decomposition(self):
        q = [[-.2, -.3, -.5], [-1.2, -.7, -.2]]
        p = [[-.4, -.9, -.1], [-.3, -.6, -.8]]
        v, null = [-.6, -.7, -.8], [-1., -2., -3.]
        result = native_coordinates(q, p, v, null)
        for i, step in enumerate(result):
            denominator = math.exp(sum(null[:i + 1]))
            q_probability = sum(math.exp(sum(x[:i + 1])) for x in q) / len(q)
            p_probability = sum(math.exp(x[i]) for x in p) / len(p)
            self.assertAlmostEqual(step["QCD_t"], math.log2(q_probability / denominator), places=12)
            self.assertAlmostEqual(step["PCD_t"], math.log2(p_probability / denominator), places=12)
        summary = window_summary(result, q[0], [0, 1, 2])
        self.assertAlmostEqual(summary["PCD"], summary["prefix_support_token_bits"] - summary["null_cumulative_bits"], places=12)
        tiny = native_coordinates([[-1000.] * 3], [[-1001.] * 3], [-1002.] * 3, [-1010.] * 3)
        self.assertTrue(all(math.isfinite(row["QCD_t"]) for row in tiny))

    def test_mixture_matches_existing_core(self):
        path = base / "compute_mavis_drive_scores_logdomain.py"
        if not path.exists():
            path = experiments / "who-drives-vlm/source/compute_mavis_drive_scores_logdomain.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "log_weighted_mean")
        namespace = {"math": math}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        for vector in ([-.1, -.2, -.3], [-100., -500., -1000.]):
            self.assertAlmostEqual(logmean(vector), namespace["log_weighted_mean"](vector, [1.] * len(vector)), places=12)

    def test_windows_and_equal_length_prefix(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(x) for x in text]
            def decode(self, ids):
                return "".join(chr(x) for x in ids)
        prefix = prefix_conditions(Tokenizer(), "dog", "a long animal name")
        self.assertEqual(len({len(p["ids"]) for p in prefix.values()}), 1)
        tokens = ["Initial ", "estimate ", "{cat", "}", " It", " has", " fur", "<EOS>"]
        selected = windows(tokens, {7}, list(range(8)), offset=2)
        self.assertEqual(selected["answer"], [2])
        self.assertEqual(selected["explanation"], [4, 5, 6])
        ambiguous = windows(["{a}", " {b}"], set(), [0, 1])
        self.assertEqual(ambiguous["answer"], [])

    def test_adapter_alignment_immutable_cache_and_greedy_generation(self):
        import torch
        from PIL import Image
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.model = SimpleNamespace(rope_deltas="stale")
                self.generation_config = SimpleNamespace(eos_token_id=[4])
                self.calls = 0
            def forward(self, **kwargs):
                self.calls += 1
                self.last = kwargs
                assert kwargs["use_cache"] is False and self.model.rope_deltas is None
                n = kwargs["logits_to_keep"]
                return SimpleNamespace(logits=torch.arange(8, dtype=torch.float).view(1, 1, 8).expand(1, n, 8))
            def generate(self, **kwargs):
                assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 256
                return torch.cat([kwargs["input_ids"], torch.tensor([[3, 4]])], dim=1)
        model = Model()
        adapter = HFAdapter.__new__(HFAdapter)
        adapter.torch, adapter.model = torch, model
        adapter.score_cache = {}
        prompt = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2), dtype=torch.long),
                  "mm_token_type_ids": torch.ones((1, 2), dtype=torch.long), "pixel_values": torch.ones((1, 3, 2, 2))}
        adapter.prompt = lambda image, question: ((question,), prompt)
        adapter.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kwargs: "test")
        image = Image.new("RGB", (2, 2))
        actual = adapter.score(image, "question", [3, 4])
        expected = [float(i - torch.logsumexp(torch.arange(8, dtype=torch.float), dim=0)) for i in (3, 4)]
        self.assertEqual(actual, expected)
        self.assertEqual(model.last["input_ids"].tolist(), [[1, 2, 3]])
        self.assertEqual(model.last["mm_token_type_ids"].tolist(), [[1, 1, 0]])
        self.assertEqual(prompt["input_ids"].tolist(), [[1, 2]])
        adapter.score(image, "question", [3, 4])
        self.assertEqual(model.calls, 1)
        generated = adapter.generate(image, "question", [5, 6])
        self.assertEqual(generated["ids"], [3, 4])
        self.assertTrue(generated["ended_with_eos"])

    def test_manifest_fixed_cohort_and_support_exclusions(self):
        from PIL import Image
        data, original = [], []
        for topic in range(7):
            for number in range(40):
                index = len(data)
                row = {"ID": str(index), "prompt": f"Question {index}", "ground_truth": "cat",
                       "expected_bias": "dog", "topic": str(topic), "image": Image.new("RGB", (2, 2))}
                data.append(row)
                if number < 30:
                    original.append({"index": index, "sample_id": str(index), "question": row["prompt"],
                                     "ground_truth": "cat", "cue_answer": "dog", "cue_candidate": "dog",
                                     "visual_candidate": "cat", "topic": str(topic),
                                     "counterfactual_index": topic * 40 + 39, "sham_index": topic * 40 + 38})
        # A corrupt support-only image is excluded before inference, never a target outcome.
        data[30]["image"] = {"bytes": b"not an image", "path": None}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"records": original}), encoding="utf-8")
            actual = build_manifest(path, data, Path(directory) / "data.parquet")
            self.assertEqual(actual["n"], 210)
            self.assertIn("30", actual["support_image_integrity_exclusions"])
            for before, after in zip(original, actual["records"]):
                self.assertEqual(after["index"], before["index"])
                self.assertEqual(len(set(after["context_ids"])), 20)
                self.assertFalse(set(after["context_ids"]) & {before["index"], before["counterfactual_index"], before["sham_index"]})
                self.assertNotIn(30, after["context_ids"])
            self.assertEqual(actual, build_manifest(path, data, Path(directory) / "data.parquet"))


if __name__ == "__main__":
    unittest.main()
