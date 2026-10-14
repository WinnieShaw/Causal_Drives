"""CPU engineering checks only; synthetic checks are not paper evidence."""
import unittest
import numpy as np
import json
from pathlib import Path
import tempfile
import os
import shutil
import subprocess
import sys
from unittest.mock import patch

from source_response_common import classify, content_indices, evaluate, weighted_metrics
from run_source_response_validation import pad_prefix_ids
from source_response_runtime import DirectParquetRows, checkpoint_dtype_name, gpu_model_kwargs, write_state


class Helpers(unittest.TestCase):
    def test_launcher_retains_signal_exit(self):
        # Kill a CPU-only stub, not a model: verify the actual Bash launcher's
        # previously missing signal/exit diagnostics and fail-fast behavior.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script_dir = root / "experiments/source_response_validation"
            script_dir.mkdir(parents=True)
            source = Path(__file__).parent
            for name in ["run_source_response_validation_8gpu_front.sh", "source_response_runtime.py"]:
                shutil.copyfile(source / name, script_dir / name)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake = bin_dir / "cpu-python-stub"
            fake.write_text("#!" + sys.executable + "\n" + '''import os, signal, sys
if len(sys.argv) > 2 and sys.argv[1] == "-c" and "torch.cuda.device_count" in sys.argv[2]:
    print(8)
    sys.exit(0)
if len(sys.argv) > 1 and sys.argv[1].endswith("run_source_response_validation.py"):
    if "--output" in sys.argv:
        print("CPU engineering stub: deliberate SIGKILL at model initialization", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    sys.exit(0)
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
''', encoding="utf-8")
            fake.chmod(0o755)
            smi = bin_dir / "nvidia-smi"
            smi.write_text("#!/bin/sh\necho 'CPU engineering stub; no GPU query'\n", encoding="utf-8")
            smi.chmod(0o755)
            env = {**os.environ, "MODEL_PATH": str(root / "unused_model"), "RUN_TAG": "cpu_signal_fixture",
                   "PYTHON_BIN": str(fake), "PROBE_MB": "64", "MIN_FREE_KB": "0",
                   "MODEL_INIT_TIMEOUT": "10", "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
            result = subprocess.run(["bash", str(script_dir / "run_source_response_validation_8gpu_front.sh"), str(root)],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout)
            logs = root / "outputs/source_response_validation/cpu_signal_fixture/logs"
            status = json.loads((logs / "worker_0_exit.json").read_text(encoding="utf-8"))
            self.assertEqual(status["exit_code"], 137)
            self.assertEqual(status["signal_number"], 9)
            self.assertEqual(json.loads((logs / "foreground_exit.json").read_text(encoding="utf-8"))["exit_code"], 1)
            self.assertIn("deliberate SIGKILL", (logs / "foreground.log").read_text(encoding="utf-8"))
            self.assertFalse((logs / "worker_1.log").exists())

    def test_runtime_contract(self):
        # Real Parquet API and atomic status persistence, but synthetic contents.
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pq.write_table(pa.table({"ID": ["a", "b", "c", "d", "e"], "answer": [1, 2, 3, 4, 5]}),
                           tmp / "rows.parquet", row_group_size=2)
            ds = DirectParquetRows(tmp / "rows.parquet")
            self.assertEqual(len(ds), 5)
            self.assertEqual([ds[i]["answer"] for i in [4, 0, 2, 1, 3]], [5, 1, 3, 2, 4])
            with self.assertRaises(IndexError):
                ds[-1]
            (tmp / "config.json").write_text(json.dumps({"text_config": {"dtype": "bfloat16"}}), encoding="utf-8")
            self.assertEqual(checkpoint_dtype_name(tmp), "bfloat16")
            kwargs = gpu_model_kwargs("dtype_fixture")
            self.assertEqual(kwargs["device_map"], {"": 0})
            self.assertTrue(kwargs["local_files_only"])
            self.assertNotIn("cpu", kwargs)
            write_state(tmp / "status.json", "model_loading", worker_id=3)
            state = json.loads((tmp / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["stage"], "model_loading")
            self.assertEqual(state["worker_id"], 3)
            self.assertEqual(sorted(p.name for p in tmp.glob("status.json*")), ["status.json"])

    def test_content_and_strict_match(self):
        self.assertEqual(content_indices(["{", "4", "}", "<|im_end|>"]), [1])
        self.assertEqual(content_indices(["{cat", "}"]), [0])
        self.assertEqual(classify("{four}", "5", "4"), "cue_answer_match")
        self.assertEqual(classify("not four, but five", "5", "4"), "other")
        self.assertEqual(classify("{4} and {5}", "5", "4"), "other")
        self.assertEqual(classify("{5}", "five", "5"), "ambiguous_match")

    def test_weighted_ties(self):
        y = np.array([0, 1, 0, 1])
        s = np.array([.1, .1, .4, .9])
        weights = np.array([[1., 1., 1., 1.], [2., 1., 3., 2.]])
        auc, ap = weighted_metrics(y, s, weights)
        for i, w in enumerate(weights.astype(int)):
            yy, ss = np.repeat(y, w), np.repeat(s, w)
            p, n = ss[yy == 1], ss[yy == 0]
            expected = np.mean((p[:, None] > n[None, :]) + .5 * (p[:, None] == n[None, :]))
            expected_ap = sum(yy[ss >= t].mean() * np.sum((ss == t) & (yy == 1)) / yy.sum() for t in np.unique(ss))
            self.assertAlmostEqual(auc[i], expected)
            self.assertAlmostEqual(ap[i], expected_ap)
        result = evaluate(y, {"a": s, "b": s}, [0, 0, 1, 2], rounds=100)
        self.assertEqual(result["paired_differences"]["a minus b"]["ci95"], [0., 0.])

    def test_exact_prefix_length(self):
        class Tokenizer:
            def encode(self, x, add_special_tokens=False):
                return [ord(c) for c in x]
            def decode(self, x, skip_special_tokens=False):
                return "".join(map(chr, x))
        result = pad_prefix_ids(Tokenizer(), {"visual": "5", "cue": "four", "neutral": "unknown"})
        self.assertEqual({v["token_count"] for v in result.values()}, {7})
        self.assertTrue(result["visual"]["decoded_text"].endswith("5"))

    def test_analysis_contract(self):
        # Synthetic schema fixture only, isolated and discarded; never evidence.
        from analyze_source_response_validation import main
        rows = []
        for i in range(2):
            conditions = {}
            for name, numerator in [("neutral", -2.), ("cue", -4.), ("visual", -1.)]:
                conditions[name] = {
                    "answer_match": "cue_answer_match" if name == "cue" and i == 0 else "visual_answer_match",
                    "fixed_candidate": {"content_summary": {
                        "pcd_level": numerator + 5., "prefix_numerator_bits": numerator,
                        "null_sequence_bits": -5., "full_content_logprob": numerator}}}
            rows.append({"status": "ok", "index": i, "sample_id": str(i), "topic": "fixture",
                         "prefix": conditions, "visual": None})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "shard.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            (tmp / "manifest.json").write_text(json.dumps({"mode": "prefix", "records": [
                {"index": i, "eligible_prefix": True, "eligible_visual": False} for i in range(2)]}), encoding="utf-8")
            with patch("sys.argv", ["analysis", "--input", str(tmp / "shard.jsonl"), "--manifest",
                                    str(tmp / "manifest.json"), "--output_dir", str(tmp / "out"), "--bootstrap", "100"]):
                main()
            result = json.loads((tmp / "out/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["prefix_completed"], 2)
            self.assertEqual(result["paired_effects"]["prefix"]["cue_delta_prefix_numerator_bits"]["mean"], -2.)


if __name__ == "__main__":
    unittest.main()
