#!/usr/bin/env python3
"""Build a neutral-name deployment archive; never change the old experiment directory."""
from pathlib import Path
import hashlib
import io
import json
import tarfile


def main():
    source = Path(__file__).resolve().parent
    build = source.parent / "build"
    build.mkdir(exist_ok=True)
    names = ("run_mavis_complete_answers.py", "run_mavis_complete_answers_8gpu_front.sh",
             "run_mavis_support_stability.py", "run_mavis_support_stability_8gpu_front.sh",
             "mavis_supplement_runtime.py", "test_mavis_supplements.py",
             "run_mavis_model_scaling.py", "compute_mavis_drive_scores_logdomain.py",
             "summarize_mavis_drive_scores.py", "mavis_safe_scoring.py", "mavis_local_data.py")
    identity = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    archive = build / "mavis_supplemental_experiments.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        for name in names:
            package.add(source / name, arcname=f"experiments/answer_source_validation/{name}")
        payload = json.dumps(identity, indent=2).encode("utf-8")
        record = tarfile.TarInfo("experiments/answer_source_validation/package_sha256.json")
        record.size = len(payload)
        package.addfile(record, io.BytesIO(payload))
    print(archive)
    print(hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
