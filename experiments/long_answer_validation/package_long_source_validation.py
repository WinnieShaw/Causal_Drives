#!/usr/bin/env python3
"""Package new probes without overwriting modules used by existing jobs."""
from pathlib import Path
import hashlib
import io
import json
import tarfile


def main():
    source = Path(__file__).resolve().parent
    experiments = source.parents[1]
    build = source.parent / "build"
    build.mkdir(exist_ok=True)
    inputs = [source / name for name in ("long_source_common.py", "run_long_source_validation.py",
                                         "analyze_long_source_validation.py", "run_long_source_validation_8gpu_front.sh",
                                         "test_long_source_validation.py", "README.txt",
                                         "run_long_source_suite_8gpu_front.sh")]
    inputs += [experiments / "source-response-validation/source" / name for name in
               ("source_response_common.py", "source_response_runtime.py")]
    inputs += [experiments / "who-drives-vlm/source" / name for name in
               ("mavis_local_data.py", "download_internvl_checkpoint.py", "compute_mavis_drive_scores_logdomain.py")]
    identity = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    path = build / "long_source_validation_code.tar.gz"
    with tarfile.open(path, "w:gz") as package:
        for item in inputs:
            package.add(item, arcname=f"experiments/long_answer_source_validation/{item.name}")
        payload = json.dumps(identity, indent=2).encode("utf-8")
        metadata = tarfile.TarInfo("experiments/long_answer_source_validation/package_sha256.json")
        metadata.size = len(payload)
        package.addfile(metadata, io.BytesIO(payload))
    print(path)
    print(hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
