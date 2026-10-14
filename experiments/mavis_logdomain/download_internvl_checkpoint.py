#!/usr/bin/env python3
"""Download a pinned public checkpoint and verify LFS sizes and SHA256 values."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = "OpenGVLab/InternVL3-8B-hf"
REVISION = "259a3b64a14623c0ec91a045cb43f7c5af5fa6af"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()
    root = args.project_root.resolve()
    destination = root / "models" / "OpenGVLab" / "InternVL3-8B-hf"
    destination.mkdir(parents=True, exist_ok=True)
    import requests
    response = requests.get(f"{args.endpoint}/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=30)
    response.raise_for_status()
    info = response.json()
    if info.get("sha") != REVISION:
        raise ValueError("Repository revision does not match the pinned revision")
    files = [x for x in info["siblings"] if x["rfilename"].endswith((".json", ".jinja", ".txt", ".safetensors"))]
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download
    print(json.dumps({"stage": "downloading", "repo": REPO, "revision": REVISION,
                      "destination": str(destination), "files": len(files)}), flush=True)
    snapshot_download(REPO, revision=REVISION, local_dir=str(destination),
                      allow_patterns=[x["rfilename"] for x in files], max_workers=2)
    verified = []
    for item in files:
        path = (destination / item["rfilename"]).resolve()
        if not path.is_relative_to(destination.resolve()) or not path.is_file():
            raise ValueError("Missing or unsafe checkpoint member")
        size = path.stat().st_size
        expected = item.get("size")
        if expected is not None and size != expected:
            raise ValueError(f"Size mismatch: {item['rfilename']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        expected_hash = item.get("lfs", {}).get("sha256")
        if expected_hash and actual != expected_hash:
            raise ValueError(f"SHA256 mismatch: {item['rfilename']}")
        verified.append({"name": item["rfilename"], "size": size, "sha256": actual,
                         "verified_against_lfs": bool(expected_hash)})
        print(f"[verified] {item['rfilename']} bytes={size}", flush=True)
    payload = {"repo": REPO, "revision": REVISION, "endpoint": args.endpoint,
               "complete": True, "files": verified}
    temporary = destination / "download_manifest.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination / "download_manifest.json")
    print(json.dumps({"stage": "download_complete", "total_bytes": sum(x["size"] for x in verified),
                      "model_loaded": False}), flush=True)


if __name__ == "__main__":
    main()
