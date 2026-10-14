#!/usr/bin/env python3
"""Read the supplied MAVIS parquet directly, without creating an Arrow cache.

Only a CPU in-memory table is constructed. Image decoding mirrors the
datasets.Image behavior (load, EXIF transpose), then produces RGB inputs for
the established processor. This module performs no downloads or GPU work.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile

REQUIRED_COLUMNS = {"image", "Q", "verifiable_facts", "pos_document_ids", "meta"}
DATA_BACKEND = "direct_parquet_in_memory_no_hf_arrow_cache"


def decode_image(value, dataset_path=None):
    from PIL import Image, ImageOps
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    if not isinstance(value, dict):
        raise TypeError("MAVIS image must contain embedded bytes or a readable path")
    if value.get("bytes") is not None:
        source = io.BytesIO(value["bytes"])
    elif value.get("path"):
        source = Path(value["path"])
        if not source.is_absolute() and dataset_path is not None:
            source = Path(dataset_path).parent / source
    else:
        raise ValueError("MAVIS row has no image bytes/path")
    with Image.open(source) as image:
        image.load()
        converted = ImageOps.exif_transpose(image).convert("RGB")
        if converted.width <= 0 or converted.height <= 0:
            raise ValueError("Invalid MAVIS image dimensions")
        return converted


class MavisParquetDataset:
    def __init__(self, path):
        import pyarrow.parquet as pq
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        # read_table reads the existing file into RAM. Unlike load_dataset,
        # it does not invoke download_and_prepare or inspect free disk space.
        table = pq.read_table(self.path)
        if not REQUIRED_COLUMNS <= set(table.column_names):
            raise ValueError(f"Missing MAVIS columns: {sorted(REQUIRED_COLUMNS - set(table.column_names))}")
        self.records = table.to_pylist()
        del table

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row = dict(self.records[index])
        row["image"] = decode_image(row["image"], self.path)
        return row


def load_mavis_parquet(path):
    return MavisParquetDataset(path)


def verify_output_writable(directory, num_bytes=65536):
    """Probe real writes, not CubeFS free-space reports; clean only this probe."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=".mavis-write-probe-", dir=directory)
    probe = Path(filename)
    payload = bytes((index % 251 for index in range(num_bytes)))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        with probe.open("rb") as handle:
            if handle.read() != payload:
                raise OSError("Output write/readback mismatch")
    finally:
        probe.unlink(missing_ok=True)
    return {"directory": str(directory), "bytes_written_and_verified": num_bytes,
            "fsync": True, "probe_removed": True}
