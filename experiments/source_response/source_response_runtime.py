"""Runtime safeguards; no changes to probability scores or drive definitions."""
from bisect import bisect_right
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time


def resource_snapshot():
    result = {"pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for name in ("/proc/self/status", "/proc/meminfo"):
        try:
            lines = Path(name).read_text(encoding="utf-8").splitlines()
            result[name] = {k: v.strip() for k, v in (line.split(":", 1) for line in lines if ":" in line)
                            if k in {"VmRSS", "VmPeak", "VmSize", "MemAvailable", "MemTotal"}}
        except OSError:
            pass
    roots = [Path("/sys/fs/cgroup")]
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, location = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                roots.insert(0, Path("/sys/fs/cgroup") / location.lstrip("/"))
    except (OSError, ValueError):
        pass
    for root in roots:
        for field in ("memory.current", "memory.max", "memory.events", "memory.peak",
                      "memory/memory.limit_in_bytes", "memory/memory.failcnt"):
            try:
                result[str(root / field)] = (root / field).read_text(encoding="utf-8").strip()
            except OSError:
                pass
    return result


def write_state(path, stage, **details):
    """Fsync an atomic status record so abrupt worker loss leaves a last stage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "unix_time": time.time(), "resources": resource_snapshot(), **details}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(json.dumps(record, ensure_ascii=False), flush=True)


class DirectParquetRows:
    """Preserve Parquet row order; cache just one row group, never HF Arrow."""
    def __init__(self, path):
        import pyarrow.parquet as pq
        self.file = pq.ParquetFile(path)
        self.offsets = [0]
        for i in range(self.file.num_row_groups):
            self.offsets.append(self.offsets[-1] + self.file.metadata.row_group(i).num_rows)
        self.cached_group = None
        self.cached_table = None

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        group = bisect_right(self.offsets, index) - 1
        if self.cached_group != group:
            self.cached_table = self.file.read_row_group(group, use_threads=False)
            self.cached_group = group
        return self.cached_table.slice(index - self.offsets[group], 1).to_pylist()[0]


@contextmanager
def model_loading_lock(path):
    # Linux workers share this run-local lock; it is not a shared global cache.
    import fcntl
    with Path(path).open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def checkpoint_dtype_name(model_path):
    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    value = config.get("dtype") or config.get("torch_dtype")
    if value is None:
        text = config.get("text_config", {})
        value = text.get("dtype") or text.get("torch_dtype")
    if value not in {"bfloat16", "float16"}:
        raise ValueError(f"Explicit 16-bit checkpoint dtype required, got {value!r}; no silent conversion.")
    return value


def gpu_model_kwargs(dtype):
    return {"dtype": dtype, "attn_implementation": "sdpa", "device_map": {"": 0},
            "low_cpu_mem_usage": True, "local_files_only": True, "trust_remote_code": True}


def load_full_gpu_model(model_path, max_memory_per_gpu):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly one allocated GPU.")
    dtype_name = checkpoint_dtype_name(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, **gpu_model_kwargs(getattr(torch, dtype_name)))
    model.eval()
    if any(str(device) in {"cpu", "disk"} for device in getattr(model, "hf_device_map", {}).values()):
        raise RuntimeError("CPU/disk offload is prohibited for this experiment.")
    if any(p.device.type != "cuda" or p.device.index != 0 for p in model.parameters()):
        raise RuntimeError("The complete model must reside on this worker's GPU 0.")
    cap = str(max_memory_per_gpu)
    if not cap.endswith("GiB"):
        raise ValueError("max_memory_per_gpu must be expressed in GiB.")
    if torch.cuda.memory_allocated(0) > float(cap[:-3]) * 1024 ** 3:
        raise RuntimeError("Complete loaded model exceeds the requested GPU budget.")
    return model
