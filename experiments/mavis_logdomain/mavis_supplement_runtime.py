"""CPU checks and serial initialization for supplemental foreground jobs."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_state(directory, worker_id, stage, **details):
    directory = Path(directory)
    record = {"stage": stage, "worker_id": worker_id, "unix_time": time.time(), **details}
    for name in ("/proc/self/status", "/proc/meminfo"):
        path = Path(name)
        if path.exists():
            record[name] = {k: v.strip() for k, v in
                            (line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
                            if k in {"VmRSS", "VmPeak", "MemAvailable", "MemTotal"}}
    for name in ("memory.current", "memory.max", "memory.events"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            record[name] = path.read_text().strip()
    temporary = directory / f"worker_{worker_id}_state.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / f"worker_{worker_id}_state.json")
    print(json.dumps(record, ensure_ascii=False), flush=True)


@contextmanager
def loading_lock(directory, worker_id):
    import fcntl
    write_state(directory, worker_id, "waiting_for_serial_model_load")
    with (Path(directory) / "model_load.lock").open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            write_state(directory, worker_id, "loading_model_weights")
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def eos_ids(model_path):
    path = Path(model_path) / "generation_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    values = config.get("eos_token_id")
    if isinstance(values, int):
        values = [values]
    if not values or any(not isinstance(x, int) for x in values):
        raise ValueError("A nonempty EOS token list is required in generation_config.json")
    return set(values)


def check_checkpoint(model_path):
    root = Path(model_path).resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl" or config.get("quantization_config"):
        raise ValueError("Require an unquantized full Qwen3-VL checkpoint")
    index = root / "model.safetensors.index.json"
    names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())) if index.exists() else ["model.safetensors"]
    if not names:
        raise ValueError("Empty checkpoint index")
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing or unsafe weight shard: {name}")
    eos_ids(root)
    return len(names)


def load_model(args):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly one GPU")
    with loading_lock(args.output_dir, args.worker_id):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.model_path), torch_dtype=torch.float16, attn_implementation="sdpa",
            device_map={"": 0}, low_cpu_mem_usage=True, local_files_only=True,
            trust_remote_code=True,
        ).eval()
        if any(p.device.type != "cuda" or p.device.index != 0 for p in model.parameters()):
            raise RuntimeError("The entire model must reside on this worker's GPU; no offload")
        count = sum(p.numel() for p in model.parameters())
        low, high = {"8b": (6_000_000_000, 12_000_000_000), "32b": (30_000_000_000, 40_000_000_000)}[args.model_size]
        if not low <= count <= high:
            raise ValueError(f"Checkpoint scale mismatch: {count}")
        import run_mavis_model_scaling as engine
        if torch.cuda.memory_allocated(0) > engine.memory_bytes(args.gpu_max_memory):
            raise RuntimeError("Loaded weights exceed the GPU budget")
        write_state(args.output_dir, args.worker_id, "model_loaded", parameters=count, dtype="float16")
    return model


def load_core(args):
    import run_mavis_model_scaling as engine
    import mavis_safe_scoring as safe
    core = engine.load_module(args.core_script, "supplement_mavis_core")
    engine.normalize_allocator_environment()
    safe.install(core)
    core.run_numerical_self_test()
    return core
