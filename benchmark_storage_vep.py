#!/usr/bin/env python3
"""Benchmark four HDF5 compression filters in all- and per-channel layouts.

The script uses one canonical float32 signal as the reference, writes eight
HDF5 outputs, and compares storage, I/O, memory, copy time, waveform error,
and event-locked VEP results.

Marker CSV format:
    sample,tag
    12000,6
    18000,4

The sample column uses indices relative to the loaded signal. The default VEP
settings mirror the GUI LetterModeData path: epoch -500..800 ms, baseline
-200..0 ms, response 0..300 ms, mean aggregation, and first 30/60 trials.

Storage-only BIN example:
    python benchmark_storage_vep.py --bin-file data.bin --output-dir result
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

try:
    from compare_bin_storage import BinReader, inspect_bin, read_timing_metadata
except ImportError:  # pragma: no cover
    BinReader = None
    inspect_bin = None
    read_timing_metadata = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None
    pq = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import openpyxl
except ImportError:  # pragma: no cover
    openpyxl = None

try:
    import hdf5plugin
except ImportError:  # pragma: no cover
    hdf5plugin = None


CHANNELS = 512
CHANNEL_NAMES = [f"ch{i:03d}" for i in range(1, CHANNELS + 1)]
DATASET_NAME = "signal"


def load_signal(path: Path) -> tuple[np.ndarray, float, float, int, dict]:
    with h5py.File(path, "r") as handle:
        key = "rawData512" if "rawData512" in handle else DATASET_NAME
        if key not in handle:
            raise KeyError(f"No rawData512 or signal dataset in {path}")
        data = np.asarray(handle[key][:], dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D signal array, got {data.shape}")
        if data.shape[1] == CHANNELS:
            pass
        elif data.shape[0] == CHANNELS:
            data = data.T
        else:
            raise ValueError(f"Cannot identify channel axis in {data.shape}")
        fs = float(np.asarray(handle["FS"]).squeeze()) if "FS" in handle else float(handle.attrs.get("fs", 6490.0))
        t0 = float(np.asarray(handle["t0"]).squeeze()) if "t0" in handle else float(handle.attrs.get("t0", 0.0))
        delta_t1 = float(np.asarray(handle["deltaT1"]).squeeze()) if "deltaT1" in handle else float(handle.attrs.get("deltaT1_ms", handle.attrs.get("deltaT1", 0.0)))
        delta_unit = _read_h5_text(handle, "deltaT1_unit") or str(handle.attrs.get("deltaT1_unit", ""))
        if delta_unit.strip().lower() in {"ms", "millisecond", "milliseconds"} or (not delta_unit and abs(delta_t1) >= 0.5):
            delta_t1 /= 1000.0
        attrs = {str(k): _json_value(v) for k, v in handle.attrs.items()}
        if "channel_quality" in handle:
            quality = np.asarray(handle["channel_quality"][:]).ravel().astype(np.uint8)
        else:
            quality = np.ones(CHANNELS, dtype=np.uint8)
    if quality.size != CHANNELS:
        raise ValueError(f"channel_quality must contain 512 values, got {quality.size}")
    return data, fs, t0, int(data.shape[0]), {"channel_quality": quality, "deltaT1_sec": delta_t1, **attrs}


def _read_h5_text(handle: h5py.File, key: str) -> str:
    if key not in handle:
        return ""
    value = np.asarray(handle[key])
    if value.dtype.kind == "u" and value.size:
        try:
            return bytes(value.astype(np.uint8).ravel().tolist()).decode("utf-8", errors="replace")
        except (ValueError, OverflowError):
            pass
    if value.dtype.kind in {"S", "U"}:
        value = value.tobytes() if value.dtype.kind == "S" else value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    return value


def load_quality(path: Path | None, default: np.ndarray) -> np.ndarray:
    if path is None:
        return default.astype(np.uint8, copy=True)
    text = path.read_text(encoding="utf-8-sig")
    values = []
    for token in text.replace(",", " ").split():
        values.append(int(float(token)))
    quality = np.asarray(values, dtype=np.uint8)
    if quality.size != CHANNELS or np.any((quality != 0) & (quality != 1)):
        raise ValueError("Quality file must contain exactly 512 values, each 0 or 1.")
    return quality


def load_reference_from_bin(path: Path, start_sec: float, duration_sec: float, chunk_rows: int) -> tuple[np.ndarray, float, float, int, dict, float]:
    if BinReader is None or inspect_bin is None:
        raise RuntimeError("compare_bin_storage.py is required for BIN decoding.")
    start = time.perf_counter()
    layout = inspect_bin(path)
    reader = BinReader(layout, start_sec, duration_sec, chunk_rows)
    blocks = [block for _, block in reader.iter_chunks()]
    if not blocks:
        raise ValueError("BIN produced no decoded data blocks.")
    data = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    timing = read_timing_metadata(path) if read_timing_metadata is not None else {}
    metadata = {"channel_quality": np.ones(CHANNELS, dtype=np.uint8), **timing}
    if "deltaT1_ms" in timing:
        metadata["deltaT1_sec"] = float(timing["deltaT1_ms"]) / 1000.0
    t0 = reader.start_frame / 6490.0
    return data, 6490.0, t0, int(data.shape[0]), metadata, time.perf_counter() - start


def load_markers(path: Path) -> np.ndarray:
    rows = list(csv.reader(path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError("Marker CSV is empty.")
    header = [cell.strip().lower() for cell in rows[0]]
    sample_idx = next((i for i, name in enumerate(header) if name in {"sample", "samples", "marker_sample"}), None)
    tag_idx = next((i for i, name in enumerate(header) if name in {"tag", "stimtag", "stim_tag"}), None)
    start = 1 if sample_idx is not None and tag_idx is not None else 0
    if sample_idx is None:
        sample_idx, tag_idx = 0, 1
    values = []
    for row in rows[start:]:
        if len(row) <= max(sample_idx, tag_idx):
            continue
        try:
            values.append((int(round(float(row[sample_idx]))), int(round(float(row[tag_idx])))))
        except ValueError:
            continue
    if not values:
        raise ValueError("Marker CSV has no usable sample/tag rows.")
    return np.asarray(values, dtype=np.int64)[np.argsort(np.asarray(values)[:, 0])]


def read_gui_log_timing(path: Path, delta_t1_sec: float) -> dict[str, float | str]:
    text = None
    for encoding in ("utf-8-sig", "gbk", "utf-16"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = path.read_text(errors="ignore")
    lines = text.splitlines()
    values = {"rec_start_text": "", "stim_start_text": "", "rec_end_text": "", "stim_end_text": ""}
    for index, line in enumerate(lines):
        previous = lines[index - 1] if index else line
        if "Recording ON" in line:
            values["rec_start_text"] = extract_gui_time(previous) or extract_gui_time(line) or ""
        elif "Recording OFF" in line:
            values["rec_end_text"] = extract_gui_time(previous) or extract_gui_time(line) or ""
        elif "PROGRAM_START" in line:
            values["stim_start_text"] = extract_gui_time(line) or ""
        elif "PROGRAM_STOP" in line:
            values["stim_end_text"] = extract_gui_time(line) or ""
    if not values["rec_start_text"] or not values["stim_start_text"]:
        raise ValueError("GUI log.txt must contain Recording ON and PROGRAM_START times.")
    rec_sec = gui_time_to_seconds(str(values["rec_start_text"]))
    stim_sec = gui_time_to_seconds(str(values["stim_start_text"]))
    raw_delay = stim_sec - rec_sec
    if raw_delay < 0:
        raw_delay += 24 * 3600
    values.update({
        "rec_start_sec": rec_sec,
        "stim_start_sec": stim_sec,
        "stim_delay_raw_sec": raw_delay,
        "deltaT1_sec": float(delta_t1_sec),
        "stim_delay_sec": raw_delay + float(delta_t1_sec),
    })
    return values


def extract_gui_time(text: str) -> str | None:
    matches = re.findall(r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?", text)
    return matches[-1] if matches else None


def gui_time_to_seconds(text: str) -> float:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Cannot parse time in GUI log.txt: {text}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def normalize_letter_mode(value: object) -> str:
    text = str(value).replace("\ufeff", "").replace("\u3000", "").strip()
    compact = re.sub(r"\s+", "", text)
    if not compact or compact.lower() == "nan":
        return ""
    if compact.upper() in {"O", "0"}:
        return "O"
    if compact in {"|", "｜", "∣", "│"}:
        return "|"
    if compact in {"闪烁光点", "flashdot", "flash_dot"}:
        return "闪烁光点"
    if compact and all(char in {"-", "‐", "‑", "‒", "–", "—", "―", "−", "－"} for char in compact):
        return "-"
    return compact


def build_gui_letter_markers(path: Path, timing: dict[str, float | str], fs: float) -> np.ndarray:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Letter Event CSV parsing requires pandas.") from exc
    try:
        table = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        table = pd.read_csv(path, encoding="gbk")
    if table.shape[1] < 15:
        raise ValueError("Letter Event CSV must have at least 15 columns, matching the GUI logic.")
    stim_mode = table.iloc[:, 1].astype(str).to_numpy()
    dot_offset_sec = pd.to_numeric(table.iloc[:, 12], errors="coerce").to_numpy()
    letter_offset_sec = pd.to_numeric(table.iloc[:, 13], errors="coerce").to_numpy()
    delay_sec = float(timing["stim_delay_sec"])
    tag_map = {"O": 6, "|": 4, "-": 7, "闪烁光点": 5}
    samples, tags = [], []
    for mode_text, offset_sec in zip(stim_mode, letter_offset_sec):
        if np.isfinite(offset_sec):
            samples.append(round(fs * (delay_sec + float(offset_sec))))
            tags.append(tag_map.get(normalize_letter_mode(mode_text), 100))
    for offset_sec in dot_offset_sec:
        if np.isfinite(offset_sec):
            samples.append(round(fs * (delay_sec + float(offset_sec))))
            tags.append(100)
    if not samples:
        raise ValueError("No Letter markers could be built from the Event CSV.")
    markers = np.column_stack([samples, tags]).astype(np.int64)
    return markers[np.argsort(markers[:, 0])]


COMPRESSION_VARIANTS = {
    "gzip1": {"label": "gzip level 1", "filename": "all_channels_gzip1.h5"},
    "gzip4": {"label": "gzip level 4", "filename": "all_channels_gzip4.h5"},
    "lzf_shuffle": {"label": "LZF + shuffle", "filename": "all_channels_lzf_shuffle.h5"},
    "blosc_lz4_bitshuffle": {"label": "Blosc/LZ4 + bitshuffle", "filename": "all_channels_blosc_lz4_bitshuffle.h5"},
}


def h5_compression_options(variant: str) -> dict:
    if variant == "gzip1":
        return {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    if variant == "gzip4":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    if variant == "lzf_shuffle":
        return {"compression": "lzf", "shuffle": True}
    if variant == "blosc_lz4_bitshuffle":
        if hdf5plugin is None:
            raise RuntimeError(
                "Blosc/LZ4 + bitshuffle requires hdf5plugin. "
                "Install it with: python -m pip install hdf5plugin "
                "-i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        return dict(hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE))
    raise ValueError(f"Unknown HDF5 compression variant: {variant}")


def write_compressed_h5(path: Path, data: np.ndarray, quality: np.ndarray, fs: float, t0: float, delta_t1_sec: float, chunk_rows: int, variant: str) -> dict:
    start = time.perf_counter()
    options = h5_compression_options(variant)
    with h5py.File(path, "w") as handle:
        rows = min(chunk_rows, data.shape[0])
        handle.create_dataset(DATASET_NAME, data=data, dtype="float32", chunks=(rows, CHANNELS), **options)
        handle.create_dataset("channel_quality", data=quality.astype(np.uint8), dtype="uint8")
        handle.attrs.update({"t0": t0, "fs": fs, "n": int(data.shape[0]), "deltaT1_sec": delta_t1_sec, "data_unit": "mV", "storage_dtype": "float32", "compression_variant": variant, "compression_label": COMPRESSION_VARIANTS[variant]["label"]})
    return {"seconds": time.perf_counter() - start, "bytes": path.stat().st_size, "files": 1, "compression_variant": variant, "compression_label": COMPRESSION_VARIANTS[variant]["label"]}


def write_compressed_per_channel_h5(root: Path, data: np.ndarray, quality: np.ndarray, fs: float, t0: float, delta_t1_sec: float, chunk_rows: int, variant: str) -> dict:
    start = time.perf_counter()
    options = h5_compression_options(variant)
    root.mkdir(parents=True, exist_ok=True)
    rows = min(chunk_rows, data.shape[0])
    for index, name in enumerate(CHANNEL_NAMES):
        with h5py.File(root / f"{name}.h5", "w") as handle:
            handle.create_dataset(DATASET_NAME, data=data[:, index], dtype="float32", chunks=(rows,), **options)
            handle.attrs.update({"good_channel": int(quality[index]), "t0": t0, "fs": fs, "n": int(data.shape[0]), "deltaT1_sec": delta_t1_sec, "data_unit": "mV", "storage_dtype": "float32", "compression_variant": variant, "compression_label": COMPRESSION_VARIANTS[variant]["label"]})
    return {"seconds": time.perf_counter() - start, "bytes": directory_bytes(root), "files": CHANNELS, "compression_variant": variant, "compression_label": COMPRESSION_VARIANTS[variant]["label"]}


def write_layout_per_channel(root: Path, data: np.ndarray, quality: np.ndarray, fs: float, t0: float, chunk_rows: int, gzip_level: int) -> dict:
    start = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    rows = min(chunk_rows, data.shape[0])
    for index, name in enumerate(CHANNEL_NAMES):
        with h5py.File(root / f"{name}.h5", "w") as handle:
            handle.create_dataset(DATASET_NAME, data=data[:, index], dtype="float32", chunks=(rows,), compression="gzip", compression_opts=gzip_level, shuffle=True)
            handle.attrs.update({"good_channel": int(quality[index]), "t0": t0, "fs": fs, "n": int(data.shape[0]), "data_unit": "mV", "storage_dtype": "float32"})
    return {"seconds": time.perf_counter() - start, "bytes": directory_bytes(root), "files": CHANNELS}


def write_layout_parquet(path: Path, data: np.ndarray, quality: np.ndarray, fs: float, t0: float, chunk_rows: int, compression: str) -> dict:
    if pa is None or pq is None:
        raise RuntimeError("Parquet requires pyarrow: python -m pip install pyarrow")
    start = time.perf_counter()
    fields = [pa.field(name, pa.float32()) for name in CHANNEL_NAMES]
    metadata = {b"t0": str(t0).encode(), b"fs": str(fs).encode(), b"n": str(data.shape[0]).encode(), b"data_unit": b"mV", b"storage_dtype": b"float32", b"channel_quality": json.dumps(quality.tolist()).encode()}
    schema = pa.schema(fields, metadata=metadata)
    writer = pq.ParquetWriter(path, schema, compression=compression, use_dictionary=False)
    try:
        for first in range(0, data.shape[0], chunk_rows):
            block = data[first : first + chunk_rows]
            arrays = [pa.array(block[:, col]) for col in range(CHANNELS)]
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
    finally:
        writer.close()
    return {"seconds": time.perf_counter() - start, "bytes": path.stat().st_size, "files": 1}


def write_layout_per_channel_parquet(root: Path, data: np.ndarray, quality: np.ndarray, fs: float, t0: float, chunk_rows: int, compression: str) -> dict:
    if pa is None or pq is None:
        raise RuntimeError("Parquet requires pyarrow: python -m pip install pyarrow")
    start = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(CHANNEL_NAMES):
        field = pa.field("signal", pa.float32())
        metadata = {
            b"t0": str(t0).encode(),
            b"fs": str(fs).encode(),
            b"n": str(data.shape[0]).encode(),
            b"good_channel": str(int(quality[index])).encode(),
            b"data_unit": b"mV",
            b"storage_dtype": b"float32",
        }
        schema = pa.schema([field], metadata=metadata)
        path = root / f"{name}.parquet"
        writer = pq.ParquetWriter(path, schema, compression=compression, use_dictionary=False)
        try:
            for first in range(0, data.shape[0], chunk_rows):
                block = pa.array(data[first : first + chunk_rows, index])
                writer.write_batch(pa.RecordBatch.from_arrays([block], schema=schema))
        finally:
            writer.close()
    return {"seconds": time.perf_counter() - start, "bytes": directory_bytes(root), "files": CHANNELS}


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def read_all_h5(path: Path, channels: np.ndarray | None = None) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        data = handle[DATASET_NAME]
        if channels is None:
            return np.asarray(data[:], dtype=np.float32)
        return np.asarray(data[:, channels], dtype=np.float32)


def read_per_channel(root: Path, channels: np.ndarray | None = None) -> np.ndarray:
    selected = np.arange(CHANNELS) if channels is None else np.asarray(channels, dtype=int)
    columns = []
    for channel in selected:
        with h5py.File(root / f"ch{channel + 1:03d}.h5", "r") as handle:
            columns.append(np.asarray(handle[DATASET_NAME][:], dtype=np.float32))
    return np.column_stack(columns).astype(np.float32, copy=False)


def read_parquet(path: Path, channels: np.ndarray | None = None) -> np.ndarray:
    if pq is None:
        raise RuntimeError("Parquet requires pyarrow.")
    selected = np.arange(CHANNELS) if channels is None else np.asarray(channels, dtype=int)
    names = [CHANNEL_NAMES[index] for index in selected]
    table = pq.read_table(path, columns=names, use_threads=True)
    return table.to_pandas().to_numpy(dtype=np.float32, copy=False)


def read_per_channel_parquet(root: Path, channels: np.ndarray | None = None) -> np.ndarray:
    selected = np.arange(CHANNELS) if channels is None else np.asarray(channels, dtype=int)
    columns = []
    for channel in selected:
        table = pq.read_table(root / f"ch{channel + 1:03d}.parquet", columns=["signal"], use_threads=True)
        columns.append(table.column(0).to_numpy(zero_copy_only=False).astype(np.float32, copy=False))
    return np.column_stack(columns).astype(np.float32, copy=False)


def current_rss() -> int | None:
    return None if psutil is None else int(psutil.Process().memory_info().rss)


def measure(function: Callable[[], object]) -> tuple[object, dict]:
    before = current_rss()
    peak = before
    stop = threading.Event()

    def sample_memory():
        nonlocal peak
        while not stop.wait(0.01):
            value = current_rss()
            if value is not None:
                peak = max(peak or value, value)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    start = time.perf_counter()
    value = function()
    elapsed = time.perf_counter() - start
    stop.set()
    sampler.join(timeout=1)
    after = current_rss()
    return value, {"seconds": elapsed, "rss_before": before, "rss_after": after, "peak_rss": peak, "delta_rss": None if before is None else peak - before}


def open_latency(layout: str, path: Path, channels_root: Path | None = None) -> dict:
    start = time.perf_counter()
    if path.is_dir():
        for child in sorted(path.glob("*.h5")):
            with h5py.File(child, "r") as handle:
                _ = dict(handle.attrs)
                _ = handle[DATASET_NAME].shape
    elif layout == "all_h5" or layout == "h5":
        with h5py.File(path, "r") as handle:
            _ = dict(handle.attrs)
            _ = handle[DATASET_NAME].shape
    elif layout == "parquet":
        _ = pq.ParquetFile(path).schema
    elif layout == "per_channel_parquet":
        assert channels_root is not None
        for index in range(CHANNELS):
            _ = pq.ParquetFile(channels_root / f"ch{index + 1:03d}.parquet").schema
    else:
        assert channels_root is not None
        for index in range(CHANNELS):
            with h5py.File(channels_root / f"ch{index + 1:03d}.h5", "r") as handle:
                _ = dict(handle.attrs)
                _ = handle[DATASET_NAME].shape
    return {"seconds": time.perf_counter() - start}


def open_one_metadata_latency(layout: str, path: Path) -> dict:
    start = time.perf_counter()
    if layout == "per_channel_parquet":
        _ = pq.ParquetFile(path).schema
    else:
        with h5py.File(path, "r") as handle:
            _ = dict(handle.attrs)
            _ = handle[DATASET_NAME].shape
    return {"seconds": time.perf_counter() - start}


def matlab_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def matlab_smoke_test(matlab_exe: str | None, h5_paths: dict[str, Path], per_channel_names: set[str] | None = None) -> dict:
    if not matlab_exe:
        return {"available": False, "reason": "matlab executable was not found in PATH"}
    per_channel_names = per_channel_names or set()
    tests = {
        name: (
            f"h5read('{matlab_quote(path)}','/signal',[1],[1]);"
            if name in per_channel_names
            else f"h5read('{matlab_quote(path)}','/signal',[1 1],[1 1]);"
        )
        for name, path in h5_paths.items()
    }
    result = {"available": True, "executable": matlab_exe, "tests": {}}
    for name, expression in tests.items():
        start = time.perf_counter()
        try:
            proc = subprocess.run([matlab_exe, "-batch", expression], capture_output=True, text=True, timeout=180)
            result["tests"][name] = {
                "ok": proc.returncode == 0,
                "seconds": time.perf_counter() - start,
                "returncode": proc.returncode,
                "stderr": proc.stderr[-2000:],
            }
        except Exception as exc:
            result["tests"][name] = {"ok": False, "seconds": time.perf_counter() - start, "error": str(exc)}
    return result


def rmse_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {"rmse": float(np.sqrt(np.mean(diff * diff))), "max_abs": float(np.max(np.abs(diff))), "mean_abs": float(np.mean(np.abs(diff)))}


def gui_make_epochs(
    channels_data: np.ndarray,
    markers: np.ndarray,
    fs: float,
    t0: float,
    epoch_start: float,
    epoch_end: float,
    tag: int | None = None,
    max_trials: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror integrated_pipeline_gui.make_epochs for a loaded segment."""
    start_offset = int(round(epoch_start * fs / 1000.0))
    end_offset = int(round(epoch_end * fs / 1000.0))
    if end_offset <= start_offset:
        raise ValueError("Epoch end must be after epoch start.")
    offset = int(round(t0 * fs))
    use_markers = markers if tag is None else markers[markers[:, 1] == int(tag)]
    epochs = []
    used = []
    for marker in use_markers:
        center = int(marker[0]) - offset
        begin = center + start_offset
        finish = center + end_offset
        if begin < 0 or finish > channels_data.shape[0]:
            continue
        epochs.append(channels_data[begin:finish, :])
        used.append(marker)
        if max_trials > 0 and len(epochs) >= max_trials:
            break
    if not epochs:
        raise RuntimeError("No valid epochs were found in the loaded data segment.")
    t_ms = np.arange(start_offset, end_offset, dtype=np.float64) / fs * 1000.0
    return np.stack(epochs, axis=0).astype(np.float32, copy=False), np.asarray(used, dtype=np.int64), t_ms


def gui_baseline_correct_epochs(epochs: np.ndarray, t_ms: np.ndarray, baseline_start: float, baseline_end: float) -> np.ndarray:
    """Mirror integrated_pipeline_gui.baseline_correct_epochs."""
    mask = (t_ms >= baseline_start) & (t_ms <= baseline_end)
    if not np.any(mask):
        return epochs
    base = np.nanmean(epochs[:, mask, :], axis=1, keepdims=True)
    return np.asarray(epochs - base, dtype=np.float32)


def aggregate_epochs(data: np.ndarray, axis: int | tuple[int, ...], method: str) -> np.ndarray:
    return np.nanmedian(data, axis=axis) if method == "median" else np.nanmean(data, axis=axis)


def parse_int_list(text: str | None) -> list[int]:
    if text is None or not str(text).strip():
        return []
    values = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(int(float(item)))
    return values


def marker_epoch_availability(
    markers: np.ndarray,
    fs: float,
    t0: float,
    sample_count: int,
    epoch_start: float,
    epoch_end: float,
) -> dict[int, dict[str, int | float]]:
    start_offset = int(round(epoch_start * fs / 1000.0))
    end_offset = int(round(epoch_end * fs / 1000.0))
    segment_start_sample = int(round(t0 * fs))
    result = {}
    for tag in sorted(int(value) for value in np.unique(markers[:, 1])):
        rows = markers[markers[:, 1] == tag]
        valid = 0
        for marker in rows:
            center = int(marker[0]) - segment_start_sample
            if center + start_offset >= 0 and center + end_offset <= sample_count:
                valid += 1
        result[tag] = {
            "marker_count": int(rows.shape[0]),
            "valid_epoch_count": valid,
            "first_marker_sample": int(rows[0, 0]),
            "last_marker_sample": int(rows[-1, 0]),
        }
    return result


def epoch_sample_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    result = rmse_metrics(reference, candidate)
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if finite.sum() > 2 and np.std(reference[finite]) > 0 and np.std(candidate[finite]) > 0:
        result["correlation"] = float(np.corrcoef(reference[finite], candidate[finite])[0, 1])
    else:
        result["correlation"] = float("nan")
    return result


def vep_analysis(data: np.ndarray, markers: np.ndarray, fs: float, t0: float, epoch_start: float, epoch_end: float, baseline_start: float, baseline_end: float, response_start: float, response_end: float, trial_a: int, trial_b: int, aggregate: str = "mean", tags: list[int] | None = None) -> tuple[dict, dict[int, np.ndarray]]:
    start_offset = int(round(epoch_start * fs / 1000.0))
    end_offset = int(round(epoch_end * fs / 1000.0))
    t_ms = np.arange(start_offset, end_offset, dtype=np.float64) / fs * 1000.0
    tags = sorted(int(tag) for tag in (np.unique(markers[:, 1]) if tags is None else tags))
    results = {}
    waves = {}
    base_mask = (t_ms >= baseline_start) & (t_ms <= baseline_end)
    resp_mask = (t_ms >= response_start) & (t_ms <= response_end)
    for tag in tags:
        try:
            epochs, _, _ = gui_make_epochs(data, markers, fs, t0, epoch_start, epoch_end, tag=tag, max_trials=trial_b)
        except RuntimeError:
            continue
        corrected = gui_baseline_correct_epochs(epochs, t_ms, baseline_start, baseline_end)
        avg_a = aggregate_epochs(corrected[: min(trial_a, corrected.shape[0])], axis=(0, 2), method=aggregate)
        avg_b = aggregate_epochs(corrected[: min(trial_b, corrected.shape[0])], axis=(0, 2), method=aggregate)
        finite = np.isfinite(avg_a) & np.isfinite(avg_b)
        corr = float(np.corrcoef(avg_a[finite], avg_b[finite])[0, 1]) if finite.sum() > 2 else float("nan")
        peak_a = float(np.nanmax(np.abs(avg_a)))
        peak_b = float(np.nanmax(np.abs(avg_b)))
        change = (peak_b - peak_a) / max(abs(peak_a), np.finfo(np.float32).eps) * 100.0
        base_power = np.nanmean(corrected[:, base_mask, :] ** 2, axis=(0, 1)) if np.any(base_mask) else np.full(data.shape[1], np.nan)
        resp_power = np.nanmean(corrected[:, resp_mask, :] ** 2, axis=(0, 1)) if np.any(resp_mask) else np.full(data.shape[1], np.nan)
        snr = 10.0 * np.log10((resp_power + np.finfo(np.float32).eps) / (base_power + np.finfo(np.float32).eps))
        results[tag] = {"tag": tag, "valid_trials": int(corrected.shape[0]), "trial_a": min(trial_a, corrected.shape[0]), "trial_b": min(trial_b, corrected.shape[0]), "corr": corr, "peak_a": peak_a, "peak_b": peak_b, "change_percent": float(change), "event_snr_median_db": float(np.nanmedian(snr)), "epoch_samples": int(end_offset - start_offset), "epoch_start_ms": epoch_start, "epoch_end_ms": epoch_end, "baseline_start_ms": baseline_start, "baseline_end_ms": baseline_end, "response_start_ms": response_start, "response_end_ms": response_end, "aggregate": aggregate}
        waves[tag] = np.asarray(aggregate_epochs(corrected, axis=0, method=aggregate), dtype=np.float32)
    return results, waves


def save_vep_plot(path: Path, layout_waves: dict[str, dict[int, np.ndarray]], tag: int, channel_index: int, t_ms: np.ndarray) -> None:
    if plt is None:
        return
    available = [waves[tag] for waves in layout_waves.values() if tag in waves]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for layout, waves in layout_waves.items():
        if tag in waves:
            ax.plot(t_ms, waves[tag][:, channel_index], label=layout)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"LetterModeData-style VEP | tag {tag} | ch{channel_index + 1:03d}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_epoch_sample_plot(
    path: Path,
    samples_by_layout: dict[str, dict[int, dict]],
    tag: int,
    trial_index: int,
    channel_position: int,
    corrected: bool,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for layout, tag_samples in samples_by_layout.items():
        sample = tag_samples.get(tag)
        if sample is None or trial_index >= sample["raw_epochs"].shape[0]:
            continue
        values = sample["corrected_epochs"] if corrected else sample["raw_epochs"]
        ax.plot(sample["t_ms"], values[trial_index, :, channel_position], label=layout, linewidth=1.0)
    ax.axvline(0, color="black", linewidth=0.8)
    first_sample = next((tag_samples[tag] for tag_samples in samples_by_layout.values() if tag in tag_samples), None)
    channel_label = int(first_sample["channels"][channel_position]) if first_sample is not None else channel_position + 1
    ax.set_title(
        f"LetterModeData epoch sample | tag {tag} | trial {trial_index + 1} | "
        f"ch{channel_label:03d}"
    )
    ax.set_xlabel("Time from Letter onset (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def copy_benchmark(source: Path, destination: Path) -> dict:
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    start = time.perf_counter()
    if source.is_dir():
        shutil.copytree(source, destination)
        copied_bytes = directory_bytes(destination)
        copied_files = sum(1 for item in destination.rglob("*") if item.is_file())
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_bytes = destination.stat().st_size
        copied_files = 1
    return {"seconds": time.perf_counter() - start, "bytes": copied_bytes, "files": copied_files}


def build_overview_rows(results: dict) -> tuple[list[str], list[dict]]:
    subset_sizes = [1, 8, 32, 128, CHANNELS]
    headers = [
        "layout", "layout_label", "layout_mode", "compression_variant", "files", "bytes", "size_GiB", "compression_ratio",
        "write_seconds", "end_to_end_estimated_seconds", "open_metadata_seconds",
        "open_one_channel_metadata_seconds", "full_read_seconds",
        "full_read_peak_rss_GiB", "full_read_delta_rss_GiB",
        "full_read_rmse", "full_read_max_abs", "copy_seconds", "copy_files",
        "vep_read_seconds", "vep_valid_tags", "matlab_smoke_ok",
    ]
    for subset_size in subset_sizes:
        headers.extend([f"read_{subset_size}ch_seconds", f"read_{subset_size}ch_GiB_per_second"])

    matlab_tests = results.get("matlab_smoke", {}).get("tests", {}) if isinstance(results.get("matlab_smoke"), dict) else {}
    rows = []
    for name, info in results.get("layouts", {}).items():
        full_read = info.get("full_read", {})
        copy_info = info.get("copy_backup") or {}
        vep_info = results.get("vep", {}).get(name, {}) if isinstance(results.get("vep"), dict) else {}
        row = {
            "layout": name,
            "layout_label": info.get("layout_label", name),
            "layout_mode": info.get("layout_mode"),
            "compression_variant": info.get("compression_variant"),
            "files": info.get("files"),
            "bytes": info.get("bytes"),
            "size_GiB": (info.get("bytes") or 0) / 1024**3,
            "compression_ratio": info.get("compression_ratio_vs_float32"),
            "write_seconds": info.get("seconds"),
            "end_to_end_estimated_seconds": info.get("end_to_end_estimated_seconds"),
            "open_metadata_seconds": info.get("open_metadata_seconds"),
            "open_one_channel_metadata_seconds": info.get("open_one_channel_metadata_seconds"),
            "full_read_seconds": full_read.get("seconds"),
            "full_read_peak_rss_GiB": (full_read.get("peak_rss") or 0) / 1024**3 if full_read.get("peak_rss") is not None else None,
            "full_read_delta_rss_GiB": (full_read.get("delta_rss") or 0) / 1024**3 if full_read.get("delta_rss") is not None else None,
            "full_read_rmse": info.get("rmse_vs_reference", {}).get("rmse"),
            "full_read_max_abs": info.get("rmse_vs_reference", {}).get("max_abs"),
            "copy_seconds": copy_info.get("seconds"),
            "copy_files": copy_info.get("files"),
            "vep_read_seconds": vep_info.get("read_for_vep", {}).get("seconds") if isinstance(vep_info, dict) else None,
            "vep_valid_tags": len(vep_info.get("tags", {})) if isinstance(vep_info, dict) and isinstance(vep_info.get("tags"), dict) else None,
            "matlab_smoke_ok": matlab_tests.get(name, {}).get("ok") if isinstance(matlab_tests.get(name), dict) else None,
        }
        for subset_size in subset_sizes:
            subset = info.get("subset_reads", {}).get(str(subset_size), {})
            row[f"read_{subset_size}ch_seconds"] = subset.get("seconds")
            row[f"read_{subset_size}ch_GiB_per_second"] = subset.get("GiB_per_second")
        rows.append(row)
    return headers, rows


def write_summary_workbook(results: dict, path: Path) -> bool:
    if openpyxl is None:
        return False
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    headers, overview_rows = build_overview_rows(results)
    workbook = openpyxl.Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)

    def excel_value(value: object) -> object:
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, ensure_ascii=False, default=_json_value)
        if isinstance(value, Path):
            return str(value)
        return value

    def add_table(name: str, table_headers: list[str], table_rows: list[list[object]]) -> None:
        sheet = workbook.create_sheet(name)
        sheet.append(table_headers)
        for row in table_rows:
            sheet.append([excel_value(value) for value in row])
        sheet.freeze_panes = "A2"
        if table_headers:
            sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
        for index, column in enumerate(sheet.columns, start=1):
            values = [str(cell.value) if cell.value is not None else "" for cell in column[:200]]
            width = min(32, max(12, max((len(value) for value in values), default=12) + 2))
            sheet.column_dimensions[get_column_letter(index)].width = width

    add_table("Overview", headers, [[row.get(header) for header in headers] for row in overview_rows])

    subset_headers = ["layout", "channel_count", "seconds", "GiB_per_second", "shape", "dtype", "peak_rss_GiB", "delta_rss_GiB"]
    subset_rows = []
    for name, info in results.get("layouts", {}).items():
        for count, value in sorted(info.get("subset_reads", {}).items(), key=lambda item: int(item[0])):
            subset_rows.append([
                name, int(count), value.get("seconds"), value.get("GiB_per_second"),
                str(value.get("shape")), value.get("dtype"),
                (value.get("peak_rss") or 0) / 1024**3 if value.get("peak_rss") is not None else None,
                (value.get("delta_rss") or 0) / 1024**3 if value.get("delta_rss") is not None else None,
            ])
    add_table("Subset Reads", subset_headers, subset_rows)

    vep_headers = ["layout", "tag", "valid_trials", "trial_a", "trial_b", "corr", "peak_a", "peak_b", "change_percent", "event_snr_median_db", "epoch_samples", "read_seconds"]
    vep_rows = []
    vep_data = results.get("vep", {})
    if isinstance(vep_data, dict) and vep_data.get("status") == "skipped":
        vep_rows.append(["STATUS", "", "", "", "", "", "", "", "", vep_data.get("reason", ""), "", ""])
    else:
        for name, payload in vep_data.items() if isinstance(vep_data, dict) else []:
            if not isinstance(payload, dict):
                continue
            read_seconds = payload.get("read_for_vep", {}).get("seconds")
            for tag, row in payload.get("tags", {}).items():
                vep_rows.append([
                    name, tag, row.get("valid_trials"), row.get("trial_a"), row.get("trial_b"),
                    row.get("corr"), row.get("peak_a"), row.get("peak_b"), row.get("change_percent"),
                    row.get("event_snr_median_db"), row.get("epoch_samples"), read_seconds,
                ])
    add_table("VEP Metrics", vep_headers, vep_rows)

    epoch_headers = [
        "layout", "tag", "trial_count", "epoch_samples", "channels", "raw_rmse", "raw_max_abs",
        "raw_correlation", "baseline_corrected_rmse", "baseline_corrected_max_abs",
        "baseline_corrected_correlation",
    ]
    epoch_rows = [[row.get(header) for header in epoch_headers] for row in results.get("vep_epoch_compare_vs_reference", results.get("vep_epoch_compare_vs_all_h5", []))]
    add_table("Epoch Compare", epoch_headers, epoch_rows)

    metadata_rows = []
    for key in ["source", "source_kind", "samples", "channels", "fs", "t0", "deltaT1_sec", "bin_parse_seconds", "marker_source", "marker_count", "marker_tags", "quality_good", "quality_bad"]:
        metadata_rows.append([key, results.get(key)])
    for key, value in results.get("timing_info", {}).items():
        metadata_rows.append([f"timing_info.{key}", value])
    for key, value in results.get("parameters", {}).items():
        metadata_rows.append([f"parameter.{key}", value])
    add_table("Metadata", ["key", "value"], metadata_rows)

    layout_map = {row["layout"]: row for row in overview_rows}
    best_storage = min(overview_rows, key=lambda row: row["bytes"] or float("inf")) if overview_rows else None
    best_write = min(overview_rows, key=lambda row: row["write_seconds"] or float("inf")) if overview_rows else None
    best_full_read = min(overview_rows, key=lambda row: row["full_read_seconds"] or float("inf")) if overview_rows else None
    if best_storage and best_write and best_storage["layout"] == best_write["layout"]:
        recommendation = f"Recommended for storage and write speed: {best_storage['layout']}"
        reasons = (
            f"It has the smallest file ({best_storage['size_GiB']:.3f} GiB) and fastest write "
            f"({best_write['write_seconds']:.2f} s). The fastest full read is "
            f"{best_full_read['layout']} ({best_full_read['full_read_seconds']:.2f} s)."
        )
    elif best_storage and best_write:
        recommendation = f"Storage: {best_storage['layout']}; write speed: {best_write['layout']}"
        reasons = (
            f"The smallest file is {best_storage['size_GiB']:.3f} GiB, while the fastest write is "
            f"{best_write['write_seconds']:.2f} s. The fastest full read is "
            f"{best_full_read['layout']} ({best_full_read['full_read_seconds']:.2f} s)."
        )
    else:
        recommendation = "No layout was generated"
        reasons = "Use the generated Overview values to choose the smallest and fastest layout."
    conclusion_rows = [
        ["recommendation", recommendation],
        ["reasons", reasons],
        ["smallest_storage", f"{best_storage['layout']} ({best_storage['size_GiB']:.3f} GiB)" if best_storage else ""],
        ["fastest_write", f"{best_write['layout']} ({best_write['write_seconds']:.2f} s)" if best_write else ""],
        ["fastest_full_read", f"{best_full_read['layout']} ({best_full_read['full_read_seconds']:.2f} s)" if best_full_read else ""],
        ["data_integrity_rule", "full_read_rmse and epoch RMSE should be approximately 0 for lossless layouts"],
        ["migration_note", "All layouts are native HDF5. gzip1/gzip4 use standard HDF5 filters; LZF and Blosc/LZ4 depend on filter support. Python needs hdf5plugin for Blosc/LZ4, and MATLAB h5read compatibility should be checked with the smoke-test results."],
    ]
    add_table("Conclusion", ["item", "value"], conclusion_rows)
    workbook.save(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark four HDF5 compression filters in all-channel and per-channel layouts with simplified LetterModeData VEP validation.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--bin-file", type=Path, help="Original BIN file; decode it before writing all layouts")
    source_group.add_argument("--source-h5", type=Path, help="Canonical float32 H5 containing rawData512 or signal")
    parser.add_argument("--markers", type=Path, default=None, help="Fallback CSV with sample,tag columns")
    parser.add_argument("--log-txt", type=Path, default=None, help="GUI experiment log.txt")
    parser.add_argument("--event-csv", type=Path, default=None, help="GUI Letter Event CSV")
    parser.add_argument("--deltaT1-ms", type=float, default=None, help="Optional override when the source H5 lacks deltaT1 metadata")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=0.0, help="BIN segment start in seconds")
    parser.add_argument("--duration-sec", type=float, default=0.0, help="BIN segment duration; 0 means full length")
    parser.add_argument("--quality", type=Path, default=None, help="Optional file with 512 quality values")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--epoch-start-ms", type=float, default=-500)
    parser.add_argument("--epoch-end-ms", type=float, default=800)
    parser.add_argument("--baseline-start-ms", type=float, default=-200)
    parser.add_argument("--baseline-end-ms", type=float, default=0)
    parser.add_argument("--response-start-ms", type=float, default=0)
    parser.add_argument("--response-end-ms", type=float, default=300)
    parser.add_argument("--trial-a", type=int, default=30)
    parser.add_argument("--trial-b", type=int, default=60)
    parser.add_argument("--plot-channel", type=int, default=1)
    parser.add_argument("--aggregate", choices=("mean", "median"), default="mean", help="LetterModeData task_response_aggregate mode")
    parser.add_argument("--vep-tags", default="", help="Optional comma-separated Letter tags, e.g. 4,5,6,7; empty means all marker tags")
    parser.add_argument("--epoch-sample-trials", type=int, default=3, help="Number of valid epochs per tag to save for cross-layout comparison")
    parser.add_argument("--epoch-sample-channels", default="", help="Optional 1-based channels to save, e.g. 1,2,17; empty means --plot-channel")
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument("--skip-matlab", action="store_true")
    parser.add_argument("--run-vep", action="store_true", help="Optional: run the LetterModeData-style VEP validation")
    args = parser.parse_args()

    if hdf5plugin is None:
        parser.error(
            "Blosc/LZ4 + bitshuffle comparison requires hdf5plugin. "
            "Install it with: python -m pip install hdf5plugin "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple. "
            "If no wheel exists for Python 3.14, use Python 3.11 or 3.12."
        )
    if args.bin_file is not None and not args.bin_file.is_file():
        parser.error(f"BIN file does not exist: {args.bin_file}")
    if args.source_h5 is not None and not args.source_h5.is_file():
        parser.error(f"H5 file does not exist: {args.source_h5}")
    if args.run_vep and args.markers is None and (args.log_txt is None or args.event_csv is None):
        parser.error("--run-vep requires --log-txt + --event-csv, or --markers as fallback.")
    if args.markers is not None and not args.markers.is_file():
        parser.error(f"Marker file does not exist: {args.markers}")
    if args.run_vep and args.log_txt is not None and not args.log_txt.is_file():
        parser.error(f"log.txt does not exist: {args.log_txt}")
    if args.run_vep and args.event_csv is not None and not args.event_csv.is_file():
        parser.error(f"Event CSV does not exist: {args.event_csv}")
    if not 1 <= args.plot_channel <= CHANNELS:
        parser.error("--plot-channel must be between 1 and 512")
    if args.epoch_sample_trials <= 0:
        parser.error("--epoch-sample-trials must be positive")
    requested_sample_channels = parse_int_list(args.epoch_sample_channels)
    if not requested_sample_channels:
        requested_sample_channels = [args.plot_channel]
    if any(channel < 1 or channel > CHANNELS for channel in requested_sample_channels):
        parser.error("--epoch-sample-channels must contain values between 1 and 512")
    requested_vep_tags = parse_int_list(args.vep_tags)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.bin_file is not None:
        data, fs, t0, n, source_meta, bin_parse_seconds = load_reference_from_bin(
            args.bin_file, args.start_sec, args.duration_sec, args.chunk_rows
        )
        source_path = args.bin_file
        source_kind = "BIN"
    else:
        data, fs, t0, n, source_meta = load_signal(args.source_h5)
        bin_parse_seconds = None
        source_path = args.source_h5
        source_kind = "H5 reference"
    quality = load_quality(args.quality, source_meta["channel_quality"])
    delta_t1_sec = float(source_meta.get("deltaT1_sec", 0.0))
    if args.deltaT1_ms is not None:
        delta_t1_sec = float(args.deltaT1_ms) / 1000.0
    markers = None
    timing_info = {}
    marker_source = "not used"
    if args.run_vep:
        if args.log_txt is not None and args.event_csv is not None:
            timing_info = read_gui_log_timing(args.log_txt, delta_t1_sec)
            markers = build_gui_letter_markers(args.event_csv, timing_info, fs)
            marker_source = "GUI log.txt + Letter Event CSV"
        else:
            timing_info = {"deltaT1_sec": delta_t1_sec, "stim_delay_sec": None}
            markers = load_markers(args.markers)
            marker_source = "fallback sample/tag CSV"
    good = np.flatnonzero(quality == 1)
    if good.size == 0:
        raise RuntimeError("No good channels are available.")
    print(f"Reference: {data.shape}, float32, fs={fs}, t0={t0}")
    if markers is None:
        print(f"Markers: not used; good channels={good.size}")
    else:
        print(f"Markers: {len(markers)}; tags={sorted(set(markers[:, 1].tolist()))}; good channels={good.size}")
    marker_availability = {}
    if args.run_vep:
        marker_availability = marker_epoch_availability(
            markers, fs, t0, n,
            args.epoch_start_ms, args.epoch_end_ms,
        )
        print(f"Marker sample range: {int(markers[:, 0].min())}..{int(markers[:, 0].max())}")
        print(f"Segment absolute sample range: {int(round(t0 * fs))}..{int(round(t0 * fs)) + n - 1}")
        print(f"Valid epoch availability by tag: {marker_availability}")

    layout_specs = {}
    for variant, spec in COMPRESSION_VARIANTS.items():
        all_name = f"all_{variant}"
        per_name = f"per_{variant}"
        all_path = args.output_dir / spec["filename"]
        per_path = args.output_dir / f"per_channel_{variant}"
        for path in (all_path, per_path):
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        layout_specs[all_name] = {"variant": variant, "mode": "all", "path": all_path}
        layout_specs[per_name] = {"variant": variant, "mode": "per", "path": per_path}

    results = {"source": str(source_path.resolve()), "source_kind": source_kind, "source_float32_bytes": int(data.nbytes), "samples": n, "channels": CHANNELS, "fs": fs, "t0": t0, "deltaT1_sec": delta_t1_sec, "bin_parse_seconds": bin_parse_seconds, "marker_source": marker_source, "timing_info": timing_info, "marker_count": None if markers is None else int(len(markers)), "marker_tags": [] if markers is None else sorted(set(markers[:, 1].tolist())), "marker_epoch_availability": marker_availability, "quality_good": int(good.size), "quality_bad": int(CHANNELS - good.size), "parameters": vars(args).copy(), "layouts": {}, "vep": {}}
    results["parameters"]["source_h5"] = str(args.source_h5) if args.source_h5 else ""
    results["parameters"]["bin_file"] = str(args.bin_file) if args.bin_file else ""
    results["parameters"]["markers"] = str(args.markers) if args.markers else ""
    results["parameters"]["log_txt"] = str(args.log_txt) if args.log_txt else ""
    results["parameters"]["event_csv"] = str(args.event_csv) if args.event_csv else ""

    if markers is not None:
        marker_csv = args.output_dir / "vep_markers_used.csv"
        with marker_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            selected_tags = set(requested_vep_tags)
            writer.writerow(["sample", "time_sec", "tag", "selected_for_vep"])
            for sample, tag in markers:
                writer.writerow([int(sample), float(sample) / fs, int(tag), not selected_tags or int(tag) in selected_tags])
        results["vep_marker_csv"] = str(marker_csv.resolve())

    for name, spec in layout_specs.items():
        variant = spec["variant"]
        layout_label = COMPRESSION_VARIANTS[variant]["label"]
        if spec["mode"] == "all":
            print(f"Writing all-channel {layout_label}...")
            results["layouts"][name] = write_compressed_h5(
                spec["path"], data, quality, fs, t0, delta_t1_sec, args.chunk_rows, variant
            )
        else:
            print(f"Writing per-channel {layout_label}...")
            results["layouts"][name] = write_compressed_per_channel_h5(
                spec["path"], data, quality, fs, t0, delta_t1_sec, args.chunk_rows, variant
            )
        results["layouts"][name]["layout_mode"] = spec["mode"]
        results["layouts"][name]["layout_label"] = f"{'all-channel' if spec['mode'] == 'all' else 'per-channel'} {layout_label}"

    def make_reader(spec: dict) -> Callable[[np.ndarray | None], np.ndarray]:
        if spec["mode"] == "all":
            return lambda channels=None, path=spec["path"]: read_all_h5(path, channels)
        return lambda channels=None, path=spec["path"]: read_per_channel(path, channels)

    readers = {name: make_reader(spec) for name, spec in layout_specs.items()}
    layout_paths = {name: spec["path"] for name, spec in layout_specs.items()}
    matlab_paths = {
        name: spec["path"] if spec["mode"] == "all" else spec["path"] / "ch001.h5"
        for name, spec in layout_specs.items()
    }
    subset_sizes = [1, 8, 32, 128, CHANNELS]
    for name, reader in readers.items():
        info = results["layouts"][name]
        if bin_parse_seconds is not None:
            info["end_to_end_estimated_seconds"] = bin_parse_seconds + info["seconds"]
        info["compression_ratio_vs_float32"] = results["source_float32_bytes"] / max(info["bytes"], 1)
        info["open_metadata_seconds"] = open_latency("h5", layout_paths[name], None)["seconds"]
        if layout_specs[name]["mode"] == "per":
            info["open_one_channel_metadata_seconds"] = open_one_metadata_latency("h5", layout_paths[name] / "ch001.h5")["seconds"]
        info["subset_reads"] = {}
        for subset_size in subset_sizes:
            columns = good[:subset_size] if subset_size < CHANNELS else np.arange(CHANNELS)
            value, timing = measure(lambda r=reader, c=columns: r(c))
            info["subset_reads"][str(subset_size)] = {**timing, "shape": list(value.shape), "dtype": str(value.dtype), "GiB_per_second": value.nbytes / max(timing["seconds"], 1e-12) / 1024**3}
            del value
            gc.collect()
        full_value, full_timing = measure(lambda r=reader: r(None))
        info["full_read"] = {**full_timing, "shape": list(full_value.shape), "dtype": str(full_value.dtype), "GiB_per_second": full_value.nbytes / max(full_timing["seconds"], 1e-12) / 1024**3}
        info["rmse_vs_reference"] = rmse_metrics(data, full_value)
        del full_value

        if not args.skip_copy:
            copy_source = layout_paths[name]
            copy_destination = args.output_dir / "copy_benchmark" / name
            info["copy_backup"] = copy_benchmark(copy_source, copy_destination)
        else:
            info["copy_backup"] = None

    if not args.run_vep:
        results["vep"] = {"status": "skipped", "reason": "VEP validation is disabled; use --run-vep to enable it."}
    else:
        print("Running simplified LetterModeData VEP...")
        vep_layouts = {}
        layout_waves = {}
        layout_epoch_samples = {}
        layout_epoch_meta = {}
        sample_channels = [channel for channel in requested_sample_channels if (channel - 1) in set(good.tolist())]
        if not sample_channels:
            sample_channels = [int(good[0]) + 1]
        sample_positions = np.asarray([int(np.flatnonzero(good == channel - 1)[0]) for channel in sample_channels], dtype=int)
        vep_tags = requested_vep_tags or None
        epoch_offsets = np.arange(int(round(args.epoch_start_ms * fs / 1000.0)), int(round(args.epoch_end_ms * fs / 1000.0)), dtype=np.float32)
        t_ms = epoch_offsets / np.float32(fs) * np.float32(1000.0)
        epoch_sample_dir = args.output_dir / "vep_epoch_samples"
        epoch_sample_dir.mkdir(exist_ok=True)
        for name, reader in readers.items():
            value, timing = measure(lambda r=reader, c=good: r(c))
            vep_result, waves = vep_analysis(
                value, markers, fs, t0,
                args.epoch_start_ms, args.epoch_end_ms,
                args.baseline_start_ms, args.baseline_end_ms,
                args.response_start_ms, args.response_end_ms,
                args.trial_a, args.trial_b,
                aggregate=args.aggregate,
                tags=vep_tags,
            )
            vep_layouts[name] = {"read_for_vep": timing, "tags": vep_result}
            layout_waves[name] = waves
            layout_epoch_samples[name] = {}
            layout_epoch_meta[name] = {}
            tags_to_save = sorted(vep_result)
            for tag in tags_to_save:
                try:
                    raw_epochs, used_markers, sample_t_ms = gui_make_epochs(
                        value, markers, fs, t0,
                        args.epoch_start_ms, args.epoch_end_ms,
                        tag=tag, max_trials=args.epoch_sample_trials,
                    )
                except RuntimeError:
                    continue
                corrected_epochs = gui_baseline_correct_epochs(
                    raw_epochs, sample_t_ms, args.baseline_start_ms, args.baseline_end_ms
                )
                sample = {
                    "raw_epochs": np.asarray(raw_epochs[:, :, sample_positions], dtype=np.float32),
                    "corrected_epochs": np.asarray(corrected_epochs[:, :, sample_positions], dtype=np.float32),
                    "used_markers": np.asarray(used_markers, dtype=np.int64),
                    "t_ms": np.asarray(sample_t_ms, dtype=np.float64),
                    "channels": np.asarray(sample_channels, dtype=np.int64),
                }
                sample_path = epoch_sample_dir / f"{name}_tag_{tag}.npz"
                np.savez_compressed(sample_path, **sample)
                layout_epoch_samples[name][tag] = sample
                layout_epoch_meta[name][str(tag)] = {
                    "path": str(sample_path.resolve()),
                    "shape": list(sample["raw_epochs"].shape),
                    "channels": sample_channels,
                    "used_marker_samples": sample["used_markers"][:, 0].tolist(),
                }
            del value
            gc.collect()
        reference_layout = "all_gzip4"
        if not vep_layouts.get(reference_layout, {}).get("tags"):
            requested_text = ",".join(str(tag) for tag in (requested_vep_tags or sorted(marker_availability))) or "none"
            raise RuntimeError(
                "No valid VEP epochs were found. "
                f"Requested tags: {requested_text}. "
                f"Marker/epoch availability: {marker_availability}. "
                "Check --vep-tags, --duration-sec, epoch window, and log/Event CSV timing."
            )
        results["vep"] = vep_layouts
        results["vep_epoch_samples"] = layout_epoch_meta
        # Compare the actual epoch waveforms against the all_gzip4 reference.
        reference_waves = layout_waves.get(reference_layout, {})
        waveform_compare = {}
        for layout, waves in layout_waves.items():
            if layout == reference_layout:
                continue
            layout_rows = {}
            for tag in sorted(set(reference_waves) & set(waves)):
                ref_wave = reference_waves[tag]
                test_wave = waves[tag]
                difference = test_wave.astype(np.float64) - ref_wave.astype(np.float64)
                correlation = []
                for channel in range(ref_wave.shape[1]):
                    a = ref_wave[:, channel]
                    b = test_wave[:, channel]
                    if np.std(a) > 0 and np.std(b) > 0:
                        correlation.append(float(np.corrcoef(a, b)[0, 1]))
                layout_rows[str(tag)] = {
                    "epoch_wave_rmse_all_good_channels": float(np.sqrt(np.mean(difference * difference))),
                    "epoch_wave_max_abs_all_good_channels": float(np.max(np.abs(difference))),
                    "epoch_wave_corr_median": float(np.nanmedian(correlation)) if correlation else float("nan"),
                }
            waveform_compare[layout] = layout_rows
        results["vep_waveform_compare_vs_reference"] = waveform_compare
        results["vep_waveform_compare_vs_all_h5"] = waveform_compare

        reference_samples = layout_epoch_samples.get(reference_layout, {})
        epoch_compare_rows = []
        for layout, tag_samples in layout_epoch_samples.items():
            if layout == reference_layout:
                continue
            for tag in sorted(set(reference_samples) & set(tag_samples)):
                reference_sample = reference_samples[tag]
                candidate_sample = tag_samples[tag]
                raw_metrics = epoch_sample_metrics(reference_sample["raw_epochs"], candidate_sample["raw_epochs"])
                corrected_metrics = epoch_sample_metrics(reference_sample["corrected_epochs"], candidate_sample["corrected_epochs"])
                epoch_compare_rows.append({
                    "layout": layout,
                    "tag": int(tag),
                    "trial_count": int(candidate_sample["raw_epochs"].shape[0]),
                    "epoch_samples": int(candidate_sample["raw_epochs"].shape[1]),
                    "channels": ",".join(str(channel) for channel in sample_channels),
                    "raw_rmse": raw_metrics["rmse"],
                    "raw_max_abs": raw_metrics["max_abs"],
                    "raw_correlation": raw_metrics["correlation"],
                    "baseline_corrected_rmse": corrected_metrics["rmse"],
                    "baseline_corrected_max_abs": corrected_metrics["max_abs"],
                    "baseline_corrected_correlation": corrected_metrics["correlation"],
                })
        results["vep_epoch_compare_vs_reference"] = epoch_compare_rows
        results["vep_epoch_compare_vs_all_h5"] = epoch_compare_rows
        epoch_compare_csv = args.output_dir / "vep_epoch_compare.csv"
        with epoch_compare_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = [
                "layout", "tag", "trial_count", "epoch_samples", "channels",
                "raw_rmse", "raw_max_abs", "raw_correlation",
                "baseline_corrected_rmse", "baseline_corrected_max_abs",
                "baseline_corrected_correlation",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(epoch_compare_rows)

        plot_dir = args.output_dir / "vep_plots"
        plot_dir.mkdir(exist_ok=True)
        plot_position = int(np.flatnonzero(good == args.plot_channel - 1)[0]) if np.any(good == args.plot_channel - 1) else 0
        plot_label = int(good[plot_position]) + 1
        for tag in sorted(set(markers[:, 1].tolist())):
            save_vep_plot(plot_dir / f"tag_{tag}_ch{plot_label:03d}.png", layout_waves, int(tag), plot_position, t_ms)
        sample_plot_position = 0
        sample_plot_channel = sample_channels[sample_plot_position]
        for tag in sorted(reference_samples):
            for trial_index in range(min(args.epoch_sample_trials, reference_samples[tag]["raw_epochs"].shape[0])):
                save_epoch_sample_plot(
                    plot_dir / f"epoch_tag_{tag}_trial_{trial_index + 1:03d}_ch{sample_plot_channel:03d}_raw.png",
                    layout_epoch_samples, int(tag), trial_index, sample_plot_position, corrected=False,
                )
                save_epoch_sample_plot(
                    plot_dir / f"epoch_tag_{tag}_trial_{trial_index + 1:03d}_ch{sample_plot_channel:03d}_baseline.png",
                    layout_epoch_samples, int(tag), trial_index, sample_plot_position, corrected=True,
                )

        vep_csv = args.output_dir / "vep_metrics.csv"
        with vep_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layout", "tag", "valid_trials", "trial_a", "trial_b", "corr", "peak_a", "peak_b", "change_percent", "event_snr_median_db", "epoch_samples"])
            for layout, payload in vep_layouts.items():
                for tag, row in payload["tags"].items():
                    writer.writerow([layout, tag, row["valid_trials"], row["trial_a"], row["trial_b"], row["corr"], row["peak_a"], row["peak_b"], row["change_percent"], row["event_snr_median_db"], row["epoch_samples"]])
        results["matlab_smoke"] = None if args.skip_matlab else matlab_smoke_test(
            shutil.which("matlab"), matlab_paths,
            {name for name, spec in layout_specs.items() if spec["mode"] == "per"},
        )

    if not args.run_vep:
        results["matlab_smoke"] = None if args.skip_matlab else matlab_smoke_test(
            shutil.which("matlab"), matlab_paths,
            {name for name, spec in layout_specs.items() if spec["mode"] == "per"},
        )

    results["environment"] = {"psutil_available": psutil is not None, "pyarrow_version": getattr(pa, "__version__", None), "h5py_version": h5py.__version__}
    summary_csv = args.output_dir / "storage_summary.csv"
    overview_headers, overview_rows = build_overview_rows(results)
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=overview_headers)
        writer.writeheader()
        writer.writerows(overview_rows)
    summary_xlsx = args.output_dir / "storage_summary.xlsx"
    workbook_written = write_summary_workbook(results, summary_xlsx)
    report = args.output_dir / "storage_vep_benchmark.json"
    report.write_text(json.dumps(results, indent=2, default=_json_value), encoding="utf-8")
    print(f"Report: {report}")
    print(f"Summary CSV: {summary_csv}")
    if workbook_written:
        print(f"Summary Excel: {summary_xlsx}")
    print("HDF5 outputs:")
    for name, path in layout_paths.items():
        print(f"  {name}: {path}")
    if args.run_vep:
        print(f"VEP markers: {marker_csv}")
        print(f"VEP metrics: {vep_csv}")
        print(f"VEP epoch comparison: {epoch_compare_csv}")
        print(f"VEP epoch samples: {epoch_sample_dir}")
        print(f"VEP plots: {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
