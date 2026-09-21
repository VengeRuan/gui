#!/usr/bin/env python3
"""Compare HDF5 and Parquet read speed for the generated signal files.

The benchmark reads the signal as the GUI would consume it: float32 NumPy
arrays with shape samples x 512. It reports both the raw table read and the
extra Parquet-to-NumPy conversion cost.

Example:
    python compare_read_speed.py \
        --h5 compare_result/D0000_float32_chunked_gzip.h5 \
        --parquet compare_result/D0000_float32.parquet
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pyarrow.parquet as pq


CHANNELS = 512
CHANNEL_NAMES = [f"ch{channel:03d}" for channel in range(1, CHANNELS + 1)]


def h5_signal_axis(dataset: h5py.Dataset) -> int:
    if len(dataset.shape) != 2:
        raise ValueError(f"Expected a 2D HDF5 rawData512 dataset, got {dataset.shape}")
    if dataset.shape[1] == CHANNELS:
        return 1
    if dataset.shape[0] == CHANNELS:
        return 0
    raise ValueError(f"Cannot identify channel axis in HDF5 shape {dataset.shape}")


def read_h5_all(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dataset = handle["rawData512"]
        axis = h5_signal_axis(dataset)
        values = np.asarray(dataset[:], dtype=np.float32)
    if axis == 0:
        values = values.T
    return values


def read_h5_channel(path: Path, channel_index: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dataset = handle["rawData512"]
        axis = h5_signal_axis(dataset)
        if axis == 1:
            values = np.asarray(dataset[:, channel_index], dtype=np.float32)
        else:
            values = np.asarray(dataset[channel_index, :], dtype=np.float32)
    return values


def parquet_table(path: Path, columns: list[str]):
    return pq.read_table(path, columns=columns, use_threads=True)


def parquet_all_table(path: Path):
    return parquet_table(path, CHANNEL_NAMES)


def parquet_all_numpy(path: Path) -> np.ndarray:
    table = parquet_all_table(path)
    # This represents the extra conversion needed by NumPy/scipy-based GUI code.
    return table.to_pandas().to_numpy(dtype=np.float32, copy=False)


def parquet_channel(path: Path, channel_index: int) -> np.ndarray:
    table = parquet_table(path, [CHANNEL_NAMES[channel_index]])
    return table.column(0).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)


def measure(function: Callable[[], object], repeats: int, warmup: int) -> dict[str, object]:
    for _ in range(warmup):
        value = function()
        del value
        gc.collect()

    timings: list[float] = []
    shape = None
    dtype = None
    checksum = None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = function()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        if isinstance(value, np.ndarray):
            shape = list(value.shape)
            dtype = str(value.dtype)
            checksum = float(np.asarray(value.reshape(-1)[: min(value.size, 1000)], dtype=np.float64).sum())
        else:
            shape = [value.num_rows, value.num_columns]
            dtype = str(value.column(0).type)
        del value

    median = statistics.median(timings)
    return {
        "seconds": timings,
        "min_seconds": min(timings),
        "median_seconds": median,
        "mean_seconds": statistics.mean(timings),
        "shape": shape,
        "dtype": dtype,
        "checksum_first_1000": checksum,
    }


def throughput_gib(shape: list[int] | None, seconds: float | None) -> float | None:
    if not shape or seconds is None or seconds <= 0:
        return None
    elements = int(np.prod(shape))
    return elements * 4 / seconds / 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare HDF5 and Parquet signal read speed.")
    parser.add_argument("--h5", type=Path, required=True, help="HDF5 file containing rawData512")
    parser.add_argument("--parquet", type=Path, required=True, help="Parquet file containing ch001...ch512")
    parser.add_argument("--channel", type=int, default=1, help="1-based channel for the single-channel test")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    if not args.h5.is_file():
        parser.error(f"HDF5 file does not exist: {args.h5}")
    if not args.parquet.is_file():
        parser.error(f"Parquet file does not exist: {args.parquet}")
    if not 1 <= args.channel <= CHANNELS:
        parser.error("--channel must be between 1 and 512")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup cannot be negative")

    channel_index = args.channel - 1
    results: dict[str, object] = {
        "h5": str(args.h5.resolve()),
        "parquet": str(args.parquet.resolve()),
        "h5_bytes": args.h5.stat().st_size,
        "parquet_bytes": args.parquet.stat().st_size,
        "channel_test": args.channel,
        "repeats": args.repeats,
        "warmup": args.warmup,
    }

    print(f"HDF5:    {args.h5} ({args.h5.stat().st_size / 1024**3:.3f} GiB)")
    print(f"Parquet: {args.parquet} ({args.parquet.stat().st_size / 1024**3:.3f} GiB)")
    print(f"Repeats: {args.repeats}; warmup: {args.warmup}; single channel: ch{args.channel:03d}")

    tests: list[tuple[str, Callable[[], object]]] = [
        ("h5_all_numpy", lambda: read_h5_all(args.h5)),
        ("parquet_all_table_signal_only", lambda: parquet_all_table(args.parquet)),
        ("parquet_all_numpy_signal_only", lambda: parquet_all_numpy(args.parquet)),
        ("h5_single_channel_numpy", lambda: read_h5_channel(args.h5, channel_index)),
        ("parquet_single_channel_numpy", lambda: parquet_channel(args.parquet, channel_index)),
    ]

    for name, function in tests:
        print(f"Running {name}...")
        result = measure(function, args.repeats, args.warmup)
        result["median_GiB_per_second"] = throughput_gib(result["shape"], result["median_seconds"])
        results[name] = result
        print(
            f"  median={result['median_seconds']:.4f} s; "
            f"min={result['min_seconds']:.4f} s; "
            f"shape={result['shape']}; dtype={result['dtype']}"
        )

    print("\nSummary:")
    for name in ("h5_all_numpy", "parquet_all_numpy_signal_only", "h5_single_channel_numpy", "parquet_single_channel_numpy"):
        result = results[name]
        print(f"  {name}: {result['median_seconds']:.4f} s ({result['median_GiB_per_second']:.3f} GiB/s)")

    output = args.json or args.parquet.with_name(args.parquet.stem + "_read_speed.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
