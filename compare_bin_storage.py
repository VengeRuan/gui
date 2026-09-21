#!/usr/bin/env python3
"""Compare float32 HDF5 and Parquet storage for the SD-card BIN format.

This is an independent benchmark tool. It does not change the GUI or the
the legacy acquisition frame layout:
four interleaved 64 KiB chip blocks, 128 channels per frame, and a two-word
frame header followed by 128 uint16 ADC values.

Examples:
    python compare_bin_storage.py data.bin --output-dir compare_out
    python compare_bin_storage.py data.bin --duration-sec 10 --output-dir compare_out

Install the optional Parquet dependency before using Parquet output:
    python -m pip install pyarrow
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from h5_provenance import build_h5_provenance, write_h5_provenance

try:
    import hdf5plugin
except ImportError:  # pragma: no cover - reported when HDF5 output is requested
    hdf5plugin = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - reported when Parquet is requested
    pa = None
    pq = None


FS = 6490.0
OFFSET_BYTES = 0
CHANNELS_PER_CHIP = 128
NUM_CHIPS = 4
TOTAL_CHANNELS = CHANNELS_PER_CHIP * NUM_CHIPS
BLOCK_SIZE_BYTES = 64 * 1024
BLOCK_SIZE_WORDS = BLOCK_SIZE_BYTES // 2
WORDS_PER_FRAME = 2 + CHANNELS_PER_CHIP
SYNC_WORD = np.uint16(0xFFFF)
FRAME_SECOND_WORD = np.uint16(0x0000)
ADC_TO_MV = np.float32(1000.0 * 5.0 / 32768.0)#65536  32768.0
OFFSET_MV = np.float32(2500.0)
TIMING_HEADER = b"Hello SD Card via FatFs!\n"


@dataclass
class BinLayout:
    path: Path
    sync_start_bytes: int
    num_cycles: int
    frame_starts: np.ndarray
    frame_counts: np.ndarray
    common_frames: int


def find_sync_start(path: Path, probe_bytes: int = 10 * 1024 * 1024) -> int:
    """Find the first run of nine FFFF words in the acquisition stream."""
    with path.open("rb") as handle:
        handle.seek(OFFSET_BYTES)
        words = np.fromfile(handle, dtype="<u2", count=probe_bytes // 2)
    if words.size < 9:
        raise ValueError("BIN is too small to contain the sync header.")

    candidates = np.flatnonzero(words[:-8] == SYNC_WORD)
    for index in candidates:
        if np.all(words[index : index + 9] == SYNC_WORD):
            return max(OFFSET_BYTES, int(index) * 2 - 2)
    raise ValueError("Sync header was not found in the first 10 MB.")


def read_chip_probe(path: Path, sync_start: int, cycles: int) -> np.ndarray:
    count = cycles * NUM_CHIPS * BLOCK_SIZE_WORDS
    with path.open("rb") as handle:
        handle.seek(sync_start)
        raw = np.fromfile(handle, dtype="<u2", count=count)
    expected = cycles * NUM_CHIPS * BLOCK_SIZE_WORDS
    if raw.size != expected:
        raise ValueError("BIN ended before the frame probe was complete.")
    blocks = raw.reshape(cycles, NUM_CHIPS, BLOCK_SIZE_WORDS)
    return blocks.transpose(1, 0, 2).reshape(NUM_CHIPS, cycles * BLOCK_SIZE_WORDS)


def find_frame_start(chip_stream: np.ndarray) -> int:
    """Return the zero-based word offset of the first frame header."""
    sync_candidates = np.flatnonzero(chip_stream[:-8] == SYNC_WORD)
    start_search = 0
    for index in sync_candidates:
        if np.all(chip_stream[index : index + 9] == SYNC_WORD):
            start_search = int(index) + 10
            break

    candidates = np.flatnonzero(chip_stream[start_search:-1] == SYNC_WORD)
    for relative in candidates:
        index = start_search + int(relative)
        if chip_stream[index + 1] == FRAME_SECOND_WORD:
            return index
    raise ValueError("A chip frame header (FFFF, 0000) was not found.")


def inspect_bin(path: Path) -> BinLayout:
    path = path.resolve()
    file_size = path.stat().st_size
    sync_start = find_sync_start(path)
    cycle_bytes = NUM_CHIPS * BLOCK_SIZE_BYTES
    num_cycles = (file_size - sync_start) // cycle_bytes
    if num_cycles <= 0:
        raise ValueError("BIN has no complete four-chip block cycle after sync.")

    probe_cycles = min(num_cycles, 128)
    probe = read_chip_probe(path, sync_start, probe_cycles)
    frame_starts = np.array([find_frame_start(probe[chip]) for chip in range(NUM_CHIPS)], dtype=np.int64)
    stream_words = num_cycles * BLOCK_SIZE_WORDS
    frame_counts = (stream_words - frame_starts) // WORDS_PER_FRAME
    if np.any(frame_counts <= 0):
        raise ValueError(f"Invalid frame counts: {frame_counts.tolist()}")

    common_frames = int(frame_counts.min())
    return BinLayout(path, sync_start, num_cycles, frame_starts, frame_counts, common_frames)


def parse_filename_metadata(path: Path) -> dict[str, object]:
    """Extract date, animal, and block when the standard filename is used."""
    text = path.stem
    match = re.search(r"(?P<date>\d{8})[_-](?P<animal>\d+)[_-]block[_-]?(?P<block>\d+)", text, re.I)
    if not match:
        match = re.search(r"(?P<date>\d{8}).*?(?P<animal>\d+).*?block[_-]?(?P<block>\d+)", text, re.I)
    if not match:
        return {"source_bin": str(path), "date": "", "animal": 0, "block": 0}
    return {
        "source_bin": str(path),
        "date": match.group("date"),
        "animal": int(match.group("animal")),
        "block": int(match.group("block")),
    }


def read_timing_metadata(path: Path) -> dict[str, object]:
    """Read dt1/dt2 and deltaT1/deltaT2 from BIN timing metadata offsets."""
    scan_bytes = min(path.stat().st_size, 100 * 1024 * 1024)
    scan_start = max(0, path.stat().st_size - scan_bytes)
    with path.open("rb") as handle:
        handle.seek(scan_start)
        data = handle.read(scan_bytes)
    index = data.find(TIMING_HEADER)
    if index < 0:
        return {}

    base = index + len(TIMING_HEADER)
    off_a = 8 + 24
    off_b = off_a + 8 + 24
    off_t2 = off_b + 8 + 88
    required = off_t2 + 8
    if base + required > len(data):
        return {}

    def u32(offset: int) -> int:
        return int.from_bytes(data[base + offset : base + offset + 4], "little")

    def u64(offset: int) -> int:
        return int.from_bytes(data[base + offset : base + offset + 8], "little")

    t1_raw = u64(0)
    t2_raw = u64(off_t2)
    diff_a = (u32(off_a) - u32(off_a + 4)) & 0xFFFFFFFF
    diff_b = (u32(off_b) - u32(off_b + 4)) & 0xFFFFFFFF

    def format_local_time(raw_microseconds: int) -> str:
        value = datetime.fromtimestamp(raw_microseconds / 1e6, tz=timezone.utc) + timedelta(hours=8)
        return value.strftime("%Y-%m-%d %H:%M:%S.%f")

    return {
        "dt1": format_local_time(t1_raw),
        "dt2": format_local_time(t2_raw),
        "dt1_date": format_local_time(t1_raw)[:10],
        # The counter difference is in microseconds.  Convert it to the
        # declared millisecond unit once, rather than storing a value that is
        # 1000 times too large and then forcing readers to compensate.
        "deltaT1_ms": float(diff_a / 216000.0),
        "deltaT2_ms": float(diff_b / 216000.0),
        "deltaT_unit": "ms",
    }


class BinReader:
    def __init__(self, layout: BinLayout, start_sec: float, duration_sec: float, chunk_rows: int):
        self.layout = layout
        self.start_frame = max(0, int(np.floor(start_sec * FS)))
        if self.start_frame >= layout.common_frames:
            raise ValueError("Start time is beyond the available BIN duration.")
        if duration_sec > 0:
            requested = max(1, int(np.floor(duration_sec * FS)))
            self.num_frames = min(requested, layout.common_frames - self.start_frame)
        else:
            self.num_frames = layout.common_frames - self.start_frame
        self.chunk_rows = max(1, int(chunk_rows))

    @property
    def selected_duration_sec(self) -> float:
        return self.num_frames / FS

    def iter_chunks(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (first sample index, float32 samples x 512 channels)."""
        for output_start in range(0, self.num_frames, self.chunk_rows):
            rows = min(self.chunk_rows, self.num_frames - output_start)
            yield output_start, self.read_chunk(output_start, rows)

    def read_chunk(self, output_start: int, rows: int) -> np.ndarray:
        first_frame = self.start_frame + output_start
        # Keep the two-word frame header in the slice.  The values are removed
        # exactly once below with frames[:, 2:], after retaining the header.
        word_starts = self.layout.frame_starts + first_frame * WORDS_PER_FRAME
        word_ends = word_starts + rows * WORDS_PER_FRAME
        cycle_start = int(np.floor(word_starts.min() / BLOCK_SIZE_WORDS))
        cycle_end = int(np.ceil(word_ends.max() / BLOCK_SIZE_WORDS))
        cycles = cycle_end - cycle_start

        byte_offset = self.layout.sync_start_bytes + cycle_start * NUM_CHIPS * BLOCK_SIZE_BYTES
        word_count = cycles * NUM_CHIPS * BLOCK_SIZE_WORDS
        with self.layout.path.open("rb") as handle:
            handle.seek(byte_offset)
            raw = np.fromfile(handle, dtype="<u2", count=word_count)
        expected = word_count
        if raw.size != expected:
            raise ValueError("BIN ended while reading a data chunk.")
        blocks = raw.reshape(cycles, NUM_CHIPS, BLOCK_SIZE_WORDS)

        result = np.empty((rows, TOTAL_CHANNELS), dtype=np.float32)
        for chip in range(NUM_CHIPS):
            stream = blocks[:, chip, :].reshape(-1)
            local_start = int(word_starts[chip] - cycle_start * BLOCK_SIZE_WORDS)
            local_end = int(word_ends[chip] - cycle_start * BLOCK_SIZE_WORDS)
            frames = stream[local_start:local_end].reshape(rows, WORDS_PER_FRAME)
            values = frames[:, 2:].astype(np.float32, copy=False)
            values = values * ADC_TO_MV - OFFSET_MV
            col_start = chip * CHANNELS_PER_CHIP
            result[:, col_start : col_start + CHANNELS_PER_CHIP] = values
        return result


def create_h5(
    path: Path,
    reader: BinReader,
    metadata: dict[str, object],
    compression_level: int = 5,
    progress_callback=None,
) -> dict[str, object]:
    if hdf5plugin is None:
        raise RuntimeError(
            "Blosc/LZ4 + bitshuffle HDF5 output requires hdf5plugin. "
            "Install it with: python -m pip install hdf5plugin"
        )
    start = time.perf_counter()
    chunk_rows = min(reader.chunk_rows, reader.num_frames)
    compression = dict(
        hdf5plugin.Blosc(
            cname="lz4",
            clevel=int(compression_level),
            shuffle=hdf5plugin.Blosc.BITSHUFFLE,
        )
    )
    with h5py.File(path, "w") as handle:
        raw = handle.create_dataset(
            "rawData512",
            shape=(reader.num_frames, TOTAL_CHANNELS),
            dtype="float32",
            chunks=(chunk_rows, TOTAL_CHANNELS),
            **compression,
        )
        time_ds = handle.create_dataset(
            "time",
            shape=(reader.num_frames,),
            dtype="float32",
            chunks=(chunk_rows,),
            **compression,
        )
        handle.create_dataset("FS", data=np.float32(FS))
        handle.create_dataset("selected_start_sec", data=np.float32(reader.start_frame / FS))
        handle.create_dataset("selected_duration_sec", data=np.float32(reader.selected_duration_sec))
        handle.create_dataset("frame_counts", data=reader.layout.frame_counts.astype(np.int64))
        for key, value in metadata.items():
            handle.attrs[key] = value
        handle.attrs["data_unit"] = "mV"
        handle.attrs["storage_dtype"] = "float32"
        handle.attrs["hdf5_compression"] = "Blosc/LZ4 + bitshuffle"
        handle.attrs["hdf5_compression_filter"] = "blosc:lz4"
        handle.attrs["hdf5_compression_level"] = int(compression_level)
        handle.attrs["hdf5_compression_shuffle"] = "bitshuffle"
        handle.attrs["hdf5_chunk_shape"] = f"({chunk_rows}, {TOTAL_CHANNELS})"
        write_h5_provenance(handle, build_h5_provenance(
            stage="raw",
            fs=FS,
            unit="mV",
            channel_ids=np.arange(1, TOTAL_CHANNELS + 1, dtype=np.int64),
            source={
                "kind": "bin", "path": str(metadata.get("source_bin", "")),
                "selected_start_sec": float(reader.start_frame / FS),
                "selected_duration_sec": float(reader.selected_duration_sec),
            },
            operation={
                "name": "bin_to_hdf5", "adc_scale_mv": float(ADC_TO_MV),
                "adc_offset_mv": float(OFFSET_MV),
            },
        ))

        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("data_unit", data="mV", dtype=string_dtype)
        if "deltaT1_ms" in metadata:
            handle.create_dataset("deltaT1", data=np.float64(metadata["deltaT1_ms"]))
            handle.create_dataset("deltaT1_unit", data="ms", dtype=string_dtype)
        if "deltaT2_ms" in metadata:
            handle.create_dataset("deltaT2", data=np.float64(metadata["deltaT2_ms"]))
        if "dt1_date" in metadata:
            handle.create_dataset("dt1_date", data=str(metadata["dt1_date"]), dtype=string_dtype)
        if "dt1" in metadata:
            handle.create_dataset("dt1", data=str(metadata["dt1"]), dtype=string_dtype)
        if "dt2" in metadata:
            handle.create_dataset("dt2", data=str(metadata["dt2"]), dtype=string_dtype)

        if progress_callback is not None:
            progress_callback(0, reader.num_frames)
        for first, data in reader.iter_chunks():
            last = first + data.shape[0]
            raw[first:last, :] = data
            sample_numbers = np.arange(reader.start_frame + first, reader.start_frame + last, dtype=np.float32)
            time_ds[first:last] = sample_numbers / np.float32(FS)
            if progress_callback is not None:
                progress_callback(last, reader.num_frames)
    elapsed = time.perf_counter() - start
    return {"path": str(path), "seconds": elapsed, "bytes": path.stat().st_size}


def create_parquet(path: Path, reader: BinReader, metadata: dict[str, object], compression: str) -> dict[str, object]:
    if pa is None or pq is None:
        raise RuntimeError("Parquet output requires pyarrow. Install it with: python -m pip install pyarrow")

    start = time.perf_counter()
    fields = [pa.field("time", pa.float32())] + [pa.field(f"ch{channel:03d}", pa.float32()) for channel in range(1, TOTAL_CHANNELS + 1)]
    schema = pa.schema(fields, metadata={str(k): str(v).encode("utf-8") for k, v in metadata.items()})
    schema = schema.with_metadata({**(schema.metadata or {}), b"FS": str(FS).encode(), b"data_unit": b"mV", b"storage_dtype": b"float32"})

    writer = pq.ParquetWriter(path, schema=schema, compression=compression, use_dictionary=False)
    try:
        for first, data in reader.iter_chunks():
            sample_numbers = np.arange(reader.start_frame + first, reader.start_frame + first + data.shape[0], dtype=np.float32)
            arrays = [pa.array(sample_numbers / np.float32(FS))]
            arrays.extend(pa.array(data[:, channel]) for channel in range(TOTAL_CHANNELS))
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
    finally:
        writer.close()
    elapsed = time.perf_counter() - start
    return {"path": str(path), "seconds": elapsed, "bytes": path.stat().st_size, "compression": compression}


def benchmark_reads(h5_path: Path, parquet_path: Path | None) -> dict[str, object]:
    result: dict[str, object] = {}
    start = time.perf_counter()
    with h5py.File(h5_path, "r") as handle:
        full_shape = tuple(handle["rawData512"].shape)
        full = handle["rawData512"][:]
        full_dtype = str(full.dtype)
    result["h5_read_all_seconds"] = time.perf_counter() - start
    result["h5_read_all_shape"] = full_shape
    result["h5_read_all_dtype"] = full_dtype

    start = time.perf_counter()
    with h5py.File(h5_path, "r") as handle:
        channel = handle["rawData512"][:, 0]
        channel_shape = tuple(channel.shape)
    result["h5_read_ch001_seconds"] = time.perf_counter() - start
    result["h5_read_ch001_shape"] = channel_shape

    if parquet_path is not None:
        if pa is None or pq is None:
            raise RuntimeError("Parquet reading requires pyarrow.")
        start = time.perf_counter()
        table = pq.read_table(parquet_path)
        result["parquet_read_all_seconds"] = time.perf_counter() - start
        result["parquet_read_all_shape"] = [table.num_rows, table.num_columns]
        result["parquet_read_all_dtype"] = str(table.column(1).type)

        start = time.perf_counter()
        column = pq.read_table(parquet_path, columns=["ch001"])
        result["parquet_read_ch001_seconds"] = time.perf_counter() - start
        result["parquet_read_ch001_shape"] = [column.num_rows, column.num_columns]
    return result


def output_names(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    return output_dir / f"{stem}_float32_chunked_blosc_lz4_bitshuffle.h5", output_dir / f"{stem}_float32.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare float32 HDF5 Blosc/LZ4 + bitshuffle storage and Parquet for a BIN file.")
    parser.add_argument("bin_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 means full available duration")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--compression-level", type=int, choices=range(1, 10), default=5)
    parser.add_argument("--parquet-compression", choices=("snappy", "gzip", "zstd", "brotli", "none"), default="zstd")
    parser.add_argument("--skip-parquet", action="store_true")
    args = parser.parse_args()

    input_path = args.bin_file.resolve()
    if not input_path.is_file():
        parser.error(f"BIN file does not exist: {input_path}")
    output_dir = (args.output_dir or input_path.parent / "storage_compare").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting BIN: {input_path}")
    inspect_start = time.perf_counter()
    layout = inspect_bin(input_path)
    inspect_seconds = time.perf_counter() - inspect_start
    reader = BinReader(layout, args.start_sec, args.duration_sec, args.chunk_rows)
    metadata = parse_filename_metadata(input_path)
    metadata.update(read_timing_metadata(input_path))
    metadata.update({
        "start_sec": reader.start_frame / FS,
        "duration_sec": reader.selected_duration_sec,
        "source_bin": str(input_path),
    })
    h5_path, parquet_path = output_names(input_path, output_dir)
    if h5_path.exists():
        h5_path.unlink()
    if parquet_path.exists():
        parquet_path.unlink()

    print(f"Sync offset: {layout.sync_start_bytes} bytes")
    print(f"Frame counts: {layout.frame_counts.tolist()}; common samples: {layout.common_frames}")
    print(f"Selected: {reader.num_frames} samples x {TOTAL_CHANNELS} channels")
    print(
        f"Float type: float32; HDF5 chunk: ({min(reader.chunk_rows, reader.num_frames)}, {TOTAL_CHANNELS}); "
        f"compression: Blosc/LZ4 + bitshuffle level {args.compression_level}"
    )

    results: dict[str, object] = {
        "input": str(input_path),
        "input_bytes": input_path.stat().st_size,
        "inspect_seconds": inspect_seconds,
        "samples": reader.num_frames,
        "channels": TOTAL_CHANNELS,
        "fs": FS,
        "dtype": "float32",
        "frame_counts": layout.frame_counts.tolist(),
    }

    print("Writing HDF5...")
    results["hdf5"] = create_h5(h5_path, reader, metadata, args.compression_level)
    print(f"  {h5_path} ({results['hdf5']['bytes'] / 1024**3:.3f} GiB, {results['hdf5']['seconds']:.2f} s)")

    if not args.skip_parquet:
        print(f"Writing Parquet ({args.parquet_compression})...")
        compression = None if args.parquet_compression == "none" else args.parquet_compression
        results["parquet"] = create_parquet(parquet_path, reader, metadata, compression)
        print(f"  {parquet_path} ({results['parquet']['bytes'] / 1024**3:.3f} GiB, {results['parquet']['seconds']:.2f} s)")
    else:
        parquet_path = None

    print("Benchmarking reads...")
    results["reads"] = benchmark_reads(h5_path, parquet_path)
    print(json.dumps(results["reads"], indent=2))
    report_path = output_dir / f"{input_path.stem}_storage_compare.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
