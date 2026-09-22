"""Headless data model shared by the Qt EEG interface.

The module deliberately has no Qt import so HDF5 lazy-loading behaviour can
be tested on machines where an application-control policy prevents Qt native
DLLs from loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import atexit
import gc
import os
import re
import tempfile
import threading
import time
import weakref

import h5py
import numpy as np
from h5_provenance import read_h5_provenance

# Register the Blosc/LZ4 filter used by files written by compare_bin_storage.
try:  # pragma: no cover - depends on the user's file compression
    import hdf5plugin  # noqa: F401
except ImportError:  # Standard uncompressed HDF5 remains supported.
    hdf5plugin = None


DATASET_KEYS = ("rawData512", "raw_data", "signal", "data", "eeg")
DATA_DTYPE = np.float32
MEMMAP_THRESHOLD_BYTES = 1_500_000_000
PROCESS_CHUNK_SAMPLES = 200_000
FILTER_CHANNEL_CHUNK = 8
_PROCESS_CACHE_LOCK = threading.Lock()
_PROCESS_CACHE_PATHS: set[Path] = set()
_PROCESS_CACHE_ARRAYS: list[weakref.ReferenceType] = []
_PROCESS_CACHE_NAME = re.compile(r"_(\d+)_(\d+)\.dat$")
def _text(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _h5_scalar(h5: h5py.File, key: str, default=None):
    """Read a scalar dataset/attribute without making timing metadata mandatory."""
    value = h5[key][()] if key in h5 else h5.attrs.get(key, default)
    if value is default:
        return default
    array = np.asarray(value)
    return array.squeeze().item() if array.size == 1 else value


def read_h5_timing_metadata(h5: h5py.File) -> dict[str, object]:
    """Read legacy BIN timing with the same ms/seconds compatibility as Tk.

    ``deltaT1`` is retained in its original stored unit, while the normalized
    ``deltaT1_sec`` value is what time alignment must use.
    """
    timing: dict[str, object] = {}
    for key in ("dt1", "dt2", "dt1_date"):
        value = _h5_scalar(h5, key)
        if value is not None:
            timing[key] = _text(value).strip()
    raw_delta_t1_present = False
    for key in ("deltaT1", "deltaT2"):
        value = _h5_scalar(h5, key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            timing[key] = numeric
            raw_delta_t1_present = raw_delta_t1_present or key == "deltaT1"
    # Processed datasets from the old GUI stored normalized seconds only.
    # This value has priority over a missing raw deltaT1 field; it must never
    # be reinterpreted as milliseconds merely because it happens to exceed .5.
    legacy_sec = _h5_scalar(h5, "bin_delta_t1_sec")
    legacy_t2_sec = _h5_scalar(h5, "bin_delta_t2_sec")
    if "deltaT1" not in timing and legacy_sec is not None:
        try:
            value = float(legacy_sec)
            if np.isfinite(value):
                timing["deltaT1"] = value
                timing["deltaT1_unit"] = "s"
        except (TypeError, ValueError):
            pass
    if "deltaT2" not in timing and legacy_t2_sec is not None:
        try:
            value = float(legacy_t2_sec)
            if np.isfinite(value):
                timing["deltaT2"] = value
        except (TypeError, ValueError):
            pass
    # A few earlier Qt exports stored only this millisecond attribute.
    if "deltaT1" not in timing:
        value = _h5_scalar(h5, "deltaT1_ms")
        if value is not None:
            try:
                timing["deltaT1"] = float(value)
                timing["deltaT1_unit"] = "ms"
            except (TypeError, ValueError):
                pass
    unit = _h5_scalar(h5, "deltaT1_unit")
    if unit is not None:
        timing["deltaT1_unit"] = _text(unit).strip()
    raw_t1 = float(timing.get("deltaT1", 0.0))
    unit_text = str(timing.get("deltaT1_unit", "")).strip().lower()
    # BIN metadata's raw deltaT1/deltaT2 counter values are milliseconds.
    # Some historical exports incorrectly labelled these datasets as seconds;
    # trust the binary-data contract rather than that bad label.  The only
    # seconds-only compatibility field is bin_delta_t1_sec above.
    is_ms = raw_delta_t1_present or unit_text in {"ms", "millisecond", "milliseconds"} or (not unit_text and abs(raw_t1) >= 0.5)
    if raw_delta_t1_present:
        timing["deltaT1_unit"] = "ms"
    # The original BIN reader divided by 216 instead of 216000 and wrote
    # e.g. 161230.935185 for a real 161.230935185 ms deltaT1.  Repair only
    # implausibly large raw BIN millisecond values, never bin_delta_t1_sec.
    if raw_delta_t1_present and abs(raw_t1) >= 10_000.0:
        raw_t1 /= 1000.0
        timing["deltaT1"] = raw_t1
        timing["timing_legacy_ms_scale_repaired"] = True
    timing["deltaT1_sec"] = raw_t1 / 1000.0 if is_ms else raw_t1
    timing["deltaT1_ms"] = timing["deltaT1_sec"] * 1000.0
    raw_t2 = timing.get("deltaT2")
    if raw_t2 is not None:
        raw_t2 = float(raw_t2)
        if raw_delta_t1_present and abs(raw_t2) >= 10_000.0:
            raw_t2 /= 1000.0
            timing["deltaT2"] = raw_t2
        timing["deltaT2_sec"] = raw_t2 / 1000.0 if is_ms else raw_t2
        timing["deltaT2_ms"] = timing["deltaT2_sec"] * 1000.0
    return timing


@dataclass(frozen=True)
class H5Metadata:
    path: Path
    dataset: str
    rows: int
    channels: int
    fs: float
    chunks: tuple[int, ...] | None
    scale_to_mv: float
    time_offset: float
    channel_ids: tuple[int, ...]
    timing_metadata: dict[str, object] = field(default_factory=dict)
    provenance: dict | None = None


class LazyH5Source:
    """HDF5 EEG source that opens files only for metadata or requested slices."""

    def __init__(self) -> None:
        self.metadata: H5Metadata | None = None
        # Logical start within the physical HDF5 dataset.
        self._row_offset = 0

    @property
    def loaded(self) -> bool:
        return self.metadata is not None

    def open(
        self, filename: str | Path, start_seconds: float = 0.0,
        duration_seconds: float = 0.0,
    ) -> H5Metadata:
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        if not h5py.is_hdf5(path):
            raise ValueError("Qt preview currently accepts HDF5/H5 data files only.")
        with h5py.File(path, "r") as h5:
            dataset = next(
                (
                    key
                    for key in DATASET_KEYS
                    if key in h5
                    and isinstance(h5[key], h5py.Dataset)
                    and h5[key].ndim == 2
                    and np.issubdtype(h5[key].dtype, np.number)
                ),
                None,
            )
            if dataset is None:
                raise ValueError(f"No 2D numeric EEG dataset found; expected one of {DATASET_KEYS}.")
            raw = h5[dataset]
            fs_raw = h5["FS"][()] if "FS" in h5 else h5.attrs.get("FS", h5.attrs.get("fs", 6490.0))
            fs = float(np.asarray(fs_raw).squeeze())
            if not np.isfinite(fs) or fs <= 0:
                raise ValueError(f"Invalid sampling rate in HDF5: {fs_raw!r}")
            unit_raw = h5["data_unit"][()] if "data_unit" in h5 else h5.attrs.get("data_unit", "")
            unit = _text(unit_raw).strip().lower()
            time_dataset = h5.get("time")
            if isinstance(time_dataset, h5py.Dataset) and time_dataset.ndim == 0:
                time_offset = float(np.asarray(time_dataset[()]).squeeze())
            elif isinstance(time_dataset, h5py.Dataset) and time_dataset.shape[0]:
                time_offset = float(time_dataset[0])
            elif "time_offset_sec" in h5:
                # Earlier Qt exports used a scalar dataset; current exports
                # also write the attribute.  Accept both without altering a
                # user's existing HDF5 layout.
                time_offset = float(np.asarray(h5["time_offset_sec"][()]).squeeze())
            else:
                time_offset = float(h5.attrs.get("time_offset_sec", 0.0))
            ids_raw = h5.get("channel_ids", h5.get("channel_numbers"))
            if ids_raw is not None:
                ids = np.asarray(ids_raw[()]).reshape(-1)
                if ids.size != raw.shape[1] or not np.issubdtype(ids.dtype, np.number):
                    ids = np.arange(1, raw.shape[1] + 1, dtype=np.int64)
                else:
                    ids = ids.astype(np.int64, copy=False)
            else:
                ids = np.arange(1, raw.shape[1] + 1, dtype=np.int64)
            start_seconds = float(start_seconds or 0.0)
            duration_seconds = float(duration_seconds or 0.0)
            if not np.isfinite(start_seconds) or not np.isfinite(duration_seconds):
                raise ValueError("HDF5 time-window values must be finite numbers.")
            if start_seconds < 0.0 or duration_seconds < 0.0:
                raise ValueError("HDF5 time-window start and duration must be non-negative.")
            total_rows = int(raw.shape[0])
            row_offset = int(round(start_seconds * fs))
            if row_offset >= total_rows:
                raise ValueError(
                    f"HDF5 time-window start {start_seconds:g} s is outside the recording "
                    f"({total_rows / fs:.3f} s)."
                )
            requested_rows = int(round(duration_seconds * fs)) if duration_seconds > 0.0 else total_rows - row_offset
            rows = min(requested_rows, total_rows - row_offset)
            if rows <= 0:
                raise ValueError("HDF5 time window contains no samples.")
            self._row_offset = row_offset
            self.metadata = H5Metadata(
                path=path,
                dataset=dataset,
                rows=rows,
                channels=int(raw.shape[1]),
                fs=fs,
                chunks=raw.chunks,
                scale_to_mv=1000.0 if unit in {"v", "volt", "volts"} else 1.0,
                time_offset=time_offset + row_offset / fs,
                channel_ids=tuple(int(value) for value in ids),
                timing_metadata=read_h5_timing_metadata(h5),
                provenance=read_h5_provenance(h5),
            )
        return self.metadata

    def read(self, first: int, last: int, columns, step: int = 1) -> np.ndarray:
        """Return only ``raw[first:last:step, columns]`` in mV float32."""
        meta = self.metadata
        if meta is None:
            raise RuntimeError("No HDF5 source is loaded.")
        first = max(0, min(int(first), meta.rows))
        last = max(first, min(int(last), meta.rows))
        step = max(1, int(step))
        with h5py.File(meta.path, "r") as h5:
            values = np.asarray(
                self._read_dataset_columns(
                    h5[meta.dataset],
                    slice(self._row_offset + first, self._row_offset + last, step),
                    columns,
                ),
                dtype=np.float32,
            )
        if meta.scale_to_mv != 1.0:
            values *= meta.scale_to_mv
        return values

    @staticmethod
    def _read_dataset_columns(dataset, row_slice: slice, columns):
        """Read arbitrary column order while satisfying h5py's sorted-index rule."""
        if isinstance(columns, slice) or np.asarray(columns).ndim == 0:
            return dataset[row_slice, columns]
        indices = np.asarray(columns)
        if indices.dtype == np.bool_:
            if indices.size != dataset.shape[1]:
                raise IndexError("Boolean channel mask length does not match the HDF5 channel count.")
            indices = np.flatnonzero(indices)
        else:
            indices = indices.astype(np.int64, copy=False).ravel()
        indices = np.where(indices < 0, int(dataset.shape[1]) + indices, indices)
        if np.any(indices < 0) or np.any(indices >= dataset.shape[1]):
            raise IndexError("HDF5 channel column is out of range.")
        if not indices.size:
            return dataset[row_slice, 0:0]
        # np.unique sorts the request and also returns the map required to
        # restore the caller's original order (and duplicate columns).
        sorted_unique, restore_order = np.unique(indices, return_inverse=True)
        sorted_values = np.asarray(dataset[row_slice, sorted_unique])
        return sorted_values[:, restore_order]

    def read_many(self, ranges, columns, step: int = 1) -> list[np.ndarray]:
        """Read several independent slices while opening the HDF5 file once."""
        meta = self.metadata
        if meta is None:
            raise RuntimeError("No HDF5 source is loaded.")
        step = max(1, int(step))
        normalized = []
        for first, last in ranges:
            first = max(0, min(int(first), meta.rows))
            last = max(first, min(int(last), meta.rows))
            normalized.append((first, last))
        output = []
        with h5py.File(meta.path, "r") as h5:
            dataset = h5[meta.dataset]
            for first, last in normalized:
                values = np.asarray(
                    self._read_dataset_columns(
                        dataset,
                        slice(self._row_offset + first, self._row_offset + last, step),
                        columns,
                    ),
                    dtype=np.float32,
                )
                if meta.scale_to_mv != 1.0:
                    values *= meta.scale_to_mv
                output.append(values)
        return output

    def materialize(self, progress=None):
        """Read all samples only for an operation that genuinely needs them.

        Display code must use :meth:`read`; remapping/filtering may use this
        method, which respects native HDF5 chunks and automatically chooses a
        disk-backed output for very large recordings.
        """
        meta = self.metadata
        if meta is None:
            raise RuntimeError("No HDF5 source is loaded.")
        target, target_path = allocate_storage((meta.rows, meta.channels), DATA_DTYPE, "raw")
        with h5py.File(meta.path, "r") as h5:
            dataset = h5[meta.dataset]
            if dataset.chunks and self._row_offset == 0 and meta.rows == dataset.shape[0]:
                chunk_shape = tuple(max(1, int(value)) for value in dataset.chunks)
                chunk_grid = tuple(
                    max(1, int(np.ceil(size / chunk)))
                    for size, chunk in zip(dataset.shape, chunk_shape)
                )
                total = max(1, int(np.prod(chunk_grid, dtype=np.int64)))
                for index, selection in enumerate(dataset.iter_chunks(), start=1):
                    first_row, last_row = int(selection[0].start), int(selection[0].stop)
                    report_progress(progress, (index - 1) / total, f"HDF5 chunk 读取中 {index}/{total}（行 {first_row:,}-{last_row:,}）")
                    target[selection] = dataset[selection]
                    report_progress(progress, index / total, f"HDF5 chunk 已读取 {index}/{total}（行 {first_row:,}-{last_row:,}）")
            else:
                total = max(1, int(np.ceil(meta.rows / PROCESS_CHUNK_SAMPLES)))
                for index, first in enumerate(range(0, meta.rows, PROCESS_CHUNK_SAMPLES), start=1):
                    last = min(meta.rows, first + PROCESS_CHUNK_SAMPLES)
                    report_progress(progress, (index - 1) / total, f"HDF5 分块读取中 {index}/{total}（行 {first:,}-{last:,}）")
                    target[first:last] = dataset[self._row_offset + first:self._row_offset + last]
                    report_progress(progress, index / total, f"HDF5 分块已读取 {index}/{total}（行 {first:,}-{last:,}）")
        if meta.scale_to_mv != 1.0:
            report_progress(progress, 0.98, "HDF5 数据单位转换为 mV")
            target *= meta.scale_to_mv
        if isinstance(target, np.memmap):
            target.flush()
        report_progress(progress, 1.0, "HDF5 数据读取完成")
        return target, target_path


class ArraySource:
    """Same slice interface as :class:`LazyH5Source` for processed matrices."""

    def __init__(
        self, data, fs: float, time_offset: float = 0.0, label: str = "processed", storage_path=None,
        channel_ids=None, timing_metadata=None, provenance=None,
    ):
        array = np.asarray(data)
        if array.ndim != 2:
            raise ValueError(f"Expected samples x channels data, got {array.shape}.")
        self.data = data
        self.storage_path = Path(storage_path) if storage_path else None
        ids = np.arange(1, array.shape[1] + 1, dtype=np.int64) if channel_ids is None else np.asarray(channel_ids, dtype=np.int64).ravel()
        if ids.size != array.shape[1]:
            raise ValueError("channel_ids length must match data channels.")
        self.metadata = H5Metadata(
            path=self.storage_path or Path(f"<{label}>"),
            dataset=label,
            rows=int(array.shape[0]),
            channels=int(array.shape[1]),
            fs=float(fs),
            chunks=None,
            scale_to_mv=1.0,
            time_offset=float(time_offset),
            channel_ids=tuple(int(value) for value in ids),
            timing_metadata=dict(timing_metadata or {}),
            provenance=None if provenance is None else dict(provenance),
        )

    @property
    def loaded(self) -> bool:
        return True

    def read(self, first: int, last: int, columns, step: int = 1) -> np.ndarray:
        meta = self.metadata
        first = max(0, min(int(first), meta.rows))
        last = max(first, min(int(last), meta.rows))
        return np.asarray(self.data[first:last:max(1, int(step)), columns], dtype=DATA_DTYPE)

    def read_many(self, ranges, columns, step: int = 1) -> list[np.ndarray]:
        """Array-backed counterpart of :meth:`LazyH5Source.read_many`."""
        return [self.read(first, last, columns, step=step) for first, last in ranges]

    def materialize(self, progress=None):
        report_progress(progress, 1.0, "内存数据已就绪")
        return self.data, self.storage_path


class CustomChannelH5Source:
    """Lazy composite for the legacy one-file-per-processed-channel format."""

    def __init__(self, filenames) -> None:
        files = [Path(filename) for filename in filenames]
        if not files:
            raise ValueError("No custom-channel H5 files were selected.")
        records = []
        for path in files:
            if not path.is_file() or not h5py.is_hdf5(path):
                raise ValueError(f"Not an HDF5 custom-channel file: {path}")
            with h5py.File(path, "r") as h5:
                dataset = next(
                    (key for key in ("signal", "rawData512", "raw_data", "data")
                     if key in h5 and isinstance(h5[key], h5py.Dataset)
                     and (h5[key].ndim == 1 or (h5[key].ndim == 2 and h5[key].shape[1] == 1))),
                    None,
                )
                if dataset is None:
                    raise ValueError(f"{path.name} has no single-channel signal dataset.")
                fs_raw = h5["FS"][()] if "FS" in h5 else h5.attrs.get("FS", h5.attrs.get("fs", 6490.0))
                fs = float(np.asarray(fs_raw).squeeze())
                if not np.isfinite(fs) or fs <= 0:
                    raise ValueError(f"{path.name} has an invalid FS.")
                channel_id = None
                for key in ("target_channel_id", "physical_channel_id", "channel", "channel_id"):
                    if key in h5:
                        channel_id = int(np.asarray(h5[key][()]).squeeze()); break
                    if key in h5.attrs:
                        channel_id = int(np.asarray(h5.attrs[key]).squeeze()); break
                if channel_id is None:
                    match = re.search(r"(?:^|[_-])ch(?:annel)?[_-]?(\d+)(?:\D|$)", path.stem, re.I)
                    if match:
                        channel_id = int(match.group(1))
                if channel_id is None or channel_id < 1:
                    raise ValueError(f"{path.name} has no physical channel ID.")
                time_dataset = h5.get("time")
                if isinstance(time_dataset, h5py.Dataset) and time_dataset.ndim == 0:
                    offset = float(np.asarray(time_dataset[()]).squeeze())
                elif isinstance(time_dataset, h5py.Dataset) and time_dataset.shape[0]:
                    offset = float(time_dataset[0])
                else:
                    offset = float(h5.attrs.get("time_offset_sec", 0.0))
                unit = _text(h5["data_unit"][()] if "data_unit" in h5 else h5.attrs.get("data_unit", "")).strip().lower()
                records.append((
                    channel_id, path, dataset, int(h5[dataset].shape[0]), fs, offset,
                    1000.0 if unit in {"v", "volt", "volts"} else 1.0,
                    read_h5_timing_metadata(h5), read_h5_provenance(h5),
                ))
        records.sort(key=lambda record: record[0])
        ids = [record[0] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate physical channel IDs in custom-channel H5 files.")
        rows, fs, offset = records[0][3:6]
        if any(record[3] != rows or not np.isclose(record[4], fs) for record in records):
            raise ValueError("Custom-channel files must have identical sample count and FS.")
        self._records = records
        self.metadata = H5Metadata(
            path=records[0][1], dataset="custom_channels", rows=rows, channels=len(records), fs=fs,
            chunks=None, scale_to_mv=1.0, time_offset=offset, channel_ids=tuple(ids),
            timing_metadata=dict(records[0][7]), provenance=records[0][8],
        )

    @property
    def loaded(self) -> bool:
        return True

    def _column_indices(self, columns) -> tuple[np.ndarray, bool]:
        if isinstance(columns, slice):
            return np.arange(self.metadata.channels, dtype=np.int64)[columns], False
        values = np.asarray(columns)
        if values.ndim == 0:
            return np.asarray([int(values)], dtype=np.int64), True
        return values.astype(np.int64, copy=False).ravel(), False

    def read(self, first: int, last: int, columns, step: int = 1) -> np.ndarray:
        meta = self.metadata
        first = max(0, min(int(first), meta.rows)); last = max(first, min(int(last), meta.rows)); step = max(1, int(step))
        indices, scalar = self._column_indices(columns)
        if np.any(indices < 0) or np.any(indices >= meta.channels):
            raise IndexError("Custom-channel column is out of range.")
        values = []
        for index in indices:
            _, path, dataset, _, _, _, scale, _timing, _provenance = self._records[int(index)]
            with h5py.File(path, "r") as h5:
                column = np.asarray(h5[dataset][first:last:step], dtype=DATA_DTYPE).reshape(-1)
            if scale != 1.0:
                column *= scale
            values.append(column)
        output = np.column_stack(values) if values else np.empty(((last - first + step - 1) // step, 0), dtype=DATA_DTYPE)
        return output[:, 0] if scalar else output

    def materialize(self, progress=None):
        meta = self.metadata
        target, target_path = allocate_storage((meta.rows, meta.channels), DATA_DTYPE, "custom_channels")
        for first in range(0, meta.rows, PROCESS_CHUNK_SAMPLES):
            last = min(meta.rows, first + PROCESS_CHUNK_SAMPLES)
            target[first:last] = self.read(first, last, slice(None))
            report_progress(progress, last / max(1, meta.rows), "正在分块读取自定义通道 H5")
        if isinstance(target, np.memmap):
            target.flush()
        return target, target_path


def report_progress(callback, fraction: float, message: str) -> None:
    if callable(callback):
        callback(max(0.0, min(1.0, float(fraction))), str(message))


def _cache_root() -> Path:
    root = Path(tempfile.gettempdir()) / "sd_qt_processing_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_pid_is_running(pid: int) -> bool:
    if int(pid) == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            # Access denied means the process exists but is protected.
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def cleanup_process_cache_files(*, include_stale: bool = False) -> tuple[int, list[Path]]:
    """Close and remove this process's disk-backed arrays.

    ``include_stale`` also removes cache files whose filename PID no longer
    belongs to a running process.  It never touches exported data or cache
    files owned by another live application instance.
    """
    with _PROCESS_CACHE_LOCK:
        references = list(_PROCESS_CACHE_ARRAYS)
        paths = set(_PROCESS_CACHE_PATHS)
        _PROCESS_CACHE_ARRAYS.clear()
        _PROCESS_CACHE_PATHS.clear()
    for reference in references:
        array = reference()
        if array is None:
            continue
        try:
            array.flush()
        except (OSError, ValueError):
            pass
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            try:
                mapping.close()
            except (OSError, ValueError):
                pass
    gc.collect()
    root = Path(tempfile.gettempdir()) / "sd_qt_processing_cache"
    if root.is_dir():
        for path in root.glob("*.dat"):
            match = _PROCESS_CACHE_NAME.search(path.name)
            if match is None:
                continue
            owner_pid = int(match.group(1))
            if owner_pid == os.getpid() or (include_stale and not _cache_pid_is_running(owner_pid)):
                paths.add(path)
    removed = 0
    retained = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            retained.append(path)
    if retained:
        with _PROCESS_CACHE_LOCK:
            _PROCESS_CACHE_PATHS.update(retained)
    try:
        root.rmdir()
    except OSError:
        pass
    return removed, retained


def allocate_storage(shape, dtype=DATA_DTYPE, label: str = "data", *, force_memmap=False):
    size = int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)
    if not force_memmap and size < MEMMAP_THRESHOLD_BYTES:
        return np.empty(shape, dtype=dtype), None
    path = _cache_root() / f"{label}_{os.getpid()}_{time.time_ns()}.dat"
    array = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    with _PROCESS_CACHE_LOCK:
        _PROCESS_CACHE_PATHS.add(path)
        _PROCESS_CACHE_ARRAYS.append(weakref.ref(array))
    return array, path


atexit.register(cleanup_process_cache_files)


def read_channel_remap(path: str | Path, channel_count: int) -> np.ndarray:
    """Read the physical-to-FPC mapping from the established H2:H513 column."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Channel remapping requires pandas and openpyxl.") from exc
    try:
        table = pd.read_excel(path, header=None, usecols="H", skiprows=1, nrows=int(channel_count))
    except Exception as exc:
        raise ValueError("Cannot read channel mapping H2:H from the selected Excel file.") from exc
    values = pd.to_numeric(table.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    if values.size != int(channel_count) or np.isnan(values).any():
        valid = int(np.isfinite(values).sum())
        raise ValueError(f"H2:H{channel_count + 1} has {valid}/{channel_count} numeric channel IDs.")
    values = values.astype(np.int64)
    if np.any(values < 1) or np.unique(values).size != values.size:
        raise ValueError("Remap indices must be unique positive channel IDs.")
    return values


def read_channel_remap_for_ids(path: str | Path, channel_ids) -> np.ndarray:
    """Return Excel H-column remap destinations for physical channel IDs.

    A reloaded collection of one-channel H5 files is intentionally sparse
    (for example only physical channels 7 and 500).  The old dense reader
    cannot be applied to its compact columns directly: its first column is
    still physical ch7, not physical ch1.  This helper keeps that identity
    through a later remapping operation.
    """
    ids = np.asarray(channel_ids, dtype=np.int64).ravel()
    if not ids.size or np.any(ids < 1) or np.unique(ids).size != ids.size:
        raise ValueError("Physical channel IDs for remapping must be unique positive integers.")
    mapping = read_channel_remap(path, int(ids.max()))
    return mapping[ids - 1]


def read_channel_layout(path: str | Path, channel_count: int) -> np.ndarray:
    """Read the legacy Excel 20×26 physical-channel placement if present.

    Invalid/missing cells are kept as zero, and callers may fill remaining
    positions with the default remapped order.  This mirrors the old GUI's
    graceful fallback for incomplete layout sheets.
    """
    try:
        import pandas as pd
        table = pd.read_excel(path, header=None, usecols="B:AA", skiprows=1, nrows=20)
    except Exception:
        return np.arange(1, int(channel_count) + 1, dtype=np.int64)
    values = pd.to_numeric(table.to_numpy().reshape(-1), errors="coerce")
    layout = np.zeros(20 * 26, dtype=np.int64)
    limit = min(layout.size, values.size)
    part = values[:limit]
    valid = np.isfinite(part)
    layout[np.flatnonzero(valid)] = part[valid].astype(np.int64)
    return layout


def remap_array(
    source, mapping, progress=None, *, max_output_channels: int | None = None,
):
    """Return mapped columns in target-ID order without materializing ID gaps.

    ``max_output_channels`` is retained only for call compatibility.  It no
    longer truncates target IDs: a target ID is an identity, not an array
    position, so large IDs and gaps must not discard data or create zero
    columns.  Callers that need the real output IDs should use
    :func:`remap_source_from_excel`, which returns them explicitly.
    """
    source = np.asarray(source)
    if source.ndim != 2:
        raise ValueError(f"Remapping needs a 2D matrix, got {source.shape}.")
    mapping = np.asarray(mapping, dtype=np.int64).ravel()
    if mapping.size != source.shape[1]:
        raise ValueError(f"Mapping has {mapping.size} values for {source.shape[1]} data channels.")
    if np.any(mapping < 1) or np.unique(mapping).size != mapping.size:
        raise ValueError("Mapping must contain unique positive target channel IDs.")
    source_order = np.argsort(mapping, kind="stable")
    target, target_path = allocate_storage(
        (source.shape[0], mapping.size), source.dtype, "remapped",
    )
    for first in range(0, source.shape[0], PROCESS_CHUNK_SAMPLES):
        last = min(source.shape[0], first + PROCESS_CHUNK_SAMPLES)
        target[first:last] = source[first:last, source_order]
        report_progress(progress, last / max(1, source.shape[0]), "正在进行通道重映射")
    if isinstance(target, np.memmap):
        target.flush()
    return target, target_path


def remap_source_streaming(
    source, mapping, progress=None, *, max_output_channels: int | None = None,
    force_memmap: bool = False, source_columns=None,
):
    """Remap a source directly into one output store, one time chunk at a time.

    Unlike ``source.materialize()`` followed by :func:`remap_array`, this
    never creates a full raw-data cache before remapping.  Output is compact:
    columns are sorted by actual target ID, absent IDs create no columns, and
    no target ID is dropped because its numeric value is large.

    ``max_output_channels`` is retained only for call compatibility and is
    intentionally ignored; target IDs are identities rather than dense array
    positions.
    """
    meta = getattr(source, "metadata", None)
    if meta is None:
        raise RuntimeError("Load a source before remapping.")
    selected_columns = (
        np.arange(meta.channels, dtype=np.int64)
        if source_columns is None else np.asarray(source_columns, dtype=np.int64).ravel()
    )
    if (not selected_columns.size or np.any(selected_columns < 0)
            or np.any(selected_columns >= meta.channels)
            or np.unique(selected_columns).size != selected_columns.size):
        raise ValueError("Remap source columns must be unique valid column indices.")
    mapping = np.asarray(mapping, dtype=np.int64).ravel()
    if mapping.size != selected_columns.size:
        raise ValueError(f"Mapping has {mapping.size} values for {selected_columns.size} selected data channels.")
    if np.any(mapping < 1) or np.unique(mapping).size != mapping.size:
        raise ValueError("Mapping must contain unique positive target channel IDs.")
    source_order = np.argsort(mapping, kind="stable")
    ordered_columns = selected_columns[source_order]
    output_channels = int(mapping.size)
    target, target_path = allocate_storage(
        (meta.rows, output_channels), DATA_DTYPE, "remapped",
        force_memmap=force_memmap,
    )
    chunk_rows = int(meta.chunks[0]) if meta.chunks and meta.chunks[0] else PROCESS_CHUNK_SAMPLES
    chunk_rows = max(1, min(meta.rows, chunk_rows))
    total_chunks = max(1, int(np.ceil(meta.rows / chunk_rows)))

    def write_chunk(first: int, last: int, values) -> None:
        target[first:last] = np.asarray(values, dtype=DATA_DTYPE)

    # Keep one HDF5 handle for the whole stream.  This both respects the
    # source chunk layout and avoids repeatedly opening a large compressed H5.
    if isinstance(source, LazyH5Source):
        with h5py.File(meta.path, "r") as h5:
            dataset = h5[meta.dataset]
            for index, first in enumerate(range(0, meta.rows, chunk_rows), start=1):
                last = min(meta.rows, first + chunk_rows)
                report_progress(progress, (index - 1) / total_chunks, f"HDF5 chunk 读取/重映射中 {index}/{total_chunks}（行 {first:,}-{last:,}）")
                if np.array_equal(ordered_columns, np.arange(ordered_columns.size)):
                    values = np.asarray(dataset[first:last, :ordered_columns.size], dtype=DATA_DTYPE)
                else:
                    values = np.asarray(
                        LazyH5Source._read_dataset_columns(
                            dataset, slice(first, last), ordered_columns,
                        ),
                        dtype=DATA_DTYPE,
                    )
                if meta.scale_to_mv != 1.0:
                    values *= meta.scale_to_mv
                write_chunk(first, last, values)
                report_progress(progress, index / total_chunks, f"HDF5 chunk 已读取并重映射 {index}/{total_chunks}（行 {first:,}-{last:,}）")
    else:
        for index, first in enumerate(range(0, meta.rows, chunk_rows), start=1):
            last = min(meta.rows, first + chunk_rows)
            report_progress(progress, (index - 1) / total_chunks, f"分块读取/重映射中 {index}/{total_chunks}（行 {first:,}-{last:,}）")
            write_chunk(first, last, source.read(first, last, ordered_columns))
            report_progress(progress, index / total_chunks, f"分块已读取并重映射 {index}/{total_chunks}（行 {first:,}-{last:,}）")
    if isinstance(target, np.memmap):
        target.flush()
    report_progress(progress, 1.0, "通道重映射完成")
    return target, target_path


def filter_array(
    data,
    fs: float,
    mode: str,
    low: float = 0.5,
    high: float = 300.0,
    progress=None,
    *,
    highpass_order: int = 3,
    lowpass_order: int = 5,
    notch: bool = False,
    notch_frequency: float = 50.0,
    notch_q: float = 30.0,
    notch_harmonics: int = 1,
    channels=None,
    channel_input=None,
    parallel_workers: int = 1,
    parallel_memory_budget_bytes: int = 512 * 1024 * 1024,
):
    """Apply preprocessing filters without silently discarding UI settings.

    ``channels`` is a zero-based sequence.  When it is supplied only those
    columns are filtered; all other columns are copied unchanged.  This is
    important for the old application's "filter only selected channels"
    workflow and avoids accidental modification of unselected recordings.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"", "off", "none"} and not notch:
        report_progress(progress, 1.0, "滤波未启用")
        return data, None
    try:
        from scipy import signal
    except ImportError as exc:
        raise ImportError("Preprocessing filters require scipy.") from exc
    source = np.asarray(data)
    if source.ndim != 2:
        raise ValueError(f"Filtering needs a 2D matrix, got {source.shape}.")
    nyquist = float(fs) / 2.0
    hp_order = max(1, int(highpass_order))
    lp_order = max(1, int(lowpass_order))
    filters = []
    if normalized_mode == "highpass":
        if not 0 < low < nyquist:
            raise ValueError(f"High-pass cutoff must be between 0 and {nyquist:g} Hz.")
        filters.append(signal.butter(hp_order, low, btype="highpass", fs=fs, output="sos"))
    elif normalized_mode == "lowpass":
        if not 0 < high < nyquist:
            raise ValueError(f"Low-pass cutoff must be between 0 and {nyquist:g} Hz.")
        filters.append(signal.butter(lp_order, high, btype="lowpass", fs=fs, output="sos"))
    elif normalized_mode == "bandpass":
        if not 0 < low < high < nyquist:
            raise ValueError(f"Band-pass requires 0 < low < high < {nyquist:g} Hz.")
        # Cascade the two legacy sections so each of their order settings is
        # respected instead of replacing both by a hard-coded fifth order.
        filters.append(signal.butter(hp_order, low, btype="highpass", fs=fs, output="sos"))
        filters.append(signal.butter(lp_order, high, btype="lowpass", fs=fs, output="sos"))
    elif normalized_mode not in {"", "off", "none"}:
        raise ValueError(f"Unknown filter mode: {mode!r}")
    if notch:
        if not 0 < float(notch_frequency) < nyquist:
            raise ValueError(f"Notch frequency must be between 0 and {nyquist:g} Hz.")
        harmonic_count = max(1, int(notch_harmonics))
        for harmonic in range(1, harmonic_count + 1):
            frequency = float(notch_frequency) * harmonic
            # There is no representable notch at/above Nyquist.  Silently
            # stop at that physical limit while preserving all valid lower
            # harmonics (50, 100, 150 Hz, ...).
            if frequency >= nyquist:
                break
            b, a = signal.iirnotch(frequency, max(0.1, float(notch_q)), fs=fs)
            filters.append(signal.tf2sos(b, a))

    if channels is None:
        selected = np.arange(source.shape[1], dtype=np.int64)
    else:
        selected = np.unique(np.asarray(channels, dtype=np.int64).ravel())
        selected = selected[(selected >= 0) & (selected < source.shape[1])]
        if selected.size == 0:
            raise ValueError("指定通道为空或不在当前数据范围内。")
    replacement_input = None
    if channel_input is not None:
        replacement_input = np.asarray(channel_input)
        expected_shape = (source.shape[0], selected.size)
        if replacement_input.shape != expected_shape:
            raise ValueError(
                f"Replacement filter input must have shape {expected_shape}, "
                f"got {replacement_input.shape}."
            )
    target, target_path = allocate_storage(source.shape, source.dtype, "filtered")
    all_channels_selected = selected.size == source.shape[1] and np.array_equal(selected, np.arange(source.shape[1]))
    # When every channel is filtered, each block can be written once after
    # filtering.  This avoids an otherwise redundant full target[:] = source
    # copy for large disk-backed remapped recordings.
    if not all_channels_selected:
        target[:] = source
    channel_groups = [
        (first_channel, selected[first_channel:first_channel + FILTER_CHANNEL_CHUNK])
        for first_channel in range(0, selected.size, FILTER_CHANNEL_CHUNK)
    ]
    requested_workers = max(1, int(parallel_workers))
    # sosfiltfilt commonly promotes intermediates to float64 and keeps
    # several work arrays.  Use a deliberately conservative estimate so the
    # ordinary full-record filter never gains speed by risking an OOM.
    largest_group = max((columns.size for _first, columns in channel_groups), default=1)
    estimated_task_bytes = max(1, int(source.shape[0]) * largest_group * 8 * 6)
    memory_limited_workers = max(
        1, int(parallel_memory_budget_bytes) // estimated_task_bytes,
    )
    actual_workers = min(requested_workers, len(channel_groups), memory_limited_workers)

    def filter_group(first_channel, active_columns):
        if replacement_input is None:
            block = np.asarray(source[:, active_columns], dtype=DATA_DTYPE)
        else:
            local_columns = slice(
                first_channel, first_channel + active_columns.size,
            )
            block = np.asarray(replacement_input[:, local_columns], dtype=DATA_DTYPE)
        output_block = block.copy()
        finite_columns = np.isfinite(block).any(axis=0)
        if finite_columns.any():
            finite_block = block[:, finite_columns]
            try:
                for sos in filters:
                    finite_block = signal.sosfiltfilt(sos, finite_block, axis=0)
                output_block[:, finite_columns] = finite_block
            except ValueError:
                # Very short records cannot be zero-phase filtered safely;
                # retain the samples instead of failing the whole operation.
                output_block[:, finite_columns] = finite_block
        return first_channel, active_columns, output_block

    completed_channels = 0
    if actual_workers == 1:
        completed_groups = (
            filter_group(first_channel, active_columns)
            for first_channel, active_columns in channel_groups
        )
        for _first_channel, active_columns, output_block in completed_groups:
            target[:, active_columns] = output_block
            completed_channels += active_columns.size
            report_progress(
                progress, completed_channels / max(1, selected.size),
                "正在分通道组滤波（单线程）",
            )
    else:
        # Worker threads only read source data and calculate their private
        # blocks.  This thread alone writes the shared ndarray/memmap target,
        # avoiding concurrent storage writes and h5py thread-safety issues.
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit only one worker-sized batch at a time.  Besides bounding
            # memory, this preserves the preprocessing worker's opportunity
            # to observe pause/cancel requests between small channel batches.
            for batch_first in range(0, len(channel_groups), actual_workers):
                batch = channel_groups[batch_first:batch_first + actual_workers]
                futures = [
                    executor.submit(filter_group, first_channel, active_columns)
                    for first_channel, active_columns in batch
                ]
                for future in as_completed(futures):
                    _first_channel, active_columns, output_block = future.result()
                    target[:, active_columns] = output_block
                    completed_channels += active_columns.size
                    report_progress(
                        progress, completed_channels / max(1, selected.size),
                        f"正在分通道组滤波（{actual_workers}线程）",
                    )
    if isinstance(target, np.memmap):
        target.flush()
    return target, target_path


def remap_source_from_excel(
    source, mapping_path: str | Path, progress=None, *, force_memmap: bool = False,
):
    """Run the GUI's canonical “执行通道重映射” operation.

    All callers—including parameter experiments—must use this entry point so
    the Excel interpretation, physical-ID lookup, output limit and streaming
    data movement remain identical to the preprocessing button.
    """
    meta = getattr(source, "metadata", None)
    if meta is None:
        raise RuntimeError("Load a source before remapping.")
    source_ids = np.asarray(meta.channel_ids, dtype=np.int64).ravel()
    if (
        source_ids.size == meta.channels
        and np.all(source_ids >= 1)
        and np.unique(source_ids).size == source_ids.size
    ):
        mapping = read_channel_remap_for_ids(mapping_path, source_ids)
    else:
        mapping = read_channel_remap(mapping_path, meta.channels)
    # Compact output: keep only real mapped channels. Sort columns by their
    # target channel ID, but preserve those actual IDs separately instead of
    # allocating every integer position up to mapping.max() and zero-filling
    # gaps.
    source_order = np.argsort(mapping, kind="stable")
    target_channel_ids = np.asarray(mapping[source_order], dtype=np.int64)
    compact_positions = np.arange(1, target_channel_ids.size + 1, dtype=np.int64)
    data, storage_path = remap_source_streaming(
        source,
        compact_positions,
        progress,
        force_memmap=force_memmap,
        source_columns=source_order,
    )
    return data, storage_path, mapping, source_ids, target_channel_ids


def parse_alignment_log(path: str | Path, delta_t1_sec: float = 0.0) -> dict:
    """Read Recording ON and PROGRAM_START times from the experiment log."""
    path = Path(path)
    text = None
    for encoding in ("utf-8-sig", "gbk", "utf-16"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = path.read_text(errors="ignore")
    found = {"rec_start_text": None, "stim_start_text": None}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        previous = lines[index - 1] if index else line
        if "Recording ON" in line:
            found["rec_start_text"] = _extract_time(previous) or _extract_time(line)
        elif "PROGRAM_START" in line:
            found["stim_start_text"] = _extract_time(line)
    if not found["rec_start_text"] or not found["stim_start_text"]:
        raise ValueError("log.txt must contain Recording ON and PROGRAM_START timestamps.")
    rec_seconds = _clock_seconds(found["rec_start_text"])
    stim_seconds = _clock_seconds(found["stim_start_text"])
    raw_delay = stim_seconds - rec_seconds
    if raw_delay < 0:
        raw_delay += 24 * 3600
    found.update(
        rec_start_sec=rec_seconds,
        stim_start_sec=stim_seconds,
        stim_delay_raw_sec=raw_delay,
        deltaT1_sec=float(delta_t1_sec),
        stim_delay_sec=raw_delay + float(delta_t1_sec),
    )
    return found


def build_flash_markers(path: str | Path, timing: dict, fs: float) -> np.ndarray:
    """Build [sample, tag] markers from the established Flash event CSV."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Event CSV alignment requires pandas.") from exc
    try:
        table = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        table = pd.read_csv(path, encoding="gbk")
    if table.shape[1] < 6:
        raise ValueError("Flash Event CSV must contain at least six columns.")
    frequency = pd.to_numeric(table.iloc[:, 1], errors="coerce").to_numpy()
    count = pd.to_numeric(table.iloc[:, 3], errors="coerce").fillna(0).astype(int).to_numpy()
    tag = pd.to_numeric(table.iloc[:, 1], errors="coerce").to_numpy()
    offset_ms = pd.to_numeric(table.iloc[:, 5], errors="coerce").fillna(0).to_numpy()
    samples, tags = [], []
    for freq, repeats, label, offset in zip(frequency, count, tag, offset_ms):
        if not np.isfinite(freq) or freq <= 0 or repeats <= 0:
            continue
        interval = round(fs / freq)
        first = round(fs * (float(timing["stim_delay_sec"]) + float(offset) / 1000.0))
        samples.extend(first + interval * index for index in range(repeats))
        tags.extend([int(label) if np.isfinite(label) else int(freq)] * repeats)
    if not samples:
        raise ValueError("No usable Flash markers were found in the Event CSV.")
    markers = np.column_stack([samples, tags]).astype(int)
    return markers[np.argsort(markers[:, 0])]


def _normalize_letter_stim_mode(value) -> str:
    """Normalize the Letter symbols exactly as the legacy GUI does."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    text = text.replace("\ufeff", "").replace("\u3000", "").strip()
    compact = re.sub(r"\s+", "", text)
    if compact.upper() in {"O", "0"}:
        return "o"
    if compact in {"|", "\uff5c", "\u2223", "\u2502"}:
        return "|"
    if compact.lower() in {"flashdot", "flash_dot"}:
        return "\u95ea\u70c1\u5149\u70b9"
    # CSV files frequently use a double em dash (\u2014\u2014) for tag 7.
    dash_chars = {"-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212", "\uff0d"}
    if compact and all(char in dash_chars for char in compact):
        return "-"
    return compact.lower()


def build_task_markers(path: str | Path, timing: dict, fs: float, mode: str = "Flash") -> np.ndarray:
    """Port of the old Flash/Letter Event-CSV marker builder."""
    normalized = str(mode).strip().lower()
    if normalized == "flash":
        return build_flash_markers(path, timing, fs)
    if normalized != "letter":
        raise ValueError(f"Unknown task mode: {mode!r}")
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Letter markers require pandas.") from exc
    try:
        table = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        table = pd.read_csv(path, encoding="gbk")
    if table.shape[1] < 15:
        raise ValueError("Letter Event CSV must have at least 15 columns, matching VEPAverager_experiment2_v3.")
    stim_mode = table.iloc[:, 1].astype(str).to_numpy()
    dot_offset_sec = pd.to_numeric(table.iloc[:, 12], errors="coerce").to_numpy()
    letter_offset_sec = pd.to_numeric(table.iloc[:, 13], errors="coerce").to_numpy()
    tag_map = {"o": 6, "|": 4, "-": 7, "闪烁光点": 5}
    delay_sec = float(timing["stim_delay_sec"])
    samples, tags = [], []
    for mode_text, offset_sec in zip(stim_mode, letter_offset_sec):
        if not np.isfinite(offset_sec):
            continue
        key = _normalize_letter_stim_mode(mode_text)
        samples.append(round(float(fs) * (delay_sec + float(offset_sec))))
        tags.append(tag_map.get(key, 100))
    for offset_sec in dot_offset_sec:
        if np.isfinite(offset_sec):
            samples.append(round(float(fs) * (delay_sec + float(offset_sec))))
            tags.append(100)
    if not samples:
        raise ValueError("No Letter stim markers could be built from the CSV.")
    markers = np.column_stack([samples, tags]).astype(int)
    return markers[np.argsort(markers[:, 0])]


def read_behavior_detection_results(path: str | Path) -> tuple[float, float, list[dict]]:
    """Exact port of IntegratedPipelineGUI.read_behavior_detection_results."""
    path = Path(path)
    text = None
    for encoding in ("utf-8-sig", "gbk", "utf-16"):
        try:
            text = path.read_text(encoding=encoding); break
        except UnicodeError:
            continue
    if text is None:
        text = path.read_text(errors="ignore")
    frame_rate, recording_time, events = 30.0, np.nan, []
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        if value.lower().startswith("recordingtime"):
            match = re.findall(r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?", value)
            if match:
                recording_time = _clock_seconds(match[-1])
            continue
        if value.lower().startswith("framerate"):
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
            if numbers: frame_rate = float(numbers[-1])
            continue
        if not re.match(r"^[-+]?\d+\s*,", value):
            continue
        fields = [field.strip() for field in value.split(",")]
        if len(fields) < 5:
            continue
        try:
            tag, start_frame, end_frame = int(float(fields[0])), int(float(fields[1])), int(float(fields[2]))
        except ValueError:
            continue
        try:
            animal_start = None if fields[3].upper() in {"", "NA", "NAN", "NONE"} else int(float(fields[3]))
        except ValueError:
            animal_start = None
        events.append({"tag": tag, "start_frame": start_frame, "end_frame": end_frame, "animal_start_frame": animal_start, "validity": fields[4].upper()})
    if not events or not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("Detection results contain no valid events or frame rate.")
    return float(frame_rate), float(recording_time), events


def match_behavior_events_to_markers(
    markers, fs: float, frame_rate: float, video_recording_time_sec: float,
    eeg_recording_time_sec: float, detection_events: list[dict],
) -> dict[int, dict]:
    """Exact port of IntegratedPipelineGUI's first-Letter-trial matching.

    The recording-time arguments are retained for interface parity with the
    established implementation.  Its calibration is deliberately based on
    the first matching Letter trial, not wall-clock timestamps.
    """
    letter_markers = [
        row for row in np.asarray(markers, dtype=int)
        if int(row[1]) in {4, 5, 6, 7}
    ]
    letter_markers.sort(key=lambda row: int(row[0]))
    if not letter_markers:
        return {}
    first_marker = letter_markers[0]
    first_marker_sec = int(first_marker[0]) / fs
    first_tag = int(first_marker[1])
    first_event_index = next((
        index for index, event in sorted(
            enumerate(detection_events), key=lambda item: int(item[1].get("start_frame", 0))
        ) if int(event.get("tag", -1)) == first_tag
    ), None)
    if first_event_index is None:
        raise ValueError(f"No detection event matches the first Letter tag {first_tag}.")

    def frame_time_sec(frame: int) -> float:
        # Detection files use 1-based frame numbering; frame 1 is t=0.
        return max(0, int(frame) - 1) / frame_rate

    first_frame_sec = frame_time_sec(detection_events[first_event_index]["start_frame"])
    video_eeg_offset_sec = first_marker_sec - first_frame_sec

    def detection_eeg_time(frame: int) -> float:
        return video_eeg_offset_sec + frame_time_sec(frame)

    unused, matched = set(range(len(detection_events))), {}
    for marker_index, marker in enumerate(letter_markers):
        sample, tag = int(marker[0]), int(marker[1])
        marker_sec = sample / fs
        candidates = [
            index for index in unused
            if int(detection_events[index].get("tag", -1)) == tag
            and abs(detection_eeg_time(detection_events[index]["start_frame"]) - marker_sec) <= 3.0
        ]
        if marker_index == 0:
            candidates = [first_event_index] if first_event_index in unused else []
        if not candidates:
            continue
        index = min(candidates, key=lambda item: abs(detection_eeg_time(detection_events[item]["start_frame"]) - marker_sec))
        unused.remove(index)
        event = dict(detection_events[index])
        detection_start_video_sec = frame_time_sec(event["start_frame"])
        detection_start_eeg_sec = detection_eeg_time(event["start_frame"])
        detection_end_video_sec = frame_time_sec(event["end_frame"])
        detection_end_eeg_sec = detection_eeg_time(event["end_frame"])
        animal_frame = event.get("animal_start_frame")
        animal_latency = (
            detection_eeg_time(animal_frame) - marker_sec
            if animal_frame is not None and int(animal_frame) > int(event["start_frame"])
            else np.nan
        )
        matched[sample] = {
            "tag": tag, "marker_sec": marker_sec,
            "visual_start_latency_sec": detection_start_eeg_sec - marker_sec,
            "animal_latency_sec": animal_latency, "validity": event.get("validity", ""),
            "start_frame": event["start_frame"], "animal_start_frame": animal_frame,
            "end_frame": event["end_frame"],
            "start_frame_video_sec": detection_start_video_sec,
            "start_frame_eeg_sec": detection_start_eeg_sec,
            "end_frame_video_sec": detection_end_video_sec,
            "end_frame_eeg_sec": detection_end_eeg_sec,
            "end_latency_sec": detection_end_eeg_sec - marker_sec,
            "animal_start_video_sec": frame_time_sec(animal_frame) if animal_frame is not None else np.nan,
            "animal_start_eeg_sec": detection_eeg_time(animal_frame) if animal_frame is not None else np.nan,
            "video_eeg_offset_sec": video_eeg_offset_sec,
            "alignment_basis": "first_letter_trial",
        }
    return matched


def _extract_time(value: str) -> str | None:
    matches = re.findall(r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?", value)
    return matches[-1] if matches else None


def _clock_seconds(value: str) -> float:
    hour, minute, second = value.split(":")
    return int(hour) * 3600 + int(minute) * 60 + float(second)
