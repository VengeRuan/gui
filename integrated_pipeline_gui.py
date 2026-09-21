#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated acquisition-analysis launcher.

Python handles the SD-card BIN parser, HDF5 storage, preview, remapping, channel
export, and LFP/Spike analysis.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import pickle
import queue
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional runtime dependency
    h5py = None

try:
    import hdf5plugin
except ImportError:  # pragma: no cover - optional runtime dependency
    hdf5plugin = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional runtime dependency
    pd = None

try:
    import scipy.io as sio
except ImportError:  # pragma: no cover - optional runtime dependency
    sio = None

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from snr_gui import apply_notch_filter, compute_batch_channel_snr, compute_resting_snr
from compare_bin_storage import BinReader, create_h5, inspect_bin, read_timing_metadata

try:
    import scipy.signal as scisig
except ImportError:  # pragma: no cover - optional runtime dependency
    scisig = None

try:
    import scipy.fft as scifft
except ImportError:  # pragma: no cover - optional runtime dependency
    scifft = None

try:
    import mne
    from mne.preprocessing import ICA as MNEICA
except ImportError:  # pragma: no cover - optional runtime dependency
    mne = None
    MNEICA = None


FS_DEFAULT = 6490.0
OFFSET_BYTES = int("400000", 16)
BYTES_PER_GLOBAL_SAMPLE_APPROX = 4 * (2 + 128) * 2
DATA_DTYPE = np.float32
DATA_UNIT = "mV"
V_TO_MV = 1000.0
LEGACY_V_ABS_MAX_GUESS = 20.0
HDF5_COMPRESSION_LABEL = "Blosc/LZ4 + bitshuffle"


def hdf5_compression_kwargs() -> dict:
    """Return the native HDF5 filter settings used by Python-written files."""
    if hdf5plugin is None:
        raise ImportError(
            "Blosc/LZ4 + bitshuffle requires hdf5plugin. "
            "Install it with: python -m pip install hdf5plugin"
        )
    return dict(
        hdf5plugin.Blosc(
            cname="lz4",
            clevel=5,
            shuffle=hdf5plugin.Blosc.BITSHUFFLE,
        )
    )


def parse_float(value: str, default: float = 0.0) -> float:
    value = str(value).strip()
    if not value:
        return default
    return float(value)


def parse_int(value: str, default: int = 0) -> int:
    value = str(value).strip()
    if not value:
        return default
    return int(float(value))


def parse_bool_like(value, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def normalize_samples_by_channels(data: np.ndarray, expected_channels: int | None = None) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D signal array, got shape {data.shape}.")

    # HDF5 benchmark files use (samples, channels), while legacy MAT files
    # can use (channels, samples).  The old size-only rule transposed a short
    # (500 samples, 512 channels) H5 file incorrectly.
    try:
        expected = int(expected_channels) if expected_channels is not None else 0
    except (TypeError, ValueError):
        expected = 0
    if expected > 0:
        if data.shape[0] == expected and data.shape[1] != expected:
            data = data.T
    else:
        common_channel_counts = {1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 384, 512, 520}
        if data.shape[0] in common_channel_counts and data.shape[1] not in common_channel_counts:
            data = data.T
        elif data.shape[1] not in common_channel_counts and data.shape[0] <= 520 and data.shape[1] > data.shape[0]:
            data = data.T
    return np.asarray(data, dtype=DATA_DTYPE)


def data_unit_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip().lower().replace(" ", "")


def normalize_letter_stim_mode(value) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    text = text.replace("\ufeff", "").replace("\u3000", "").strip()
    compact = re.sub(r"\s+", "", text)
    if compact.upper() in {"O", "0"}:
        return "O"
    if compact in {"|", "\uff5c", "\u2223", "\u2502"}:
        return "|"
    if compact in {"\u95ea\u70c1\u5149\u70b9", "flashdot", "flash_dot"}:
        return "\u95ea\u70c1\u5149\u70b9"
    dash_chars = {"-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212", "\uff0d"}
    if compact and all(ch in dash_chars for ch in compact):
        return "-"
    return compact


def looks_like_legacy_volts(data: np.ndarray) -> bool:
    arr = np.asarray(data)
    if arr.size == 0:
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    sample = finite
    if sample.size > 200000:
        step = max(1, sample.size // 200000)
        sample = sample[::step]
    robust_abs = float(np.nanpercentile(np.abs(sample), 99.9))
    return np.isfinite(robust_abs) and robust_abs <= LEGACY_V_ABS_MAX_GUESS


def sanitize_filename_part(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("._-")


class IntegratedPipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SD BIN -> Python Analysis")
        self.geometry("1320x860")

        self.raw_data: np.ndarray | None = None
        self.remapped_data: np.ndarray | None = None
        self.remapped_channel_ids: np.ndarray | None = None
        self.layout_channel_ids: np.ndarray | None = None
        self.layout_grid: np.ndarray | None = None
        self.time: np.ndarray | None = None
        self.fs: float = FS_DEFAULT
        self.mat_path: Path | None = None
        self.bin_delta_t1_sec: float = 0.0
        self.bin_delta_t2_sec: float = np.nan
        self.last_snr_rows: list[dict] = []
        self.selected_channels: set[int] = set()
        self.timing_info: dict | None = None
        self.stim_markers: np.ndarray | None = None
        self.resting_cache: dict | None = None
        self.resting_page_index: int = 0
        self.tvep_cache: dict | None = None
        self.stim_task_cache: dict | None = None
        self.tvep_tag_index: int = 0
        self.stim_task_tag_index: int = 0
        self.tvep_page_index: int = 0
        self.psd_page_index: int = 0
        self.spike_results: list[dict] = []
        self.lfp_filtered_data: np.ndarray | None = None
        self.lfp_filtered_signature: str = ""
        self.motion_artifact_data: np.ndarray | None = None
        self.motion_artifact_signature: str = ""
        self.motion_artifact_channels: list[int] = []
        self.motion_artifact_info: dict = {}
        self.spike_filtered_data: np.ndarray | None = None
        self.spike_filtered_signature: str = ""
        self.manual_review_channels: list[int] = []
        self.manual_review_index: int = 0
        self.all_raw_preview_channels: list[int] = []
        self.all_raw_preview_index: int = 0
        self.all_raw_preview_enabled: bool = True
        self.task_analysis_source: str = "lfp"
        self.task_window: tk.Toplevel | None = None
        self.pending_task_after_spike: tuple[str, str] | None = None
        self.behavior_detection_path: Path | None = None
        self.behavior_detection_events: list[dict] = []
        self.behavior_detection_frame_rate: float = 30.0
        self.behavior_detection_recording_time_sec: float = np.nan
        self.behavior_detection_video_eeg_offset_sec: float = np.nan
        self._ui_queue: queue.Queue = queue.Queue()

        self._build_variables()
        self._build_ui()
        self.after(50, self._poll_ui_queue)

    def _build_variables(self):
        cwd = Path.cwd()
        self.bin_path_var = tk.StringVar()
        self.mat_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(cwd))
        self.channel_dir_var = tk.StringVar(value=str(cwd / "channel_exports"))
        self.remap_path_var = tk.StringVar()
        self.animal_var = tk.StringVar(value="1109")
        self.block_var = tk.StringVar(value="3")
        self.record_time_var = tk.StringVar()
        self.start_sec_var = tk.StringVar(value="0")
        self.duration_sec_var = tk.StringVar(value="60")
        self.parse_full_length_var = tk.BooleanVar(value=True)
        self.save_mat_var = tk.BooleanVar(value=True)
        self.file_info_var = tk.StringVar(value="Select a BIN file.")
        self.bin_timing_status_var = tk.StringVar(value="BIN timing: deltaT1 not loaded")
        self.preview_channels_var = tk.StringVar(value="1,92,204,325")
        self.preview_start_var = tk.StringVar(value="0")
        self.preview_duration_var = tk.StringVar(value="5")
        self.preview_display_var = tk.StringVar(value="raw")
        self.preview_band_low_var = tk.StringVar(value="0.5")
        self.preview_band_high_var = tk.StringVar(value="300")
        self.preview_linewidth_var = tk.StringVar(value="0.45")
        self.preview_notch_var = tk.BooleanVar(value=False)
        self.snr_mode_var = tk.StringVar(value="resting")
        self.snr_threshold_var = tk.StringVar(value="5")
        self.signal_band_low_var = tk.StringVar(value="1")
        self.signal_band_high_var = tk.StringVar(value="30")
        self.noise_band_low_var = tk.StringVar(value="1")
        self.noise_band_high_var = tk.StringVar(value="200")
        self.stim_interval_var = tk.StringVar(value="2")
        self.stim_duration_var = tk.StringVar(value="0.1")
        self.first_onset_var = tk.StringVar(value="0.5")
        self.stim_freq_var = tk.StringVar(value="10")
        self.harmonics_var = tk.StringVar(value="3")
        self.neighbor_bins_var = tk.StringVar(value="4")
        self.fft_len_var = tk.StringVar(value="2")
        self.notch_var = tk.BooleanVar(value=False)
        self.skip_lfp_snr_var = tk.BooleanVar(value=False)
        self.snr_highpass_var = tk.StringVar(value="off")
        self.snr_band_low_var = tk.StringVar(value="0.5")
        self.snr_band_high_var = tk.StringVar(value="300")
        self.filter_highpass_order_var = tk.StringVar(value="3")
        self.filter_lowpass_order_var = tk.StringVar(value="5")
        self.motion_artifact_enable_var = tk.BooleanVar(value=False)
        self.motion_ica_components_var = tk.StringVar(value="32")
        self.motion_ica_exclude_var = tk.StringVar(value="0,1")
        self.motion_ica_lfreq_var = tk.StringVar(value="1")
        self.motion_ica_hfreq_var = tk.StringVar(value="100")
        self.motion_ica_decim_var = tk.StringVar(value="3")
        self.motion_ica_max_iter_var = tk.StringVar(value="800")
        self.motion_artifact_status_var = tk.StringVar(value="Motion ICA: not applied")
        self.motion_preview_start_var = tk.StringVar(value="0")
        self.motion_preview_duration_var = tk.StringVar(value="6")
        self.motion_preview_mode_var = tk.StringVar(value="Stack")
        self.motion_preview_channel_var = tk.StringVar()
        self.bad_channel_check_var = tk.BooleanVar(value=True)
        self.bad_check_parallel_var = tk.BooleanVar(value=True)
        self.bad_check_workers_var = tk.StringVar(value="0")
        self.bad_flat_std_var = tk.StringVar(value="1e-4")
        self.bad_flat_ratio_var = tk.StringVar(value="30")
        self.bad_window_check_var = tk.BooleanVar(value=False)
        self.bad_window_snr_threshold_var = tk.StringVar(value="1")
        self.bad_window_ptp_threshold_var = tk.StringVar(value="0.01")
        self.selected_summary_var = tk.StringVar(value="Selected channels: 0")
        self.snr_status_var = tk.StringVar(value="SNR status: not run")
        self.snr_progress_var = tk.DoubleVar(value=0.0)
        self.snr_select_action_var = tk.StringVar(value="Select > threshold dB")
        self.snr_view_action_var = tk.StringVar(value="SNR boxplot")
        self.snr_export_action_var = tk.StringVar(value="Export current LFP SNR CSV")
        self.psd_channel_source_var = tk.StringVar(value="selected channels")
        self.psd_page_size_var = tk.StringVar(value="12")
        self.psd_welch_sec_var = tk.StringVar(value="2")
        self.psd_overlap_pct_var = tk.StringVar(value="50")
        self.psd_freq_min_var = tk.StringVar(value="0")
        self.psd_freq_max_var = tk.StringVar(value="300")
        self.psd_scale_var = tk.StringVar(value="dB")
        self.psd_powerline_notch_var = tk.BooleanVar(value=False)
        self.psd_powerline_freq_var = tk.StringVar(value="50")
        self.psd_powerline_harmonics_var = tk.StringVar(value="6")
        self.psd_notch_q_var = tk.StringVar(value="30")
        self.psd_mark_stim_harmonics_var = tk.BooleanVar(value=True)
        self.psd_stim_freq_var = tk.StringVar(value="10")
        self.psd_stim_harmonics_var = tk.StringVar(value="5")
        self.psd_mask_stim_harmonics_var = tk.BooleanVar(value=False)
        self.psd_harmonic_bandwidth_var = tk.StringVar(value="0.25")
        self.psd_status_var = tk.StringVar(value="PSD: not plotted")
        self.log_txt_path_var = tk.StringVar()
        self.event_csv_path_var = tk.StringVar()
        self.task_mode_var = tk.StringVar(value="Flash")
        self.state_start_var = tk.StringVar(value="0")
        self.state_duration_var = tk.StringVar(value="5")
        self.epoch_start_ms_var = tk.StringVar(value="-500")
        self.epoch_end_ms_var = tk.StringVar(value="800")
        self.baseline_start_ms_var = tk.StringVar(value="-200")
        self.baseline_end_ms_var = tk.StringVar(value="0")
        self.response_start_ms_var = tk.StringVar(value="0")
        self.response_end_ms_var = tk.StringVar(value="300")
        self.trial_count_var = tk.StringVar(value="60")
        self.n_stim_types_var = tk.StringVar(value="4")
        self.trials_per_stim_var = tk.StringVar(value="30")
        self.stimtag_var = tk.StringVar(value="2,5,10,30")
        self.compare_trial_a_var = tk.StringVar(value="30")
        self.compare_trial_b_var = tk.StringVar(value="60")
        self.analysis_view_mode_var = tk.StringVar(value="raw")
        self.task_response_aggregate_var = tk.StringVar(value="Mean")
        self.analysis_notch_var = tk.BooleanVar(value=False)
        self.analysis_bandpass_var = tk.BooleanVar(value=False)
        self.analysis_band_low_var = tk.StringVar(value="0.5")
        self.analysis_band_high_var = tk.StringVar(value="300")
        self.smooth_signal_var = tk.BooleanVar(value=False)
        self.smooth_window_size_var = tk.StringVar(value="0.02")
        self.zscoredata_var = tk.BooleanVar(value=False)
        self.resting_page_size_var = tk.StringVar(value="12")
        self.resting_page_var = tk.StringVar(value="Resting page: none")
        self.tvep_page_size_var = tk.StringVar(value="12")
        self.tvep_page_var = tk.StringVar(value="TVEP page: none")
        self.tvep_tag_var = tk.StringVar(value="")
        self.auto_plot_marker_alignment_var = tk.BooleanVar(value=False)
        self.spike_filter_low_var = tk.StringVar(value="300")
        self.spike_filter_high_var = tk.StringVar(value="3000")
        self.spike_threshold_factor_var = tk.StringVar(value="4.5")
        self.spike_window_sec_var = tk.StringVar(value="1")
        self.spike_step_ms_var = tk.StringVar(value="100")
        self.spike_refractory_ms_var = tk.StringVar(value="2")
        self.spike_pre_samples_var = tk.StringVar(value="13")
        self.spike_post_samples_var = tk.StringVar(value="19")
        self.spike_min_count_var = tk.StringVar(value="3")
        self.spike_summary_var = tk.StringVar(value="Spike SNR: not run")
        self.manual_review_status_var = tk.StringVar(value="Manual review: none")
        self.all_raw_preview_status_var = tk.StringVar(value="All raw preview: on, load data to view.")
        self.all_raw_preview_button_var = tk.StringVar(value="Hide all raw")

    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.parse_tab = ttk.Frame(self.notebook)
        self.data_tab = self.parse_tab
        self.snr_tab = ttk.Frame(self.notebook)
        self.spike_tab = ttk.Frame(self.notebook)
        self.psd_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.parse_tab, text="1 BIN/MAT + Preview")
        self.notebook.add(self.snr_tab, text="2 LFP SNR")
        self.notebook.add(self.spike_tab, text="3 Spike SNR")
        self.notebook.add(self.psd_tab, text="4 LFP PSD")

        self._build_parse_tab()
        self._build_data_tab()
        self._build_snr_tab()
        self._build_spike_tab()
        self._build_psd_tab()

    def _row(self, parent, row, label, variable, browse=None, width=72):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        if browse:
            ttk.Button(parent, text="Browse", command=browse).grid(row=row, column=2, padx=6, pady=4)
        return entry

    def _build_parse_tab(self):
        top = ttk.LabelFrame(self.parse_tab, text="Python BIN parser")
        top.pack(fill="x", padx=10, pady=10)
        top.columnconfigure(1, weight=1)

        self._row(top, 0, "BIN file", self.bin_path_var, self.choose_bin)
        self._row(top, 1, "Output folder", self.output_dir_var, self.choose_output_dir)
        self._row(top, 2, "Existing MAT/HDF5", self.mat_path_var, self.choose_mat)

        grid = ttk.Frame(top)
        grid.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=4)
        for i in range(12):
            grid.columnconfigure(i, weight=1)
        fields = [
            ("animal", self.animal_var),
            ("block", self.block_var),
            ("date", self.record_time_var),
            ("start sec", self.start_sec_var),
            ("duration sec", self.duration_sec_var),
        ]
        self.parse_start_entry = None
        self.parse_duration_entry = None
        for i, (label, var) in enumerate(fields):
            ttk.Label(grid, text=label).grid(row=0, column=i * 2, sticky="e", padx=3)
            entry = ttk.Entry(grid, textvariable=var, width=12)
            entry.grid(row=0, column=i * 2 + 1, sticky="w", padx=3)
            if var is self.start_sec_var:
                self.parse_start_entry = entry
            elif var is self.duration_sec_var:
                self.parse_duration_entry = entry
        ttk.Checkbutton(
            grid,
            text="full length",
            variable=self.parse_full_length_var,
            command=self.on_parse_full_length_changed,
        ).grid(row=0, column=10, columnspan=2, sticky="w", padx=(14, 3))
        self.on_parse_full_length_changed()

        ttk.Label(top, textvariable=self.file_info_var, foreground="#445").grid(
            row=4, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        ttk.Label(top, textvariable=self.bin_timing_status_var, foreground="#245").grid(
            row=5, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))
        ttk.Label(
            top,
            text="date is auto-filled from BIN dt1 when available and tags the output filename.",
            foreground="#666",
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))

        btns = ttk.Frame(top)
        btns.grid(row=7, column=0, columnspan=3, sticky="w", padx=6, pady=8)
        ttk.Button(btns, text="Run Python parser", command=self.run_python_parser).pack(side="left", padx=4)
        ttk.Button(btns, text="Load existing MAT/HDF5", command=self.load_existing_mat).pack(side="left", padx=4)
        ttk.Label(btns, text="Output: HDF5").pack(side="left", padx=(18, 4))
        ttk.Button(btns, text="Save processed H5", command=self.save_processed_dataset).pack(side="left", padx=(18, 4))
        ttk.Button(btns, text="Load processed H5", command=self.load_processed_dataset).pack(side="left", padx=4)

    def _build_data_tab(self):
        controls = ttk.LabelFrame(self.data_tab, text="Preview, remapping, and channel export")
        controls.pack(fill="x", padx=10, pady=10)
        controls.columnconfigure(1, weight=1)

        self._row(controls, 0, "Plot layout xlsx", self.remap_path_var, self.choose_remap)
        self._row(controls, 1, "Channel output folder", self.channel_dir_var, self.choose_channel_dir)

        preview = ttk.Frame(controls)
        preview.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=4)
        for i in range(8):
            preview.columnconfigure(i, weight=1)
        preview_fields = [
            ("channels", self.preview_channels_var, 24),
            ("start sec", self.preview_start_var, 10),
            ("duration sec", self.preview_duration_var, 10),
        ]
        col = 0
        for label, var, width in preview_fields:
            ttk.Label(preview, text=label).grid(row=0, column=col, sticky="e", padx=3)
            ttk.Entry(preview, textvariable=var, width=width).grid(row=0, column=col + 1, sticky="w", padx=3)
            col += 2
        ttk.Button(preview, text="Plot raw/remapped", command=self.plot_preview).grid(row=0, column=col, padx=6)
        ttk.Button(preview, text="Apply channel mapping", command=self.apply_remapping).grid(row=0, column=col + 1, padx=6)
        ttk.Button(
            preview,
            text="Batch BIN -> HDF5 files",
            command=self.export_channel_mats,
        ).grid(row=0, column=col + 2, padx=6)

        all_raw_row = ttk.Frame(controls)
        all_raw_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 4))
        all_raw_row.columnconfigure(4, weight=1)
        ttk.Label(all_raw_row, textvariable=self.all_raw_preview_status_var, foreground="#245", width=34, anchor="w").grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Button(
            all_raw_row,
            textvariable=self.all_raw_preview_button_var,
            command=self.toggle_all_raw_preview,
        ).grid(row=0, column=1, padx=(12, 4))
        ttk.Button(
            all_raw_row,
            text="Large 10x10 overview",
            command=self.open_all_raw_overview_popup,
        ).grid(row=0, column=2, padx=4)
        ttk.Label(
            all_raw_row,
            text="Uses start/duration and display settings below; one channel per page.",
            foreground="#666",
        ).grid(row=0, column=3, sticky="w", padx=8)

        display_row = ttk.Frame(controls)
        display_row.grid(row=4, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 4))
        ttk.Label(display_row, text="display").pack(side="left", padx=(0, 4))
        ttk.Combobox(
            display_row,
            textvariable=self.preview_display_var,
            values=("raw", "demean", "highpass 0.5Hz", "highpass 1Hz", "bandpass"),
            state="readonly",
            width=16,
        ).pack(side="left", padx=4)
        ttk.Label(display_row, text="band").pack(side="left", padx=(12, 4))
        ttk.Entry(display_row, textvariable=self.preview_band_low_var, width=6).pack(side="left", padx=2)
        ttk.Label(display_row, text="-").pack(side="left")
        ttk.Entry(display_row, textvariable=self.preview_band_high_var, width=6).pack(side="left", padx=2)
        ttk.Label(display_row, text="Hz").pack(side="left", padx=(2, 8))
        ttk.Label(display_row, text="line width").pack(side="left", padx=(16, 4))
        ttk.Entry(display_row, textvariable=self.preview_linewidth_var, width=7).pack(side="left", padx=4)
        ttk.Checkbutton(display_row, text="50Hz notch", variable=self.preview_notch_var).pack(side="left", padx=(16, 4))
        ttk.Label(
            display_row,
            text="Use the toolbar to zoom, pan, stretch, and save. Preview filters do not change the data.",
            foreground="#666",
        ).pack(side="left", padx=14)

        review_row = ttk.Frame(controls)
        review_row.grid(row=5, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 4))
        review_row.columnconfigure(6, weight=1)
        ttk.Label(review_row, textvariable=self.manual_review_status_var, foreground="#245", width=28, anchor="w").grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Button(review_row, text="< Prev channel", command=self.preview_previous_channel).grid(row=0, column=1, padx=4)
        ttk.Button(review_row, text="Next channel >", command=self.preview_next_channel).grid(row=0, column=2, padx=4)
        ttk.Button(review_row, text="Mark current as Good", command=self.mark_manual_review_current_good).grid(
            row=0, column=3, padx=(12, 4))
        ttk.Button(review_row, text="Mark current as Bad", command=self.mark_manual_review_current_bad).grid(
            row=0, column=4, padx=4)
        ttk.Label(
            review_row,
            text="Prev/Next follows the loaded raw channels, or the manual review list when one is active.",
            foreground="#666",
        ).grid(row=0, column=5, sticky="w", padx=8)

        fig_frame = ttk.LabelFrame(self.data_tab, text="Signal preview")
        fig_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.preview_fig = Figure(figsize=(9, 5), dpi=130)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=fig_frame)
        self.preview_toolbar = NavigationToolbar2Tk(self.preview_canvas, fig_frame, pack_toolbar=False)
        self.preview_toolbar.update()
        self.preview_toolbar.pack(fill="x", padx=6, pady=(4, 0))
        self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.preview_canvas.mpl_connect("button_press_event", self.on_preview_click_popup)
        self.preview_canvas.mpl_connect("scroll_event", self.on_preview_scroll_zoom)

    def _build_snr_tab(self):
        controls = ttk.LabelFrame(self.snr_tab, text="LFP SNR for all loaded channels")
        controls.pack(fill="x", padx=10, pady=10)

        self.snr_mode_var.set("resting")
        row1 = ttk.Frame(controls)
        row1.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Button(row1, text="Set params / Run SNR", command=self.open_snr_params_dialog).pack(side="left", padx=(0, 10))
        ttk.Label(row1, text="red if > dB").pack(side="left", padx=(0, 4))
        ttk.Entry(row1, textvariable=self.snr_threshold_var, width=6).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(row1, text="50Hz notch", variable=self.notch_var).pack(side="left", padx=(0, 14))

        ttk.Label(row1, text="Selection").pack(side="left", padx=(0, 4))
        ttk.Combobox(
            row1,
            textvariable=self.snr_select_action_var,
            values=(
                "Select > threshold dB",
                "Select all finite",
                "Select good channels",
                "Select bad channels",
                "Invert selection",
                "Clear selection",
            ),
            state="readonly",
            width=22,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(row1, text="Apply", command=self.apply_snr_select_action).pack(side="left", padx=(0, 12))
        ttk.Button(row1, text="Preview selected", command=self.preview_selected_channels).pack(side="left", padx=(0, 6))
        ttk.Button(row1, text="Manual review", command=self.start_manual_review_selected).pack(side="left", padx=(0, 12))

        row2 = ttk.Frame(controls)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="View").pack(side="left", padx=(0, 4))
        ttk.Combobox(
            row2,
            textvariable=self.snr_view_action_var,
            values=("SNR boxplot", "2s window SNR table", "10s dynamic SNR"),
            state="readonly",
            width=22,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(row2, text="Open", command=self.open_snr_view_action).pack(side="left", padx=(0, 16))

        ttk.Label(row2, text="Export").pack(side="left", padx=(0, 4))
        ttk.Combobox(
            row2,
            textvariable=self.snr_export_action_var,
            values=("Export current LFP SNR CSV", "Run parameter CSV batch"),
            state="readonly",
            width=26,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(row2, text="Export", command=self.run_snr_export_action).pack(side="left", padx=(0, 16))
        ttk.Button(row2, text="SI-style Bad Check", command=self.run_spikeinterface_style_bad_check).pack(side="left", padx=(0, 8))

        row3 = ttk.Frame(controls)
        row3.pack(fill="x", padx=8, pady=2)
        ttk.Label(row3, text="SNR filter").pack(side="left", padx=(0, 4))
        ttk.Combobox(
            row3,
            textvariable=self.snr_highpass_var,
            values=("off", "0.5", "1.0", "bandpass"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=(0, 10))
        ttk.Label(row3, text="band").pack(side="left", padx=(0, 4))
        ttk.Entry(row3, textvariable=self.snr_band_low_var, width=6).pack(side="left", padx=(0, 2))
        ttk.Label(row3, text="-").pack(side="left")
        ttk.Entry(row3, textvariable=self.snr_band_high_var, width=6).pack(side="left", padx=(2, 2))
        ttk.Label(row3, text="Hz").pack(side="left", padx=(0, 14))
        ttk.Label(row3, text="HP order").pack(side="left", padx=(0, 4))
        ttk.Entry(row3, textvariable=self.filter_highpass_order_var, width=5).pack(side="left", padx=(0, 8))
        ttk.Label(row3, text="LP order").pack(side="left", padx=(0, 4))
        ttk.Entry(row3, textvariable=self.filter_lowpass_order_var, width=5).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(row3, text="Skip SNR scoring", variable=self.skip_lfp_snr_var).pack(side="left", padx=(0, 10))

        row_motion = ttk.Frame(controls)
        row_motion.pack(fill="x", padx=8, pady=2)
        ttk.Checkbutton(
            row_motion,
            text="Motion artifact ICA",
            variable=self.motion_artifact_enable_var,
            command=self.clear_motion_artifact_cache,
        ).pack(side="left", padx=(0, 10))
        ttk.Label(row_motion, text="components").pack(side="left", padx=(0, 4))
        ttk.Entry(row_motion, textvariable=self.motion_ica_components_var, width=5).pack(side="left", padx=(0, 8))
        ttk.Label(row_motion, text="exclude IC").pack(side="left", padx=(0, 4))
        ttk.Entry(row_motion, textvariable=self.motion_ica_exclude_var, width=8).pack(side="left", padx=(0, 8))
        ttk.Label(row_motion, text="fit").pack(side="left", padx=(0, 4))
        ttk.Entry(row_motion, textvariable=self.motion_ica_lfreq_var, width=5).pack(side="left")
        ttk.Label(row_motion, text="-").pack(side="left")
        ttk.Entry(row_motion, textvariable=self.motion_ica_hfreq_var, width=5).pack(side="left", padx=(2, 2))
        ttk.Label(row_motion, text="Hz").pack(side="left", padx=(0, 8))
        ttk.Label(row_motion, text="decim").pack(side="left", padx=(0, 4))
        ttk.Entry(row_motion, textvariable=self.motion_ica_decim_var, width=4).pack(side="left", padx=(0, 8))
        ttk.Button(row_motion, text="Apply selected ICA", command=self.apply_motion_artifact_removal).pack(side="left", padx=(4, 8))
        ttk.Button(row_motion, text="Preview ICA", command=self.open_motion_ica_preview).pack(side="left", padx=(0, 8))
        ttk.Button(row_motion, text="Score ICA SNR", command=self.open_ica_snr_dialog).pack(side="left", padx=(0, 8))
        ttk.Button(row_motion, text="Clear ICA", command=self.clear_motion_artifact_cache).pack(side="left", padx=(0, 8))
        ttk.Label(row_motion, textvariable=self.motion_artifact_status_var, foreground="#245").pack(side="left", padx=(4, 0))

        row4 = ttk.Frame(controls)
        row4.pack(fill="x", padx=8, pady=2)
        ttk.Checkbutton(row4, text="Bad check", variable=self.bad_channel_check_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(row4, text="Parallel", variable=self.bad_check_parallel_var).pack(side="left", padx=(0, 8))
        ttk.Label(row4, text="workers").pack(side="left", padx=(0, 4))
        ttk.Entry(row4, textvariable=self.bad_check_workers_var, width=4).pack(side="left", padx=(0, 10))
        ttk.Label(row4, text="flat std < mV").pack(side="left", padx=(0, 4))
        ttk.Entry(row4, textvariable=self.bad_flat_std_var, width=9).pack(side="left", padx=(0, 8))
        ttk.Label(row4, text="flat ratio % >").pack(side="left", padx=(0, 4))
        ttk.Entry(row4, textvariable=self.bad_flat_ratio_var, width=7).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(row4, text="2s win bad check", variable=self.bad_window_check_var).pack(side="left", padx=(0, 10))
        ttk.Label(row4, text="valid win dB >").pack(side="left", padx=(0, 4))
        ttk.Entry(row4, textvariable=self.bad_window_snr_threshold_var, width=7).pack(side="left", padx=(0, 8))
        ttk.Label(row4, text="flat win PTP < mV").pack(side="left", padx=(0, 4))
        ttk.Entry(row4, textvariable=self.bad_window_ptp_threshold_var, width=8).pack(side="left", padx=(0, 4))

        hint = ttk.Label(
            controls,
            text="Bad check defaults to fast flat/dead checks. Enable 2s win bad check only when window-level SNR should mark bad channels; Show window SNR can still compute details on demand.",
            foreground="#556",
            wraplength=1080,
        )
        hint.pack(fill="x", padx=8, pady=(2, 8))

        ttk.Label(self.snr_tab, textvariable=self.snr_status_var, foreground="#245").pack(
            anchor="w", padx=14, pady=(0, 4))
        self.snr_progress = ttk.Progressbar(
            self.snr_tab,
            variable=self.snr_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.snr_progress.pack(fill="x", padx=14, pady=(0, 6))

        table_frame = ttk.LabelFrame(self.snr_tab, text="SNR scores")
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)
        columns = ("selected", "channel", "snr", "mode", "detail")
        self.snr_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        self.snr_tree.heading("selected", text="Use")
        self.snr_tree.heading("channel", text="Channel")
        self.snr_tree.heading("snr", text="SNR dB")
        self.snr_tree.heading("mode", text="Mode")
        self.snr_tree.heading("detail", text="Detail")
        self.snr_tree.column("selected", width=60, anchor="center")
        self.snr_tree.column("channel", width=90, anchor="center")
        self.snr_tree.column("snr", width=100, anchor="center")
        self.snr_tree.column("mode", width=100, anchor="center")
        self.snr_tree.column("detail", width=720, anchor="w")
        self.snr_tree.tag_configure("high", foreground="red")
        self.snr_tree.tag_configure("selected", background="#eaf3ff")
        self.snr_tree.tag_configure("bad", foreground="#777")
        self.snr_tree.tag_configure("si_candidate", foreground="#b45309")
        self.snr_tree.bind("<ButtonRelease-1>", self.on_snr_tree_click)
        self.snr_tree.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.snr_tree.yview)
        scroll.pack(side="right", fill="y")
        self.snr_tree.configure(yscrollcommand=scroll.set)

        ttk.Label(self.snr_tab, textvariable=self.selected_summary_var, foreground="#245").pack(
            anchor="w", padx=14, pady=(0, 8))
        self._build_inline_task_controls(self.snr_tab, source="lfp")

    def apply_snr_select_action(self):
        action = self.snr_select_action_var.get()
        if action == "Select > threshold dB":
            self.select_snr_above_threshold()
        elif action == "Select all finite":
            self.select_all_snr_channels()
        elif action == "Select good channels":
            self.select_good_snr_channels()
        elif action == "Select bad channels":
            self.select_bad_snr_channels()
        elif action == "Invert selection":
            self.invert_snr_selection()
        elif action == "Clear selection":
            self.clear_selected_channels()
        else:
            messagebox.showwarning("Unknown action", f"Unknown selection action: {action}")

    def open_snr_view_action(self):
        action = self.snr_view_action_var.get()
        if action == "SNR boxplot":
            self.plot_snr_db_by_channel()
        elif action == "2s window SNR table":
            self.show_selected_window_snr()
        elif action == "10s dynamic SNR":
            self.plot_lfp_dynamic_snr()
        else:
            messagebox.showwarning("Unknown action", f"Unknown view action: {action}")

    def run_snr_export_action(self):
        action = self.snr_export_action_var.get()
        if action == "Export current LFP SNR CSV":
            self.export_snr_csv()
        elif action == "Run parameter CSV batch":
            self.run_lfp_snr_param_csv()
        else:
            messagebox.showwarning("Unknown action", f"Unknown export action: {action}")

    def _build_inline_task_controls(self, parent, source: str):
        task = ttk.LabelFrame(parent, text="Task timing / downstream analysis")
        task.pack(fill="x", padx=10, pady=(0, 10))
        task.columnconfigure(1, weight=1)
        self._row(task, 0, "log.txt", self.log_txt_path_var, self.choose_log_txt)
        self._row(task, 1, "Event CSV", self.event_csv_path_var, self.choose_event_csv)
        button_row = ttk.Frame(task)
        button_row.grid(row=2, column=0, columnspan=3, sticky="e", padx=6, pady=4)
        ttk.Button(
            button_row,
            text="Flash_Task",
            command=lambda: self.open_task_entry(source, "Flash"),
        ).pack(side="left", padx=8)
        ttk.Button(
            button_row,
            text="LettermodeData",
            command=lambda: self.open_task_entry(source, "Letter"),
        ).pack(side="left", padx=4)
        source_label = "LFP SNR" if source == "lfp" else "Spike SNR"
        ttk.Label(
            task,
            text=(
                f"Task analysis reuses only this page's {source_label} settings. "
                "If needed, this page's SNR is computed first; no cross-page SNR or extra task filter is applied."
            ),
            foreground="#666",
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))

    def _build_state_tab(self, parent=None):
        parent = parent if parent is not None else self
        controls = ttk.LabelFrame(parent, text="Task result navigation")
        controls.pack(fill="x", padx=10, pady=10)
        ttk.Label(controls, textvariable=self.selected_summary_var, foreground="#245").pack(
            side="left", padx=8, pady=8)
        self.state_prev_button = ttk.Button(controls, text="< Prev", command=self.prev_state_page)
        self.state_prev_button.pack(side="left", padx=(18, 4))
        self.state_next_button = ttk.Button(controls, text="Next >", command=self.next_state_page)
        self.state_next_button.pack(side="left", padx=4)
        self.state_nav_info_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.state_nav_info_var, foreground="#245").pack(side="left", padx=8)
        ttk.Label(controls, text="tag/freq").pack(side="left", padx=(18, 4))
        self.tvep_tag_combo = ttk.Combobox(
            controls,
            textvariable=self.tvep_tag_var,
            values=(),
            state="disabled",
            width=10,
        )
        self.tvep_tag_combo.pack(side="left", padx=4)
        self.tvep_tag_combo.bind("<<ComboboxSelected>>", self.on_tvep_tag_selected)
        ttk.Checkbutton(
            controls,
            text="Auto plot alignment",
            variable=self.auto_plot_marker_alignment_var,
        ).pack(side="left", padx=(10, 4))
        ttk.Button(
            controls,
            text="Long behavior window",
            command=self.show_long_behavior_linked_window,
        ).pack(side="left", padx=(4, 0))
        ttk.Button(
            controls,
            text="Save current figure",
            command=self.save_current_task_figure,
        ).pack(side="left", padx=(14, 4))

        output = ttk.PanedWindow(parent, orient="horizontal")
        output.pack(fill="both", expand=True, padx=10, pady=10)
        text_frame = ttk.LabelFrame(output, text="Analysis report")
        fig_frame = ttk.LabelFrame(output, text="Analysis figure")
        output.add(text_frame, weight=2)
        output.add(fig_frame, weight=3)
        report_frame = ttk.Frame(text_frame)
        report_frame.pack(fill="both", expand=True, padx=6, pady=6)
        report_frame.rowconfigure(0, weight=1)
        report_frame.columnconfigure(0, weight=1)
        report_y_scroll = ttk.Scrollbar(report_frame, orient="vertical")
        report_x_scroll = ttk.Scrollbar(report_frame, orient="horizontal")
        self.state_text = tk.Text(
            report_frame,
            wrap="none",
            xscrollcommand=report_x_scroll.set,
            yscrollcommand=report_y_scroll.set,
        )
        self.state_text.configure(font=("Consolas", 9))
        report_y_scroll.configure(command=self.state_text.yview)
        report_x_scroll.configure(command=self.state_text.xview)
        self.state_text.grid(row=0, column=0, sticky="nsew")
        report_y_scroll.grid(row=0, column=1, sticky="ns")
        report_x_scroll.grid(row=1, column=0, sticky="ew")
        self.state_fig = Figure(figsize=(9, 6), dpi=100)
        self.state_canvas = FigureCanvasTkAgg(self.state_fig, master=fig_frame)
        self.state_canvas.mpl_connect("button_press_event", self.on_state_figure_click)
        self.state_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.current_paged_plot: str | None = None
        self.update_state_nav(None)
        self.write_state_text(
            "Choose channels in the LFP SNR or Spike SNR page first.\n\n"
            "Flash_Task: builds Flash markers from log.txt + Event CSV, then runs stimulus-locked TVEP/Flash analysis.\n\n"
            "LettermodeData: builds Letter/Dot markers from log.txt + Event CSV, then runs the task analysis.\n"
        )

    def show_task_analysis_window(self):
        if self.task_window is not None and self.task_window.winfo_exists():
            self.task_window.deiconify()
            self.task_window.lift()
            return
        self.task_window = tk.Toplevel(self)
        self.task_window.title("Flash_Task / LettermodeData analysis")
        self.task_window.geometry("1280x820")
        self.task_window.protocol("WM_DELETE_WINDOW", self.task_window.withdraw)
        self._build_state_tab(parent=self.task_window)

    def save_figure_with_dialog(self, fig: Figure, parent, title: str, initialfile: str, log_label: str):
        path = filedialog.asksaveasfilename(
            parent=parent,
            title=title,
            defaultextension=".png",
            initialfile=initialfile,
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF file", "*.pdf"),
                ("SVG file", "*.svg"),
                ("JPEG image", "*.jpg;*.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            self.log(f"Saved {log_label}: {path}")
            messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=parent)
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Save failed", str(exc), parent=parent)

    def save_current_task_figure(self):
        if not hasattr(self, "state_fig"):
            messagebox.showinfo("No figure", "No task figure is available yet.")
            return
        mode = self.current_paged_plot or self.task_mode_var.get() or "task"
        tag = self.tvep_tag_var.get().strip()
        name_parts = ["task", mode]
        if tag:
            name_parts.append(tag)
        initialfile = sanitize_filename_part("_".join(name_parts)) or "task_figure"
        parent = self.task_window if self.task_window is not None and self.task_window.winfo_exists() else self
        self.save_figure_with_dialog(
            self.state_fig,
            parent,
            "Save current task figure",
            f"{initialfile}.png",
            "current task figure",
        )

    def open_task_entry(self, source: str, mode: str):
        self.task_analysis_source = source
        if source == "lfp" and not self.last_snr_rows:
            self.run_lfp_snr(on_done=lambda: self.open_task_entry_ready(source, mode))
            return
        if source == "spike" and not self.spike_results:
            self.pending_task_after_spike = (source, mode)
            self.run_spike_snr()
            return
        self.open_task_entry_ready(source, mode)

    def open_task_entry_ready(self, source: str, mode: str):
        self.task_analysis_source = source
        self.task_mode_var.set(mode)
        self.show_task_analysis_window()
        if self.log_txt_path_var.get().strip() and self.event_csv_path_var.get().strip():
            try:
                self.load_stim_timing()
            except Exception:
                return
        title = "TVEP / Flash" if mode == "Flash" else "Stimulus / Task"
        callback = self.run_tvep_analysis if mode == "Flash" else self.run_stim_task_analysis
        self.open_analysis_params_dialog(title, callback)

    def _build_psd_tab(self):
        controls = ttk.LabelFrame(self.psd_tab, text="PSD from LFP SNR filtered data")
        controls.pack(fill="x", padx=10, pady=10)

        row1 = ttk.Frame(controls)
        row1.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(row1, text="channels").pack(side="left", padx=(0, 4))
        ttk.Combobox(
            row1,
            textvariable=self.psd_channel_source_var,
            values=("selected channels", "healthy LFP channels", "all channels"),
            state="readonly",
            width=20,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(row1, text="channels/page").pack(side="left", padx=(0, 4))
        ttk.Entry(row1, textvariable=self.psd_page_size_var, width=6).pack(side="left", padx=(0, 12))
        ttk.Label(row1, text="Welch sec").pack(side="left", padx=(0, 4))
        ttk.Entry(row1, textvariable=self.psd_welch_sec_var, width=7).pack(side="left", padx=(0, 8))
        ttk.Label(row1, text="overlap %").pack(side="left", padx=(0, 4))
        ttk.Entry(row1, textvariable=self.psd_overlap_pct_var, width=6).pack(side="left", padx=(0, 12))
        ttk.Label(row1, text="freq").pack(side="left", padx=(0, 4))
        ttk.Entry(row1, textvariable=self.psd_freq_min_var, width=6).pack(side="left")
        ttk.Label(row1, text="-").pack(side="left")
        ttk.Entry(row1, textvariable=self.psd_freq_max_var, width=6).pack(side="left", padx=(0, 4))
        ttk.Label(row1, text="Hz").pack(side="left", padx=(0, 12))
        ttk.Combobox(
            row1,
            textvariable=self.psd_scale_var,
            values=("dB", "linear"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=(0, 12))

        row2 = ttk.Frame(controls)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Checkbutton(row2, text="extra powerline notch", variable=self.psd_powerline_notch_var).pack(side="left", padx=(0, 8))
        ttk.Entry(row2, textvariable=self.psd_powerline_freq_var, width=6).pack(side="left")
        ttk.Label(row2, text="Hz x").pack(side="left", padx=(4, 2))
        ttk.Entry(row2, textvariable=self.psd_powerline_harmonics_var, width=5).pack(side="left")
        ttk.Label(row2, text="harmonics, Q").pack(side="left", padx=(4, 2))
        ttk.Entry(row2, textvariable=self.psd_notch_q_var, width=6).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row2, text="mark stim harmonics", variable=self.psd_mark_stim_harmonics_var).pack(side="left", padx=(0, 8))
        ttk.Label(row2, text="stim").pack(side="left", padx=(0, 4))
        ttk.Entry(row2, textvariable=self.psd_stim_freq_var, width=6).pack(side="left")
        ttk.Label(row2, text="Hz x").pack(side="left", padx=(4, 2))
        ttk.Entry(row2, textvariable=self.psd_stim_harmonics_var, width=5).pack(side="left")
        ttk.Checkbutton(row2, text="mask +/-", variable=self.psd_mask_stim_harmonics_var).pack(side="left", padx=(16, 2))
        ttk.Entry(row2, textvariable=self.psd_harmonic_bandwidth_var, width=6).pack(side="left")
        ttk.Label(row2, text="Hz around stim harmonics").pack(side="left", padx=(4, 0))

        row3 = ttk.Frame(controls)
        row3.pack(fill="x", padx=8, pady=(2, 8))
        ttk.Button(row3, text="Plot PSD", command=self.render_psd_page).pack(side="left", padx=(0, 6))
        ttk.Button(row3, text="< Prev page", command=lambda: self.change_psd_page(-1)).pack(side="left", padx=4)
        ttk.Button(row3, text="Next page >", command=lambda: self.change_psd_page(1)).pack(side="left", padx=4)
        ttk.Button(row3, text="Open large / fullscreen", command=self.open_psd_popup).pack(side="left", padx=(14, 6))
        ttk.Label(row3, textvariable=self.psd_status_var, foreground="#245").pack(side="left", padx=12)

        fig_frame = ttk.LabelFrame(self.psd_tab, text="LFP PSD")
        fig_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.psd_fig = Figure(figsize=(9, 5), dpi=120)
        self.psd_canvas = FigureCanvasTkAgg(self.psd_fig, master=fig_frame)
        self.psd_toolbar = NavigationToolbar2Tk(self.psd_canvas, fig_frame, pack_toolbar=False)
        self.psd_toolbar.update()
        self.psd_toolbar.pack(fill="x", padx=6, pady=(4, 0))
        self.psd_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_spike_tab(self):
        controls = ttk.LabelFrame(self.spike_tab, text="Adaptive spike detection and spike SNR")
        controls.pack(fill="x", padx=10, pady=10)
        for i in range(8):
            controls.columnconfigure(i, weight=1)

        fields = [
            ("band low Hz", self.spike_filter_low_var, "band high Hz", self.spike_filter_high_var),
            ("threshold x noise", self.spike_threshold_factor_var, "noise window s", self.spike_window_sec_var),
            ("step ms", self.spike_step_ms_var, "refractory ms", self.spike_refractory_ms_var),
            ("pre samples", self.spike_pre_samples_var, "post samples", self.spike_post_samples_var),
            ("min spikes", self.spike_min_count_var, "", None),
        ]
        for row, (label_a, var_a, label_b, var_b) in enumerate(fields):
            ttk.Label(controls, text=label_a).grid(row=row, column=0, sticky="e", padx=4, pady=4)
            ttk.Entry(controls, textvariable=var_a, width=10).grid(row=row, column=1, sticky="w", padx=4, pady=4)
            if var_b is not None:
                ttk.Label(controls, text=label_b).grid(row=row, column=2, sticky="e", padx=4, pady=4)
                ttk.Entry(controls, textvariable=var_b, width=10).grid(row=row, column=3, sticky="w", padx=4, pady=4)

        buttons = ttk.Frame(controls)
        buttons.grid(row=0, column=4, rowspan=5, columnspan=4, sticky="nsew", padx=10, pady=4)
        ttk.Label(
            buttons,
            text="Uses selected channels from page 3. Filtered spike band -> adaptive MAD threshold -> waveform template -> residual-noise SNR.",
            wraplength=520,
            foreground="#445",
        ).pack(anchor="w", pady=(0, 8))
        ttk.Button(buttons, text="Run spike detection / SNR", command=self.run_spike_snr).pack(side="left", padx=4)
        ttk.Button(buttons, text="Plot spike SNR", command=self.plot_spike_snr_summary).pack(side="left", padx=4)
        ttk.Button(buttons, text="10s dynamic Spike SNR", command=self.plot_spike_dynamic_snr).pack(side="left", padx=4)
        ttk.Button(buttons, text="Save spike results+fig", command=self.save_spike_results_bundle).pack(side="left", padx=4)
        ttk.Button(buttons, text="Export CSV", command=self.export_spike_snr_csv).pack(side="left", padx=4)

        ttk.Label(self.spike_tab, textvariable=self.spike_summary_var, foreground="#245").pack(
            anchor="w", padx=14, pady=(0, 4)
        )
        self._build_inline_task_controls(self.spike_tab, source="spike")

        table_frame = ttk.LabelFrame(self.spike_tab, text="Spike SNR results")
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)
        columns = ("channel", "spikes", "rate", "snr", "template_pp", "noise_std")
        self.spike_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "channel": "Channel",
            "spikes": "Spike count",
            "rate": "Rate Hz",
            "snr": "Spike SNR dB",
            "template_pp": "Template p-p mV",
            "noise_std": "Residual noise mV",
        }
        widths = {
            "channel": 90,
            "spikes": 120,
            "rate": 110,
            "snr": 120,
            "template_pp": 130,
            "noise_std": 130,
        }
        for col in columns:
            self.spike_tree.heading(col, text=headings[col])
            self.spike_tree.column(col, width=widths[col], anchor="center")
        self.spike_tree.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.spike_tree.yview)
        self.spike_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

    def choose_bin(self):
        path = filedialog.askopenfilename(filetypes=[("BIN files", "*.bin"), ("All files", "*.*")])
        if path:
            self.bin_path_var.set(path)
            self.autofill_block_from_bin(path)
            self.autofill_date_from_bin(path)
            self.update_file_info(path)

    def autofill_block_from_bin(self, path: str):
        stem = Path(path).stem
        match = re.search(r"(\d+)$", stem)
        if match:
            self.block_var.set(str(int(match.group(1))))

    def read_bin_dt1_date(self, path: str) -> str | None:
        header = b"Hello SD Card via FatFs!\n"
        with open(path, "rb") as f:
            data = f.read(4096)
            idx = data.find(header)
            if idx < 0:
                return None
            base = idx + 25
            if base + 8 > len(data):
                return None
            t1_raw = int.from_bytes(data[base:base + 8], byteorder="little", signed=False)
        ts1_sec = t1_raw / 1e6 + 8 * 3600
        return datetime.fromtimestamp(ts1_sec, tz=timezone.utc).strftime("%Y-%m-%d")

    def autofill_date_from_bin(self, path: str):
        try:
            date_text = self.read_bin_dt1_date(path)
        except Exception as exc:
            self.log(f"Could not read dt1 date from BIN: {exc}")
            return
        if date_text:
            self.record_time_var.set(date_text)
            self.log(f"Auto-filled date from BIN dt1: {date_text}")

    def choose_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir_var.set(path)

    def choose_channel_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.channel_dir_var.set(path)

    def choose_mat(self):
        path = filedialog.askopenfilename(
            filetypes=[("MAT/HDF5 files", ("*.mat", "*.h5", "*.hdf5")), ("All files", "*.*")]
        )
        if path:
            self.mat_path_var.set(path)

    def choose_remap(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")])
        if path:
            self.remap_path_var.set(path)

    def prompt_for_remap_file(self, current_path: str = "", title: str = "Select remapping Excel file") -> str:
        current = Path(current_path) if current_path else Path.cwd()
        initialdir = current.parent if current.parent.exists() else Path.cwd()
        path = filedialog.askopenfilename(
            title=title,
            initialdir=str(initialdir),
            filetypes=[("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")],
        )
        if path:
            self.remap_path_var.set(path)
        return path

    def choose_log_txt(self):
        path = filedialog.askopenfilename(filetypes=[("Log TXT files", "*.txt"), ("All files", "*.*")])
        if path:
            self.log_txt_path_var.set(path)

    def choose_event_csv(self):
        path = filedialog.askopenfilename(filetypes=[("Event CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.event_csv_path_var.set(path)

    def open_analysis_params_dialog(self, title: str, run_callback):
        fields_by_title = {
            "Resting": [
                ("start s", self.state_start_var, "Start time in loaded data, seconds"),
                ("duration s", self.state_duration_var, "Analysis duration, seconds"),
            ],
            "TVEP / Flash": [
                ("win_ms start", self.epoch_start_ms_var, "Matches MATLAB win_ms(1), epoch start in ms"),
                ("win_ms end", self.epoch_end_ms_var, "Matches MATLAB win_ms(2), epoch end in ms"),
                ("baseline start", self.baseline_start_ms_var, "Baseline window start"),
                ("baseline end", self.baseline_end_ms_var, "Baseline window end"),
                ("response start", self.response_start_ms_var, "Response window start"),
                ("response end", self.response_end_ms_var, "Response window end"),
                ("nStimTypes", self.n_stim_types_var, "Matches MATLAB params.nStimTypes"),
                ("trialsPerStim", self.trials_per_stim_var, "Matches MATLAB params.trialsPerStim; 0 means use all available"),
                ("stimtag", self.stimtag_var, "MATLAB tags: 4=vertical line, 5=flash dot, 6=O, 7=horizontal line, 100=other/Square"),
                ("compare A", self.compare_trial_a_var, "Fewer trials for robustness comparison"),
                ("compare B", self.compare_trial_b_var, "More trials for robustness comparison"),
            ],
            "SSVEP": [
                ("fft len sec", self.fft_len_var, "PSD/FFT window length in seconds"),
                ("stim freq", self.stim_freq_var, "Target stimulation frequency in Hz"),
                ("harmonics", self.harmonics_var, "Number of harmonics to include"),
                ("neighbor bins", self.neighbor_bins_var, "Neighbor FFT bins used as noise reference"),
            ],
            "Stimulus / Task": [
                ("win_ms start", self.epoch_start_ms_var, "Matches MATLAB win_ms(1), epoch start in ms"),
                ("win_ms end", self.epoch_end_ms_var, "Matches MATLAB win_ms(2), epoch end in ms"),
                ("baseline start", self.baseline_start_ms_var, "Baseline window start"),
                ("baseline end", self.baseline_end_ms_var, "Baseline window end"),
                ("nStimTypes", self.n_stim_types_var, "Matches MATLAB params.nStimTypes"),
                ("trialsPerStim", self.trials_per_stim_var, "Matches MATLAB params.trialsPerStim; 0 means use all available"),
                ("stimtag", self.stimtag_var, "MATLAB tags: 4=vertical line, 5=flash dot, 6=O, 7=horizontal line, 100=other/Square"),
                ("compare A", self.compare_trial_a_var, "Fewer trials, e.g. 30"),
                ("compare B", self.compare_trial_b_var, "More trials, e.g. 60"),
            ],
        }
        dialog = tk.Toplevel(self)
        dialog.title(f"{title} parameters")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        frame = ttk.LabelFrame(dialog, text=f"{title} parameter settings")
        frame.pack(fill="both", expand=True, padx=12, pady=12)
        fields = fields_by_title.get(title, [])
        ttk.Label(frame, textvariable=self.selected_summary_var, foreground="#245").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 10))
        preprocess = ttk.LabelFrame(frame, text="Display post-processing")
        preprocess.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 8))
        ttk.Label(
            preprocess,
            text=(
                "No extra notch/bandpass here. Task analysis uses the LFP SNR or Spike SNR filtered data source."
            ),
            foreground="#445",
        ).grid(row=0, column=0, columnspan=8, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(preprocess, text="smooth_signal", variable=self.smooth_signal_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        ttk.Label(preprocess, text="smooth_window_size s").grid(row=1, column=2, columnspan=2, sticky="e", padx=4, pady=4)
        ttk.Entry(preprocess, textvariable=self.smooth_window_size_var, width=8).grid(
            row=1, column=4, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(preprocess, text="zscoredata", variable=self.zscoredata_var).grid(
            row=1, column=5, columnspan=2, sticky="w", padx=8, pady=4)
        ttk.Label(preprocess, text="response aggregate").grid(row=2, column=0, columnspan=2, sticky="e", padx=4, pady=4)
        ttk.Combobox(
            preprocess,
            textvariable=self.task_response_aggregate_var,
            values=("Mean", "Median"),
            state="readonly",
            width=9,
        ).grid(row=2, column=2, sticky="w", padx=4, pady=4)

        for i, (label, var, hint) in enumerate(fields, start=2):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="e", padx=6, pady=4)
            ttk.Entry(frame, textvariable=var, width=14).grid(row=i, column=1, sticky="w", padx=6, pady=4)
            ttk.Label(frame, text=hint, foreground="#556").grid(row=i, column=2, sticky="w", padx=6, pady=4)
        note_row = len(fields) + 2
        if title != "Resting":
            ttk.Label(
                frame,
                text="TVEP/SSVEP/Stimulus tasks need log.txt and Event CSV timing first.",
                foreground="#774",
            ).grid(row=note_row, column=0, columnspan=3, sticky="w", padx=6, pady=(8, 4))
            note_row += 1
        else:
            ttk.Label(frame, text="channels/page").grid(row=note_row, column=0, sticky="e", padx=6, pady=4)
            ttk.Entry(frame, textvariable=self.resting_page_size_var, width=14).grid(
                row=note_row, column=1, sticky="w", padx=6, pady=4)
            ttk.Label(frame, text="Channels shown per page; use Previous/Next to page.", foreground="#556").grid(
                row=note_row, column=2, sticky="w", padx=6, pady=4)
            note_row += 1

        if title == "TVEP / Flash":
            ttk.Label(frame, text="channels/page").grid(row=note_row, column=0, sticky="e", padx=6, pady=4)
            ttk.Entry(frame, textvariable=self.tvep_page_size_var, width=14).grid(
                row=note_row, column=1, sticky="w", padx=6, pady=4)
            ttk.Label(frame, text="channels shown per TVEP page", foreground="#556").grid(
                row=note_row, column=2, sticky="w", padx=6, pady=4)
            note_row += 1

        buttons = ttk.Frame(frame)
        buttons.grid(row=note_row, column=0, columnspan=3, sticky="e", padx=6, pady=(10, 4))

        def run_and_close():
            dialog.destroy()
            run_callback()

        ttk.Button(buttons, text="Run", command=run_and_close).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)
        dialog.bind("<Return>", lambda _event: run_and_close())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.wait_window()

    def open_snr_params_dialog(self):
        mode = self.snr_mode_var.get()
        fields_by_mode = {
            "resting": [
                ("signal band low", self.signal_band_low_var, "Signal band low Hz"),
                ("signal band high", self.signal_band_high_var, "Signal band high Hz"),
                ("noise band low", self.noise_band_low_var, "Noise/background band low Hz"),
                ("noise band high", self.noise_band_high_var, "Noise/background band high Hz"),
            ],
            "evoked": [
                ("stim duration", self.stim_duration_var, "Response window after each marker, seconds"),
                ("signal band low", self.signal_band_low_var, "Response band low Hz"),
                ("signal band high", self.signal_band_high_var, "Response band high Hz"),
                ("noise band low", self.noise_band_low_var, "Background band low Hz"),
                ("noise band high", self.noise_band_high_var, "Background band high Hz"),
            ],
            "ssvep": [
                ("stim freq", self.stim_freq_var, "Target stimulation frequency Hz"),
                ("harmonics", self.harmonics_var, "Number of harmonics, e.g. 3 = f/2f/3f"),
                ("neighbor bins", self.neighbor_bins_var, "Neighbor FFT bins used as noise reference"),
                ("fft len sec", self.fft_len_var, "FFT window length, seconds"),
            ],
        }
        dialog = tk.Toplevel(self)
        dialog.title(f"{mode} SNR parameters")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        frame = ttk.LabelFrame(dialog, text=f"{mode} SNR parameter settings")
        frame.pack(fill="both", expand=True, padx=12, pady=12)
        ttk.Label(
            frame,
            text=f"Shared settings: red if > {self.snr_threshold_var.get()} dB; "
                 f"50Hz notch {'on' if self.notch_var.get() else 'off'}",
            foreground="#245",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 10))
        ttk.Checkbutton(
            frame,
            text="Skip SNR scoring",
            variable=self.skip_lfp_snr_var,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6))
        start_row = 2

        fields = fields_by_mode.get(mode, [])
        for i, (label, var, hint) in enumerate(fields, start=start_row):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="e", padx=6, pady=4)
            ttk.Entry(frame, textvariable=var, width=14).grid(row=i, column=1, sticky="w", padx=6, pady=4)
            ttk.Label(frame, text=hint, foreground="#556").grid(row=i, column=2, sticky="w", padx=6, pady=4)

        buttons = ttk.Frame(frame)
        buttons.grid(row=start_row + len(fields), column=0, columnspan=3, sticky="e", padx=6, pady=(10, 4))

        def run_and_close():
            dialog.destroy()
            self.run_lfp_snr()

        ttk.Button(buttons, text="Run SNR", command=run_and_close).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)
        dialog.bind("<Return>", lambda _event: run_and_close())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.wait_window()

    def open_ica_snr_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("ICA SNR parameters")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        frame = ttk.LabelFrame(dialog, text="Score current ICA-cleaned selected channels")
        frame.pack(fill="both", expand=True, padx=12, pady=12)
        ttk.Label(
            frame,
            text=(
                f"Uses current Motion ICA cache only; no bad check or full Run SNR. "
                f"50Hz notch {'on' if self.notch_var.get() else 'off'}."
            ),
            foreground="#245",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 10))

        fields = [
            ("signal band low", self.signal_band_low_var, "Signal band low Hz"),
            ("signal band high", self.signal_band_high_var, "Signal band high Hz"),
            ("noise band low", self.noise_band_low_var, "Noise/background band low Hz"),
            ("noise band high", self.noise_band_high_var, "Noise/background band high Hz"),
        ]
        for i, (label, var, hint) in enumerate(fields, start=1):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="e", padx=6, pady=4)
            ttk.Entry(frame, textvariable=var, width=14).grid(row=i, column=1, sticky="w", padx=6, pady=4)
            ttk.Label(frame, text=hint, foreground="#556").grid(row=i, column=2, sticky="w", padx=6, pady=4)

        buttons = ttk.Frame(frame)
        buttons.grid(row=1 + len(fields), column=0, columnspan=3, sticky="e", padx=6, pady=(10, 4))

        def run_and_close():
            dialog.destroy()
            self.run_ica_snr_score()

        ttk.Button(buttons, text="Run ICA SNR", command=run_and_close).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)
        dialog.bind("<Return>", lambda _event: run_and_close())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.wait_window()

    def update_file_info(self, path: str):
        size = os.path.getsize(path)
        usable = max(0, size - OFFSET_BYTES)
        approx_sec = usable / (FS_DEFAULT * BYTES_PER_GLOBAL_SAMPLE_APPROX)
        self.file_info_var.set(
            f"File size {size / 1024 / 1024:.2f} MB; rough available duration {approx_sec:.1f} sec "
            f"(precise duration is calculated by the Python BIN reader)."
        )

    def on_parse_full_length_changed(self):
        state = "disabled" if self.parse_full_length_var.get() else "normal"
        for entry in (getattr(self, "parse_start_entry", None), getattr(self, "parse_duration_entry", None)):
            if entry is not None:
                entry.configure(state=state)

    def log(self, msg: str):
        text = str(msg).strip()
        if text:
            print(text)

    def _queue_ui(self, callback) -> None:
        """Queue Tk work for the main thread; worker threads must not call Tk."""
        self._ui_queue.put(callback)

    def process_ui_queue(self) -> None:
        """Execute queued UI callbacks from the Tk main thread."""
        for _ in range(100):
            try:
                callback = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except Exception:
                self.log(traceback.format_exc())

    def _poll_ui_queue(self) -> None:
        self.process_ui_queue()
        try:
            self.after(50, self._poll_ui_queue)
        except tk.TclError:
            pass

    def update_bin_timing_status(self):
        delta_t1 = float(getattr(self, "bin_delta_t1_sec", 0.0) or 0.0)
        delta_t2 = float(getattr(self, "bin_delta_t2_sec", np.nan))
        if np.isfinite(delta_t2):
            self.bin_timing_status_var.set(
                f"BIN timing: deltaT1={delta_t1 * 1000.0:.3f} ms; deltaT2={delta_t2 * 1000.0:.3f} ms"
            )
        else:
            self.bin_timing_status_var.set(
                f"BIN timing: deltaT1={delta_t1 * 1000.0:.3f} ms; deltaT2=nan"
            )

    def ensure_data_millivolts(
        self,
        data: np.ndarray,
        unit: str = "",
        source_label: str = "data",
        expected_channels: int | None = None,
    ) -> np.ndarray:
        arr = normalize_samples_by_channels(data, expected_channels=expected_channels)
        unit_text = data_unit_text(unit)
        if unit_text in {"mv", "millivolt", "millivolts"}:
            return arr
        if unit_text in {"v", "volt", "volts"} or looks_like_legacy_volts(arr):
            self.log(f"Converted {source_label} from V to mV.")
            return np.asarray(arr * V_TO_MV, dtype=DATA_DTYPE)
        return arr

    def run_python_parser(self):
        bin_path = self.bin_path_var.get().strip()
        if not bin_path:
            messagebox.showerror("Missing BIN", "Please select a BIN file.")
            return
        out_dir = self.output_dir_var.get().strip() or str(Path(bin_path).parent)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        record_time_tag = sanitize_filename_part(self.record_time_var.get())
        if self.parse_full_length_var.get():
            segment_start_sec = 0.0
            segment_duration_sec = 0.0
        else:
            segment_start_sec = parse_float(self.start_sec_var.get())
            segment_duration_sec = parse_float(self.duration_sec_var.get())

        def worker():
            try:
                source_path = Path(bin_path).resolve()
                layout = inspect_bin(source_path)
                reader = BinReader(layout, segment_start_sec, segment_duration_sec, chunk_rows=4096)
                timing = read_timing_metadata(source_path)
                date_tag = record_time_tag or str(timing.get("dt1_date", ""))
                metadata = {
                    "source_bin": str(source_path),
                    "date": date_tag,
                    "animal": parse_int(self.animal_var.get()),
                    "block": parse_int(self.block_var.get()),
                    "start_sec": reader.start_frame / FS_DEFAULT,
                    "duration_sec": reader.selected_duration_sec,
                }
                metadata.update(timing)
                segment_tag = f"{reader.start_frame / FS_DEFAULT:.3f}s_{reader.selected_duration_sec:.3f}s".replace(".", "p")
                out_path = Path(out_dir) / (
                    f"Dog_{metadata['animal']}_Block-{metadata['block']}_"
                    f"Rec_allChans_segment_{segment_tag}.h5"
                )
                if date_tag:
                    out_path = out_path.with_name(f"{out_path.stem}_Date-{sanitize_filename_part(date_tag)}{out_path.suffix}")
                self.log(f"Running Python BIN parser: {source_path}")
                create_h5(out_path, reader, metadata, compression_level=5)
                self.log(f"Python parser wrote Blosc/LZ4 + bitshuffle H5: {out_path}")

                def finish_load():
                    self.mat_path_var.set(str(out_path))
                    self.load_existing_mat()

                self._queue_ui(finish_load)
            except Exception as exc:
                self.log(traceback.format_exc())
                self._queue_ui(lambda error=str(exc): messagebox.showerror("Python parser failed", error))

        threading.Thread(target=worker, daemon=True).start()

    def load_existing_mat(self):
        path = self.mat_path_var.get().strip()
        if not path:
            messagebox.showerror("Missing MAT", "Please select or generate a MAT file.")
            return
        try:
            data, fs, time, meta = self.read_mat(Path(path))
            self.raw_data = self.ensure_data_millivolts(
                data,
                meta.get("data_unit", ""),
                source_label="MAT/HDF5 data",
                expected_channels=meta.get("channel_count"),
            )
            self.remapped_data = None
            self.remapped_channel_ids = None
            self.layout_channel_ids = None
            self.layout_grid = None
            self.clear_processed_filter_cache()
            self.clear_selected_channels(update_tree=False)
            self.fs = float(fs)
            self.time = np.asarray(time).ravel() if time is not None else np.arange(self.raw_data.shape[0]) / self.fs
            self.mat_path = Path(path)
            delta_t1 = float(meta.get("deltaT1", 0.0))
            delta_unit = str(meta.get("deltaT1_unit", "")).strip().lower()
            # Older files may contain the newer millisecond-valued deltaT1
            # without a unit field; normal BIN timing offsets are sub-second.
            # Raw BIN deltaT1/deltaT2 fields are milliseconds.  Historical
            # files with a malformed "s" tag must not turn 161 ms into 161 s.
            delta_is_ms = "deltaT1" in meta or delta_unit in {"ms", "millisecond", "milliseconds"}
            if delta_is_ms and abs(delta_t1) >= 10_000.0:
                delta_t1 /= 1000.0
            if delta_is_ms:
                delta_t1 /= 1000.0
            self.bin_delta_t1_sec = delta_t1 if np.isfinite(delta_t1) else 0.0
            delta_t2 = float(meta.get("deltaT2", np.nan))
            if delta_is_ms and np.isfinite(delta_t2) and abs(delta_t2) >= 10_000.0:
                delta_t2 /= 1000.0
            if delta_is_ms:
                delta_t2 /= 1000.0
            self.bin_delta_t2_sec = delta_t2
            dt1_date = str(meta.get("dt1_date", "")).strip()
            if dt1_date:
                self.record_time_var.set(dt1_date)
            self.log(f"Loaded MAT/HDF5: {path}")
            self.log(f"Data shape: {self.raw_data.shape[0]} samples x {self.raw_data.shape[1]} channels; FS={self.fs}; unit={DATA_UNIT}")
            self.log(
                f"BIN timing deltaT1={self.bin_delta_t1_sec * 1000.0:.3f} ms; "
                f"deltaT2={self.bin_delta_t2_sec * 1000.0:.3f} ms"
            )
            if self.all_raw_preview_enabled:
                self.start_all_raw_preview()
            else:
                self.all_raw_preview_status_var.set("All raw preview: off")
            self.notebook.select(self.data_tab)
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("MAT/HDF5 load failed", self.h5_open_error_message(path, exc))

    def clear_processed_filter_cache(self):
        self.lfp_filtered_data = None
        self.lfp_filtered_signature = ""
        self.motion_artifact_data = None
        self.motion_artifact_signature = ""
        self.motion_artifact_channels = []
        self.motion_artifact_info = {}
        if hasattr(self, "motion_artifact_status_var"):
            self.motion_artifact_status_var.set("Motion ICA: not applied")
        self.spike_filtered_data = None
        self.spike_filtered_signature = ""

    def clear_motion_artifact_cache(self):
        self.motion_artifact_data = None
        self.motion_artifact_signature = ""
        self.motion_artifact_channels = []
        self.motion_artifact_info = {}
        self.lfp_filtered_data = None
        self.lfp_filtered_signature = ""
        self.motion_artifact_status_var.set("Motion ICA: cache cleared")

    def parse_motion_ica_params(self) -> dict:
        l_freq = parse_float(self.motion_ica_lfreq_var.get(), 1.0)
        h_freq = parse_float(self.motion_ica_hfreq_var.get(), 100.0)
        decim = max(1, parse_int(self.motion_ica_decim_var.get(), 3))
        max_iter = max(1, parse_int(self.motion_ica_max_iter_var.get(), 800))
        components = max(1, parse_int(self.motion_ica_components_var.get(), 32))
        exclude: list[int] = []
        for part in re.split(r"[,;\s]+", self.motion_ica_exclude_var.get().strip()):
            if not part:
                continue
            exclude.append(parse_int(part, 0))
        nyq = self.fs / 2.0
        if l_freq <= 0 or l_freq >= nyq:
            raise ValueError(f"Invalid ICA low cutoff: {l_freq:g} Hz; Nyquist={nyq:g} Hz.")
        if h_freq <= l_freq or h_freq >= nyq:
            raise ValueError(f"Invalid ICA high cutoff: {h_freq:g} Hz; require {l_freq:g} < high < {nyq:g}.")
        return {
            "l_freq": float(l_freq),
            "h_freq": float(h_freq),
            "decim": int(decim),
            "max_iter": int(max_iter),
            "components": int(components),
            "exclude": exclude,
            "random_state": 97,
        }

    def current_motion_artifact_signature(self) -> str:
        params = self.parse_motion_ica_params()
        settings = self.current_lfp_snr_settings()
        return self.json_dumps_safe({
            "base": self.current_base_signature(),
            "kind": "motion_ica",
            "enabled": int(self.motion_artifact_enable_var.get()),
            "channels": self.get_selected_channels(),
            "output_snr_filter": settings.get("snr_filter"),
            "output_snr_band_low": settings.get("snr_band_low"),
            "output_snr_band_high": settings.get("snr_band_high"),
            "output_hp_order": settings.get("hp_order"),
            "output_lp_order": settings.get("lp_order"),
            "params": params,
        })

    def json_dumps_safe(self, value) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    def h5_write_json_attr(self, h5_obj, key: str, value):
        h5_obj.attrs[key] = self.json_dumps_safe(value)

    def h5_read_json_attr(self, h5_obj, key: str, default=None):
        if key not in h5_obj.attrs:
            return default
        raw = h5_obj.attrs[key]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(str(raw))
        except Exception:
            return default

    def open_h5_readonly(self, path: str | Path):
        try:
            return h5py.File(path, "r")
        except OSError as exc:
            msg = str(exc).lower()
            if "lock" not in msg and "unable to synchronously open file" not in msg:
                raise
            try:
                return h5py.File(path, "r", locking=False)
            except TypeError:
                raise exc

    def h5_open_error_message(self, path: str | Path, exc: Exception) -> str:
        msg = str(exc)
        lower = msg.lower()
        if "addr overflow" in lower or "bad object header" in lower or "truncated" in lower:
            size_text = ""
            try:
                size_text = f"\nFile size: {Path(path).stat().st_size / 1024 / 1024:.2f} MB"
            except OSError:
                pass
            return (
                "This H5 file appears incomplete or corrupted, so it cannot be recovered by the loader.\n"
                "This often happens if saving was interrupted or the same file was read while it was still being written."
                f"{size_text}\n\n"
                f"Original error:\n{msg}"
            )
        if "lock" in lower:
            return (
                "This H5 file is locked by another process. Close any other GUI, Python, MATLAB, or HDF5 viewer "
                f"using the file and try again.\n\nOriginal error:\n{msg}"
            )
        if "unknown filter" in lower or "filter" in lower and ("32001" in lower or "plugin" in lower):
            return (
                "This H5 file uses the Blosc/LZ4 + bitshuffle filter, but the HDF5 filter plugin is unavailable. "
                "Install hdf5plugin or rebuild the packaged application with the current build_exe.ps1.\n\n"
                f"Original error:\n{msg}"
            )
        return msg

    def h5_write_json_dataset(self, h5_obj, key: str, value):
        text = self.json_dumps_safe(value)
        if key in h5_obj:
            del h5_obj[key]
        dtype = h5py.string_dtype(encoding="utf-8")
        h5_obj.create_dataset(key, data=text, dtype=dtype)

    def h5_read_json_dataset(self, h5_obj, key: str, default=None):
        if key not in h5_obj:
            return default
        raw = h5_obj[key][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(str(raw))
        except Exception:
            return default

    def current_base_signature(self) -> str:
        data = self.current_data()
        source = "remapped" if self.remapped_data is not None else "raw"
        return self.json_dumps_safe({
            "source": source,
            "shape": list(data.shape),
            "fs": float(self.fs),
            "unit": DATA_UNIT,
        })

    def current_lfp_filter_signature(self) -> str:
        settings = self.current_lfp_snr_settings()
        return self.json_dumps_safe({
            "base": self.current_base_signature(),
            "kind": "lfp",
            "snr_filter": settings.get("snr_filter"),
            "snr_band_low": settings.get("snr_band_low"),
            "snr_band_high": settings.get("snr_band_high"),
            "hp_order": settings.get("hp_order"),
            "lp_order": settings.get("lp_order"),
            "notch_50hz": settings.get("notch_50hz"),
            "motion_ica": self.current_motion_artifact_signature() if self.motion_artifact_enable_var.get() else "",
        })

    def spike_filter_signature_from_params(self, params: dict | None = None) -> str:
        if params is None:
            params = self.parse_spike_params()
        return self.json_dumps_safe({
            "base": self.current_base_signature(),
            "kind": "spike",
            "low": params.get("low"),
            "high": params.get("high"),
            "filter_order": 4,
        })

    def run_motion_ica_on_matrix(
        self,
        data: np.ndarray,
        channels: list[int],
        params: dict,
    ) -> tuple[np.ndarray, dict]:
        if mne is None or MNEICA is None:
            raise ImportError("mne is required for ICA motion artifact removal.")
        arr = np.asarray(data, dtype=DATA_DTYPE)
        if arr.ndim != 2 or arr.shape[1] != len(channels):
            raise ValueError("ICA input must be samples x selected_channels.")
        if len(channels) < 2:
            raise ValueError("ICA motion artifact removal needs at least 2 selected channels.")
        n_samples, n_channels = arr.shape
        if n_samples < max(64, len(channels) * 4):
            raise ValueError("Selected data is too short for ICA motion artifact removal.")

        n_components = min(int(params["components"]), n_channels, max(1, n_samples - 1))
        exclude = [idx for idx in params["exclude"] if 0 <= idx < n_components]
        ch_names = [f"ch_{ch}" for ch in channels]
        info = mne.create_info(ch_names=ch_names, sfreq=float(self.fs), ch_types=["ecog"] * n_channels)
        raw = mne.io.RawArray(arr.T, info, verbose="ERROR")
        raw_car, _ = mne.set_eeg_reference(
            raw.copy(),
            ref_channels="average",
            projection=False,
            verbose="ERROR",
        )
        raw_filtered = raw_car.filter(
            l_freq=float(params["l_freq"]),
            h_freq=float(params["h_freq"]),
            fir_design="firwin",
            n_jobs=1,
            verbose="ERROR",
        )
        target_data = self.snr_preprocess_highpass(raw_car.get_data().T)
        raw_target = mne.io.RawArray(np.asarray(target_data, dtype=DATA_DTYPE).T, info, verbose="ERROR")
        ica = MNEICA(
            n_components=n_components,
            method="fastica",
            random_state=int(params["random_state"]),
            max_iter=int(params["max_iter"]),
        )
        ica.fit(raw_filtered, decim=int(params["decim"]), verbose="ERROR")
        ica.exclude = exclude
        raw_clean = raw_target.copy()
        ica.apply(raw_clean, verbose="ERROR")
        cleaned = raw_clean.get_data().T
        output_filter = self.snr_filter_label() or "unfiltered full band"
        info_dict = {
            "channels": channels,
            "n_samples": int(n_samples),
            "n_channels": int(n_channels),
            "n_components": int(n_components),
            "exclude": exclude,
            "fit_l_freq": float(params["l_freq"]),
            "fit_h_freq": float(params["h_freq"]),
            "l_freq": float(params["l_freq"]),
            "h_freq": float(params["h_freq"]),
            "output_filter": output_filter,
            "decim": int(params["decim"]),
        }
        return np.asarray(cleaned, dtype=DATA_DTYPE), info_dict

    def get_motion_artifact_cleaned_selected_data(self) -> tuple[np.ndarray, list[int], str]:
        if not self.motion_artifact_enable_var.get():
            raise RuntimeError("Motion artifact ICA is not enabled.")
        signature = self.current_motion_artifact_signature()
        channels = self.get_selected_channels()
        if not channels:
            raise RuntimeError("Select channels before running Motion artifact ICA.")
        if (
            self.motion_artifact_data is not None
            and self.motion_artifact_signature == signature
            and self.motion_artifact_channels == channels
            and self.motion_artifact_data.shape[1] == len(channels)
        ):
            info = self.motion_artifact_info
            label = (
                f"Motion ICA fit {info.get('fit_l_freq', info.get('l_freq', self.motion_ica_lfreq_var.get()))}-"
                f"{info.get('fit_h_freq', info.get('h_freq', self.motion_ica_hfreq_var.get()))}Hz, "
                f"output {info.get('output_filter', self.snr_filter_label() or 'unfiltered full band')}, "
                f"exclude IC {','.join(str(x) for x in info.get('exclude', [])) or 'none'}"
            )
            return self.motion_artifact_data, channels, label

        data, channels = self.selected_base_data()
        params = self.parse_motion_ica_params()
        cleaned, info = self.run_motion_ica_on_matrix(data, channels, params)
        self.motion_artifact_data = cleaned
        self.motion_artifact_signature = signature
        self.motion_artifact_channels = list(channels)
        self.motion_artifact_info = info
        label = (
            f"Motion ICA fit {info['fit_l_freq']:g}-{info['fit_h_freq']:g}Hz, "
            f"output {info['output_filter']}, "
            f"components {info['n_components']}, exclude IC "
            f"{','.join(str(x) for x in info['exclude']) or 'none'}"
        )
        self.motion_artifact_status_var.set(
            f"Motion ICA: ready; fit {info['fit_l_freq']:g}-{info['fit_h_freq']:g}Hz -> {info['output_filter']}"
        )
        return cleaned, channels, label

    def apply_motion_artifact_removal(self):
        try:
            self.motion_artifact_enable_var.set(True)
            data, channels = self.selected_base_data()
            params = self.parse_motion_ica_params()
            signature = self.current_motion_artifact_signature()
        except Exception as exc:
            messagebox.showerror("Motion ICA setup failed", str(exc))
            return

        self.motion_artifact_status_var.set(f"Motion ICA: running on {len(channels)} selected channels...")
        self.set_snr_progress(5.0, "Motion ICA: starting selected-channel artifact removal")

        def worker():
            try:
                cleaned, info = self.run_motion_ica_on_matrix(data, channels, params)

                def finish():
                    self.motion_artifact_data = cleaned
                    self.motion_artifact_signature = signature
                    self.motion_artifact_channels = list(channels)
                    self.motion_artifact_info = info
                    self.lfp_filtered_data = None
                    self.lfp_filtered_signature = ""
                    self.motion_artifact_status_var.set(
                        f"Motion ICA: ready; fit {info['fit_l_freq']:g}-{info['fit_h_freq']:g}Hz -> {info['output_filter']}"
                    )
                    self.set_snr_progress(100.0, "Motion ICA: done; LFP cache will use cleaned selected channels")
                    self.log(
                        f"Motion ICA applied to selected channels {channels[:10]}"
                        f"{'...' if len(channels) > 10 else ''}; "
                        f"fit={info['fit_l_freq']:g}-{info['fit_h_freq']:g}Hz, "
                        f"output={info['output_filter']}, components={info['n_components']}, "
                        f"excluded={info['exclude'] or 'none'}."
                    )

                self._queue_ui(finish)
            except Exception as exc:
                err = str(exc)
                tb = traceback.format_exc()

                def fail():
                    self.log(tb)
                    self.motion_artifact_status_var.set("Motion ICA: failed")
                    self.set_snr_progress(0.0, "Motion ICA: failed")
                    messagebox.showerror("Motion ICA failed", err)

                self._queue_ui(fail)

        threading.Thread(target=worker, daemon=True).start()

    def run_ica_snr_score(self):
        try:
            if not self.motion_artifact_enable_var.get():
                raise RuntimeError("Enable Motion artifact ICA first.")
            channels = self.get_selected_channels()
            if not channels:
                raise RuntimeError("Select channels before scoring ICA SNR.")
            signature = self.current_motion_artifact_signature()
            if (
                self.motion_artifact_data is None
                or self.motion_artifact_signature != signature
                or self.motion_artifact_channels != channels
                or self.motion_artifact_data.shape[1] != len(channels)
            ):
                raise RuntimeError(
                    "Motion ICA cache is not ready for the current channels/settings. "
                    "Click Apply selected ICA first, then score ICA SNR."
                )
            signal_band = (
                parse_float(self.signal_band_low_var.get()),
                parse_float(self.signal_band_high_var.get()),
            )
            noise_band = (
                parse_float(self.noise_band_low_var.get()),
                parse_float(self.noise_band_high_var.get()),
            )
            nyq = self.fs / 2.0
            if not (0 <= signal_band[0] < signal_band[1] < nyq):
                raise ValueError(f"Invalid signal band: require 0 <= low < high < {nyq:g} Hz.")
            if not (0 <= noise_band[0] < noise_band[1] < nyq):
                raise ValueError(f"Invalid noise band: require 0 <= low < high < {nyq:g} Hz.")
            cleaned = np.asarray(self.motion_artifact_data, dtype=DATA_DTYPE).copy()
            label = (
                f"Motion ICA fit {self.motion_artifact_info.get('fit_l_freq', self.motion_ica_lfreq_var.get())}-"
                f"{self.motion_artifact_info.get('fit_h_freq', self.motion_ica_hfreq_var.get())}Hz, "
                f"output {self.motion_artifact_info.get('output_filter', self.snr_filter_label() or 'unfiltered full band')}, "
                f"exclude IC {','.join(str(x) for x in self.motion_artifact_info.get('exclude', [])) or 'none'}"
            )
            notch_enabled = bool(self.notch_var.get())
            threshold = parse_float(self.snr_threshold_var.get(), 5)
        except Exception as exc:
            messagebox.showerror("ICA SNR unavailable", str(exc))
            return

        self.set_snr_progress(5.0, f"ICA SNR: scoring {len(channels)} cleaned channels...")

        def worker():
            try:
                score_data = cleaned
                if notch_enabled:
                    self._queue_ui(lambda: self.set_snr_progress(30.0, "ICA SNR: applying 50Hz notch"))
                    score_data = self.apply_notch_filter_matrix(score_data, freq=50.0, q=30.0, harmonics=1, axis=0)
                self._queue_ui(lambda: self.set_snr_progress(60.0, "ICA SNR: computing Welch band-power SNR"))
                snr_values = self.compute_resting_snr_vectorized(score_data, signal_band, noise_band)
                detail_parts = [
                    "ICA-only SNR",
                    label,
                    f"signal {signal_band[0]:g}-{signal_band[1]:g}Hz / noise {noise_band[0]:g}-{noise_band[1]:g}Hz",
                ]
                if notch_enabled:
                    detail_parts.append("50Hz notch")
                detail = "; ".join(detail_parts)
                rows = [
                    {
                        "channel": int(ch),
                        "snr_db": float(snr_values[idx]),
                        "mode": "ica_resting",
                        "detail": detail,
                    }
                    for idx, ch in enumerate(channels)
                ]
                finite_count = int(sum(1 for row in rows if np.isfinite(row["snr_db"])))
                high_count = int(sum(1 for row in rows if np.isfinite(row["snr_db"]) and row["snr_db"] > threshold))

                def finish():
                    updated_by_channel = {int(row["channel"]): dict(row) for row in rows}
                    merged_rows: list[dict] = []
                    seen_channels: set[int] = set()
                    previous_total = len(self.last_snr_rows)
                    for existing in self.last_snr_rows:
                        try:
                            ch = int(existing["channel"])
                        except Exception:
                            merged_rows.append(existing)
                            continue
                        seen_channels.add(ch)
                        if ch not in updated_by_channel:
                            merged_rows.append(existing)
                            continue
                        old_row = dict(existing)
                        new_row = dict(updated_by_channel[ch])
                        old_snr = self.snr_db_value(old_row)
                        if np.isfinite(old_snr):
                            new_row["pre_ica_snr_db"] = float(old_snr)
                        if old_row.get("mode"):
                            new_row["pre_ica_mode"] = old_row.get("mode")
                        if old_row.get("detail"):
                            new_row["pre_ica_detail"] = old_row.get("detail")
                        merged_rows.append(new_row)
                    for ch, row in updated_by_channel.items():
                        if ch not in seen_channels:
                            merged_rows.append(row)
                    self.last_snr_rows = sorted(merged_rows, key=lambda row: int(row["channel"]))
                    preserved_count = max(0, len(self.last_snr_rows) - len(rows))
                    self.refresh_snr_tree(threshold=threshold)
                    self.update_selected_summary()
                    self.snr_status_var.set(
                        f"ICA SNR status: scored {finite_count}/{len(rows)} selected cleaned channels | "
                        f">{threshold:g} dB: {high_count} | preserved other rows: {preserved_count}"
                    )
                    self.set_snr_progress(100.0, self.snr_status_var.get())
                    self.notebook.select(self.snr_tab)
                    self.log(
                        f"Computed ICA-only resting SNR for {len(rows)} selected channels; "
                        f"finite={finite_count}, >{threshold:g}dB={high_count}; "
                        f"merged with previous SNR table rows {previous_total}->{len(self.last_snr_rows)}; {detail}."
                    )

                self._queue_ui(finish)
            except Exception as exc:
                err = str(exc)
                tb = traceback.format_exc()

                def fail():
                    self.log(tb)
                    self.set_snr_progress(0.0, "ICA SNR: failed")
                    messagebox.showerror("ICA SNR failed", err)

                self._queue_ui(fail)

        threading.Thread(target=worker, daemon=True).start()

    def compute_motion_ica_preview_original(self, channels: list[int]) -> np.ndarray:
        base_data, base_channels = self.selected_base_data()
        if list(base_channels) != list(channels):
            channel_to_col = {ch: idx for idx, ch in enumerate(base_channels)}
            base_data = np.column_stack([base_data[:, channel_to_col[ch]] for ch in channels])
        arr = np.asarray(base_data, dtype=DATA_DTYPE)
        car = arr - np.nanmean(arr, axis=1, keepdims=True)
        original = self.snr_preprocess_highpass(car)
        if self.notch_var.get():
            original = self.apply_notch_filter_matrix(original, freq=50.0, q=30.0, harmonics=1, axis=0)
        return np.asarray(original, dtype=DATA_DTYPE)

    def open_motion_ica_preview(self):
        try:
            if not self.motion_artifact_enable_var.get():
                raise RuntimeError("Enable Motion artifact ICA first.")
            channels = self.get_selected_channels()
            if not channels:
                raise RuntimeError("Select channels before previewing ICA.")
            signature = self.current_motion_artifact_signature()
            if (
                self.motion_artifact_data is None
                or self.motion_artifact_signature != signature
                or self.motion_artifact_channels != channels
            ):
                raise RuntimeError("Motion ICA cache is not ready for the current channels/settings. Click Apply selected ICA first.")
            cleaned = np.asarray(self.motion_artifact_data, dtype=DATA_DTYPE)
            original = self.compute_motion_ica_preview_original(channels)
            if cleaned.shape != original.shape:
                raise RuntimeError(f"ICA preview shape mismatch: original {original.shape}, cleaned {cleaned.shape}. Re-apply ICA.")
            if self.notch_var.get():
                cleaned_for_preview = self.apply_notch_filter_matrix(cleaned, freq=50.0, q=30.0, harmonics=1, axis=0)
            else:
                cleaned_for_preview = cleaned
        except Exception as exc:
            messagebox.showerror("ICA preview unavailable", str(exc))
            return

        if not self.motion_preview_channel_var.get() and channels:
            self.motion_preview_channel_var.set(str(channels[0]))
        elif parse_int(self.motion_preview_channel_var.get(), channels[0]) not in channels:
            self.motion_preview_channel_var.set(str(channels[0]))

        dialog = tk.Toplevel(self)
        dialog.title("Motion ICA preview")
        dialog.geometry("1120x820")
        dialog.transient(self)

        controls = ttk.Frame(dialog)
        controls.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(controls, text="Channel").pack(side="left", padx=(0, 4))
        channel_combo = ttk.Combobox(
            controls,
            textvariable=self.motion_preview_channel_var,
            values=[str(ch) for ch in channels],
            state="readonly",
            width=8,
        )
        channel_combo.pack(side="left", padx=(0, 8))
        ttk.Button(
            controls,
            text="< Prev",
            command=lambda: step_channel(-1),
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            controls,
            text="Next >",
            command=lambda: step_channel(1),
        ).pack(side="left", padx=(0, 14))
        ttk.Label(controls, text="Start sec").pack(side="left", padx=(0, 4))
        ttk.Entry(controls, textvariable=self.motion_preview_start_var, width=8).pack(side="left", padx=(0, 8))
        ttk.Label(controls, text="Duration").pack(side="left", padx=(0, 4))
        ttk.Entry(controls, textvariable=self.motion_preview_duration_var, width=8).pack(side="left", padx=(0, 12))
        ttk.Label(controls, text="Mode").pack(side="left", padx=(0, 4))
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.motion_preview_mode_var,
            values=("Stack", "Overlay", "PSD"),
            state="readonly",
            width=10,
        )
        mode_combo.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Apply window", command=lambda: render()).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Fullscreen", command=lambda: dialog.state("zoomed")).pack(side="left", padx=(0, 8))

        info_var = tk.StringVar()
        ttk.Label(dialog, textvariable=info_var, foreground="#245").pack(anchor="w", padx=12, pady=(0, 4))

        fig = Figure(figsize=(10.5, 7.2), dpi=110)
        canvas = FigureCanvasTkAgg(fig, master=dialog)
        toolbar = NavigationToolbar2Tk(canvas, dialog, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x", padx=8, pady=(2, 0))
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

        def current_channel_index() -> tuple[int, int]:
            ch = parse_int(self.motion_preview_channel_var.get(), channels[0])
            if ch not in channels:
                ch = channels[0]
                self.motion_preview_channel_var.set(str(ch))
            return ch, channels.index(ch)

        def step_channel(delta: int):
            ch, idx = current_channel_index()
            next_idx = (idx + delta) % len(channels)
            self.motion_preview_channel_var.set(str(channels[next_idx]))
            render()

        def render():
            ch, local_idx = current_channel_index()
            start_sec = max(0.0, parse_float(self.motion_preview_start_var.get(), 0.0))
            duration_sec = max(1.0 / max(self.fs, 1.0), parse_float(self.motion_preview_duration_var.get(), 6.0))
            start_sample = int(round(start_sec * self.fs))
            end_sample = min(original.shape[0], start_sample + int(round(duration_sec * self.fs)))
            if start_sample >= original.shape[0]:
                start_sample = max(0, original.shape[0] - int(round(duration_sec * self.fs)))
                end_sample = original.shape[0]
                self.motion_preview_start_var.set(f"{start_sample / self.fs:.3f}")
            if end_sample <= start_sample:
                end_sample = min(original.shape[0], start_sample + 1)
            t = np.arange(start_sample, end_sample, dtype=DATA_DTYPE) / self.fs
            y0 = original[start_sample:end_sample, local_idx]
            y1 = cleaned_for_preview[start_sample:end_sample, local_idx]
            removed = y0 - y1

            fig.clear()
            mode = self.motion_preview_mode_var.get()
            label = (
                f"ch{ch} | window {start_sample / self.fs:.3f}-{end_sample / self.fs:.3f}s | "
                f"fit {self.motion_artifact_info.get('fit_l_freq', self.motion_ica_lfreq_var.get())}-"
                f"{self.motion_artifact_info.get('fit_h_freq', self.motion_ica_hfreq_var.get())}Hz | "
                f"output {self.motion_artifact_info.get('output_filter', self.snr_filter_label() or 'unfiltered full band')} | "
                f"exclude IC {','.join(str(x) for x in self.motion_artifact_info.get('exclude', [])) or 'none'}"
            )
            info_var.set(label)

            if mode == "PSD":
                axes = [fig.add_subplot(1, 1, 1)]
            elif mode == "Overlay":
                axes = [fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)]
            else:
                axes = [fig.add_subplot(3, 1, 1), fig.add_subplot(3, 1, 2), fig.add_subplot(3, 1, 3)]

            if mode in {"Stack", "Overlay"}:
                ax = axes[0]
                ax.plot(t, y0, color="#1f77b4", linewidth=0.75, label="Original target")
                ax.plot(t, y1, color="#2ca02c", linewidth=0.75, label="ICA cleaned")
                ax.set_title("Original vs ICA cleaned")
                ax.set_ylabel("mV")
                ax.grid(True, alpha=0.25)
                ax.legend(loc="upper right", fontsize=8)
                if mode == "Stack":
                    ax_removed = axes[1]
                    ax_removed.plot(t, removed, color="#9467bd", linewidth=0.75, label="Removed")
                    ax_removed.set_title("Removed = original - cleaned")
                    ax_removed.set_ylabel("mV")
                    ax_removed.grid(True, alpha=0.25)
                    ax_removed.legend(loc="upper right", fontsize=8)
                    psd_ax = axes[2]
                else:
                    psd_ax = axes[1]
                psd_ax.set_xlabel("Frequency (Hz)")
            else:
                psd_ax = axes[0]

            if scisig is not None and y0.size >= 8:
                nperseg = min(y0.size, max(64, int(round(min(2.0, duration_sec) * self.fs))))
                freqs0, psd0 = scisig.welch(y0, fs=self.fs, nperseg=nperseg)
                freqs1, psd1 = scisig.welch(y1, fs=self.fs, nperseg=nperseg)
                freqs0 = np.asarray(freqs0, dtype=DATA_DTYPE)
                freqs1 = np.asarray(freqs1, dtype=DATA_DTYPE)
                psd0 = np.asarray(psd0, dtype=DATA_DTYPE)
                psd1 = np.asarray(psd1, dtype=DATA_DTYPE)
                psd_ax.plot(freqs0, 10.0 * np.log10(psd0 + np.finfo(DATA_DTYPE).eps), color="#d62728", linewidth=1.0, label="Original PSD")
                psd_ax.plot(freqs1, 10.0 * np.log10(psd1 + np.finfo(DATA_DTYPE).eps), color="#1f77b4", linewidth=1.0, label="Cleaned PSD")
                psd_ax.set_xlim(0, min(300.0, self.fs / 2.0))
                psd_ax.set_ylabel("dB/Hz")
                psd_ax.grid(True, alpha=0.25)
                psd_ax.legend(loc="upper right", fontsize=8)
                psd_ax.set_title("PSD before vs after ICA")
            else:
                psd_ax.text(0.5, 0.5, "PSD unavailable", ha="center", va="center", transform=psd_ax.transAxes)
            if mode == "Stack":
                axes[1].set_xlabel("Time (s)")
            elif mode == "Overlay":
                axes[0].set_xlabel("Time (s)")
            fig.tight_layout()
            canvas.draw_idle()

        channel_combo.bind("<<ComboboxSelected>>", lambda _event: render())
        mode_combo.bind("<<ComboboxSelected>>", lambda _event: render())
        render()

    def get_lfp_filtered_full_data(self) -> np.ndarray:
        data = self.current_data()
        signature = self.current_lfp_filter_signature()
        if (
            self.lfp_filtered_data is not None
            and self.lfp_filtered_signature == signature
            and self.lfp_filtered_data.shape == data.shape
        ):
            return self.lfp_filtered_data
        filtered = self.snr_preprocess_highpass(data)
        if self.notch_var.get():
            filtered = self.apply_notch_filter_matrix(filtered, freq=50.0, q=30.0, harmonics=1, axis=0)
        if self.motion_artifact_enable_var.get():
            cleaned, channels, _label = self.get_motion_artifact_cleaned_selected_data()
            if self.notch_var.get():
                cleaned = self.apply_notch_filter_matrix(cleaned, freq=50.0, q=30.0, harmonics=1, axis=0)
            for local_idx, ch in enumerate(channels):
                if 1 <= ch <= filtered.shape[1]:
                    filtered[:, ch - 1] = cleaned[:, local_idx]
        self.lfp_filtered_data = np.asarray(filtered, dtype=DATA_DTYPE)
        self.lfp_filtered_signature = signature
        return self.lfp_filtered_data

    def get_spike_filtered_full_data(self, params: dict | None = None) -> np.ndarray:
        if params is None:
            params = self.parse_spike_params()
        data = self.current_data()
        signature = self.spike_filter_signature_from_params(params)
        if (
            self.spike_filtered_data is not None
            and self.spike_filtered_signature == signature
            and self.spike_filtered_data.shape == data.shape
        ):
            return self.spike_filtered_data
        if scisig is None:
            raise ImportError("scipy.signal is required for spike filtering.")
        low = float(params["low"])
        high = float(params["high"])
        nyq = self.fs / 2.0
        if not (0 < low < high < nyq):
            raise RuntimeError(f"Invalid spike bandpass: {low:g}-{high:g} Hz, Nyquist={nyq:g} Hz.")
        sos = scisig.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
        self.spike_filtered_data = np.asarray(
            scisig.sosfiltfilt(sos, np.asarray(data, dtype=DATA_DTYPE), axis=0),
            dtype=DATA_DTYPE,
        )
        self.spike_filtered_signature = signature
        return self.spike_filtered_data

    def save_processed_dataset(self):
        if h5py is None:
            messagebox.showerror("Missing dependency", "h5py is required to save processed H5 files.")
            return
        if not self.has_loaded_data():
            messagebox.showerror("No data", "Load BIN/MAT data first.")
            return
        default_name = f"processed_Dog_{self.animal_var.get()}_Block-{self.block_var.get()}_{time.strftime('%Y%m%d_%H%M%S')}.h5"
        path = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("Processed H5", "*.h5"), ("All files", "*.*")],
            initialdir=self.output_dir_var.get().strip() or str(Path.cwd()),
            initialfile=default_name,
            title="Save processed dataset",
        )
        if not path:
            return

        def worker():
            target_path = Path(path)
            tmp_path = target_path.with_name(f".{target_path.name}.tmp")
            try:
                self._queue_ui(lambda: self.file_info_var.set("Saving processed H5: computing LFP/Spike filtered caches..."))
                lfp_filtered = self.get_lfp_filtered_full_data()
                spike_params = self.parse_spike_params()
                spike_filtered = self.get_spike_filtered_full_data(spike_params)
                time_vec = self.time
                if time_vec is None or len(time_vec) != self.current_data().shape[0]:
                    time_vec = np.arange(self.current_data().shape[0]) / self.fs
                if tmp_path.exists():
                    tmp_path.unlink()
                with h5py.File(tmp_path, "w") as f:
                    f.attrs["format"] = "SD_copy_processed_dataset"
                    f.attrs["version"] = "1"
                    f.attrs["data_unit"] = DATA_UNIT
                    f.attrs["compression"] = HDF5_COMPRESSION_LABEL
                    f.attrs["compression_filter"] = "blosc:lz4"
                    f.attrs["compression_level"] = 5
                    f.attrs["compression_shuffle"] = "bitshuffle"
                    f.attrs["fs"] = float(self.fs)
                    f.attrs["bin_delta_t1_sec"] = float(self.bin_delta_t1_sec)
                    f.attrs["bin_delta_t2_sec"] = float(self.bin_delta_t2_sec) if np.isfinite(self.bin_delta_t2_sec) else np.nan
                    for key, var in (
                        ("bin_path", self.bin_path_var),
                        ("mat_path", self.mat_path_var),
                        ("output_dir", self.output_dir_var),
                        ("remap_path", self.remap_path_var),
                        ("animal", self.animal_var),
                        ("block", self.block_var),
                        ("date", self.record_time_var),
                        ("start_sec", self.start_sec_var),
                        ("duration_sec", self.duration_sec_var),
                        ("log_txt_path", self.log_txt_path_var),
                        ("event_csv_path", self.event_csv_path_var),
                    ):
                        f.attrs[key] = var.get()
                    f.attrs["parse_full_length"] = bool(self.parse_full_length_var.get())
                    self.h5_write_json_attr(f, "lfp_settings", self.current_lfp_snr_settings())
                    self.h5_write_json_attr(f, "selected_channels", sorted(self.selected_channels))
                    self.h5_write_json_dataset(f, "last_snr_rows_json", self.last_snr_rows)
                    self.h5_write_json_dataset(f, "spike_results_json", self.spike_results)
                    self.h5_write_json_dataset(f, "timing_info_json", self.timing_info or {})
                    compression = hdf5_compression_kwargs()
                    if self.raw_data is not None:
                        f.create_dataset("raw_data", data=np.asarray(self.raw_data, dtype=DATA_DTYPE), chunks=True, **compression)
                    if self.remapped_data is not None:
                        f.create_dataset("remapped_data", data=np.asarray(self.remapped_data, dtype=DATA_DTYPE), chunks=True, **compression)
                    if self.remapped_channel_ids is not None:
                        f.create_dataset("remapped_channel_ids", data=np.asarray(self.remapped_channel_ids))
                    if self.layout_channel_ids is not None:
                        f.create_dataset("layout_channel_ids", data=np.asarray(self.layout_channel_ids))
                    if self.layout_grid is not None:
                        f.create_dataset("layout_grid", data=np.asarray(self.layout_grid))
                    f.create_dataset("time", data=np.asarray(time_vec, dtype=DATA_DTYPE), chunks=True, **compression)
                    f.create_dataset("lfp_filtered_data", data=np.asarray(lfp_filtered, dtype=DATA_DTYPE), chunks=True, **compression)
                    f["lfp_filtered_data"].attrs["signature"] = self.lfp_filtered_signature
                    f.create_dataset("spike_filtered_data", data=np.asarray(spike_filtered, dtype=DATA_DTYPE), chunks=True, **compression)
                    f["spike_filtered_data"].attrs["signature"] = self.spike_filtered_signature
                    if self.stim_markers is not None:
                        f.create_dataset("stim_markers", data=np.asarray(self.stim_markers, dtype=int))
                    f.flush()
                tmp_path.replace(target_path)
                self._queue_ui(lambda: messagebox.showinfo("Saved", f"Processed dataset saved:\n{target_path}"))
                self._queue_ui(lambda: self.file_info_var.set(f"Saved processed H5: {target_path}"))
            except Exception as exc:
                error_msg = str(exc)
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self.log(traceback.format_exc())
                self._queue_ui(lambda: messagebox.showerror("Save processed H5 failed", error_msg))

        threading.Thread(target=worker, daemon=True).start()

    def load_processed_dataset(self):
        if h5py is None:
            messagebox.showerror("Missing dependency", "h5py is required to load processed H5 files.")
            return
        path = filedialog.askopenfilename(
            filetypes=[("Processed H5", "*.h5"), ("All files", "*.*")],
            initialdir=self.output_dir_var.get().strip() or str(Path.cwd()),
            title="Load processed dataset",
        )
        if not path:
            return
        try:
            with self.open_h5_readonly(path) as f:
                if f.attrs.get("format", "") != "SD_copy_processed_dataset":
                    raise ValueError("This is not a saved SD_copy processed dataset H5 file.")
                self.fs = float(f.attrs.get("fs", FS_DEFAULT))
                data_unit = f.attrs.get("data_unit", "")
                self.raw_data = (
                    self.ensure_data_millivolts(np.asarray(f["raw_data"], dtype=DATA_DTYPE), data_unit, "H5 raw_data")
                    if "raw_data" in f else None
                )
                self.remapped_data = (
                    self.ensure_data_millivolts(np.asarray(f["remapped_data"], dtype=DATA_DTYPE), data_unit, "H5 remapped_data")
                    if "remapped_data" in f else None
                )
                if self.raw_data is None and self.remapped_data is None:
                    raise ValueError("Processed H5 does not contain raw_data or remapped_data.")
                self.remapped_channel_ids = np.asarray(f["remapped_channel_ids"]) if "remapped_channel_ids" in f else None
                self.layout_channel_ids = np.asarray(f["layout_channel_ids"]) if "layout_channel_ids" in f else None
                self.layout_grid = np.asarray(f["layout_grid"]) if "layout_grid" in f else None
                data_for_shape = self.current_data()
                self.time = np.asarray(f["time"]).ravel() if "time" in f else np.arange(data_for_shape.shape[0]) / self.fs
                self.bin_delta_t1_sec = float(f.attrs.get("bin_delta_t1_sec", 0.0))
                self.bin_delta_t2_sec = float(f.attrs.get("bin_delta_t2_sec", np.nan))
                self.lfp_filtered_data = (
                    self.ensure_data_millivolts(np.asarray(f["lfp_filtered_data"], dtype=DATA_DTYPE), data_unit, "H5 lfp_filtered_data")
                    if "lfp_filtered_data" in f else None
                )
                self.lfp_filtered_signature = str(f["lfp_filtered_data"].attrs.get("signature", "")) if "lfp_filtered_data" in f else ""
                self.spike_filtered_data = (
                    self.ensure_data_millivolts(np.asarray(f["spike_filtered_data"], dtype=DATA_DTYPE), data_unit, "H5 spike_filtered_data")
                    if "spike_filtered_data" in f else None
                )
                self.spike_filtered_signature = str(f["spike_filtered_data"].attrs.get("signature", "")) if "spike_filtered_data" in f else ""
                self.motion_artifact_data = (
                    self.ensure_data_millivolts(np.asarray(f["motion_artifact_data"], dtype=DATA_DTYPE), data_unit, "H5 motion_artifact_data")
                    if "motion_artifact_data" in f else None
                )
                self.stim_markers = np.asarray(f["stim_markers"], dtype=int) if "stim_markers" in f else None
                for key, var in (
                    ("bin_path", self.bin_path_var),
                    ("mat_path", self.mat_path_var),
                    ("output_dir", self.output_dir_var),
                    ("remap_path", self.remap_path_var),
                    ("animal", self.animal_var),
                    ("block", self.block_var),
                    ("date", self.record_time_var),
                    ("start_sec", self.start_sec_var),
                    ("duration_sec", self.duration_sec_var),
                    ("log_txt_path", self.log_txt_path_var),
                    ("event_csv_path", self.event_csv_path_var),
                ):
                    if key in f.attrs:
                        var.set(str(f.attrs[key]))
                self.parse_full_length_var.set(parse_bool_like(f.attrs.get("parse_full_length", True), True))
                self.on_parse_full_length_changed()
                settings = self.h5_read_json_attr(f, "lfp_settings", {})
                if isinstance(settings, dict):
                    self.apply_lfp_snr_param_row(settings)
                self.selected_channels = {int(ch) for ch in self.h5_read_json_attr(f, "selected_channels", []) or []}
                self.last_snr_rows = self.h5_read_json_dataset(f, "last_snr_rows_json", []) or []
                self.spike_results = self.h5_read_json_dataset(f, "spike_results_json", []) or []
                timing = self.h5_read_json_dataset(f, "timing_info_json", {}) or {}
                self.timing_info = timing if timing else None
            self.refresh_snr_tree(threshold=parse_float(self.snr_threshold_var.get(), 5))
            self.refresh_spike_tree()
            self.update_selected_summary()
            if self.all_raw_preview_enabled:
                self.start_all_raw_preview()
            self.file_info_var.set(
                f"Loaded processed H5: {path}; data {self.current_data().shape[0]} samples x {self.current_data().shape[1]} channels"
            )
            self.notebook.select(self.data_tab)
            messagebox.showinfo("Loaded", "Processed dataset loaded. Filtered LFP/Spike caches will be reused when parameters match.")
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Load processed H5 failed", self.h5_open_error_message(path, exc))

    def read_mat(self, path: Path):
        meta = {}
        if h5py is not None:
            # Native HDF5 (including MATLAB v7.3 and Blosc/LZ4 H5) should not
            # fall through to scipy.loadmat after a read error.  That would
            # hide a missing filter plugin behind a misleading MAT error.
            if h5py.is_hdf5(str(path)):
                with self.open_h5_readonly(path) as f:
                    def h5_value(key: str, default=None):
                        if key in f:
                            return np.array(f[key])
                        if key in f.attrs:
                            return f.attrs[key]
                        return default

                    fs_value = h5_value("FS", None)
                    if fs_value is None:
                        fs_value = h5_value("fs", FS_DEFAULT)
                    fs = float(np.asarray(fs_value).squeeze())

                    time_value = h5_value("time", None)
                    time = np.asarray(time_value).squeeze() if time_value is not None else None
                    if "dt1" in f:
                        meta["dt1"] = np.array(f["dt1"])
                    elif "dt1" in f.attrs:
                        meta["dt1"] = f.attrs["dt1"]
                    for key in ("deltaT1", "deltaT2"):
                        value = h5_value(key, None)
                        if value is not None:
                            meta[key] = float(np.asarray(value).squeeze())
                    if "deltaT1_ms" in f.attrs and "deltaT1" not in meta:
                        meta["deltaT1"] = float(f.attrs["deltaT1_ms"])
                        meta["deltaT1_unit"] = "ms"
                    delta_unit = h5_value("deltaT1_unit", None)
                    if delta_unit is not None:
                        meta["deltaT1_unit"] = self.mat_text_value(delta_unit)
                    meta["data_unit"] = self.mat_text_value(h5_value("data_unit", ""))
                    dt1_date = h5_value("dt1_date", None)
                    if dt1_date is not None:
                        meta["dt1_date"] = self.mat_text_value(dt1_date)

                    # Files written by the BIN parser use rawData512; the
                    # benchmark writes signal; processed H5 uses raw_data.
                    data = None
                    for key in ("rawData512", "raw_data", "signal", "data", "eeg"):
                        if key not in f or not isinstance(f[key], h5py.Dataset):
                            continue
                        dataset = f[key]
                        if h5py.check_dtype(ref=dataset.dtype) is None:
                            data = np.array(dataset)
                        else:
                            data = self.read_hdf5_signal_cell(dataset, f)
                        break
                    if data is None:
                        raise KeyError(
                            "HDF5 file must contain one of: rawData512, raw_data, signal, data, or eeg."
                        )

                    if "channel_quality" in f:
                        meta["channel_count"] = int(np.asarray(f["channel_quality"]).size)
                    elif np.asarray(data).ndim == 2:
                        for size in (512, 520, 384, 256, 128):
                            if size in np.asarray(data).shape:
                                meta["channel_count"] = size
                                break
                    if time is None:
                        t0 = float(np.asarray(h5_value("t0", 0.0)).squeeze())
                        time = t0 + np.arange(np.asarray(data).shape[0], dtype=np.float64) / fs
                    return data, fs, time, meta
        if sio is None:
            raise ImportError("Install h5py or scipy to load MAT files.")
        try:
            mat = sio.loadmat(path)
        except NotImplementedError as exc:
            raise ImportError(
                "This MAT file is MATLAB -v7.3/HDF5 format. Install h5py with "
                "`python -m pip install h5py`, then restart this GUI."
            ) from exc
        fs = float(np.asarray(mat.get("FS", [[FS_DEFAULT]])).squeeze())
        if "dt1" in mat:
            meta["dt1"] = np.asarray(mat["dt1"])
        for key in ("deltaT1", "deltaT2"):
            if key in mat:
                meta[key] = float(np.asarray(mat[key]).squeeze())
        if "deltaT1_unit" in mat:
            meta["deltaT1_unit"] = self.mat_text_value(mat["deltaT1_unit"])
        if "data_unit" in mat:
            meta["data_unit"] = self.mat_text_value(mat["data_unit"])
        if "dt1_date" in mat:
            meta["dt1_date"] = self.mat_text_value(mat["dt1_date"])
        if "rawData512" in mat:
            return mat["rawData512"], fs, mat.get("time"), meta
        if "signal" in mat:
            return self.read_scipy_signal_cell(mat["signal"]), fs, mat.get("time"), meta
        raise KeyError("MAT file must contain rawData512 or signal.")

    def mat_text_value(self, value) -> str:
        arr = np.asarray(value)
        if arr.dtype.kind == "O":
            vals = arr.ravel()
            if vals.size == 1:
                item = vals[0]
                if isinstance(item, bytes):
                    return item.decode("utf-8", errors="replace").strip()
                return str(item).strip()
            return "".join(
                item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
                for item in vals
            ).strip()
        if arr.dtype.kind in {"U", "S"}:
            return "".join(arr.astype(str).ravel()).strip()
        if arr.dtype.kind in {"u", "i", "f"}:
            vals = arr.ravel()
            chars = []
            for val in vals:
                code = int(val)
                if code > 0:
                    chars.append(chr(code))
            return "".join(chars).strip()
        text = str(value).strip()
        return text

    def read_hdf5_signal_cell(self, signal_dataset, h5_file) -> np.ndarray:
        refs = np.array(signal_dataset).ravel()
        chips = []
        for ref in refs:
            if not ref:
                continue
            arr = np.array(h5_file[ref])
            if arr.ndim != 2:
                continue
            if arr.shape[0] == 128 and arr.shape[1] != 128:
                arr = arr.T
            chips.append(arr)
        if not chips:
            raise KeyError("signal exists, but no chip arrays could be read.")
        min_len = min(chip.shape[0] for chip in chips)
        chips = [chip[:min_len, :] for chip in chips]
        return np.concatenate(chips, axis=1)

    def read_scipy_signal_cell(self, signal_cell) -> np.ndarray:
        cells = np.asarray(signal_cell).ravel()
        chips = []
        for cell in cells:
            arr = np.asarray(cell)
            if arr.ndim != 2 or arr.size == 0:
                continue
            if arr.shape[0] == 128 and arr.shape[1] != 128:
                arr = arr.T
            chips.append(arr)
        if not chips:
            raise KeyError("signal exists, but no chip arrays could be read.")
        min_len = min(chip.shape[0] for chip in chips)
        chips = [chip[:min_len, :] for chip in chips]
        return np.concatenate(chips, axis=1)

    def current_data(self):
        if self.remapped_data is not None:
            return self.remapped_data
        if self.raw_data is not None:
            return self.raw_data
        raise RuntimeError("No data loaded.")

    def has_loaded_data(self) -> bool:
        return self.raw_data is not None or self.remapped_data is not None

    def update_all_raw_preview_button(self):
        if self.all_raw_preview_enabled:
            self.all_raw_preview_button_var.set("Hide all raw")
        else:
            self.all_raw_preview_button_var.set("Show all raw")

    def toggle_all_raw_preview(self):
        if self.all_raw_preview_enabled:
            self.all_raw_preview_enabled = False
            self.all_raw_preview_channels = []
            self.all_raw_preview_index = 0
            self.all_raw_preview_status_var.set("All raw preview: off")
            self.update_all_raw_preview_button()
            if not self.manual_review_channels:
                self.preview_fig.clear()
                ax = self.preview_fig.add_subplot(111)
                ax.axis("off")
                ax.set_title("All raw channel preview is off")
                self.preview_canvas.draw()
            return

        self.all_raw_preview_enabled = True
        self.update_all_raw_preview_button()
        if not self.has_loaded_data():
            self.all_raw_preview_status_var.set("All raw preview: on, load data to view.")
            return
        self.start_all_raw_preview()

    def start_all_raw_preview(self):
        if not self.has_loaded_data():
            return
        data = self.current_data()
        self.all_raw_preview_enabled = True
        self.update_all_raw_preview_button()
        self.all_raw_preview_channels = list(range(1, data.shape[1] + 1))
        self.all_raw_preview_index = 0
        self.render_all_raw_preview()

    def change_all_raw_preview(self, delta: int):
        if not self.has_loaded_data():
            messagebox.showerror("No data", "Load parsed MAT data first.")
            return
        if not self.all_raw_preview_enabled:
            self.all_raw_preview_status_var.set("All raw preview: off")
            return
        if not self.all_raw_preview_channels:
            self.start_all_raw_preview()
            return
        self.all_raw_preview_index = (self.all_raw_preview_index + delta) % len(self.all_raw_preview_channels)
        self.render_all_raw_preview()

    def preview_display_data(self, segment: np.ndarray) -> tuple[np.ndarray, str]:
        y = np.asarray(segment, dtype=DATA_DTYPE)
        if y.ndim == 1:
            y = y[:, None]
        display_mode = self.preview_display_var.get().strip()
        labels: list[str] = []
        if self.preview_notch_var.get():
            y = self.apply_notch_filter_matrix(y, freq=50.0, q=30.0, harmonics=1, axis=0)
            labels.append("50Hz notch")
        if display_mode == "demean":
            y = y - np.nanmedian(y, axis=0, keepdims=True)
            labels.append("demean")
        elif display_mode.startswith("highpass"):
            cutoff = 0.5 if "0.5" in display_mode else 1.0
            if y.shape[0] > 20:
                y = self.apply_highpass_filter(y, cutoff, axis=0)
            labels.append(display_mode)
        elif display_mode == "bandpass":
            low = parse_float(self.preview_band_low_var.get(), 0.5)
            high = parse_float(self.preview_band_high_var.get(), 300.0)
            nyq = self.fs / 2.0
            if not (0 < low < high < nyq):
                raise ValueError(f"Invalid preview bandpass: require 0 < low < high < {nyq:g} Hz.")
            if y.shape[0] > 20:
                y = self.apply_bandpass_filter(y, low, high, axis=0)
            labels.append(f"bandpass {low:g}-{high:g}Hz")
        elif display_mode != "raw":
            labels.append(display_mode)
        return np.asarray(y, dtype=DATA_DTYPE), ", ".join(labels) if labels else "raw"

    def render_all_raw_preview(self):
        if not self.has_loaded_data():
            return
        data = self.current_data()
        source_label = "Remapped/FPC" if self.remapped_data is not None else "Original raw"
        if not self.all_raw_preview_channels:
            self.all_raw_preview_channels = list(range(1, data.shape[1] + 1))
        self.all_raw_preview_index = max(0, min(self.all_raw_preview_index, len(self.all_raw_preview_channels) - 1))
        ch = self.all_raw_preview_channels[self.all_raw_preview_index]
        start = max(0, int(parse_float(self.preview_start_var.get()) * self.fs))
        dur = max(1, int(parse_float(self.preview_duration_var.get(), 5) * self.fs))
        end = min(data.shape[0], start + dur)
        if end <= start:
            start = 0
            end = min(data.shape[0], dur)
        x = np.arange(start, end) / self.fs
        try:
            y_matrix, display_label = self.preview_display_data(data[start:end, ch - 1])
        except Exception as exc:
            messagebox.showerror("Preview display failed", str(exc))
            return
        y = y_matrix[:, 0]

        self.preview_fig.clear()
        ax = self.preview_fig.add_subplot(111)
        linewidth = max(0.1, parse_float(self.preview_linewidth_var.get(), 0.45))
        ax.plot(x, y, linewidth=linewidth, alpha=0.95, antialiased=True, label=f"ch{ch}")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_title(
            f"{source_label} channel preview - ch{ch} "
            f"({self.all_raw_preview_index + 1}/{len(self.all_raw_preview_channels)}) | {display_label}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
        self.preview_fig.tight_layout()
        self.preview_canvas.draw()
        self.all_raw_preview_status_var.set(
            f"All {'remapped' if self.remapped_data is not None else 'raw'} preview: "
            f"ch{ch} ({self.all_raw_preview_index + 1}/{len(self.all_raw_preview_channels)}), {display_label}"
        )

    def open_all_raw_overview_popup(self):
        if not self.has_loaded_data():
            messagebox.showerror("No data", "Load parsed MAT data first.")
            return
        data = self.current_data()
        is_remapped = self.remapped_data is not None
        source_label = "Remapped/FPC channel overview" if is_remapped else "Original raw channel overview"

        grid_rows = 10
        grid_cols = 10
        channels_per_page = grid_rows * grid_cols
        total_channels = int(data.shape[1])
        total_pages = max(1, int(np.ceil(total_channels / channels_per_page)))
        page_index = 0

        popup = tk.Toplevel(self)
        popup.title(("Remapped channels" if is_remapped else "All raw channels") + " - 10x10 overview")
        popup.geometry("1500x950")
        popup.minsize(1000, 700)
        popup.resizable(True, True)
        popup.transient(self)
        try:
            popup.state("zoomed")
        except tk.TclError:
            pass

        nav_row = ttk.Frame(popup)
        nav_row.pack(fill="x", padx=8, pady=(8, 0))
        status_var = tk.StringVar()
        is_fullscreen = tk.BooleanVar(value=False)
        plot_frame = ttk.Frame(popup)
        plot_frame.pack(fill="both", expand=True)
        overview_canvas = None
        overview_toolbar = None

        def set_overview_fullscreen(enabled: bool):
            is_fullscreen.set(bool(enabled))
            popup.attributes("-fullscreen", bool(enabled))
            fullscreen_button.configure(text="Exit fullscreen" if enabled else "Fullscreen")
            if not enabled:
                try:
                    popup.state("zoomed")
                except tk.TclError:
                    pass

        def toggle_overview_fullscreen():
            set_overview_fullscreen(not is_fullscreen.get())

        def render_overview_page():
            nonlocal overview_canvas, overview_toolbar
            start = max(0, int(parse_float(self.preview_start_var.get()) * self.fs))
            dur = max(1, int(parse_float(self.preview_duration_var.get(), 5) * self.fs))
            end = min(data.shape[0], start + dur)
            if end <= start:
                start = 0
                end = min(data.shape[0], dur)
            if end <= start:
                messagebox.showerror("No samples", "No samples are available for the selected start/duration.", parent=popup)
                return

            page_start = page_index * channels_per_page
            page_end = min(total_channels, page_start + channels_per_page)
            sample_count = end - start
            max_preview_points = 1200
            sample_step = max(1, int(np.ceil(sample_count / max_preview_points)))
            sample_indices = np.arange(start, end, sample_step)
            x = sample_indices / self.fs
            try:
                full_segment, display_label = self.preview_display_data(data[start:end, page_start:page_end])
            except Exception as exc:
                messagebox.showerror("Preview display failed", str(exc), parent=popup)
                return
            segment = np.asarray(full_segment[::sample_step, :], dtype=DATA_DTYPE)

            def get_channel_y_limits(y_values):
                finite_y = y_values[np.isfinite(y_values)]
                if not finite_y.size:
                    return None
                y_low = float(np.nanmin(finite_y))
                y_high = float(np.nanmax(finite_y))
                if not np.isfinite(y_low) or not np.isfinite(y_high):
                    return None
                if y_low == y_high:
                    pad = max(1.0, abs(y_low) * 0.05)
                else:
                    pad = (y_high - y_low) * 0.08
                return y_low - pad, y_high + pad

            def format_y_tick(value):
                abs_value = abs(float(value))
                if abs_value and (abs_value < 0.01 or abs_value >= 1000):
                    return f"{value:.1e}"
                return f"{value:.3g}"

            fig = Figure(figsize=(16.5, 9.1), dpi=100)
            axes = fig.subplots(grid_rows, grid_cols, squeeze=False, sharex=False, sharey=False)
            axes_flat = axes.ravel()
            line_width = min(0.7, max(0.1, parse_float(self.preview_linewidth_var.get(), 0.45)))
            for local_i, ax in enumerate(axes_flat):
                channel_index = page_start + local_i
                if channel_index >= page_end:
                    ax.axis("off")
                    continue
                y = segment[:, local_i]
                ax.plot(x, y, linewidth=line_width, color="black", antialiased=True)
                channel_y_limits = get_channel_y_limits(y)
                if channel_y_limits is not None:
                    ax.set_ylim(*channel_y_limits)
                    ax.set_yticks([channel_y_limits[0], channel_y_limits[1]])
                    ax.set_yticklabels([format_y_tick(channel_y_limits[0]), format_y_tick(channel_y_limits[1])])
                ax.set_title(f"ch{channel_index + 1}", fontsize=7.5, pad=2)
                ax.text(
                    0.02,
                    0.86,
                    "mV",
                    transform=ax.transAxes,
                    fontsize=5.5,
                    va="top",
                    ha="left",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.3},
                )
                ax.grid(alpha=0.18, linewidth=0.35)
                col_i = local_i % grid_cols
                ax.tick_params(axis="both", labelsize=4.5, length=2, pad=1)
                ax.tick_params(axis="y", labelleft=True)
                ax.set_xticklabels([])
                if col_i == 0:
                    ax.set_ylabel("mV", fontsize=5.5, labelpad=0)

            start_sec = start / self.fs
            end_sec = end / self.fs
            fig.suptitle(
                f"{source_label} | page {page_index + 1}/{total_pages} | "
                f"channels {page_start + 1}-{page_end} of {total_channels} | "
                f"{start_sec:.3f}-{end_sec:.3f} s | display: {display_label} | Y: auto scale/channel",
                fontsize=12,
            )
            fig.subplots_adjust(left=0.045, right=0.995, bottom=0.055, top=0.92, wspace=0.28, hspace=0.32)

            if overview_toolbar is not None:
                overview_toolbar.destroy()
            if overview_canvas is not None:
                overview_canvas.get_tk_widget().destroy()

            overview_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            overview_toolbar = NavigationToolbar2Tk(overview_canvas, plot_frame, pack_toolbar=False)
            overview_toolbar.update()
            overview_toolbar.pack(fill="x", padx=8, pady=(4, 0))
            overview_canvas.draw()
            overview_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
            status_var.set(
                f"{'Remapped' if is_remapped else 'Raw'} page {page_index + 1}/{total_pages}: "
                f"ch{page_start + 1}-ch{page_end}; "
                f"{display_label}; {sample_count} samples, plotted every {sample_step} sample(s)."
            )

        def change_overview_page(delta: int):
            nonlocal page_index
            page_index = (page_index + delta) % total_pages
            render_overview_page()

        ttk.Button(nav_row, text="< Prev page", command=lambda: change_overview_page(-1)).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(nav_row, text="Next page >", command=lambda: change_overview_page(1)).pack(
            side="left", padx=4
        )
        fullscreen_button = ttk.Button(nav_row, text="Fullscreen", command=toggle_overview_fullscreen)
        fullscreen_button.pack(side="left", padx=(12, 4))
        ttk.Label(nav_row, textvariable=status_var, foreground="#245").pack(side="left", padx=12)
        popup.bind("<Escape>", lambda _event: set_overview_fullscreen(False))
        render_overview_page()

    def plot_preview(self):
        if not self.has_loaded_data():
            return
        data = self.current_data()
        channels = []
        channel_text = self.preview_channels_var.get().strip()
        if channel_text.lower() in {"layout", "fpc"} and self.layout_channel_ids is not None:
            channels = [int(x) for x in self.layout_channel_ids[:8]]
        for part in channel_text.replace(";", ",").split(","):
            part = part.strip()
            if part:
                if part.lower() in {"layout", "fpc"}:
                    continue
                ch = int(part)
                if 1 <= ch <= data.shape[1]:
                    channels.append(ch)
        if not channels:
            channels = [1]
        start = max(0, int(parse_float(self.preview_start_var.get()) * self.fs))
        dur = max(1, int(parse_float(self.preview_duration_var.get(), 5) * self.fs))
        end = min(data.shape[0], start + dur)
        x = np.arange(start, end) / self.fs

        self.preview_fig.clear()
        ax = self.preview_fig.add_subplot(111)
        linewidth = max(0.1, parse_float(self.preview_linewidth_var.get(), 0.45))
        cols = [ch - 1 for ch in channels]
        try:
            preview_matrix, display_label = self.preview_display_data(data[start:end, cols])
        except Exception as exc:
            messagebox.showerror("Preview display failed", str(exc))
            return
        for local_idx, ch in enumerate(channels):
            label = f"ch{ch}" if display_label == "raw" else f"ch{ch} ({display_label})"
            ax.plot(x, preview_matrix[:, local_idx], linewidth=linewidth, alpha=0.95, antialiased=True, label=label)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Voltage (mV)" if display_label == "raw" else "Display voltage (mV)")
        title = "Remapped data" if self.remapped_data is not None else "Raw parsed data"
        title = f"{title} | display: {display_label}"
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
        self.preview_fig.tight_layout()
        self.preview_canvas.draw()

    def on_preview_click_popup(self, event):
        if event.button != 1 or not self.preview_fig.axes:
            return
        toolbar_mode = getattr(getattr(self, "preview_toolbar", None), "mode", "")
        toolbar_mode = getattr(toolbar_mode, "value", toolbar_mode)
        if toolbar_mode:
            return
        self.open_preview_figure_popup()

    def open_preview_figure_popup(self):
        if not self.preview_fig.axes:
            return

        popup = tk.Toplevel(self)
        popup.title("Signal preview - enlarged")
        popup.geometry("1200x820")
        popup.minsize(700, 450)
        popup.resizable(True, True)
        popup.transient(self)

        nav_row = ttk.Frame(popup)
        nav_row.pack(fill="x", padx=8, pady=(8, 0))
        status_var = tk.StringVar()

        plot_frame = ttk.Frame(popup)
        plot_frame.pack(fill="both", expand=True)
        popup_canvas = None
        popup_toolbar = None

        def current_preview_status() -> str:
            if self.manual_review_channels:
                return self.manual_review_status_var.get()
            return self.all_raw_preview_status_var.get()

        def draw_popup_from_preview():
            nonlocal popup_canvas, popup_toolbar
            try:
                popup_fig = pickle.loads(pickle.dumps(self.preview_fig))
            except Exception as exc:
                self.log(traceback.format_exc())
                messagebox.showerror("Open preview failed", str(exc), parent=popup)
                return

            width, height = popup_fig.get_size_inches()
            scale = max(1.0, min(1.7, 11.5 / max(float(width), 1.0)))
            popup_fig.set_size_inches(width * scale, height * scale, forward=True)

            if popup_toolbar is not None:
                popup_toolbar.destroy()
            if popup_canvas is not None:
                popup_canvas.get_tk_widget().destroy()

            popup_canvas = FigureCanvasTkAgg(popup_fig, master=plot_frame)
            popup_toolbar = NavigationToolbar2Tk(popup_canvas, plot_frame, pack_toolbar=False)
            popup_toolbar.update()
            popup_toolbar.pack(fill="x", padx=8, pady=(4, 0))
            popup_canvas.draw()
            popup_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
            status_var.set(current_preview_status())

        def change_popup_channel(delta: int):
            if delta < 0:
                self.preview_previous_channel()
            else:
                self.preview_next_channel()
            if popup.winfo_exists():
                draw_popup_from_preview()

        ttk.Button(nav_row, text="< Prev channel", command=lambda: change_popup_channel(-1)).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(nav_row, text="Next channel >", command=lambda: change_popup_channel(1)).pack(
            side="left", padx=4
        )
        ttk.Label(nav_row, textvariable=status_var, foreground="#245").pack(side="left", padx=12)
        draw_popup_from_preview()

    def on_preview_scroll_zoom(self, event):
        if event.inaxes is None:
            return
        key = (event.key or "").lower()
        if "control" not in key and "ctrl" not in key:
            return
        ax = event.inaxes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_center = event.xdata if event.xdata is not None else (x_min + x_max) / 2.0
        y_center = event.ydata if event.ydata is not None else (y_min + y_max) / 2.0
        scale = 0.8 if event.button == "up" else 1.25

        new_half_x = (x_max - x_min) * scale / 2.0
        new_half_y = (y_max - y_min) * scale / 2.0
        ax.set_xlim(x_center - new_half_x, x_center + new_half_x)
        ax.set_ylim(y_center - new_half_y, y_center + new_half_y)
        self.preview_canvas.draw_idle()

    def apply_remapping(self):
        if self.raw_data is None:
            if self.remapped_data is not None:
                messagebox.showinfo("Already remapped", "Data has already been remapped; original raw_data was released to save memory.")
                return
            messagebox.showerror("No data", "Load parsed MAT data first.")
            return
        if pd is None:
            messagebox.showerror("Missing dependency", "pandas/openpyxl is required for Excel remapping.")
            return
        path = self.remap_path_var.get().strip()
        if not path or not Path(path).exists():
            path = self.prompt_for_remap_file(
                path,
                "Select remapping Excel file",
            )
            if not path:
                return
        try:
            try:
                channel_map_path = self.find_channel_map_file(path)
            except FileNotFoundError:
                path = self.prompt_for_remap_file(
                    path,
                    "Select Excel file containing channel map H2:H513",
                )
                if not path:
                    return
                channel_map_path = self.find_channel_map_file(path)
            source_data = self.raw_data
            input_channels = source_data.shape[1]
            custom_map = self.read_channel_vector(channel_map_path, input_channels)
            output_channels = max(input_channels, int(np.max(custom_map)))
            remapped = np.zeros((source_data.shape[0], output_channels), dtype=source_data.dtype)
            remapped[:, :input_channels] = source_data
            remapped[:, custom_map - 1] = source_data

            layout_ids = self.read_layout_remap(path, output_channels)
            if layout_ids.size == 0:
                raise ValueError("The selected layout file does not contain usable channel IDs in Left!B2:AA21.")

            self.remapped_data = remapped
            self.remapped_channel_ids = np.arange(1, output_channels + 1)
            self.layout_channel_ids = layout_ids
            self.layout_grid = self.read_layout_grid(path, output_channels)
            self.raw_data = None
            del source_data
            gc.collect()
            self.clear_processed_filter_cache()
            self.clear_selected_channels(update_tree=False)
            if self.all_raw_preview_enabled:
                self.all_raw_preview_channels = list(range(1, self.remapped_data.shape[1] + 1))
                self.all_raw_preview_index = 0
            self.log(
                f"Applied data remapping from {channel_map_path} (H2:H{input_channels + 1}); "
                f"output has {output_channels} FPC columns; loaded plot layout from {path} "
                f"({layout_ids.size} valid IDs). raw_data released; remapped_data is now the base data."
            )
            self.plot_preview()
            if self.all_raw_preview_enabled:
                self.render_all_raw_preview()
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Remapping failed", str(exc))

    def find_channel_map_file(self, layout_path: str, n_channels: int | None = None) -> str:
        layout = Path(layout_path)
        channel_count = n_channels
        if channel_count is None:
            channel_count = self.raw_data.shape[1] if self.raw_data is not None else 512
        candidates: list[Path] = []
        if layout.exists() and layout.suffix.lower() in {".xlsx", ".xls"}:
            candidates.append(layout)
        if layout.parent.exists():
            candidates.extend(
                path for path in sorted(layout.parent.glob("*.xls*"))
                if path not in candidates and not path.name.startswith("~$")
            )
        errors: list[str] = []
        for candidate in candidates:
            try:
                self.read_channel_vector(str(candidate), channel_count)
                return str(candidate)
            except Exception as exc:
                errors.append(f"{candidate.name}: {exc}")
        detail = "\n".join(errors[:5])
        raise FileNotFoundError(
            "Could not find an Excel file with a valid data-channel map in H2:H513."
            + (f"\nChecked files:\n{detail}" if detail else "")
        )
    def read_channel_vector(self, path: str, n_channels: int) -> np.ndarray:
        """Read the MATLAB data remapping vector from H2:H513."""
        try:
            table = pd.read_excel(path, header=None, usecols="H", skiprows=1, nrows=n_channels)
        except Exception as exc:
            raise ValueError("H2:H513 could not be read from this Excel file.") from exc
        values = pd.to_numeric(table.iloc[:, 0], errors="coerce").to_numpy(dtype=DATA_DTYPE)
        valid = values[~np.isnan(values)]
        if values.size != n_channels or np.isnan(values).any():
            raise ValueError(f"H2:H{n_channels + 1} has {valid.size}/{n_channels} numeric values.")
        values = values.astype(int)
        self.validate_map(values, n_channels)
        return values

    def read_layout_remap(self, path: str, n_channels: int) -> np.ndarray:
        layout = self.read_layout_grid(path, n_channels)
        vals = layout[np.isfinite(layout)].astype(int)
        return vals

    def read_layout_grid(self, path: str, n_channels: int) -> np.ndarray:
        sheet_name = "Right"
        target_rows = 20
        target_cols = 26
        try:
            full_df = pd.read_excel(path, sheet_name=sheet_name, header=None)
        except Exception:
            full_df = pd.read_excel(path, header=None)
        if full_df.shape[0] <= 1 or full_df.shape[1] <= 1:
            self.log(
                f"Layout sheet in {path} does not contain B2:AA21; "
                "using default 20x26 remapped channel order."
            )
            return self.default_layout_grid(n_channels)

        layout_df = full_df.iloc[1:1 + target_rows, 1:1 + target_cols]
        layout = layout_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=DATA_DTYPE).copy()
        if layout.shape != (target_rows, target_cols):
            padded = np.full((target_rows, target_cols), np.nan, dtype=DATA_DTYPE)
            rows = min(target_rows, layout.shape[0])
            cols = min(target_cols, layout.shape[1])
            if rows > 0 and cols > 0:
                padded[:rows, :cols] = layout[:rows, :cols]
            layout = padded
        layout[(layout < 1) | (layout > n_channels)] = np.nan
        valid_count = int(np.isfinite(layout).sum())
        expected_count = min(n_channels, target_rows * target_cols)
        min_layout_count = max(32, int(round(expected_count * 0.75)))
        if valid_count < min_layout_count:
            self.log(
                f"Layout sheet in {path} has only {valid_count}/{expected_count} numeric channel IDs in B2:AA21; "
                "using default 20x26 remapped channel order."
            )
            return self.default_layout_grid(n_channels)
        return layout

    def default_layout_grid(self, n_channels: int) -> np.ndarray:
        rows, cols = 20, 26
        grid = np.full((rows, cols), np.nan, dtype=DATA_DTYPE)
        values = np.arange(1, min(n_channels, rows * cols) + 1, dtype=DATA_DTYPE)
        grid.flat[:values.size] = values
        return grid

    def validate_map(self, values: np.ndarray, n_channels: int):
        if np.any(values < 1):
            raise ValueError("Remap indices must be positive.")
        if np.unique(values).size != values.size:
            raise ValueError("Remap vector contains duplicate channel indices.")

    def parse_batch_bin_filename(self, path: Path) -> dict:
        match = re.fullmatch(
            r"(?P<date>\d{8})_(?P<animal>\d+)_block_?(?P<block>\d+)",
            path.stem,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(
                "filename must match YYYYMMDD_animal_block_N.bin, "
                f"for example 20260626_1102_block_1.bin: {path.name}"
            )
        date_compact = match.group("date")
        try:
            date_text = datetime.strptime(date_compact, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"invalid date in BIN filename: {path.name}") from exc
        return {
            "path": path,
            "stem": path.stem,
            "date_compact": date_compact,
            "date": date_text,
            "animal": int(match.group("animal")),
            "block": int(match.group("block")),
        }

    def remap_data_for_export(self, data: np.ndarray, layout_path: Path) -> np.ndarray:
        if pd is None:
            raise ImportError("pandas/openpyxl is required for Excel remapping.")
        source_data = normalize_samples_by_channels(data)
        input_channels = source_data.shape[1]
        channel_map_path = self.find_channel_map_file(str(layout_path), n_channels=input_channels)
        custom_map = self.read_channel_vector(channel_map_path, input_channels)
        output_channels = max(input_channels, int(np.max(custom_map)))
        remapped = np.zeros((source_data.shape[0], output_channels), dtype=DATA_DTYPE)
        remapped[:, :input_channels] = source_data
        remapped[:, custom_map - 1] = source_data
        layout_ids = self.read_layout_remap(str(layout_path), output_channels)
        if layout_ids.size == 0:
            raise ValueError("The selected Plot layout does not contain usable channel IDs.")
        return remapped

    def save_channel_mats_for_record(
        self,
        output_dir: Path,
        data: np.ndarray,
        time_vec: np.ndarray,
        fs: float,
        metadata: dict,
    ) -> int:
        if h5py is None:
            raise ImportError("h5py is required to save channel HDF5 files.")
        arr = normalize_samples_by_channels(data)
        time_arr = np.asarray(time_vec, dtype=np.float64).ravel()
        if time_arr.size != arr.shape[0]:
            time_arr = np.arange(arr.shape[0], dtype=np.float64) / float(fs)
        output_dir.mkdir(parents=True, exist_ok=True)
        date_token = re.sub(r"[^0-9]", "", str(metadata["date"]))
        file_prefix = (
            f"{date_token}_{int(metadata['animal'])}_block{int(metadata['block'])}"
        )
        for ch in range(arr.shape[1]):
            channel_id = ch + 1
            out = output_dir / f"{file_prefix}_ch{channel_id:03d}.h5"
            with h5py.File(out, "w") as h5:
                h5.attrs["format"] = "SD_batch_channel_hdf5"
                h5.attrs["compression"] = HDF5_COMPRESSION_LABEL
                h5.attrs["compression_filter"] = "blosc:lz4"
                h5.attrs["compression_level"] = 5
                h5.attrs["compression_shuffle"] = "bitshuffle"
                compression = hdf5_compression_kwargs()
                h5.create_dataset("signal", data=arr[:, ch], chunks=True, **compression)
                h5.create_dataset("time", data=time_arr, chunks=True, **compression)
                h5.create_dataset("FS", data=np.array(float(fs), dtype=np.float64))
                h5.create_dataset("channel", data=np.array(channel_id, dtype=np.int32))
                h5.create_dataset("column_index", data=np.array(channel_id, dtype=np.int32))
                string_dtype = h5py.string_dtype(encoding="utf-8")
                h5.create_dataset("source_bin", data=str(metadata["source_bin"]), dtype=string_dtype)
                h5.create_dataset("date", data=str(metadata["date"]), dtype=string_dtype)
                h5.create_dataset("animal", data=np.array(int(metadata["animal"]), dtype=np.int32))
                h5.create_dataset("block", data=np.array(int(metadata["block"]), dtype=np.int32))
                h5.create_dataset("data_unit", data=DATA_UNIT, dtype=string_dtype)
                h5.create_dataset("data_stage", data="raw", dtype=string_dtype)
                if "dt1" in metadata:
                    h5.create_dataset("dt1", data=np.asarray(metadata["dt1"]))
                if "dt1_date" in metadata:
                    h5.create_dataset("dt1_date", data=str(metadata["dt1_date"]), dtype=string_dtype)
                for key in ("deltaT1", "deltaT2"):
                    if key in metadata:
                        h5.create_dataset(key, data=np.array(float(metadata[key]), dtype=np.float64))
                if "deltaT1_unit" in metadata:
                    h5.create_dataset("deltaT1_unit", data=str(metadata["deltaT1_unit"]), dtype=string_dtype)
        return arr.shape[1]

    def save_total_h5_for_record(
        self,
        output_path: Path,
        data: np.ndarray,
        time_vec: np.ndarray,
        fs: float,
        metadata: dict,
    ) -> Path:
        if h5py is None:
            raise ImportError("h5py is required to save total HDF5 files.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arr = normalize_samples_by_channels(data)
        time_arr = np.asarray(time_vec, dtype=np.float64).ravel()
        if time_arr.size != arr.shape[0]:
            time_arr = np.arange(arr.shape[0], dtype=np.float64) / float(fs)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(output_path, "w") as h5:
            h5.attrs["format"] = "SD_batch_total_hdf5"
            h5.attrs["data_stage"] = "raw"
            h5.attrs["compression"] = HDF5_COMPRESSION_LABEL
            h5.attrs["compression_filter"] = "blosc:lz4"
            h5.attrs["compression_level"] = 5
            h5.attrs["compression_shuffle"] = "bitshuffle"
            compression = hdf5_compression_kwargs()
            h5.create_dataset("rawData512", data=np.asarray(arr, dtype=np.float64), chunks=True, **compression)
            h5.create_dataset("time", data=time_arr, chunks=True, **compression)
            h5.create_dataset("FS", data=np.array(float(fs), dtype=np.float64))
            h5.create_dataset("source_bin", data=str(metadata["source_bin"]), dtype=string_dtype)
            h5.create_dataset("date", data=str(metadata["date"]), dtype=string_dtype)
            h5.create_dataset("animal", data=np.array(int(metadata["animal"]), dtype=np.int32))
            h5.create_dataset("block", data=np.array(int(metadata["block"]), dtype=np.int32))
            h5.create_dataset("data_unit", data=DATA_UNIT, dtype=string_dtype)
            for key in ("dt1", "deltaT1", "deltaT2"):
                if key in metadata:
                    h5.create_dataset(key, data=np.asarray(metadata[key]))
            for key in ("dt1_date", "deltaT1_unit"):
                if key in metadata:
                    h5.create_dataset(key, data=str(metadata[key]), dtype=string_dtype)
        return output_path

    def run_python_parser_for_batch(
        self,
        bin_path: Path,
        output_h5: Path,
        animal: int,
        block: int,
        record_date: str,
    ) -> Path:
        output_h5.parent.mkdir(parents=True, exist_ok=True)
        source_path = Path(bin_path).resolve()
        layout = inspect_bin(source_path)
        reader = BinReader(layout, 0.0, 0.0, chunk_rows=4096)
        timing = read_timing_metadata(source_path)
        metadata = {
            "source_bin": str(source_path),
            "date": record_date,
            "animal": int(animal),
            "block": int(block),
            "start_sec": 0.0,
            "duration_sec": reader.selected_duration_sec,
        }
        metadata.update(timing)
        create_h5(output_h5, reader, metadata, compression_level=5)
        return output_h5

    def export_channel_mats(self):
        """Batch-parse BIN files into complete all-channel H5 files only."""
        if h5py is None:
            messagebox.showerror(
                "Missing dependency",
                "Batch parsing requires h5py.",
            )
            return
        input_dir_text = filedialog.askdirectory(title="Select root folder containing BIN files")
        if not input_dir_text:
            return
        input_dir = Path(input_dir_text)
        output_root = Path(self.output_dir_var.get().strip() or str(input_dir))
        output_root.mkdir(parents=True, exist_ok=True)

        bin_files = sorted(input_dir.rglob("*.bin"))
        records = []
        skipped = []
        for path in bin_files:
            try:
                record = self.parse_batch_bin_filename(path)
                parent_animal = path.parent.name
                parent_date = path.parent.parent.name if path.parent.parent != path.parent else ""
                if parent_date.isdigit() and len(parent_date) == 8 and parent_date != record["date_compact"]:
                    record["warning"] = f"parent date {parent_date} differs from filename date {record['date_compact']}"
                elif parent_animal.isdigit() and int(parent_animal) != record["animal"]:
                    record["warning"] = f"parent animal {parent_animal} differs from filename animal {record['animal']}"
                else:
                    record["warning"] = ""
                records.append(record)
            except Exception as exc:
                skipped.append(f"{path}: {exc}")
        if not records:
            detail = "\n".join(skipped[:10])
            messagebox.showerror("No valid BIN files", f"No matching BIN files were found.\n{detail}")
            return

        def worker():
            successes = []
            failures = list(skipped)
            for index, record in enumerate(records, start=1):
                bin_path = Path(record["path"])
                try:
                    self.after(
                        0,
                        lambda i=index, n=len(records), name=bin_path.name: self.file_info_var.set(
                            f"Batch parse {i}/{n}: processing {name} (full length)..."
                        ),
                    )
                    record_output_dir = output_root / record["date_compact"] / str(record["animal"])
                    record_output_dir.mkdir(parents=True, exist_ok=True)
                    final_h5 = record_output_dir / f"{record['stem']}.h5"
                    existing_h5 = False
                    if final_h5.exists():
                        try:
                            with h5py.File(final_h5, "r") as h5:
                                existing_h5 = "rawData512" in h5 and "FS" in h5
                            if not existing_h5:
                                raise ValueError("missing rawData512 or FS")
                            self.log(f"Reusing existing total HDF5: {final_h5}")
                        except Exception as exc:
                            self.log(f"Existing total HDF5 is invalid; rebuilding it: {final_h5}; {exc}")
                    if not existing_h5:
                        self.run_python_parser_for_batch(
                            bin_path,
                            final_h5,
                            record["animal"],
                            record["block"],
                            record["date"],
                        )
                    warning = f" ({record['warning']})" if record["warning"] else ""
                    successes.append(f"{record['stem']}: {final_h5}{warning}")
                    self.log(f"Batch parse completed: {final_h5}")
                except Exception as exc:
                    failures.append(f"{bin_path}: {exc}")
                    self.log(traceback.format_exc())

            summary = (
                f"Batch parse complete. Success: {len(successes)}; "
                f"failed/skipped: {len(failures)}; complete HDF5 files only."
            )
            details = "\n".join(successes[:10])
            if len(successes) > 10:
                details += f"\n... and {len(successes) - 10} more successful files."
            if failures:
                details += "\n\nFailures/skipped:\n" + "\n".join(failures[:10])
                if len(failures) > 10:
                    details += f"\n... and {len(failures) - 10} more failures/skips."
            self._queue_ui(lambda: self.file_info_var.set(summary))
            self._queue_ui(lambda: messagebox.showinfo("Batch parse", f"{summary}\n\n{details}"))

        threading.Thread(target=worker, daemon=True).start()

    def filter_orders(self) -> tuple[int, int]:
        hp_order = max(1, parse_int(self.filter_highpass_order_var.get(), 3))
        lp_order = max(1, parse_int(self.filter_lowpass_order_var.get(), 5))
        return hp_order, lp_order

    def apply_highpass_filter(self, data: np.ndarray, cutoff: float, axis: int = 0) -> np.ndarray:
        if scisig is None:
            raise ImportError("scipy.signal is required for highpass filtering.")
        nyq = self.fs / 2.0
        if cutoff <= 0 or cutoff >= nyq:
            raise ValueError(f"Invalid highpass cutoff: {cutoff:g} Hz; Nyquist is {nyq:g} Hz.")
        hp_order, _ = self.filter_orders()
        sos = scisig.butter(hp_order, cutoff / nyq, btype="highpass", output="sos")
        return np.asarray(scisig.sosfiltfilt(sos, np.asarray(data, dtype=DATA_DTYPE), axis=axis), dtype=DATA_DTYPE)

    def apply_lowpass_filter(self, data: np.ndarray, cutoff: float, axis: int = 0) -> np.ndarray:
        if scisig is None:
            raise ImportError("scipy.signal is required for lowpass filtering.")
        nyq = self.fs / 2.0
        if cutoff <= 0 or cutoff >= nyq:
            raise ValueError(f"Invalid lowpass cutoff: {cutoff:g} Hz; Nyquist is {nyq:g} Hz.")
        _, lp_order = self.filter_orders()
        sos = scisig.butter(lp_order, cutoff / nyq, btype="lowpass", output="sos")
        return np.asarray(scisig.sosfiltfilt(sos, np.asarray(data, dtype=DATA_DTYPE), axis=axis), dtype=DATA_DTYPE)

    def apply_bandpass_filter(self, data: np.ndarray, low: float, high: float, axis: int = 0) -> np.ndarray:
        nyq = self.fs / 2.0
        if not (0 < low < high < nyq):
            raise ValueError(f"Invalid bandpass: require 0 < low < high < {nyq:g} Hz.")
        filtered = self.apply_lowpass_filter(data, high, axis=axis)
        filtered = self.apply_highpass_filter(filtered, low, axis=axis)
        return filtered

    def apply_notch_filter_matrix(
        self,
        data: np.ndarray,
        freq: float = 50.0,
        q: float = 30.0,
        harmonics: int = 1,
        axis: int = 0,
    ) -> np.ndarray:
        if scisig is None:
            raise ImportError("scipy.signal is required for notch filtering.")
        filtered = np.asarray(data, dtype=DATA_DTYPE)
        for k in range(1, harmonics + 1):
            w0 = k * freq / (self.fs / 2.0)
            if w0 >= 1.0:
                continue
            b, a = scisig.iirnotch(w0, q)
            filtered = np.asarray(scisig.filtfilt(b, a, filtered, axis=axis), dtype=DATA_DTYPE)
        return np.asarray(filtered, dtype=DATA_DTYPE)

    def compute_resting_snr_vectorized(
        self,
        data: np.ndarray,
        signal_band: tuple[float, float],
        noise_band: tuple[float, float],
    ) -> np.ndarray:
        if scisig is None:
            raise ImportError("scipy.signal is required for Welch PSD.")
        arr = np.asarray(data, dtype=DATA_DTYPE)
        if arr.ndim == 1:
            arr = arr[:, None]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        nperseg = min(1024, arr.shape[0])
        if nperseg < 2:
            return np.full(arr.shape[1], np.nan, dtype=DATA_DTYPE)
        freqs, psd = scisig.welch(arr, self.fs, nperseg=nperseg, axis=0)
        freqs = np.asarray(freqs, dtype=DATA_DTYPE)
        psd = np.asarray(psd, dtype=DATA_DTYPE)
        mask_s = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
        mask_n = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
        if not mask_s.any() or not mask_n.any():
            return np.full(arr.shape[1], np.nan, dtype=DATA_DTYPE)
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        ps = trapz(psd[mask_s, :], freqs[mask_s], axis=0)
        pn = trapz(psd[mask_n, :], freqs[mask_n], axis=0) - ps
        snr = np.full(arr.shape[1], np.inf, dtype=DATA_DTYPE)
        valid = pn > 0
        snr[valid] = 10.0 * np.log10(ps[valid] / pn[valid])
        return snr

    def snr_preprocess_highpass(self, data: np.ndarray) -> np.ndarray:
        value = self.snr_highpass_var.get().strip().lower()
        if value in {"", "off", "none", "0"}:
            return np.asarray(data, dtype=DATA_DTYPE)
        if scisig is None:
            raise ImportError("scipy.signal is required for SNR filtering.")
        arr = np.asarray(data, dtype=DATA_DTYPE)
        if value == "bandpass":
            low = parse_float(self.snr_band_low_var.get(), 0.5)
            high = parse_float(self.snr_band_high_var.get(), 300.0)
            return self.apply_bandpass_filter(arr, low, high, axis=0)
        cutoff = float(value)
        return self.apply_highpass_filter(arr, cutoff, axis=0)

    def snr_filter_label(self) -> str:
        value = self.snr_highpass_var.get().strip().lower()
        if value in {"", "off", "none", "0"}:
            return ""
        hp_order, lp_order = self.filter_orders()
        if value == "bandpass":
            return f"bandpass {self.snr_band_low_var.get()}-{self.snr_band_high_var.get()}Hz, HP order {hp_order}, LP order {lp_order}"
        return f"highpass {value}Hz, order {hp_order}"

    def longest_true_run(self, mask: np.ndarray) -> int:
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0 or not mask.any():
            return 0
        padded = np.r_[False, mask, False]
        changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
        return int(np.max(changes[1::2] - changes[::2]))

    def flat_time_ratio_with_spike_tolerance(self, sig: np.ndarray, flat_std_thr: float) -> float:
        finite_mask = np.isfinite(sig)
        if finite_mask.size == 0 or not finite_mask.any():
            return 1.0
        if self.fs <= 0:
            return 0.0
        win = max(16, int(round(0.1 * self.fs)))
        step = max(1, win // 2)
        flat_mask = np.zeros(sig.size, dtype=bool)
        robust_range_thr = max(flat_std_thr * 10.0, flat_std_thr)
        robust_mad_thr = max(flat_std_thr * 2.0, flat_std_thr)
        for start in range(0, sig.size, step):
            end = min(sig.size, start + win)
            window = sig[start:end]
            window = window[np.isfinite(window)]
            if window.size < max(8, win // 10):
                continue
            q10, q90 = np.nanpercentile(window, [10, 90])
            med = np.nanmedian(window)
            robust_range = float(q90 - q10)
            robust_mad = float(np.nanmedian(np.abs(window - med)) / 0.6745)
            if robust_range <= robust_range_thr or robust_mad <= robust_mad_thr:
                flat_mask[start:end] = True
            if end >= sig.size:
                break
        return float(np.nanmean(flat_mask[finite_mask]))

    def highpass_for_artifact_check(self, sig: np.ndarray, cutoff: float = 1.0) -> np.ndarray | None:
        if scisig is None or self.fs <= 0:
            return None
        finite = np.asarray(sig, dtype=DATA_DTYPE)
        finite_mask = np.isfinite(finite)
        if finite_mask.sum() < max(32, int(round(self.fs))):
            return None
        fill = float(np.nanmedian(finite[finite_mask]))
        work = np.where(finite_mask, finite, fill)
        nyq = self.fs / 2.0
        if cutoff <= 0 or cutoff >= nyq:
            return None
        try:
            return self.apply_highpass_filter(work, cutoff, axis=0)
        except Exception:
            return None

    def high_frequency_artifact_reasons(self, sig: np.ndarray) -> list[str]:
        hp = self.highpass_for_artifact_check(sig, cutoff=1.0)
        if hp is None or hp.size < max(32, int(round(self.fs))):
            return []
        try:
            hp = apply_notch_filter(hp, self.fs, freq=50.0, q=30, harmonics=1)
        except Exception:
            pass
        centered = hp - float(np.nanmedian(hp))
        hf = centered
        if scisig is not None and self.fs > 0:
            try:
                low = min(80.0, self.fs / 2.0 * 0.25)
                high = min(300.0, self.fs / 2.0 * 0.9)
                if 0 < low < high < self.fs / 2.0:
                    hf = self.apply_bandpass_filter(centered, low, high, axis=0)
            except Exception:
                hf = centered
        hf_abs = np.abs(hf - float(np.nanmedian(hf)))
        hf_mad = float(np.nanmedian(hf_abs) / 0.6745)
        if not np.isfinite(hf_mad) or hf_mad <= 0:
            return []
        z = hf_abs / hf_mad
        z_mask = z > 8.0
        z_ratio = float(np.nanmean(z_mask))
        longest_ms = 1000.0 * self.longest_true_run(z_mask) / self.fs
        if z_ratio >= 0.05 or longest_ms >= 100.0:
            return [
                f"sustained high-frequency artifact, z>8 ratio={z_ratio * 100:.2f}%, longest={longest_ms:.0f}ms"
            ]
        return []

    def topology_neighbors_from_layout(self, n_channels: int) -> dict[int, list[int]]:
        if self.layout_grid is None:
            return {}
        grid = np.asarray(self.layout_grid, dtype=DATA_DTYPE)
        positions: dict[int, tuple[int, int]] = {}
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                val = grid[row, col]
                if np.isfinite(val):
                    ch = int(val)
                    if 1 <= ch <= n_channels:
                        positions[ch] = (row, col)
        neighbors: dict[int, list[int]] = {}
        for ch, (row, col) in positions.items():
            found: list[int] = []
            for radius in (1, 2):
                found = []
                for other, (orow, ocol) in positions.items():
                    if other == ch:
                        continue
                    if max(abs(orow - row), abs(ocol - col)) <= radius:
                        found.append(other)
                if len(found) >= 3:
                    break
            if found:
                neighbors[ch] = sorted(found)
        return neighbors

    def preprocess_for_topology_check(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=DATA_DTYPE)
        fill = np.nanmedian(arr, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        arr = np.where(np.isfinite(arr), arr, fill)
        arr = arr - np.nanmedian(arr, axis=0, keepdims=True)
        if scisig is not None and self.fs > 0 and arr.shape[0] > 32:
            nyq = self.fs / 2.0
            try:
                if 1.0 < nyq:
                    arr = self.apply_highpass_filter(arr, 1.0, axis=0)
                w0 = 50.0 / nyq
                if 0 < w0 < 1:
                    b, a = scisig.iirnotch(w0, 30)
                    arr = np.asarray(scisig.filtfilt(b, a, arr, axis=0), dtype=DATA_DTYPE)
            except Exception:
                pass
        max_samples = 120000
        if arr.shape[0] > max_samples:
            step = int(np.ceil(arr.shape[0] / max_samples))
            arr = arr[::step]
        return arr

    def detect_topology_candidate_channels(self, data: np.ndarray) -> dict[int, str]:
        neighbors = self.topology_neighbors_from_layout(data.shape[1])
        if not neighbors:
            return {}
        arr = self.preprocess_for_topology_check(data)
        if arr.shape[0] < 32:
            return {}
        win = max(32, int(round(min(self.fs, arr.shape[0]))))
        win = min(win, arr.shape[0])
        starts = list(range(0, arr.shape[0] - win + 1, win))
        if not starts:
            starts = [0]
        candidate_reasons: dict[int, str] = {}
        for ch, neigh in neighbors.items():
            col = ch - 1
            neigh_cols = [n - 1 for n in neigh if 1 <= n <= data.shape[1]]
            if len(neigh_cols) < 3:
                continue
            corrs: list[float] = []
            for start in starts:
                end = min(arr.shape[0], start + win)
                target = arr[start:end, col]
                pred = np.nanmean(arr[start:end, neigh_cols], axis=1)
                target_std = float(np.nanstd(target))
                pred_std = float(np.nanstd(pred))
                if target_std <= 1e-12 or pred_std <= 1e-12:
                    continue
                corr = float(np.corrcoef(target, pred)[0, 1])
                if np.isfinite(corr):
                    corrs.append(corr)
            if len(corrs) < 3:
                continue
            corr_arr = np.asarray(corrs)
            low_ratio = float(np.mean(corr_arr < 0.75))
            med_corr = float(np.median(corr_arr))
            if low_ratio >= 0.6 and med_corr < 0.75:
                candidate_reasons[ch] = (
                    f"Bad Channel candidate: topology outlier, "
                    f"neighbor corr median={med_corr:.2f}, <0.75 ratio={low_ratio * 100:.0f}%"
                )
        return candidate_reasons

    def compute_2s_window_snr_values(self, sig: np.ndarray) -> list[dict]:
        if self.fs <= 0:
            return []
        win_samples = int(round(2.0 * self.fs))
        if win_samples <= 0:
            return []
        n_windows = sig.size // win_samples
        if n_windows <= 0:
            return []
        signal_band = (
            parse_float(self.signal_band_low_var.get(), 30.0),
            parse_float(self.signal_band_high_var.get(), 80.0),
        )
        noise_band = (
            parse_float(self.noise_band_low_var.get(), 1.0),
            parse_float(self.noise_band_high_var.get(), 200.0),
        )
        valid_snr_threshold = parse_float(self.bad_window_snr_threshold_var.get(), 1.0)
        flat_ptp_threshold = parse_float(self.bad_window_ptp_threshold_var.get(), 0.01)
        rows: list[dict] = []
        for win_idx in range(n_windows):
            start = win_idx * win_samples
            end = start + win_samples
            segment = np.asarray(sig[start:end], dtype=DATA_DTYPE)
            finite = np.isfinite(segment)
            if finite.sum() < max(32, int(0.5 * win_samples)):
                continue
            fill = float(np.nanmedian(segment[finite]))
            segment = np.where(finite, segment, fill)
            ptp_value = float(np.nanmax(segment) - np.nanmin(segment))
            diff_abs = np.abs(np.diff(segment))
            dead_straight_ratio = float(np.nanmean(diff_abs <= 1e-6)) if diff_abs.size else 1.0
            segment = segment - float(np.nanmedian(segment))
            if self.notch_var.get():
                try:
                    segment = apply_notch_filter(segment, self.fs, freq=50.0, q=30, harmonics=1)
                except Exception:
                    pass
            try:
                snr_db = compute_resting_snr(segment, self.fs, signal_band, noise_band)
            except Exception:
                continue
            if not np.isfinite(snr_db):
                continue
            if snr_db > valid_snr_threshold:
                status_code = 1
                status_label = "normal"
            elif dead_straight_ratio > 0.20:
                status_code = 0
                status_label = "flat/no-signal"
            elif ptp_value <= flat_ptp_threshold:
                status_code = 0
                status_label = "flat/no-signal"
            else:
                status_code = 2
                status_label = "motion artifact"
            rows.append({
                "window": win_idx + 1,
                "start_sec": start / self.fs,
                "end_sec": end / self.fs,
                "snr_db": float(snr_db),
                "ptp": ptp_value,
                "dead_straight_ratio": dead_straight_ratio,
                "status": status_code,
                "status_label": status_label,
                "valid": bool(status_code != 0),
            })
        return rows

    def snr_valid_window_ratio(self, sig: np.ndarray) -> tuple[float | None, int, int]:
        rows = self.compute_2s_window_snr_values(sig)
        if not rows:
            n_windows = sig.size // max(1, int(round(2.0 * self.fs))) if self.fs > 0 else 0
            return None, 0, n_windows
        used_windows = len(rows)
        valid_windows = int(sum(1 for row in rows if row["status"] != 0))
        if used_windows <= 0:
            return None, 0, 0
        return valid_windows / used_windows, valid_windows, used_windows

    def set_snr_progress(self, value: float, message: str | None = None):
        self.snr_progress_var.set(max(0.0, min(100.0, float(value))))
        if message:
            self.snr_status_var.set(message)
        try:
            self.update()
        except tk.TclError:
            pass

    def bad_check_max_workers_from_settings(self, settings: dict, n_channels: int) -> int:
        if n_channels <= 1 or not parse_bool_like(settings.get("bad_check_parallel"), True):
            return 1
        requested = parse_int(settings.get("bad_check_workers", 0), 0)
        cpu_count = os.cpu_count() or 4
        if requested <= 0:
            return max(1, min(n_channels, cpu_count, 8))
        return max(1, min(n_channels, requested))

    def detect_2s_window_bad_channels_only(
        self,
        data: np.ndarray,
        settings: dict,
        progress_callback=None,
        progress_start: float = 0.0,
        progress_end: float = 45.0,
        label_prefix: str = "LFP SNR",
    ) -> tuple[list[int], dict[int, str]]:
        data = np.asarray(data, dtype=DATA_DTYPE)
        good_indices: list[int] = []
        bad_reasons: dict[int, str] = {}
        n_channels = max(1, data.shape[1])
        valid_snr_threshold = parse_float(settings.get("valid_win_db", 1.0), 1.0)

        def check_channel(idx: int) -> tuple[int, int, str | None]:
            ch = idx + 1
            sig = np.asarray(data[:, idx], dtype=DATA_DTYPE)
            valid_ratio, valid_windows, used_windows = self.snr_valid_window_ratio_with_settings(sig, settings)
            if valid_ratio is not None and valid_ratio < 0.5:
                return idx, ch, (
                    f"Bad Channel: 2s window SNR valid ratio={valid_ratio * 100:.1f}% "
                    f"({valid_windows}/{used_windows} windows status 1/2; "
                    f"normal if >{valid_snr_threshold:g}dB, artifact counted)"
                )
            return idx, ch, None

        max_workers = self.bad_check_max_workers_from_settings(settings, data.shape[1])
        if max_workers <= 1:
            for idx in range(data.shape[1]):
                if progress_callback:
                    value = progress_start + (progress_end - progress_start) * idx / n_channels
                    progress_callback(value, f"{label_prefix}: checking 2s windows {idx + 1}/{data.shape[1]}")
                result_idx, ch, bad_detail = check_channel(idx)
                if bad_detail:
                    bad_reasons[ch] = bad_detail
                else:
                    good_indices.append(result_idx)
        else:
            done_count = 0
            if progress_callback:
                progress_callback(
                    progress_start,
                    f"{label_prefix}: parallel 2s window bad check using {max_workers} workers",
                )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                pending = {executor.submit(check_channel, idx) for idx in range(data.shape[1])}
                last_heartbeat = time.perf_counter()
                while pending:
                    done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                    if not done:
                        if progress_callback and time.perf_counter() - last_heartbeat >= 0.5:
                            value = progress_start + (progress_end - progress_start) * done_count / n_channels
                            progress_callback(value, f"{label_prefix}: parallel 2s windows running {done_count}/{data.shape[1]}")
                            last_heartbeat = time.perf_counter()
                        continue
                    for future in done:
                        result_idx, ch, bad_detail = future.result()
                        done_count += 1
                        if bad_detail:
                            bad_reasons[ch] = bad_detail
                        else:
                            good_indices.append(result_idx)
                    if progress_callback:
                        value = progress_start + (progress_end - progress_start) * done_count / n_channels
                        progress_callback(value, f"{label_prefix}: parallel 2s windows {done_count}/{data.shape[1]}")
        good_indices.sort()
        if progress_callback:
            progress_callback(progress_end, f"{label_prefix}: 2s window bad check done ({data.shape[1]} channels)")
        return good_indices, bad_reasons

    def detect_bad_snr_channels(
        self,
        data: np.ndarray,
        progress_callback=None,
        progress_start: float = 0.0,
        progress_end: float = 45.0,
    ) -> tuple[list[int], dict[int, str], dict[int, str]]:
        if not self.bad_channel_check_var.get():
            return list(range(data.shape[1])), {}, {}
        flat_std_thr = parse_float(self.bad_flat_std_var.get(), 1e-4)
        flat_ratio_thr = parse_float(self.bad_flat_ratio_var.get(), 30.0) / 100.0
        good_indices: list[int] = []
        bad_reasons: dict[int, str] = {}
        candidate_reasons: dict[int, str] = {}
        topology_reasons: dict[int, str] = {}
        n_channels = max(1, data.shape[1])
        for idx in range(data.shape[1]):
            sig = np.asarray(data[:, idx], dtype=DATA_DTYPE)
            finite = sig[np.isfinite(sig)]
            ch = idx + 1
            if progress_callback:
                progress = progress_start + (progress_end - progress_start) * idx / n_channels
                progress_callback(progress, f"LFP SNR: checking bad channels {idx + 1}/{data.shape[1]}")
            if finite.size < 10:
                bad_reasons[ch] = "Bad Channel: too few finite samples"
                continue
            sig_std = float(np.nanstd(finite))
            sig_range = float(np.nanmax(finite) - np.nanmin(finite))
            if sig_std <= flat_std_thr or sig_range <= flat_std_thr * 6:
                bad_reasons[ch] = f"Bad Channel: flat/dead, std={sig_std:.3g}"
                continue
            quant_step = max(flat_std_thr, 1e-4)
            unique_count = int(np.unique(np.round(finite / quant_step)).size)
            unique_ratio = unique_count / finite.size
            if unique_count <= 8 or unique_ratio <= 5e-4:
                bad_reasons[ch] = f"Bad Channel: too few unique levels, unique={unique_count}"
                continue
            flat_ratio = self.flat_time_ratio_with_spike_tolerance(sig, flat_std_thr)
            if flat_ratio_thr > 0 and flat_ratio >= flat_ratio_thr:
                bad_reasons[ch] = f"Bad Channel: flat time={flat_ratio * 100:.1f}%"
                continue
            if self.bad_window_check_var.get():
                valid_ratio, valid_windows, used_windows = self.snr_valid_window_ratio(sig)
                if valid_ratio is not None and valid_ratio < 0.5:
                    valid_snr_threshold = parse_float(self.bad_window_snr_threshold_var.get(), 1.0)
                    bad_reasons[ch] = (
                        f"Bad Channel: 2s window SNR valid ratio={valid_ratio * 100:.1f}% "
                        f"({valid_windows}/{used_windows} windows status 1/2; "
                        f"normal if >{valid_snr_threshold:g}dB, artifact counted)"
                    )
                    continue
            artifact_reasons = self.high_frequency_artifact_reasons(sig)
            if artifact_reasons:
                candidate_reasons[ch] = "Bad Channel candidate: " + "; ".join(artifact_reasons)
            if ch in topology_reasons:
                if ch in candidate_reasons:
                    candidate_reasons[ch] = f"{candidate_reasons[ch]}; {topology_reasons[ch].replace('Bad Channel candidate: ', '')}"
                else:
                    candidate_reasons[ch] = topology_reasons[ch]
            good_indices.append(idx)
        if progress_callback:
            progress_callback(progress_end, f"LFP SNR: bad-channel check done ({data.shape[1]} channels)")
        return good_indices, bad_reasons, candidate_reasons

    def compute_lfp_filter_only_result(
        self,
        progress=None,
        label_prefix: str = "LFP SNR",
        raw_data: np.ndarray | None = None,
        channel_labels: list[int] | None = None,
    ) -> dict:
        raw_data = np.asarray(self.current_data() if raw_data is None else raw_data, dtype=DATA_DTYPE)
        if channel_labels is None:
            channel_labels = list(range(1, raw_data.shape[1] + 1))
        else:
            channel_labels = [int(ch) for ch in channel_labels]
        settings = self.current_lfp_snr_settings()
        mode = f"{self.snr_mode_var.get()}_filter_only"
        if progress:
            progress(5.0, f"{label_prefix}: skip SNR scoring enabled")
        if self.motion_artifact_enable_var.get():
            if progress:
                progress(8.0, f"{label_prefix}: applying selected-channel Motion ICA")
            cleaned, channels, _label = self.get_motion_artifact_cleaned_selected_data()
            raw_data = raw_data.copy()
            label_to_local = {int(ch): idx for idx, ch in enumerate(channel_labels)}
            for local_idx, ch in enumerate(channels):
                target_idx = label_to_local.get(int(ch))
                if target_idx is not None and 0 <= target_idx < raw_data.shape[1]:
                    raw_data[:, target_idx] = cleaned[:, local_idx]

        bad_reasons: dict[int, str] = {}
        candidate_reasons: dict[int, str] = {}
        if self.bad_channel_check_var.get():
            good_indices, bad_reasons, candidate_reasons = self.detect_bad_snr_channels_with_settings(
                raw_data,
                settings,
                progress_callback=progress,
                progress_start=10.0,
                progress_end=85.0,
                label_prefix=label_prefix,
            )
        elif self.bad_window_check_var.get():
            good_indices, bad_reasons = self.detect_2s_window_bad_channels_only(
                raw_data,
                settings,
                progress_callback=progress,
                progress_start=10.0,
                progress_end=85.0,
                label_prefix=label_prefix,
            )
        else:
            good_indices = list(range(raw_data.shape[1]))
            if progress:
                progress(85.0, f"{label_prefix}: bad-channel checks disabled")

        filter_label = self.snr_filter_label_from_settings(settings)
        detail_parts = ["SNR skipped by user"]
        if filter_label:
            detail_parts.append(filter_label)
        if self.notch_var.get():
            detail_parts.append("50Hz notch")
        base_detail = "; ".join(detail_parts)

        rows = []
        for idx in range(raw_data.shape[1]):
            ch = int(channel_labels[idx]) if idx < len(channel_labels) else idx + 1
            detail = base_detail
            local_ch = idx + 1
            bad_detail = bad_reasons.get(local_ch)
            if bad_detail:
                detail = f"{bad_detail}; {detail}"
            candidate_detail = candidate_reasons.get(local_ch)
            if candidate_detail:
                detail = f"{candidate_detail}; {detail}"
            rows.append({
                "channel": ch,
                "snr_db": np.nan,
                "mode": mode,
                "detail": detail,
                "snr_skipped": True,
            })
        if progress:
            progress(95.0, f"{label_prefix}: updating filter-only result table")

        finite_channels: set[int] = set()
        healthy_channels = {int(row["channel"]) for row in rows if self.is_snr_row_healthy(row)}
        total_channels = raw_data.shape[1]
        bad_count = len(bad_reasons)
        candidate_count = len(candidate_reasons)
        bad_like_count = bad_count + candidate_count
        bad_pct = 100.0 * bad_like_count / total_channels if total_channels else 0.0
        return {
            "rows": rows,
            "mode": mode,
            "valid_channels": healthy_channels,
            "finite_channels": finite_channels,
            "healthy_channels": healthy_channels,
            "total_channels": total_channels,
            "valid_count": len(healthy_channels),
            "finite_count": len(finite_channels),
            "healthy_count": len(healthy_channels),
            "bad_count": bad_count,
            "candidate_count": candidate_count,
            "bad_like_count": bad_like_count,
            "bad_pct": bad_pct,
            "snr_skipped": True,
        }

    def compute_lfp_snr_current_settings(self, progress=None, label_prefix: str = "LFP SNR") -> dict:
        if self.skip_lfp_snr_var.get():
            return self.compute_lfp_filter_only_result(progress=progress, label_prefix=label_prefix)
        raw_for_checks = self.current_data()
        preprocessed = None
        try:
            preprocessed = self.get_lfp_filtered_full_data()
            if self.motion_artifact_enable_var.get() and preprocessed is not None:
                raw_for_checks = np.asarray(raw_for_checks, dtype=DATA_DTYPE).copy()
                for ch in self.get_selected_channels():
                    if 1 <= ch <= raw_for_checks.shape[1]:
                        raw_for_checks[:, ch - 1] = preprocessed[:, ch - 1]
        except Exception:
            preprocessed = None
        return self.compute_lfp_snr_with_settings(
            raw_for_checks,
            self.current_lfp_snr_settings(),
            progress=progress,
            label_prefix=label_prefix,
            preprocessed_data=preprocessed,
        )

    def remap_lfp_snr_result_channels(self, result: dict, channel_labels: list[int]) -> dict:
        labels = [int(ch) for ch in channel_labels]

        def remap_channel(value) -> int:
            local = int(value)
            if 1 <= local <= len(labels):
                return labels[local - 1]
            return local

        rows = []
        for row in result.get("rows", []):
            mapped = dict(row)
            mapped["channel"] = remap_channel(mapped.get("channel", 0))
            rows.append(mapped)
        result = dict(result)
        result["rows"] = rows
        for key in ("valid_channels", "finite_channels", "healthy_channels"):
            result[key] = {remap_channel(ch) for ch in result.get(key, set())}
        return result

    def merge_lfp_snr_rows(self, existing_rows: list[dict], updated_rows: list[dict]) -> list[dict]:
        merged: dict[int, dict] = {}
        for row in existing_rows:
            try:
                merged[int(row["channel"])] = dict(row)
            except Exception:
                continue
        for row in updated_rows:
            try:
                merged[int(row["channel"])] = dict(row)
            except Exception:
                continue
        return [merged[ch] for ch in sorted(merged)]

    def compute_lfp_snr_selected_channels(
        self,
        channels: list[int],
        progress=None,
        label_prefix: str = "LFP SNR selected",
        respect_skip: bool = False,
    ) -> dict:
        base_data = self.current_data()
        channels = [int(ch) for ch in channels if 1 <= int(ch) <= base_data.shape[1]]
        if not channels:
            raise RuntimeError("No selected channels are valid for the loaded data.")
        cols = [ch - 1 for ch in channels]
        raw_subset = np.asarray(base_data[:, cols], dtype=DATA_DTYPE)
        if respect_skip and self.skip_lfp_snr_var.get():
            return self.compute_lfp_filter_only_result(
                progress=progress,
                label_prefix=label_prefix,
                raw_data=raw_subset,
                channel_labels=channels,
            )

        settings = self.current_lfp_snr_settings()
        preprocessed_subset = None
        raw_for_checks = raw_subset
        try:
            signature = self.current_lfp_filter_signature()
            if (
                self.lfp_filtered_data is not None
                and self.lfp_filtered_signature == signature
                and self.lfp_filtered_data.shape == base_data.shape
            ):
                preprocessed_subset = np.asarray(self.lfp_filtered_data[:, cols], dtype=DATA_DTYPE)
            elif self.motion_artifact_enable_var.get():
                full_preprocessed = self.get_lfp_filtered_full_data()
                preprocessed_subset = np.asarray(full_preprocessed[:, cols], dtype=DATA_DTYPE)
                raw_for_checks = raw_subset.copy()
                raw_for_checks[:, :] = preprocessed_subset
        except Exception:
            preprocessed_subset = None
            raw_for_checks = raw_subset

        result = self.compute_lfp_snr_with_settings(
            raw_for_checks,
            settings,
            progress=progress,
            label_prefix=label_prefix,
            preprocessed_data=preprocessed_subset,
        )
        return self.remap_lfp_snr_result_channels(result, channels)

    def run_lfp_snr(self, on_done=None):
        start_time = time.perf_counter()

        def progress(value: float, phase: str):
            elapsed = time.perf_counter() - start_time
            if value > 1:
                eta = elapsed * (100.0 - value) / value
                message = f"{phase} | {value:.0f}% | elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
            else:
                message = f"{phase} | elapsed {elapsed:.1f}s"
            self.set_snr_progress(value, message)

        try:
            self.set_snr_progress(0, "LFP SNR: starting...")
            selected_for_update = self.get_selected_channels()
            partial_update = bool(self.last_snr_rows and selected_for_update)
            if partial_update:
                result = self.compute_lfp_snr_selected_channels(
                    selected_for_update,
                    progress=progress,
                    label_prefix=f"LFP SNR selected {len(selected_for_update)}ch",
                )
                self.last_snr_rows = self.merge_lfp_snr_rows(self.last_snr_rows, result["rows"])
            else:
                result = self.compute_lfp_snr_current_settings(progress=progress, label_prefix="LFP SNR")
                self.last_snr_rows = result["rows"]
            threshold = parse_float(self.snr_threshold_var.get(), 5)
            if partial_update:
                self.selected_channels.intersection_update(set(selected_for_update))
            else:
                self.selected_channels.intersection_update(result["valid_channels"])
                if result.get("snr_skipped") and not self.selected_channels:
                    self.selected_channels = set(result["valid_channels"])
            self.refresh_snr_tree(threshold=threshold)
            self.update_selected_summary()
            if partial_update:
                if result.get("snr_skipped"):
                    self.snr_status_var.set(
                        f"SNR status: updated selected {len(selected_for_update)} channels with skip scoring | "
                        f"Bad/Candidate {result['bad_like_count']} ({result['bad_pct']:.1f}%) | other rows preserved"
                    )
                    self.log(
                        f"Updated selected {len(selected_for_update)} LFP rows with skip scoring; "
                        "unselected SNR rows were preserved."
                    )
                else:
                    self.snr_status_var.set(
                        f"SNR status: recomputed selected {len(selected_for_update)} channels | "
                        f"healthy {result['healthy_count']} | finite {result['finite_count']} | "
                        f"Bad/Candidate {result['bad_like_count']} ({result['bad_pct']:.1f}%) | other rows preserved"
                    )
                    self.log(
                        f"Recomputed LFP {result['mode']} SNR for selected channels "
                        f"{selected_for_update[:20]}{'...' if len(selected_for_update) > 20 else ''}; "
                        "unselected SNR rows were preserved."
                    )
            elif result.get("snr_skipped"):
                self.snr_status_var.set(
                    f"SNR status: skipped scoring | total {result['total_channels']} channels | "
                    f"usable after bad check {result['healthy_count']} | "
                    f"Bad/Candidate {result['bad_like_count']} ({result['bad_pct']:.1f}%)"
                )
                self.log(
                    f"Skipped LFP SNR scoring; selected {len(self.selected_channels)} usable channels; "
                    f"marked {result['bad_count']} hard bad and {result['candidate_count']} candidates "
                    f"out of {result['total_channels']} channels ({result['bad_pct']:.1f}%)."
                )
            else:
                self.snr_status_var.set(
                    f"SNR status: total {result['total_channels']} channels | healthy SNR {result['healthy_count']} "
                    f"(finite all {result['finite_count']}) | "
                    f"Bad/Candidate {result['bad_like_count']} ({result['bad_pct']:.1f}%)"
                )
                self.log(
                    f"Computed LFP {result['mode']} SNR for all {result['finite_count']} finite channels; "
                    f"summary/averages use {result['healthy_count']} healthy channels after masking; "
                    f"marked {result['bad_count']} hard bad and {result['candidate_count']} candidates "
                    f"out of {result['total_channels']} channels ({result['bad_pct']:.1f}%)."
                )
            self.set_snr_progress(100.0, self.snr_status_var.get())
            self.notebook.select(self.snr_tab)
            if callable(on_done):
                on_done()
        except Exception as exc:
            self.set_snr_progress(0.0, "SNR status: failed")
            self.log(traceback.format_exc())
            messagebox.showerror("LFP SNR failed", str(exc))

    def compute_lfp_dynamic_snr_rows(self, data: np.ndarray, channels: list[int], window_sec: float = 10.0) -> list[dict]:
        win_samples = int(round(window_sec * self.fs))
        if win_samples <= 0:
            raise ValueError("Invalid dynamic SNR window length.")
        n_windows = data.shape[0] // win_samples
        if n_windows <= 0:
            raise ValueError(f"Loaded data is shorter than {window_sec:g} seconds.")
        mode = self.snr_mode_var.get()
        full_stim_mask = None
        stim_interval = parse_float(self.stim_interval_var.get()) if mode != "evoked" else None
        if mode == "evoked":
            full_stim_mask = self.build_stim_mask_for_loaded_data(parse_float(self.stim_duration_var.get()))
        filtered_data = self.snr_preprocess_highpass(data)
        rows: list[dict] = []
        for win_idx in range(n_windows):
            start = win_idx * win_samples
            end = start + win_samples
            seg = filtered_data[start:end, :]
            seg_mask = full_stim_mask[start:end] if full_stim_mask is not None else None
            try:
                snr_rows = compute_batch_channel_snr(
                    seg,
                    self.fs,
                    mode=mode,
                    signal_band=(parse_float(self.signal_band_low_var.get()), parse_float(self.signal_band_high_var.get())),
                    noise_band=(parse_float(self.noise_band_low_var.get()), parse_float(self.noise_band_high_var.get())),
                    stim_interval=stim_interval,
                    stim_duration=parse_float(self.stim_duration_var.get()),
                    first_onset=None,
                    stim_freq=parse_float(self.stim_freq_var.get()),
                    harmonics=parse_int(self.harmonics_var.get(), 3),
                    n_neighbor=parse_int(self.neighbor_bins_var.get(), 4),
                    fft_length_sec=parse_float(self.fft_len_var.get(), 2),
                    notch=self.notch_var.get(),
                    stim_mask=seg_mask,
                )
                values = np.asarray([row["snr_db"] for row in snr_rows if np.isfinite(row["snr_db"])], dtype=DATA_DTYPE)
                status = "ok" if values.size else "no finite SNR"
            except Exception as exc:
                values = np.asarray([], dtype=DATA_DTYPE)
                status = str(exc)
            rows.append({
                "window": win_idx + 1,
                "start_sec": start / self.fs,
                "end_sec": end / self.fs,
                "median_snr_db": float(np.nanmedian(values)) if values.size else np.nan,
                "mean_snr_db": float(np.nanmean(values)) if values.size else np.nan,
                "valid_channels": int(values.size),
                "selected_channels": int(len(channels)),
                "status": status,
            })
        return rows

    def compute_lfp_dynamic_snr_rows_with_settings(
        self,
        data: np.ndarray,
        channels: list[int],
        settings: dict,
        window_sec: float = 10.0,
    ) -> list[dict]:
        win_samples = int(round(window_sec * self.fs))
        if win_samples <= 0:
            raise ValueError("Invalid dynamic SNR window length.")
        n_windows = data.shape[0] // win_samples
        if n_windows <= 0:
            raise ValueError(f"Loaded data is shorter than {window_sec:g} seconds.")
        mode = str(settings.get("mode", "resting")).strip().lower()
        full_stim_mask = None
        stim_interval = parse_float(settings.get("stim_interval", self.stim_interval_var.get())) if mode != "evoked" else None
        if mode == "evoked":
            full_stim_mask = self.build_stim_mask_for_loaded_data(parse_float(settings.get("stim_duration", 0.1), 0.1))
        filtered_data = self.snr_preprocess_highpass_with_settings(data, settings)
        signal_band = (
            parse_float(settings.get("signal_band_low", 1.0), 1.0),
            parse_float(settings.get("signal_band_high", 30.0), 30.0),
        )
        noise_band = (
            parse_float(settings.get("noise_band_low", 1.0), 1.0),
            parse_float(settings.get("noise_band_high", 200.0), 200.0),
        )
        rows: list[dict] = []
        for win_idx in range(n_windows):
            start = win_idx * win_samples
            end = start + win_samples
            seg = filtered_data[start:end, :]
            seg_mask = full_stim_mask[start:end] if full_stim_mask is not None else None
            try:
                if mode == "resting":
                    score_seg = seg
                    if parse_bool_like(settings.get("notch_50hz"), False):
                        score_seg = self.apply_notch_filter_matrix(score_seg, freq=50.0, q=30.0, harmonics=1, axis=0)
                    values = self.compute_resting_snr_vectorized(score_seg, signal_band, noise_band)
                    values = np.asarray(values[np.isfinite(values)], dtype=DATA_DTYPE)
                else:
                    snr_rows = compute_batch_channel_snr(
                        seg,
                        self.fs,
                        mode=mode,
                        signal_band=signal_band,
                        noise_band=noise_band,
                        stim_interval=stim_interval,
                        stim_duration=parse_float(settings.get("stim_duration", 0.1), 0.1),
                        first_onset=None,
                        stim_freq=parse_float(settings.get("stim_freq", 10.0), 10.0),
                        harmonics=parse_int(settings.get("harmonics", 3), 3),
                        n_neighbor=parse_int(settings.get("neighbor_bins", 4), 4),
                        fft_length_sec=parse_float(settings.get("fft_len_sec", 2.0), 2.0),
                        notch=parse_bool_like(settings.get("notch_50hz"), False),
                        stim_mask=seg_mask,
                    )
                    values = np.asarray([row["snr_db"] for row in snr_rows if np.isfinite(row["snr_db"])], dtype=DATA_DTYPE)
                status = "ok" if values.size else "no finite SNR"
            except Exception as exc:
                values = np.asarray([], dtype=DATA_DTYPE)
                status = str(exc)
            rows.append({
                "window": win_idx + 1,
                "start_sec": start / self.fs,
                "end_sec": end / self.fs,
                "median_snr_db": float(np.nanmedian(values)) if values.size else np.nan,
                "mean_snr_db": float(np.nanmean(values)) if values.size else np.nan,
                "valid_channels": int(values.size),
                "selected_channels": int(len(channels)),
                "status": status,
            })
        return rows

    def plot_lfp_dynamic_snr(self):
        try:
            data, channels = self.selected_data()
            rows = self.compute_lfp_dynamic_snr_rows(data, channels, window_sec=10.0)
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Dynamic LFP SNR failed", str(exc))
            return
        self.show_dynamic_snr_dialog(
            title="LFP SNR 10s dynamic median",
            rows=rows,
            y_label="LFP median SNR (dB)",
            default_csv="lfp_dynamic_median_snr.csv",
            default_fig="lfp_dynamic_median_snr.png",
        )

    def show_dynamic_snr_dialog(
        self,
        title: str,
        rows: list[dict],
        y_label: str,
        default_csv: str,
        default_fig: str,
    ):
        if not rows:
            messagebox.showwarning("No rows", "No dynamic SNR rows are available.")
            return
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry("980x680")
        dialog.transient(self)

        fig = self.make_dynamic_snr_figure(rows, title, y_label, dpi=100)

        canvas = FigureCanvasTkAgg(fig, master=dialog)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=False, padx=10, pady=(10, 4))

        table_frame = ttk.LabelFrame(dialog, text="10s window dynamic SNR table")
        table_frame.pack(fill="both", expand=True, padx=10, pady=8)
        columns = ("window", "start", "end", "median", "mean", "valid", "selected", "spikes", "status")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "window": "Window",
            "start": "Start s",
            "end": "End s",
            "median": "Median SNR dB",
            "mean": "Mean SNR dB",
            "valid": "Valid ch",
            "selected": "Selected ch",
            "spikes": "Spikes",
            "status": "Status",
        }
        widths = {
            "window": 70,
            "start": 90,
            "end": 90,
            "median": 120,
            "mean": 110,
            "valid": 80,
            "selected": 90,
            "spikes": 80,
            "status": 220,
        }
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="center" if col != "status" else "w")
        tree.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        scroll.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scroll.set)

        for row in rows:
            tree.insert("", "end", values=(
                row["window"],
                f"{row['start_sec']:.3f}",
                f"{row['end_sec']:.3f}",
                f"{row['median_snr_db']:.3f}" if np.isfinite(row["median_snr_db"]) else "nan",
                f"{row['mean_snr_db']:.3f}" if np.isfinite(row["mean_snr_db"]) else "nan",
                row.get("valid_channels", ""),
                row.get("selected_channels", ""),
                row.get("spike_count", ""),
                row.get("status", ""),
            ))

        def export_dynamic_csv():
            path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Export dynamic SNR CSV",
                defaultextension=".csv",
                initialfile=default_csv,
                filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            try:
                self.save_dynamic_snr_rows_csv(Path(path), rows)
                self.log(f"Exported dynamic SNR CSV: {path}")
                messagebox.showinfo("Exported", f"CSV exported:\n{path}", parent=dialog)
            except Exception as exc:
                self.log(traceback.format_exc())
                messagebox.showerror("Export failed", str(exc), parent=dialog)

        def save_dynamic_figure():
            path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Save dynamic SNR figure",
                defaultextension=".png",
                initialfile=default_fig,
                filetypes=[
                    ("PNG image", "*.png"),
                    ("PDF file", "*.pdf"),
                    ("SVG file", "*.svg"),
                    ("JPEG image", "*.jpg;*.jpeg"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return
            try:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                self.log(f"Saved dynamic SNR figure: {path}")
                messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=dialog)
            except Exception as exc:
                self.log(traceback.format_exc())
                messagebox.showerror("Save failed", str(exc), parent=dialog)

        button_bar = ttk.Frame(dialog)
        button_bar.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(button_bar, text="Export CSV", command=export_dynamic_csv).pack(side="right", padx=4)
        ttk.Button(button_bar, text="Save figure", command=save_dynamic_figure).pack(side="right", padx=4)

    def make_dynamic_snr_figure(self, rows: list[dict], title: str, y_label: str, dpi: int = 120) -> Figure:
        fig = Figure(figsize=(9.4, 3.8), dpi=dpi)
        ax = fig.add_subplot(111)
        centers = np.asarray([(row["start_sec"] + row["end_sec"]) / 2.0 for row in rows], dtype=DATA_DTYPE)
        medians = np.asarray([row["median_snr_db"] for row in rows], dtype=DATA_DTYPE)
        ax.plot(centers, medians, marker="o", linewidth=1.5, color="#1f77b4")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.35)
        if np.isfinite(medians).any():
            ax.axhline(float(np.nanmedian(medians)), color="#e15759", linestyle="--", linewidth=1.1, label="overall median")
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def save_dynamic_snr_rows_csv(self, path: Path, rows: list[dict]):
        fieldnames = [
            "window", "start_sec", "end_sec", "median_snr_db", "mean_snr_db",
            "valid_channels", "selected_channels", "spike_count", "status",
        ]
        numeric_fields = {
            "window": "int",
            "start_sec": "float",
            "end_sec": "float",
            "median_snr_db": "float",
            "mean_snr_db": "float",
            "valid_channels": "int",
            "selected_channels": "int",
            "spike_count": "int",
        }
        export_rows = []
        for row in rows:
            export_row = dict(row)
            for key, kind in numeric_fields.items():
                if key in export_row:
                    export_row[key] = self.csv_numeric_value(export_row.get(key), kind=kind)
            export_rows.append(export_row)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(export_rows)

    def build_stim_mask_for_loaded_data(self, stim_duration_sec: float) -> np.ndarray:
        if self.stim_markers is None:
            raise RuntimeError(
                "Evoked SNR now follows the MATLAB Flash timing logic. "
                "Select log.txt and Event CSV, then run Flash_Task or LettermodeData."
            )
        data = self.current_data()
        mask = np.zeros(data.shape[0], dtype=bool)
        duration_samples = max(1, int(round(stim_duration_sec * self.fs)))
        offset = self.marker_offset_samples()
        markers_inside = 0
        for marker_sample in np.asarray(self.stim_markers)[:, 0]:
            start = int(marker_sample) - offset
            end = start + duration_samples
            if end <= 0 or start >= mask.size:
                continue
            markers_inside += 1
            mask[max(0, start):min(mask.size, end)] = True
        if not mask.any():
            raise RuntimeError(
                "stimMarkers were built, but no stimulus points fall inside the loaded data segment. "
                "Check whether page 1 start/duration covers the stimulus timing."
            )
        coverage = float(mask.mean())
        if mask.all() or coverage > 0.95:
            raise RuntimeError(
                f"Stimulus windows cover {coverage * 100:.1f}% of the loaded data, so evoked SNR is not valid. "
                f"Markers inside segment: {markers_inside}; stim duration={stim_duration_sec:g} sec. "
                "For Flash, use a smaller stim duration such as 0.05-0.1 sec."
            )
        return mask

    def refresh_snr_tree(self, threshold: float | None = None):
        if threshold is None:
            threshold = parse_float(self.snr_threshold_var.get(), 5)
        self.snr_tree.delete(*self.snr_tree.get_children())
        for row in self.last_snr_rows:
            ch = int(row["channel"])
            snr = self.snr_db_value(row)
            tags = []
            if np.isfinite(snr) and snr > threshold:
                tags.append("high")
            if str(row.get("detail", "")).startswith("Bad Channel"):
                tags.append("bad")
            if "SpikeInterface-style candidate" in str(row.get("detail", "")):
                tags.append("si_candidate")
            if ch in self.selected_channels:
                tags.append("selected")
            self.snr_tree.insert("", "end", iid=f"ch{ch}", values=(
                "[x]" if ch in self.selected_channels else "[ ]",
                ch,
                f"{snr:.3f}" if np.isfinite(snr) else str(snr),
                row["mode"],
                row["detail"],
            ), tags=tuple(tags))

    def on_snr_tree_click(self, event):
        item_id = self.snr_tree.identify_row(event.y)
        if not item_id:
            return
        values = self.snr_tree.item(item_id, "values")
        if len(values) < 2:
            return
        try:
            ch = int(values[1])
        except ValueError:
            return
        if ch in self.selected_channels:
            self.selected_channels.remove(ch)
        else:
            self.selected_channels.add(ch)
        self.refresh_snr_tree()
        self.update_selected_summary()

    def select_snr_above_threshold(self):
        if not self.last_snr_rows:
            messagebox.showerror("No SNR rows", "Run LFP SNR first.")
            return
        threshold = parse_float(self.snr_threshold_var.get(), 5)
        self.selected_channels = {
            int(row["channel"])
            for row in self.last_snr_rows
            if np.isfinite(self.snr_db_value(row)) and self.snr_db_value(row) > threshold
        }
        self.refresh_snr_tree(threshold=threshold)
        self.update_selected_summary()
        self.log(f"Selected {len(self.selected_channels)} channels with SNR > {threshold:g} dB.")

    def select_all_snr_channels(self):
        if not self.last_snr_rows:
            messagebox.showerror("No SNR rows", "Run LFP SNR first.")
            return
        self.selected_channels = {int(row["channel"]) for row in self.last_snr_rows}
        self.refresh_snr_tree()
        self.update_selected_summary()
        self.log(f"Selected all {len(self.selected_channels)} SNR channels.")

    def select_bad_snr_channels(self):
        if not self.last_snr_rows:
            messagebox.showerror("No SNR rows", "Run LFP SNR first.")
            return
        self.selected_channels = {
            int(row["channel"])
            for row in self.last_snr_rows
            if str(row.get("detail", "")).startswith("Bad Channel")
        }
        self.refresh_snr_tree()
        self.update_selected_summary()
        self.log(f"Selected {len(self.selected_channels)} bad channels.")

    def detect_spikeinterface_style_bad_channels(self, data: np.ndarray) -> dict[int, str]:
        arr = np.asarray(data, dtype=DATA_DTYPE)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return {}
        fill = np.nanmedian(arr, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        work = np.where(np.isfinite(arr), arr, fill)
        work = work - np.nanmedian(work, axis=0, keepdims=True)
        if scisig is not None and self.fs > 0 and work.shape[0] > 32:
            try:
                if 1.0 < self.fs / 2.0:
                    work = self.apply_highpass_filter(work, 1.0, axis=0)
                work = self.apply_notch_filter_matrix(work, freq=50.0, q=30.0, harmonics=1, axis=0)
            except Exception:
                pass
        max_samples = 120000
        if work.shape[0] > max_samples:
            step = int(np.ceil(work.shape[0] / max_samples))
            work = work[::step, :]

        reasons: dict[int, list[str]] = {}
        flat_std_thr = parse_float(self.bad_flat_std_var.get(), 1e-4)
        robust_std = np.full(work.shape[1], np.nan, dtype=DATA_DTYPE)
        for idx in range(work.shape[1]):
            sig = work[:, idx]
            finite = sig[np.isfinite(sig)]
            ch = idx + 1
            if finite.size < 32:
                reasons.setdefault(ch, []).append("too few finite samples")
                continue
            med = float(np.nanmedian(finite))
            mad_std = float(np.nanmedian(np.abs(finite - med)) / 0.6745)
            robust_std[idx] = mad_std
            sig_range = float(np.nanmax(finite) - np.nanmin(finite))
            if mad_std <= max(flat_std_thr, 1e-12) or sig_range <= flat_std_thr * 10:
                reasons.setdefault(ch, []).append(f"dead/flat robust std={mad_std:.3g}")

        valid_std = robust_std[np.isfinite(robust_std) & (robust_std > 0)]
        if valid_std.size:
            std_med = float(np.nanmedian(valid_std))
            std_mad = float(np.nanmedian(np.abs(valid_std - std_med)) / 0.6745)
            noisy_thr = max(std_med * 5.0, std_med + 6.0 * std_mad)
            quiet_thr = max(flat_std_thr * 10.0, std_med / 20.0)
            for idx, std_val in enumerate(robust_std):
                if not np.isfinite(std_val):
                    continue
                ch = idx + 1
                if std_val >= noisy_thr and noisy_thr > 0:
                    reasons.setdefault(ch, []).append(
                        f"noisy robust std={std_val:.3g}, group median={std_med:.3g}"
                    )
                elif std_val <= quiet_thr:
                    reasons.setdefault(ch, []).append(
                        f"abnormally quiet robust std={std_val:.3g}, group median={std_med:.3g}"
                    )

        neighbors = self.topology_neighbors_from_layout(work.shape[1])
        win = max(32, int(round(min(self.fs, work.shape[0])))) if self.fs > 0 else min(1024, work.shape[0])
        starts = list(range(0, max(1, work.shape[0] - win + 1), win))
        if not starts:
            starts = [0]
        for idx in range(work.shape[1]):
            ch = idx + 1
            if neighbors.get(ch):
                ref_cols = [n - 1 for n in neighbors[ch] if 1 <= n <= work.shape[1]]
            else:
                ref_cols = [j for j in range(work.shape[1]) if j != idx]
            if len(ref_cols) < 3:
                continue
            corrs: list[float] = []
            for start in starts:
                end = min(work.shape[0], start + win)
                target = work[start:end, idx]
                ref = np.nanmedian(work[start:end, ref_cols], axis=1)
                target_std = float(np.nanstd(target))
                ref_std = float(np.nanstd(ref))
                if target_std <= 1e-12 or ref_std <= 1e-12:
                    continue
                corr = float(np.corrcoef(target, ref)[0, 1])
                if np.isfinite(corr):
                    corrs.append(corr)
            if len(corrs) < 2:
                continue
            corr_arr = np.asarray(corrs, dtype=DATA_DTYPE)
            med_corr = float(np.nanmedian(corr_arr))
            low_ratio = float(np.nanmean(corr_arr < 0.2))
            if med_corr < 0.2 and low_ratio >= 0.7:
                ref_kind = "neighbor" if neighbors.get(ch) else "group"
                reasons.setdefault(ch, []).append(
                    f"low {ref_kind} correlation median={med_corr:.2f}, <0.2 ratio={low_ratio * 100:.0f}%"
                )

        return {
            ch: "SpikeInterface-style candidate: " + "; ".join(parts)
            for ch, parts in reasons.items()
            if parts
        }

    def run_spikeinterface_style_bad_check(self):
        if not self.last_snr_rows:
            messagebox.showerror("Run LFP SNR first", "Run LFP SNR first, then use SI-style Bad Check.")
            return
        try:
            self.set_snr_progress(0, "SI-style bad check: preprocessing and comparing channels...")
            reasons = self.detect_spikeinterface_style_bad_channels(self.current_data())
            for row in self.last_snr_rows:
                ch = int(row["channel"])
                reason = reasons.get(ch)
                if not reason:
                    continue
                detail = str(row.get("detail", ""))
                if "SpikeInterface-style candidate" not in detail:
                    row["detail"] = f"{detail}; {reason}" if detail else reason
            self.selected_channels = set(reasons.keys())
            self.refresh_snr_tree()
            self.update_selected_summary()
            self.set_snr_progress(
                100,
                f"SI-style bad check done: {len(reasons)} candidate channels selected for review.",
            )
            self.log(f"SI-style bad check selected {len(reasons)} candidate channels: {sorted(reasons.keys())}")
            messagebox.showinfo(
                "SI-style Bad Check",
                f"Found {len(reasons)} candidate channels.\nThey are selected in the SNR table for review.",
            )
        except Exception as exc:
            self.set_snr_progress(0, "SI-style bad check failed")
            self.log(traceback.format_exc())
            messagebox.showerror("SI-style Bad Check failed", str(exc))

    def select_good_snr_channels(self):
        if not self.last_snr_rows:
            messagebox.showerror("No SNR rows", "Run LFP SNR first.")
            return
        self.selected_channels = {
            int(row["channel"])
            for row in self.last_snr_rows
            if self.is_snr_row_healthy(row)
        }
        self.refresh_snr_tree()
        self.update_selected_summary()
        self.log(f"Selected {len(self.selected_channels)} non-bad channels for downstream Spike SNR.")

    def start_manual_review_selected(self):
        channels = self.get_selected_channels()
        if not channels:
            messagebox.showerror("No selected channels", "Select channels in the LFP SNR table first.")
            return
        self.manual_review_channels = channels
        self.manual_review_index = 0
        self.show_manual_review_channel()
        self.notebook.select(self.data_tab)
        self.log(f"Started manual review for {len(channels)} selected channels.")

    def show_manual_review_channel(self):
        if not self.manual_review_channels:
            self.manual_review_status_var.set("Manual review: none")
            return
        self.manual_review_index = max(0, min(self.manual_review_index, len(self.manual_review_channels) - 1))
        ch = self.manual_review_channels[self.manual_review_index]
        self.preview_channels_var.set(str(ch))
        self.manual_review_status_var.set(
            f"Manual review: ch{ch} ({self.manual_review_index + 1}/{len(self.manual_review_channels)})"
        )
        self.plot_preview()

    def manual_review_prev(self):
        if not self.manual_review_channels:
            self.change_all_raw_preview(-1)
            return
        self.manual_review_index = (self.manual_review_index - 1) % len(self.manual_review_channels)
        self.show_manual_review_channel()

    def manual_review_next(self):
        if not self.manual_review_channels:
            self.change_all_raw_preview(1)
            return
        self.manual_review_index = (self.manual_review_index + 1) % len(self.manual_review_channels)
        self.show_manual_review_channel()

    def preview_previous_channel(self):
        if self.manual_review_channels:
            self.manual_review_prev()
        else:
            self.change_all_raw_preview(-1)

    def preview_next_channel(self):
        if self.manual_review_channels:
            self.manual_review_next()
        else:
            self.change_all_raw_preview(1)

    def mark_manual_review_current_good(self):
        if not self.manual_review_channels:
            messagebox.showerror("No review list", "Start Manual review from the LFP SNR page first.")
            return
        ch = int(self.manual_review_channels[self.manual_review_index])
        changed = False
        for row in self.last_snr_rows:
            if int(row.get("channel", -1)) != ch:
                continue
            detail = str(row.get("detail", ""))
            parts = [part.strip() for part in detail.split(";") if part.strip()]
            kept = [
                part for part in parts
                if not part.startswith("Bad Channel") and not part.startswith("Manual Good")
            ]
            row["detail"] = "Manual Good override" + (f"; {'; '.join(kept)}" if kept else "")
            changed = True
            break
        self.selected_channels.add(ch)
        self.refresh_snr_tree()
        self.update_selected_summary()
        self.manual_review_status_var.set(
            f"Manual review: ch{ch} marked Good ({self.manual_review_index + 1}/{len(self.manual_review_channels)})"
        )
        self.log(f"Manual review marked ch{ch} as Good; Bad Channel label removed.")
        if not changed:
            messagebox.showwarning("Channel not in SNR table", f"ch{ch} is not present in the current LFP SNR rows.")

    def mark_manual_review_current_bad(self):
        if not self.manual_review_channels:
            messagebox.showerror("No review list", "Start Manual review from the LFP SNR page first.")
            return
        ch = int(self.manual_review_channels[self.manual_review_index])
        changed = False
        for row in self.last_snr_rows:
            if int(row.get("channel", -1)) != ch:
                continue
            detail = str(row.get("detail", ""))
            parts = [part.strip() for part in detail.split(";") if part.strip()]
            kept = [
                part for part in parts
                if not part.startswith("Bad Channel") and not part.startswith("Manual Good") and not part.startswith("Manual Bad")
            ]
            row["detail"] = "Bad Channel: Manual Bad override" + (f"; {'; '.join(kept)}" if kept else "")
            changed = True
            break
        self.selected_channels.discard(ch)
        self.refresh_snr_tree()
        self.update_selected_summary()
        self.manual_review_status_var.set(
            f"Manual review: ch{ch} marked Bad ({self.manual_review_index + 1}/{len(self.manual_review_channels)})"
        )
        self.log(f"Manual review marked ch{ch} as Bad; it was removed from selected downstream channels.")
        if not changed:
            messagebox.showwarning("Channel not in SNR table", f"ch{ch} is not present in the current LFP SNR rows.")

    def invert_snr_selection(self):
        if not self.last_snr_rows:
            messagebox.showerror("No SNR rows", "Run LFP SNR first.")
            return
        all_channels = {
            int(row["channel"])
            for row in self.last_snr_rows
            if np.isfinite(row.get("snr_db", np.nan))
        }
        self.selected_channels = all_channels - self.selected_channels
        self.refresh_snr_tree()
        self.update_selected_summary()
        self.log(f"Inverted selection: {len(self.selected_channels)} channels selected.")

    def show_selected_window_snr(self):
        if not self.selected_channels:
            messagebox.showerror("No selected channels", "Select channels in the SNR table first.")
            return
        data = self.current_data()
        channels = sorted(ch for ch in self.selected_channels if 1 <= ch <= data.shape[1])
        if not channels:
            messagebox.showerror("No valid channels", "Selected channels are outside the loaded data range.")
            return

        threshold = parse_float(self.bad_window_snr_threshold_var.get(), 1.0)
        ptp_threshold = parse_float(self.bad_window_ptp_threshold_var.get(), 0.01)
        dialog = tk.Toplevel(self)
        dialog.title("2s window SNR values")
        dialog.geometry("820x560")
        dialog.transient(self)

        header = ttk.Label(
            dialog,
            text=(
                f"2s windows, tail shorter than 2s ignored | valid if SNR > {threshold:g} dB | "
                f"if lower: PTP <= {ptp_threshold:g}mV => 0 flat, otherwise 2 artifact | "
                "flatline abs(diff)<=1e-6mV >20% => 0 flat | "
                f"signal {self.signal_band_low_var.get()}-{self.signal_band_high_var.get()} Hz / "
                f"noise {self.noise_band_low_var.get()}-{self.noise_band_high_var.get()} Hz"
            ),
            foreground="#245",
        )
        header.pack(anchor="w", padx=10, pady=(10, 4))

        table_frame = ttk.Frame(dialog)
        table_frame.pack(fill="both", expand=True, padx=10, pady=8)
        columns = ("channel", "window", "start", "end", "snr", "ptp", "dead", "status", "valid")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "channel": "Channel",
            "window": "Window",
            "start": "Start s",
            "end": "End s",
            "snr": "SNR dB",
            "ptp": "PTP mV",
            "dead": "Dead %",
            "status": "Code",
            "valid": "Valid",
        }
        widths = {
            "channel": 90,
            "window": 80,
            "start": 100,
            "end": 100,
            "snr": 110,
            "ptp": 110,
            "dead": 90,
            "status": 80,
            "valid": 80,
        }
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="center")
        tree.tag_configure("valid", foreground="#116611")
        tree.tag_configure("invalid", foreground="#aa2222")
        tree.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        scroll.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scroll.set)

        export_rows: list[dict] = []
        for ch in channels:
            rows = self.compute_2s_window_snr_values(np.asarray(data[:, ch - 1], dtype=DATA_DTYPE))
            for row in rows:
                valid = bool(row["valid"])
                status = int(row["status"])
                values = (
                    ch,
                    row["window"],
                    f"{row['start_sec']:.3f}",
                    f"{row['end_sec']:.3f}",
                    f"{row['snr_db']:.3f}",
                    f"{row['ptp']:.6g}",
                    f"{row['dead_straight_ratio'] * 100:.2f}",
                    status,
                    "yes" if valid else "no",
                )
                tree.insert("", "end", values=values, tags=("valid" if valid else "invalid",))
                export_rows.append({
                    "channel": ch,
                    "window": row["window"],
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "snr_db": row["snr_db"],
                    "ptp": row["ptp"],
                    "dead_straight_ratio": row["dead_straight_ratio"],
                    "status": status,
                    "status_label": row["status_label"],
                    "valid": valid,
                })

        def export_window_snr_csv():
            if not export_rows:
                messagebox.showwarning("No rows", "No window SNR rows are available.", parent=dialog)
                return
            path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Export window SNR CSV",
                defaultextension=".csv",
                initialfile="selected_channel_window_snr.csv",
                filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            try:
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "channel", "window", "start_sec", "end_sec",
                            "snr_db", "ptp", "dead_straight_ratio",
                            "status", "status_label", "valid",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(export_rows)
                self.log(f"Exported selected channel window SNR CSV: {path}")
                messagebox.showinfo("Exported", f"CSV exported:\n{path}", parent=dialog)
            except Exception as exc:
                self.log(traceback.format_exc())
                messagebox.showerror("Export failed", str(exc), parent=dialog)

        button_bar = ttk.Frame(dialog)
        button_bar.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(button_bar, text="Export CSV", command=export_window_snr_csv).pack(side="right")

    def clear_selected_channels(self, update_tree: bool = True):
        self.selected_channels.clear()
        if update_tree and hasattr(self, "snr_tree"):
            self.refresh_snr_tree()
        self.update_selected_summary()

    def update_selected_summary(self):
        selected = self.get_selected_channels()
        if selected:
            preview = ",".join(str(ch) for ch in selected[:20])
            suffix = "..." if len(selected) > 20 else ""
            self.selected_summary_var.set(f"Selected channels: {len(selected)} [{preview}{suffix}]")
        else:
            self.selected_summary_var.set("Selected channels: 0")

    def get_selected_channels(self) -> list[int]:
        data = self.current_data() if self.has_loaded_data() else None
        max_channel = data.shape[1] if data is not None else 0
        return sorted(ch for ch in self.selected_channels if 1 <= ch <= max_channel)

    def selected_base_data(self) -> tuple[np.ndarray, list[int]]:
        """Return selected channels from the loaded raw/remapped base data.

        LFP and Spike analysis must both start from this base matrix. Do not
        feed LFP-filtered data into Spike SNR or Spike task analysis.
        """
        data = self.current_data()
        channels = self.get_selected_channels()
        if not channels:
            raise RuntimeError("No channels selected. Select channels in the LFP SNR page first.")
        return data[:, [ch - 1 for ch in channels]], channels

    def selected_data(self) -> tuple[np.ndarray, list[int]]:
        return self.selected_base_data()

    def psd_page_size(self) -> int:
        return max(1, parse_int(self.psd_page_size_var.get(), 12))

    def get_psd_channels(self) -> list[int]:
        data = self.current_data()
        n_channels = data.shape[1]
        source = self.psd_channel_source_var.get().strip().lower()
        if source == "all channels":
            return list(range(1, n_channels + 1))
        if source == "healthy lfp channels":
            channels = [
                int(row["channel"])
                for row in self.last_snr_rows
                if self.is_snr_row_healthy(row) and 1 <= int(row["channel"]) <= n_channels
            ]
            if not channels:
                raise RuntimeError("No healthy LFP channels are available. Run LFP SNR first or choose all channels.")
            return sorted(set(channels))
        channels = self.get_selected_channels()
        if not channels:
            raise RuntimeError("No channels selected. Select channels in page 2, or choose all channels in page 4.")
        return channels

    def psd_harmonic_frequencies(self, base_freq: float, harmonics: int, fmin: float, fmax: float) -> list[float]:
        if base_freq <= 0 or harmonics <= 0:
            return []
        upper = min(fmax, self.fs / 2.0)
        return [base_freq * k for k in range(1, harmonics + 1) if fmin <= base_freq * k <= upper]

    def compute_lfp_psd_for_channels(self, channels: list[int]) -> tuple[np.ndarray, np.ndarray, str]:
        if scisig is None:
            raise ImportError("scipy.signal is required for PSD plotting.")
        if not self.has_loaded_data():
            raise RuntimeError("Load BIN/MAT data first.")
        if not channels:
            raise RuntimeError("No PSD channels selected.")
        full_lfp = self.get_lfp_filtered_full_data()
        data = np.asarray(full_lfp[:, [ch - 1 for ch in channels]], dtype=DATA_DTYPE)
        label_parts = ["LFP SNR filtered data"]
        filter_label = self.snr_filter_label()
        if filter_label:
            label_parts.append(filter_label)
        if self.notch_var.get():
            label_parts.append("LFP 50Hz notch")
        if self.psd_powerline_notch_var.get():
            freq = parse_float(self.psd_powerline_freq_var.get(), 50.0)
            harmonics = max(1, parse_int(self.psd_powerline_harmonics_var.get(), 1))
            q = max(1.0, parse_float(self.psd_notch_q_var.get(), 30.0))
            data = self.apply_notch_filter_matrix(data, freq=freq, q=q, harmonics=harmonics, axis=0)
            label_parts.append(f"extra notch {freq:g}Hz x{harmonics}, Q={q:g}")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        nperseg = int(round(parse_float(self.psd_welch_sec_var.get(), 2.0) * self.fs))
        nperseg = min(max(2, nperseg), data.shape[0])
        overlap_pct = min(95.0, max(0.0, parse_float(self.psd_overlap_pct_var.get(), 50.0)))
        noverlap = min(nperseg - 1, int(round(nperseg * overlap_pct / 100.0)))
        freqs, psd = scisig.welch(data, fs=self.fs, nperseg=nperseg, noverlap=noverlap, axis=0)
        freqs = np.asarray(freqs, dtype=DATA_DTYPE)
        psd = np.asarray(psd, dtype=DATA_DTYPE)
        label_parts.append(f"Welch {nperseg / self.fs:g}s, overlap {overlap_pct:g}%")
        return freqs, psd, "; ".join(label_parts)

    def render_psd_figure(self, fig: Figure, channels: list[int]) -> tuple[int, int, int, str]:
        page_size = self.psd_page_size()
        total = len(channels)
        total_pages = max(1, int(np.ceil(total / page_size)))
        self.psd_page_index = min(max(self.psd_page_index, 0), total_pages - 1)
        start = self.psd_page_index * page_size
        end = min(total, start + page_size)
        page_channels = channels[start:end]
        freqs, psd, preprocess_label = self.compute_lfp_psd_for_channels(page_channels)

        fmin = max(0.0, parse_float(self.psd_freq_min_var.get(), 0.0))
        fmax = parse_float(self.psd_freq_max_var.get(), 300.0)
        nyq = self.fs / 2.0
        if fmax <= fmin or fmax > nyq:
            fmax = min(nyq, max(fmin + 1.0, 300.0))
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        if not freq_mask.any():
            raise RuntimeError(f"No PSD frequency bins fall inside {fmin:g}-{fmax:g} Hz.")
        plot_freqs = freqs[freq_mask]
        plot_psd = psd[freq_mask, :].copy()

        stim_freq = parse_float(self.psd_stim_freq_var.get(), parse_float(self.stim_freq_var.get(), 10.0))
        stim_harmonics = max(0, parse_int(self.psd_stim_harmonics_var.get(), 0))
        harmonic_freqs = self.psd_harmonic_frequencies(stim_freq, stim_harmonics, fmin, fmax)
        if self.psd_mask_stim_harmonics_var.get() and harmonic_freqs:
            half_width = max(0.0, parse_float(self.psd_harmonic_bandwidth_var.get(), 0.25))
            for harmonic in harmonic_freqs:
                plot_psd[np.abs(plot_freqs - harmonic) <= half_width, :] = np.nan

        if self.psd_scale_var.get().strip().lower() == "db":
            plot_y = 10.0 * np.log10(plot_psd + np.finfo(DATA_DTYPE).eps)
            y_label = "PSD (dB/Hz)"
        else:
            plot_y = plot_psd
            y_label = "PSD (mV^2/Hz)"

        fig.clear()
        n = len(page_channels)
        cols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
        rows = max(1, int(np.ceil(n / cols)))
        axes = fig.subplots(rows, cols, squeeze=False)
        for i, ax in enumerate(axes.ravel()):
            if i >= n:
                ax.axis("off")
                continue
            ax.plot(plot_freqs, plot_y[:, i], color="black", linewidth=0.75)
            if self.psd_mark_stim_harmonics_var.get():
                for harmonic in harmonic_freqs:
                    ax.axvline(harmonic, color="red", linestyle="--", linewidth=0.65, alpha=0.75)
            ax.set_title(f"ch{page_channels[i]}", fontsize=8, pad=2)
            ax.grid(alpha=0.22, linewidth=0.4)
            ax.tick_params(axis="both", labelsize=7, length=2)
            if i % cols == 0:
                ax.set_ylabel(y_label, fontsize=8)
            if i // cols == rows - 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=8)
        harmonic_text = ""
        if self.psd_mark_stim_harmonics_var.get() and harmonic_freqs:
            harmonic_text = f" | stim harmonics: {', '.join(f'{v:g}' for v in harmonic_freqs[:8])}Hz"
            if len(harmonic_freqs) > 8:
                harmonic_text += "..."
        if self.psd_mask_stim_harmonics_var.get() and harmonic_freqs:
            harmonic_text += f" | masked +/-{self.psd_harmonic_bandwidth_var.get()}Hz"
        fig.suptitle(
            f"LFP PSD | page {self.psd_page_index + 1}/{total_pages} | "
            f"channels {start + 1}-{end} of {total} | {preprocess_label}{harmonic_text}",
            fontsize=11,
        )
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.08, top=0.90, wspace=0.30, hspace=0.42)
        return total_pages, start, end, preprocess_label

    def render_psd_page(self):
        try:
            channels = self.get_psd_channels()
            total_pages, start, end, _label = self.render_psd_figure(self.psd_fig, channels)
            self.psd_canvas.draw()
            self.psd_status_var.set(
                f"PSD page {self.psd_page_index + 1}/{total_pages}; channels {start + 1}-{end} of {len(channels)}"
            )
            self.notebook.select(self.psd_tab)
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("PSD plot failed", str(exc))

    def change_psd_page(self, delta: int):
        self.psd_page_index += delta
        self.render_psd_page()

    def open_psd_popup(self):
        try:
            channels = self.get_psd_channels()
        except Exception as exc:
            messagebox.showerror("PSD plot failed", str(exc))
            return
        dialog = tk.Toplevel(self)
        dialog.title("LFP PSD - large view")
        dialog.geometry("1500x950")
        dialog.minsize(1000, 700)
        dialog.transient(self)

        controls = ttk.Frame(dialog)
        controls.pack(fill="x", padx=8, pady=(8, 0))
        status_var = tk.StringVar(value="")
        fullscreen_var = tk.BooleanVar(value=False)

        def set_fullscreen(enabled: bool):
            fullscreen_var.set(bool(enabled))
            dialog.attributes("-fullscreen", bool(enabled))
            fullscreen_btn.configure(text="Exit fullscreen" if enabled else "Fullscreen")
            if not enabled:
                try:
                    dialog.state("zoomed")
                except tk.TclError:
                    pass

        def render_popup():
            try:
                total_pages, start, end, _label = self.render_psd_figure(fig, channels)
                canvas.draw()
                status_var.set(
                    f"PSD page {self.psd_page_index + 1}/{total_pages}; channels {start + 1}-{end} of {len(channels)}"
                )
                self.psd_status_var.set(status_var.get())
            except Exception as exc:
                self.log(traceback.format_exc())
                messagebox.showerror("PSD plot failed", str(exc), parent=dialog)

        def change_page(delta: int):
            self.psd_page_index += delta
            render_popup()

        ttk.Button(controls, text="< Prev page", command=lambda: change_page(-1)).pack(side="left", padx=(0, 4))
        ttk.Button(controls, text="Next page >", command=lambda: change_page(1)).pack(side="left", padx=4)
        fullscreen_btn = ttk.Button(controls, text="Fullscreen", command=lambda: set_fullscreen(not fullscreen_var.get()))
        fullscreen_btn.pack(side="left", padx=(12, 4))
        ttk.Label(controls, textvariable=status_var, foreground="#245").pack(side="left", padx=12)

        fig = Figure(figsize=(15, 9), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=dialog)
        toolbar = NavigationToolbar2Tk(canvas, dialog, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x", padx=8, pady=(4, 0))
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        dialog.bind("<Escape>", lambda _event: set_fullscreen(False))
        try:
            dialog.state("zoomed")
        except tk.TclError:
            pass
        render_popup()

    def parse_spike_params(self) -> dict:
        fs = float(self.fs)
        low = parse_float(self.spike_filter_low_var.get(), 300.0)
        high = parse_float(self.spike_filter_high_var.get(), 3000.0)
        nyq = fs / 2.0
        if scisig is None:
            raise ImportError("scipy.signal is required for spike filtering.")
        if low <= 0 or high <= low or high >= nyq:
            raise ValueError(f"Invalid spike band: {low:g}-{high:g} Hz; Nyquist is {nyq:g} Hz.")
        window_samples = max(1, int(round(parse_float(self.spike_window_sec_var.get(), 1.0) * fs)))
        step_samples = max(1, int(round(parse_float(self.spike_step_ms_var.get(), 100.0) * fs / 1000.0)))
        refractory_samples = max(1, int(round(parse_float(self.spike_refractory_ms_var.get(), 2.0) * fs / 1000.0)))
        pre_samples = max(1, parse_int(self.spike_pre_samples_var.get(), 13))
        post_samples = max(1, parse_int(self.spike_post_samples_var.get(), 19))
        return {
            "low": low,
            "high": high,
            "threshold_factor": parse_float(self.spike_threshold_factor_var.get(), 4.5),
            "window_samples": window_samples,
            "step_samples": step_samples,
            "refractory_samples": refractory_samples,
            "pre_samples": pre_samples,
            "post_samples": post_samples,
            "min_spikes": max(1, parse_int(self.spike_min_count_var.get(), 3)),
        }

    def parse_spike_params_from_settings(self, settings: dict) -> dict:
        fs = float(self.fs)
        low = parse_float(settings.get("spike_filter_low", self.spike_filter_low_var.get()), 300.0)
        high = parse_float(settings.get("spike_filter_high", self.spike_filter_high_var.get()), 3000.0)
        nyq = fs / 2.0
        if scisig is None:
            raise ImportError("scipy.signal is required for spike filtering.")
        if low <= 0 or high <= low or high >= nyq:
            raise ValueError(f"Invalid spike band: {low:g}-{high:g} Hz; Nyquist is {nyq:g} Hz.")
        window_samples = max(1, int(round(parse_float(settings.get("spike_window_sec", 1.0), 1.0) * fs)))
        step_samples = max(1, int(round(parse_float(settings.get("spike_step_ms", 100.0), 100.0) * fs / 1000.0)))
        refractory_samples = max(1, int(round(parse_float(settings.get("spike_refractory_ms", 2.0), 2.0) * fs / 1000.0)))
        pre_samples = max(1, parse_int(settings.get("spike_pre_samples", 13), 13))
        post_samples = max(1, parse_int(settings.get("spike_post_samples", 19), 19))
        return {
            "low": low,
            "high": high,
            "threshold_factor": parse_float(settings.get("spike_threshold_factor", 4.5), 4.5),
            "window_samples": window_samples,
            "step_samples": step_samples,
            "refractory_samples": refractory_samples,
            "pre_samples": pre_samples,
            "post_samples": post_samples,
            "min_spikes": max(1, parse_int(settings.get("spike_min_count", 3), 3)),
        }

    def spike_figure_params_from_settings(self, settings: dict) -> dict:
        return {
            "low": settings.get("spike_filter_low", self.spike_filter_low_var.get()),
            "high": settings.get("spike_filter_high", self.spike_filter_high_var.get()),
            "threshold_factor": settings.get("spike_threshold_factor", self.spike_threshold_factor_var.get()),
            "window_sec": settings.get("spike_window_sec", self.spike_window_sec_var.get()),
            "step_ms": settings.get("spike_step_ms", self.spike_step_ms_var.get()),
            "refractory_ms": settings.get("spike_refractory_ms", self.spike_refractory_ms_var.get()),
            "pre_samples": settings.get("spike_pre_samples", self.spike_pre_samples_var.get()),
            "post_samples": settings.get("spike_post_samples", self.spike_post_samples_var.get()),
        }

    def adaptive_spike_detection(self, signal: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        signal = np.asarray(signal, dtype=DATA_DTYPE).ravel()
        fs = float(self.fs)
        sos = scisig.butter(
            4,
            [params["low"] / (fs / 2.0), params["high"] / (fs / 2.0)],
            btype="bandpass",
            output="sos",
        )
        filtered = np.asarray(scisig.sosfiltfilt(sos, signal), dtype=DATA_DTYPE)
        return self.adaptive_spike_detection_from_filtered(filtered, params), filtered

    def adaptive_spike_detection_from_filtered(self, filtered: np.ndarray, params: dict) -> np.ndarray:
        filtered = np.asarray(filtered, dtype=DATA_DTYPE).ravel()
        n_samples = filtered.size
        window_size = min(params["window_samples"], n_samples)
        step_size = params["step_samples"]
        threshold_factor = params["threshold_factor"]

        if n_samples == 0:
            return np.array([], dtype=int)
        starts = list(range(0, max(1, n_samples - window_size + 1), step_size))
        if starts[-1] != max(0, n_samples - window_size):
            starts.append(max(0, n_samples - window_size))

        spike_candidates: list[np.ndarray] = []
        for start in starts:
            end = min(start + window_size, n_samples)
            window_data = filtered[start:end]
            noise_std = np.nanmedian(np.abs(window_data)) / 0.6745
            if not np.isfinite(noise_std) or noise_std <= 0:
                continue
            local_threshold = threshold_factor * noise_std
            above_thresh = window_data < -local_threshold
            crossings = np.flatnonzero(np.diff(np.r_[False, above_thresh]) == 1) + start
            if crossings.size:
                spike_candidates.append(crossings)

        if not spike_candidates:
            return np.array([], dtype=int)

        spike_times = np.unique(np.concatenate(spike_candidates).astype(int))
        refractory = params["refractory_samples"]
        valid_spikes = []
        last_spike = -np.inf
        for spike in spike_times:
            if spike - last_spike > refractory:
                valid_spikes.append(int(spike))
                last_spike = spike
        return np.asarray(valid_spikes, dtype=int)

    def compute_spike_channel_snr(
        self,
        signal: np.ndarray,
        channel: int,
        params: dict,
        duration_sec: float,
    ) -> dict:
        spike_times, filtered = self.adaptive_spike_detection(signal, params)
        return self.compute_spike_channel_snr_from_filtered(filtered, spike_times, channel, params, duration_sec)

    def compute_spike_channel_snr_from_filtered(
        self,
        filtered: np.ndarray,
        spike_times: np.ndarray,
        channel: int,
        params: dict,
        duration_sec: float,
    ) -> dict:
        filtered = np.asarray(filtered, dtype=DATA_DTYPE).ravel()
        pre = params["pre_samples"]
        post = params["post_samples"]
        valid = spike_times[(spike_times >= pre) & (spike_times + post < filtered.size)]

        result = {
            "channel": int(channel),
            "spike_count": int(valid.size),
            "spike_rate_hz": float(valid.size / duration_sec) if duration_sec > 0 else np.nan,
            "snr_db": np.nan,
            "template_pp": np.nan,
            "noise_std": np.nan,
            "status": "ok",
        }
        if valid.size < params["min_spikes"]:
            result["status"] = f"too_few_spikes<{params['min_spikes']}"
            return result

        waveforms = np.vstack([filtered[spike - pre:spike + post + 1] for spike in valid])
        template = np.nanmean(waveforms, axis=0)
        residual = waveforms - template
        noise_std = float(np.nanstd(residual))
        template_pp = float(np.nanmax(template) - np.nanmin(template))
        if not np.isfinite(noise_std) or noise_std <= 0 or not np.isfinite(template_pp):
            result["status"] = "invalid_noise"
            return result

        snr_linear = template_pp / (2.0 * noise_std)
        result.update({
            "snr_db": float(20.0 * np.log10(snr_linear)) if snr_linear > 0 else np.nan,
            "template_pp": template_pp,
            "noise_std": noise_std,
            "status": "ok",
        })
        return result

    def compute_spike_snr_for_channel_set(
        self,
        data: np.ndarray,
        channels: list[int],
        params: dict,
    ) -> list[dict]:
        duration_sec = data.shape[0] / float(self.fs) if self.fs else 0.0
        results = []
        for idx, channel in enumerate(channels):
            results.append(self.compute_spike_channel_snr(data[:, idx], channel, params, duration_sec))
        return results

    def compute_spike_snr_for_filtered_channel_set(
        self,
        filtered_data: np.ndarray,
        channels: list[int],
        params: dict,
    ) -> list[dict]:
        duration_sec = filtered_data.shape[0] / float(self.fs) if self.fs else 0.0
        results = []
        for idx, channel in enumerate(channels):
            filtered = np.asarray(filtered_data[:, idx], dtype=DATA_DTYPE)
            spike_times = self.adaptive_spike_detection_from_filtered(filtered, params)
            results.append(self.compute_spike_channel_snr_from_filtered(
                filtered,
                spike_times,
                channel,
                params,
                duration_sec,
            ))
        return results

    def run_spike_snr(self):
        try:
            data, channels = self.selected_base_data()
            params = self.parse_spike_params()
        except Exception as exc:
            messagebox.showerror("Spike SNR failed", str(exc))
            return

        def worker():
            try:
                full_filtered = self.get_spike_filtered_full_data(params)
                filtered = full_filtered[:, [ch - 1 for ch in channels]]
                results = self.compute_spike_snr_for_filtered_channel_set(filtered, channels, params)
                self._queue_ui(lambda: self.finish_spike_snr(results, params))
            except Exception as exc:
                error_msg = str(exc)
                self.log(traceback.format_exc())
                self._queue_ui(lambda: messagebox.showerror("Spike SNR failed", error_msg))

        self.spike_summary_var.set("Spike SNR: running...")
        threading.Thread(target=worker, daemon=True).start()

    def finish_spike_snr(self, results: list[dict], params: dict):
        self.spike_results = results
        self.refresh_spike_tree()
        finite = np.asarray([row["snr_db"] for row in results if np.isfinite(row["snr_db"])], dtype=DATA_DTYPE)
        total_spikes = sum(int(row["spike_count"]) for row in results)
        if finite.size:
            summary = (
                f"Spike SNR: {len(results)} channels, {finite.size} valid SNR values, "
                f"median={np.nanmedian(finite):.3f} dB, total spikes={total_spikes}, "
                f"band={params['low']:g}-{params['high']:g} Hz, threshold={params['threshold_factor']:g}x"
            )
        else:
            summary = f"Spike SNR: {len(results)} channels, no valid SNR values, total spikes={total_spikes}"
        self.spike_summary_var.set(summary)
        self.log(summary)
        self.notebook.select(self.spike_tab)
        pending = self.pending_task_after_spike
        self.pending_task_after_spike = None
        if pending is not None:
            source, mode = pending
            self._queue_ui(lambda: self.open_task_entry_ready(source, mode))

    def save_spike_results_csv(self, path: Path):
        self.save_spike_results_csv_rows(path, self.spike_results)

    def save_spike_results_csv_rows(self, path: Path, rows: list[dict]):
        fieldnames = ["channel", "spike_count", "spike_rate_hz", "snr_db", "template_pp", "noise_std", "status"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def make_spike_snr_figure(self) -> Figure:
        params = {
            "low": self.spike_filter_low_var.get(),
            "high": self.spike_filter_high_var.get(),
            "threshold_factor": self.spike_threshold_factor_var.get(),
            "window_sec": self.spike_window_sec_var.get(),
            "step_ms": self.spike_step_ms_var.get(),
            "refractory_ms": self.spike_refractory_ms_var.get(),
            "pre_samples": self.spike_pre_samples_var.get(),
            "post_samples": self.spike_post_samples_var.get(),
        }
        return self.make_spike_snr_figure_for_results(self.spike_results, params)

    def make_spike_snr_figure_for_results(self, spike_results: list[dict], params: dict) -> Figure:
        if not spike_results:
            raise ValueError("No spike results. Run spike detection / SNR first.")
        finite_rows = [row for row in spike_results if np.isfinite(row["snr_db"])]
        if not finite_rows:
            raise ValueError("No finite spike SNR values are available.")

        channels = np.asarray([row["channel"] for row in finite_rows], dtype=int)
        snr_values = np.asarray([row["snr_db"] for row in finite_rows], dtype=DATA_DTYPE)
        counts = np.asarray([row["spike_count"] for row in finite_rows], dtype=DATA_DTYPE)
        median_snr = float(np.nanmedian(snr_values))

        fig = Figure(figsize=(9.8, 6.0), dpi=120)
        gs = fig.add_gridspec(1, 3, width_ratios=[2.15, 1.05, 1.25], wspace=0.42)
        ax_scatter = fig.add_subplot(gs[0])
        ax_box = fig.add_subplot(gs[1])
        ax_stats = fig.add_subplot(gs[2])

        ax_scatter.scatter(channels, snr_values, c=counts, cmap="viridis", s=34, alpha=0.8)
        ax_scatter.set_title("Spike SNR by selected non-bad channel", fontsize=11)
        ax_scatter.set_xlabel("Channel")
        ax_scatter.set_ylabel("Spike SNR (dB)")
        ax_scatter.grid(True, linestyle="--", alpha=0.3)

        ax_box.boxplot(
            [snr_values],
            labels=["All"],
            widths=0.38,
            patch_artist=True,
            boxprops={"facecolor": "#dbeafe", "edgecolor": "#1f77b4", "linewidth": 1.2},
            medianprops={"color": "#e15759", "linewidth": 1.5},
            whiskerprops={"color": "#666", "linewidth": 1.0},
            capprops={"color": "#666", "linewidth": 1.0},
            flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#666", "markersize": 4},
        )
        ax_box.set_title("Overall distribution", fontsize=10, pad=12, loc="center")
        ax_box.set_ylabel("Spike SNR (dB)")
        ax_box.grid(axis="y", linestyle="--", alpha=0.3)

        ax_stats.axis("off")
        stats_text = (
            "Spike SNR\n"
            f"selected channels: {len(spike_results)}\n"
            f"valid SNR: {snr_values.size}\n"
            f"median: {median_snr:.3f} dB\n"
            f"mean: {np.nanmean(snr_values):.3f} dB\n"
            f"total spikes: {sum(int(row['spike_count']) for row in spike_results)}\n\n"
            "Source\n"
            "Page 3 selected non-bad channels\n\n"
            "Parameters\n"
            f"band: {params.get('low')}-{params.get('high')} Hz\n"
            f"threshold: {params.get('threshold_factor')} x noise\n"
            f"window: {params.get('window_sec', params.get('window_samples', ''))} s\n"
            f"step: {params.get('step_ms', params.get('step_samples', ''))} ms\n"
            f"refractory: {params.get('refractory_ms', params.get('refractory_samples', ''))} ms\n"
            f"waveform: -{params.get('pre_samples')} to +{params.get('post_samples')} samples"
        )
        ax_stats.text(
            0.02,
            0.98,
            stats_text,
            transform=ax_stats.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.95},
        )
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.88, wspace=0.42)
        return fig

    def refresh_spike_tree(self):
        self.spike_tree.delete(*self.spike_tree.get_children())
        for row in self.spike_results:
            snr = row["snr_db"]
            self.spike_tree.insert("", "end", values=(
                row["channel"],
                row["spike_count"],
                f"{row['spike_rate_hz']:.3f}" if np.isfinite(row["spike_rate_hz"]) else "nan",
                f"{snr:.3f}" if np.isfinite(snr) else row["status"],
                f"{row['template_pp']:.4g}" if np.isfinite(row["template_pp"]) else "nan",
                f"{row['noise_std']:.4g}" if np.isfinite(row["noise_std"]) else "nan",
            ))

    def plot_spike_snr_summary(self):
        if not self.spike_results:
            messagebox.showerror("No spike results", "Run spike detection / SNR first.")
            return
        finite_rows = [row for row in self.spike_results if np.isfinite(row["snr_db"])]
        if not finite_rows:
            messagebox.showwarning("No valid SNR", "No finite spike SNR values are available to plot.")
            return

        channels = np.asarray([row["channel"] for row in finite_rows], dtype=int)
        snr_values = np.asarray([row["snr_db"] for row in finite_rows], dtype=DATA_DTYPE)
        counts = np.asarray([row["spike_count"] for row in finite_rows], dtype=DATA_DTYPE)
        median_snr = float(np.nanmedian(snr_values))

        dialog = tk.Toplevel(self)
        dialog.title("Spike SNR summary")
        dialog.geometry("960x620")
        dialog.transient(self)

        fig = Figure(figsize=(9.8, 6.0), dpi=100)
        gs = fig.add_gridspec(1, 3, width_ratios=[2.15, 1.05, 1.25], wspace=0.42)
        ax_scatter = fig.add_subplot(gs[0])
        ax_box = fig.add_subplot(gs[1])
        ax_stats = fig.add_subplot(gs[2])

        ax_scatter.scatter(channels, snr_values, c=counts, cmap="viridis", s=34, alpha=0.8)
        ax_scatter.set_title("Spike SNR by selected channel", fontsize=11)
        ax_scatter.set_xlabel("Channel")
        ax_scatter.set_ylabel("Spike SNR (dB)")
        ax_scatter.grid(True, linestyle="--", alpha=0.3)

        ax_box.boxplot(
            [snr_values],
            labels=["All"],
            widths=0.38,
            patch_artist=True,
            boxprops={"facecolor": "#dbeafe", "edgecolor": "#1f77b4", "linewidth": 1.2},
            medianprops={"color": "#e15759", "linewidth": 1.5},
            whiskerprops={"color": "#666", "linewidth": 1.0},
            capprops={"color": "#666", "linewidth": 1.0},
            flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#666", "markersize": 4},
        )
        ax_box.set_title("Overall distribution", fontsize=10, pad=12, loc="center")
        ax_box.set_ylabel("Spike SNR (dB)")
        ax_box.grid(axis="y", linestyle="--", alpha=0.3)

        ax_stats.axis("off")
        stats_text = (
            "Spike SNR\n"
            f"channels: {len(self.spike_results)}\n"
            f"valid SNR: {snr_values.size}\n"
            f"median: {median_snr:.3f} dB\n"
            f"mean: {np.nanmean(snr_values):.3f} dB\n"
            f"total spikes: {sum(int(row['spike_count']) for row in self.spike_results)}\n\n"
            "Parameters\n"
            f"band: {self.spike_filter_low_var.get()}-{self.spike_filter_high_var.get()} Hz\n"
            f"threshold: {self.spike_threshold_factor_var.get()} x noise\n"
            f"window: {self.spike_window_sec_var.get()} s\n"
            f"step: {self.spike_step_ms_var.get()} ms\n"
            f"refractory: {self.spike_refractory_ms_var.get()} ms\n"
            f"waveform: -{self.spike_pre_samples_var.get()} to +{self.spike_post_samples_var.get()} samples"
        )
        ax_stats.text(
            0.02,
            0.98,
            stats_text,
            transform=ax_stats.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.95},
        )

        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.88, wspace=0.42)
        canvas = FigureCanvasTkAgg(fig, master=dialog)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 4))

        def save_spike_figure():
            path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Save spike SNR figure",
                defaultextension=".png",
                initialfile="spike_SNR_summary.png",
                filetypes=[
                    ("PNG image", "*.png"),
                    ("PDF file", "*.pdf"),
                    ("SVG file", "*.svg"),
                    ("JPEG image", "*.jpg;*.jpeg"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return
            try:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                self.log(f"Saved spike SNR figure: {path}")
                messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=dialog)
            except Exception as exc:
                self.log(traceback.format_exc())
                messagebox.showerror("Save failed", str(exc), parent=dialog)

        button_bar = ttk.Frame(dialog)
        button_bar.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(button_bar, text="Save figure", command=save_spike_figure).pack(side="right")

    def save_spike_results_bundle(self):
        if not self.spike_results:
            messagebox.showerror("No spike results", "Run spike detection / SNR first.")
            return
        out_dir = filedialog.askdirectory(title="Select folder for Spike SNR results")
        if not out_dir:
            return
        out_root = Path(out_dir) / f"spike_snr_selected_good_{time.strftime('%Y%m%d_%H%M%S')}"
        out_root.mkdir(parents=True, exist_ok=True)
        csv_path = out_root / "spike_snr_rows.csv"
        fig_path = out_root / "spike_snr_summary.png"
        selected_path = out_root / "selected_channels.txt"
        try:
            self.save_spike_results_csv(csv_path)
            fig = self.make_spike_snr_figure()
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            channels = [str(ch) for ch in self.get_selected_channels()]
            selected_path.write_text(
                "Selected channels from page 3 after excluding Bad Channel rows:\n"
                + ",".join(channels)
                + "\n",
                encoding="utf-8",
            )
            self.log(f"Saved Spike SNR bundle: {out_root}")
            messagebox.showinfo("Saved", f"Spike SNR results saved to:\n{out_root}")
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Save spike bundle failed", str(exc))

    def compute_spike_dynamic_snr_rows(self, data: np.ndarray, channels: list[int], params: dict, window_sec: float = 10.0) -> list[dict]:
        win_samples = int(round(window_sec * self.fs))
        if win_samples <= 0:
            raise ValueError("Invalid dynamic spike SNR window length.")
        n_windows = data.shape[0] // win_samples
        if n_windows <= 0:
            raise ValueError(f"Loaded data is shorter than {window_sec:g} seconds.")
        rows: list[dict] = []
        for win_idx in range(n_windows):
            start = win_idx * win_samples
            end = start + win_samples
            values: list[float] = []
            spike_total = 0
            valid_channels = 0
            for col, channel in enumerate(channels):
                result = self.compute_spike_channel_snr(data[start:end, col], channel, params, window_sec)
                spike_total += int(result["spike_count"])
                if np.isfinite(result["snr_db"]):
                    values.append(float(result["snr_db"]))
                    valid_channels += 1
            arr = np.asarray(values, dtype=DATA_DTYPE)
            rows.append({
                "window": win_idx + 1,
                "start_sec": start / self.fs,
                "end_sec": end / self.fs,
                "median_snr_db": float(np.nanmedian(arr)) if arr.size else np.nan,
                "mean_snr_db": float(np.nanmean(arr)) if arr.size else np.nan,
                "valid_channels": int(valid_channels),
                "selected_channels": int(len(channels)),
                "spike_count": int(spike_total),
                "status": "ok" if arr.size else "no finite spike SNR",
            })
        return rows

    def plot_spike_dynamic_snr(self):
        try:
            data, channels = self.selected_data()
            params = self.parse_spike_params()
        except Exception as exc:
            messagebox.showerror("Dynamic Spike SNR failed", str(exc))
            return

        def worker():
            try:
                rows = self.compute_spike_dynamic_snr_rows(data, channels, params, window_sec=10.0)
                self._queue_ui(lambda: self.show_dynamic_snr_dialog(
                    title="Spike SNR 10s dynamic median",
                    rows=rows,
                    y_label="Spike median SNR (dB)",
                    default_csv="spike_dynamic_median_snr.csv",
                    default_fig="spike_dynamic_median_snr.png",
                ))
            except Exception as exc:
                error_msg = str(exc)
                self.log(traceback.format_exc())
                self._queue_ui(lambda: messagebox.showerror("Dynamic Spike SNR failed", error_msg))

        self.spike_summary_var.set("Spike dynamic SNR: running...")
        threading.Thread(target=worker, daemon=True).start()

    def export_spike_snr_csv(self):
        if not self.spike_results:
            messagebox.showerror("No spike results", "Run spike detection / SNR first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        fieldnames = ["channel", "spike_count", "spike_rate_hz", "snr_db", "template_pp", "noise_std", "status"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.spike_results)
        self.log(f"Exported spike SNR CSV: {path}")

    def apply_analysis_preprocess(self, data: np.ndarray) -> np.ndarray:
        filtered = np.asarray(data, dtype=DATA_DTYPE).copy()
        if filtered.ndim == 1:
            filtered = filtered[:, None]
        if self.analysis_notch_var.get():
            for ch in range(filtered.shape[1]):
                filtered[:, ch] = apply_notch_filter(filtered[:, ch], self.fs, freq=50.0, q=30, harmonics=1)
        if self.analysis_bandpass_var.get():
            if scisig is None:
                raise ImportError("scipy.signal is required for bandpass filtering.")
            low = parse_float(self.analysis_band_low_var.get(), 0.5)
            high = parse_float(self.analysis_band_high_var.get(), 300)
            nyq = self.fs / 2.0
            if low <= 0 or high <= low or high >= nyq:
                raise ValueError(f"Invalid bandpass range: {low:g}-{high:g} Hz; Nyquist is {nyq:g} Hz.")
            filtered = self.apply_bandpass_filter(filtered, low, high, axis=0)
        if self.smooth_signal_var.get():
            win = max(1, int(round(parse_float(self.smooth_window_size_var.get(), 0.02) * self.fs)))
            if win > 1:
                kernel = np.ones(win, dtype=DATA_DTYPE) / win
                filtered = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 0, filtered)
        if self.zscoredata_var.get():
            mu = np.nanmean(filtered, axis=0, keepdims=True)
            sigma = np.nanstd(filtered, axis=0, keepdims=True)
            sigma[sigma == 0] = 1.0
            filtered = (filtered - mu) / sigma
        return filtered

    def selected_analysis_data(self) -> tuple[np.ndarray, np.ndarray, list[int], str]:
        base_data, channels = self.selected_base_data()
        raw = np.asarray(base_data, dtype=DATA_DTYPE)
        col_indices = [ch - 1 for ch in channels]
        source = getattr(self, "task_analysis_source", "lfp")
        if source == "spike":
            params = self.parse_spike_params()
            full_spike = self.get_spike_filtered_full_data(params)
            analysis = full_spike[:, col_indices]
            low = params["low"]
            high = params["high"]
            label = f"base raw/remapped -> Spike SNR bandpass {low:g}-{high:g}Hz"
        else:
            if self.motion_artifact_enable_var.get():
                motion_data, motion_channels, motion_label = self.get_motion_artifact_cleaned_selected_data()
                motion_map = {ch: idx for idx, ch in enumerate(motion_channels)}
                analysis = np.column_stack([motion_data[:, motion_map[ch]] for ch in channels])
                parts = [f"base raw/remapped -> {motion_label}"]
                if self.notch_var.get():
                    analysis = self.apply_notch_filter_matrix(analysis, freq=50.0, q=30.0, harmonics=1, axis=0)
                    parts.append("50Hz notch")
            else:
                full_lfp = self.get_lfp_filtered_full_data()
                analysis = full_lfp[:, col_indices]
                filter_label = self.snr_filter_label()
                parts = ["base raw/remapped -> LFP SNR filter"]
                if filter_label:
                    parts.append(filter_label)
                if self.notch_var.get():
                    parts.append("50Hz notch")
            label = ", ".join(parts)

        if self.smooth_signal_var.get():
            win = max(1, int(round(parse_float(self.smooth_window_size_var.get(), 0.02) * self.fs)))
            if win > 1:
                kernel = np.ones(win) / win
                analysis = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 0, analysis)
                label = f"{label}, smooth {self.smooth_window_size_var.get()}s"
        if self.zscoredata_var.get():
            mu = np.nanmean(analysis, axis=0, keepdims=True)
            sigma = np.nanstd(analysis, axis=0, keepdims=True)
            sigma[sigma == 0] = 1
            analysis = (analysis - mu) / sigma
            label = f"{label}, zscoredata"
        return np.asarray(raw, dtype=DATA_DTYPE), analysis, channels, label

    def analysis_preprocess_label(self) -> str:
        parts = [self.analysis_view_mode_var.get()]
        if self.analysis_notch_var.get():
            parts.append("50Hz notch")
        if self.analysis_bandpass_var.get():
            parts.append(f"filtFre {self.analysis_band_low_var.get()}-{self.analysis_band_high_var.get()}Hz")
        if self.smooth_signal_var.get():
            parts.append(f"smooth {self.smooth_window_size_var.get()}s")
        if self.zscoredata_var.get():
            parts.append("zscoredata")
        return ", ".join(parts)

    def preview_selected_channels(self):
        channels = self.get_selected_channels()
        if not channels:
            messagebox.showerror("No selected channels", "Select channels first.")
            return
        self.preview_channels_var.set(",".join(str(ch) for ch in channels[:12]))
        self.plot_preview()
        self.notebook.select(self.data_tab)

    def write_state_text(self, text: str):
        if not hasattr(self, "state_text"):
            self.show_task_analysis_window()
        self.state_text.configure(state="normal")
        self.state_text.delete("1.0", "end")
        self.state_text.insert("end", text)
        self.state_text.configure(state="disabled")

    def event_locked_lfp_snr_rows(
        self,
        epochs: np.ndarray,
        t_ms: np.ndarray,
        channels: list[int],
        tag: int,
    ) -> list[dict]:
        b0 = parse_float(self.baseline_start_ms_var.get(), -200)
        b1 = parse_float(self.baseline_end_ms_var.get(), 0)
        r0 = parse_float(self.response_start_ms_var.get(), 0)
        r1 = parse_float(self.response_end_ms_var.get(), 300)
        base_mask = (t_ms >= b0) & (t_ms <= b1)
        resp_mask = (t_ms >= r0) & (t_ms <= r1)
        if not np.any(base_mask):
            raise RuntimeError(f"Baseline window {b0:g}-{b1:g} ms has no samples in the epoch.")
        if not np.any(resp_mask):
            raise RuntimeError(f"Response window {r0:g}-{r1:g} ms has no samples in the epoch.")

        corrected = self.baseline_correct_epochs(epochs, t_ms)
        base_power = np.nanmean(corrected[:, base_mask, :] ** 2, axis=(0, 1))
        resp_power = np.nanmean(corrected[:, resp_mask, :] ** 2, axis=(0, 1))
        eps = np.finfo(DATA_DTYPE).eps
        snr_db = 10.0 * np.log10((resp_power + eps) / (base_power + eps))
        rows: list[dict] = []
        for i, ch in enumerate(channels):
            rows.append({
                "channel": int(ch),
                "snr_db": float(snr_db[i]),
                "mode": "event_lfp",
                "tag": int(tag),
                "baseline_power": float(base_power[i]),
                "response_power": float(resp_power[i]),
                "detail": (
                    f"tag={tag}; response {r0:g}-{r1:g} ms / "
                    f"baseline {b0:g}-{b1:g} ms; "
                    f"trials={epochs.shape[0]}"
                ),
            })
        return rows

    def event_locked_spike_rate_snr_rows(
        self,
        filtered_data: np.ndarray,
        used_markers: np.ndarray,
        t_ms: np.ndarray,
        channels: list[int],
        tag: int,
    ) -> list[dict]:
        b0 = parse_float(self.baseline_start_ms_var.get(), -200)
        b1 = parse_float(self.baseline_end_ms_var.get(), 0)
        r0 = parse_float(self.response_start_ms_var.get(), 0)
        r1 = parse_float(self.response_end_ms_var.get(), 300)
        if b1 <= b0:
            raise RuntimeError(f"Invalid baseline window {b0:g}-{b1:g} ms.")
        if r1 <= r0:
            raise RuntimeError(f"Invalid response window {r0:g}-{r1:g} ms.")
        if not np.any((t_ms >= b0) & (t_ms <= b1)):
            raise RuntimeError(f"Baseline window {b0:g}-{b1:g} ms has no samples in the epoch.")
        if not np.any((t_ms >= r0) & (t_ms <= r1)):
            raise RuntimeError(f"Response window {r0:g}-{r1:g} ms has no samples in the epoch.")

        params = self.parse_spike_params()
        filtered_data = np.asarray(filtered_data, dtype=DATA_DTYPE)
        used_markers = np.asarray(used_markers, dtype=int)
        marker_centers = used_markers[:, 0] - self.marker_offset_samples()
        base_start = np.rint(marker_centers + b0 * self.fs / 1000.0).astype(int)
        base_end = np.rint(marker_centers + b1 * self.fs / 1000.0).astype(int)
        resp_start = np.rint(marker_centers + r0 * self.fs / 1000.0).astype(int)
        resp_end = np.rint(marker_centers + r1 * self.fs / 1000.0).astype(int)
        base_sec = (b1 - b0) / 1000.0
        resp_sec = (r1 - r0) / 1000.0
        eps = np.finfo(DATA_DTYPE).eps
        rows: list[dict] = []

        for i, ch in enumerate(channels):
            sig = filtered_data[:, i]
            spike_times = self.adaptive_spike_detection_from_filtered(sig, params)
            base_counts = np.asarray([
                np.count_nonzero((spike_times >= s) & (spike_times < e))
                for s, e in zip(base_start, base_end)
            ], dtype=DATA_DTYPE)
            resp_counts = np.asarray([
                np.count_nonzero((spike_times >= s) & (spike_times < e))
                for s, e in zip(resp_start, resp_end)
            ], dtype=DATA_DTYPE)
            base_rates = base_counts / base_sec
            resp_rates = resp_counts / resp_sec
            base_rate = float(np.nanmean(base_rates))
            resp_rate = float(np.nanmean(resp_rates))
            base_std = float(np.nanstd(base_rates, ddof=1)) if base_rates.size > 1 else 0.0
            rate_delta = resp_rate - base_rate
            z_snr = rate_delta / max(base_std, eps)
            snr_db = 10.0 * np.log10((resp_rate + eps) / (base_rate + eps))
            rows.append({
                "channel": int(ch),
                "snr_db": float(snr_db),
                "mode": "event_spike_rate",
                "tag": int(tag),
                "baseline_rate_hz": base_rate,
                "response_rate_hz": resp_rate,
                "rate_delta_hz": float(rate_delta),
                "rate_z": float(z_snr),
                "spike_count_baseline": int(np.nansum(base_counts)),
                "spike_count_response": int(np.nansum(resp_counts)),
                "trials": int(used_markers.shape[0]),
                "detail": (
                    f"tag={tag}; response {r0:g}-{r1:g} ms rate / "
                    f"baseline {b0:g}-{b1:g} ms rate; "
                    f"trials={used_markers.shape[0]}"
                ),
            })
        return rows

    def current_task_lfp_snr_rows(self, task_name: str) -> list[dict]:
        task = task_name.lower()
        cache = self.tvep_cache if "flash" in task else self.stim_task_cache
        if not cache:
            return []
        rows: list[dict] = []
        for result in cache.get("tag_results", []):
            rows.extend(result.get("event_lfp_snr_rows", []))
        return rows

    def current_task_spike_rate_snr_rows(self, task_name: str) -> list[dict]:
        task = task_name.lower()
        cache = self.tvep_cache if "flash" in task else self.stim_task_cache
        if not cache:
            return []
        rows: list[dict] = []
        for result in cache.get("tag_results", []):
            rows.extend(result.get("event_spike_snr_rows", []))
        return rows

    def make_event_locked_lfp_snr_figure(self, rows: list[dict], title: str) -> Figure:
        finite_rows = [row for row in rows if np.isfinite(float(row.get("snr_db", np.nan)))]
        if not finite_rows:
            raise ValueError("No finite event-locked LFP SNR values are available.")
        tags = sorted({int(row.get("tag", 0)) for row in finite_rows})
        values_by_tag = [
            np.asarray([float(row["snr_db"]) for row in finite_rows if int(row.get("tag", 0)) == tag], dtype=DATA_DTYPE)
            for tag in tags
        ]
        all_values = np.concatenate(values_by_tag) if values_by_tag else np.asarray([], dtype=DATA_DTYPE)
        b0 = parse_float(self.baseline_start_ms_var.get(), -200)
        b1 = parse_float(self.baseline_end_ms_var.get(), 0)
        r0 = parse_float(self.response_start_ms_var.get(), 0)
        r1 = parse_float(self.response_end_ms_var.get(), 300)

        fig = Figure(figsize=(8.8, 6.0), dpi=120)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.25)
        ax = fig.add_subplot(gs[0])
        stats_ax = fig.add_subplot(gs[1])
        labels = [f"{tag}" for tag in tags]
        ax.boxplot(
            values_by_tag,
            labels=labels,
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": "#dbeafe", "edgecolor": "#1f77b4", "linewidth": 1.2},
            medianprops={"color": "#e15759", "linewidth": 1.5},
            whiskerprops={"color": "#666", "linewidth": 1.0},
            capprops={"color": "#666", "linewidth": 1.0},
            flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#666", "markersize": 4},
        )
        ax.axhline(y=0, color="#888", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Stim tag / frequency")
        ax.set_ylabel("Event-locked LFP SNR (dB)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        stats_ax.axis("off")
        lines = [
            "Event-locked LFP SNR",
            f"baseline: {b0:g} to {b1:g} ms",
            f"response: {r0:g} to {r1:g} ms",
            "",
            "Formula",
            "10*log10(response power / baseline power)",
            "",
            f"channels x tags: {len(finite_rows)}",
            f"median: {np.nanmedian(all_values):.3f} dB",
            f"mean: {np.nanmean(all_values):.3f} dB",
            "",
            "Per tag median",
        ]
        for tag, vals in zip(tags, values_by_tag):
            lines.append(f"{tag}: {np.nanmedian(vals):.3f} dB (n={vals.size})")
        stats_ax.text(
            0.02, 0.98, "\n".join(lines), transform=stats_ax.transAxes,
            va="top", ha="left", fontsize=9.0, linespacing=1.3,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.95},
        )
        fig.tight_layout()
        return fig

    def make_event_locked_spike_rate_snr_figure(self, rows: list[dict], title: str) -> Figure:
        finite_rows = [row for row in rows if np.isfinite(float(row.get("snr_db", np.nan)))]
        if not finite_rows:
            raise ValueError("No finite event-locked spike rate SNR values are available.")
        tags = sorted({int(row.get("tag", 0)) for row in finite_rows})
        values_by_tag = [
            np.asarray([float(row["snr_db"]) for row in finite_rows if int(row.get("tag", 0)) == tag], dtype=DATA_DTYPE)
            for tag in tags
        ]
        z_by_tag = [
            np.asarray([float(row.get("rate_z", np.nan)) for row in finite_rows if int(row.get("tag", 0)) == tag], dtype=DATA_DTYPE)
            for tag in tags
        ]
        all_values = np.concatenate(values_by_tag) if values_by_tag else np.asarray([], dtype=DATA_DTYPE)
        b0 = parse_float(self.baseline_start_ms_var.get(), -200)
        b1 = parse_float(self.baseline_end_ms_var.get(), 0)
        r0 = parse_float(self.response_start_ms_var.get(), 0)
        r1 = parse_float(self.response_end_ms_var.get(), 300)

        fig = Figure(figsize=(8.8, 6.0), dpi=120)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.25)
        ax = fig.add_subplot(gs[0])
        stats_ax = fig.add_subplot(gs[1])
        labels = [f"{tag}" for tag in tags]
        ax.boxplot(
            values_by_tag,
            labels=labels,
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": "#dcfce7", "edgecolor": "#2ca02c", "linewidth": 1.2},
            medianprops={"color": "#e15759", "linewidth": 1.5},
            whiskerprops={"color": "#666", "linewidth": 1.0},
            capprops={"color": "#666", "linewidth": 1.0},
            flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#666", "markersize": 4},
        )
        ax.axhline(y=0, color="#888", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Stim tag / frequency")
        ax.set_ylabel("Event-locked spike rate SNR (dB)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        stats_ax.axis("off")
        lines = [
            "Event-locked spike rate SNR",
            f"baseline: {b0:g} to {b1:g} ms",
            f"response: {r0:g} to {r1:g} ms",
            "",
            "Formula",
            "10*log10(response rate / baseline rate)",
            "z = (response rate - baseline rate) / baseline-rate std",
            "",
            f"channels x tags: {len(finite_rows)}",
            f"median: {np.nanmedian(all_values):.3f} dB",
            f"mean: {np.nanmean(all_values):.3f} dB",
            "",
            "Per tag median",
        ]
        for tag, vals, zvals in zip(tags, values_by_tag, z_by_tag):
            lines.append(
                f"{tag}: {np.nanmedian(vals):.3f} dB, z={np.nanmedian(zvals):.3f} (n={vals.size})"
            )
        stats_ax.text(
            0.02, 0.98, "\n".join(lines), transform=stats_ax.transAxes,
            va="top", ha="left", fontsize=9.0, linespacing=1.3,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.95},
        )
        fig.tight_layout()
        return fig

    def show_task_source_snr_figure(self, task_name: str):
        source = getattr(self, "task_analysis_source", "lfp")
        try:
            if source == "spike":
                task_rows = self.current_task_spike_rate_snr_rows(task_name)
                if task_rows:
                    fig = self.make_event_locked_spike_rate_snr_figure(
                        task_rows,
                        f"{task_name} - event-locked spike rate SNR",
                    )
                    title = f"{task_name} - event-locked spike rate SNR"
                    default_name = f"{task_name}_event_locked_spike_rate_SNR.png"
                elif not self.spike_results:
                    return
                else:
                    fig = self.make_spike_snr_figure()
                    title = f"{task_name} - Spike SNR summary"
                    default_name = f"{task_name}_Spike_SNR_summary.png"
            else:
                task_rows = self.current_task_lfp_snr_rows(task_name)
                if task_rows:
                    fig = self.make_event_locked_lfp_snr_figure(
                        task_rows,
                        f"{task_name} - event-locked LFP SNR",
                    )
                    title = f"{task_name} - event-locked LFP SNR"
                    default_name = f"{task_name}_event_locked_LFP_SNR.png"
                elif not self.last_snr_rows:
                    return
                else:
                    selected = set(self.get_selected_channels())
                    rows = [
                        row for row in self.last_snr_rows
                        if not selected or int(row.get("channel", -1)) in selected
                    ]
                    if not rows:
                        return
                    fig = self.make_snr_boxplot_figure(
                        rows,
                        self.current_lfp_snr_settings(),
                        f"{task_name} - LFP SNR distribution",
                    )
                    title = f"{task_name} - LFP SNR summary"
                    default_name = f"{task_name}_LFP_SNR_boxplot.png"
        except Exception as exc:
            self.log(f"Skipped task SNR summary figure: {exc}")
            return

        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry("900x650")
        dialog.transient(self)
        canvas = FigureCanvasTkAgg(fig, master=dialog)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 4))

        def save_task_snr_figure():
            path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Save task SNR figure",
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[
                    ("PNG image", "*.png"),
                    ("PDF file", "*.pdf"),
                    ("SVG file", "*.svg"),
                    ("JPEG image", "*.jpg;*.jpeg"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return
            try:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                self.log(f"Saved task SNR figure: {path}")
                messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=dialog)
            except Exception as exc:
                self.log(traceback.format_exc())
                messagebox.showerror("Save failed", str(exc), parent=dialog)

        button_bar = ttk.Frame(dialog)
        button_bar.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(button_bar, text="Save figure", command=save_task_snr_figure).pack(side="right")

    def current_task_marker_tag(self) -> int | None:
        if self.current_paged_plot == "tvep" and self.tvep_cache:
            tag_results = self.tvep_cache.get("tag_results", [])
            if tag_results:
                idx = min(max(self.tvep_tag_index, 0), len(tag_results) - 1)
                return int(tag_results[idx]["tag"])
        if self.current_paged_plot == "stim_task" and self.stim_task_cache:
            tag_results = self.stim_task_cache.get("tag_results", [])
            if tag_results:
                idx = min(max(self.stim_task_tag_index, 0), len(tag_results) - 1)
                return int(tag_results[idx]["tag"])
        return None

    def show_raw_marker_alignment_figure(self):
        if not self.has_loaded_data():
            messagebox.showerror("No data", "Load parsed MAT data first.")
            return
        if self.stim_markers is None:
            messagebox.showerror("No markers", "Load log.txt and Event CSV first to build stimMarkers.")
            return
        try:
            raw_data, channels = self.selected_base_data()
            markers = np.asarray(self.stim_markers, dtype=int)
            if markers.ndim != 2 or markers.shape[1] < 2 or markers.size == 0:
                raise RuntimeError("stimMarkers are empty or invalid.")

            tag = self.current_task_marker_tag()
            plot_markers = markers
            marker_label = "all tags"
            if tag is not None:
                tag_markers = markers[markers[:, 1] == tag]
                if tag_markers.size:
                    plot_markers = tag_markers
                    marker_label = f"tag {tag}"

            sample_offset = self.marker_offset_samples()
            local_marker_samples = plot_markers[:, 0] - sample_offset
            loaded_len = raw_data.shape[0]
            valid_mask = (local_marker_samples >= 0) & (local_marker_samples < loaded_len)
            valid_markers = plot_markers[valid_mask]
            valid_local_samples = local_marker_samples[valid_mask]
            if valid_markers.size == 0:
                raise RuntimeError(
                    f"No {marker_label} markers fall inside the loaded MAT segment. "
                    "Check segment start/duration or marker timing."
                )

            duration_sec = max(0.1, parse_float(self.state_duration_var.get(), 5.0))
            duration_samples = max(1, int(round(duration_sec * self.fs)))
            start_sample = max(0, int(round(parse_float(self.state_start_var.get(), 0.0) * self.fs)))
            end_sample = min(loaded_len, start_sample + duration_samples)
            if not np.any((valid_local_samples >= start_sample) & (valid_local_samples < end_sample)):
                lead_samples = int(round(1.0 * self.fs))
                start_sample = max(0, int(valid_local_samples[0]) - lead_samples)
                end_sample = min(loaded_len, start_sample + duration_samples)
                if end_sample - start_sample < duration_samples:
                    start_sample = max(0, end_sample - duration_samples)

            window_mask = (valid_local_samples >= start_sample) & (valid_local_samples < end_sample)
            window_markers = valid_markers[window_mask]
            window_local_samples = valid_local_samples[window_mask]
            if window_markers.size == 0:
                raise RuntimeError(f"No {marker_label} markers are visible in the selected plot window.")

            max_channels = min(4, len(channels))
            channel_indices = list(range(max_channels))
            max_points = 8000
            sample_count = max(1, end_sample - start_sample)
            step = max(1, int(np.ceil(sample_count / max_points)))
            sample_indices = np.arange(start_sample, end_sample, step)
            if self.time is not None and len(self.time) == self.current_data().shape[0]:
                x = np.asarray(self.time[sample_indices]).ravel()
                marker_x = window_markers[:, 0] / self.fs
                x_label = "Recording time (sec)"
            else:
                x = sample_indices / self.fs
                marker_x = window_local_samples / self.fs
                x_label = "Loaded segment time (sec)"

            fig = Figure(figsize=(10.5, 2.2 * max_channels + 1.0), dpi=110)
            axes = fig.subplots(max_channels, 1, squeeze=False, sharex=True).ravel()
            corrected_line = None
            uncorrected_line = None
            delta_t1 = float(getattr(self, "bin_delta_t1_sec", 0.0) or 0.0)
            if self.time is not None and len(self.time) == self.current_data().shape[0]:
                uncorrected_x = (window_markers[:, 0] - int(round(delta_t1 * self.fs))) / self.fs
            else:
                uncorrected_x = (window_local_samples - int(round(delta_t1 * self.fs))) / self.fs

            for ax, ch_i in zip(axes, channel_indices):
                y = np.asarray(raw_data[sample_indices, ch_i], dtype=DATA_DTYPE)
                ax.plot(x, y, color="black", linewidth=0.75, label=f"ch{channels[ch_i]} raw")
                for mx in marker_x:
                    corrected_line = ax.axvline(mx, color="red", linewidth=0.9, alpha=0.75)
                if abs(delta_t1) > 1e-9:
                    visible_uncorrected = uncorrected_x[(uncorrected_x >= x[0]) & (uncorrected_x <= x[-1])]
                    for mx in visible_uncorrected:
                        uncorrected_line = ax.axvline(mx, color="tab:blue", linestyle="--", linewidth=0.75, alpha=0.55)
                ax.set_ylabel(f"ch{channels[ch_i]}\nmV", fontsize=8)
                ax.grid(alpha=0.22, linewidth=0.5)
                ax.tick_params(labelsize=8)
            axes[-1].set_xlabel(x_label, fontsize=9)

            handles = []
            labels = []
            if corrected_line is not None:
                handles.append(corrected_line)
                labels.append("marker with deltaT1")
            if uncorrected_line is not None:
                handles.append(uncorrected_line)
                labels.append("marker without deltaT1")
            if handles:
                axes[0].legend(handles, labels, loc="upper right", fontsize=8)

            def format_seconds(value):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return "nan"
                return f"{value:.6f}" if np.isfinite(value) else "nan"

            timing = self.timing_info or {}
            raw_delay = timing.get("stim_delay_raw_sec", np.nan)
            corrected_delay = timing.get("stim_delay_sec", np.nan)
            fig.suptitle(
                f"Raw signal + marker alignment | {marker_label} | "
                f"markers in window={len(window_markers)} | deltaT1={delta_t1 * 1000.0:.3f}ms | "
                f"raw delay={format_seconds(raw_delay)}s | corrected={format_seconds(corrected_delay)}s",
                fontsize=10,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])

            dialog = tk.Toplevel(self)
            dialog.title("Raw signal + marker alignment")
            dialog.geometry("1100x780")
            dialog.transient(self.task_window if self.task_window is not None and self.task_window.winfo_exists() else self)
            canvas = FigureCanvasTkAgg(fig, master=dialog)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, dialog, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill="x", padx=10, pady=(8, 0))
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 4))

            def save_alignment_figure():
                default_tag = f"tag_{tag}" if tag is not None else "all_tags"
                path = filedialog.asksaveasfilename(
                    parent=dialog,
                    title="Save raw marker alignment figure",
                    defaultextension=".png",
                    initialfile=f"raw_marker_alignment_{default_tag}.png",
                    filetypes=[
                        ("PNG image", "*.png"),
                        ("PDF file", "*.pdf"),
                        ("SVG file", "*.svg"),
                        ("JPEG image", "*.jpg;*.jpeg"),
                        ("All files", "*.*"),
                    ],
                )
                if not path:
                    return
                try:
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    self.log(f"Saved raw marker alignment figure: {path}")
                    messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=dialog)
                except Exception as exc:
                    self.log(traceback.format_exc())
                    messagebox.showerror("Save failed", str(exc), parent=dialog)

            info_text = (
                f"Showing selected channels {', '.join(str(channels[i]) for i in channel_indices)}; "
                f"window {x[0]:.3f}-{x[-1]:.3f} sec; "
                "red lines use the current corrected stimMarkers."
            )
            button_bar = ttk.Frame(dialog)
            button_bar.pack(fill="x", padx=10, pady=(0, 10))
            ttk.Label(button_bar, text=info_text, foreground="#445").pack(side="left")
            ttk.Button(button_bar, text="Save figure", command=save_alignment_figure).pack(side="right")
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Raw marker alignment failed", str(exc))

    def show_marker_response_figure(self):
        if not self.has_loaded_data():
            messagebox.showerror("No data", "Load parsed MAT data first.")
            return
        if self.stim_markers is None:
            messagebox.showerror("No markers", "Load log.txt and Event CSV first to build stimMarkers.")
            return
        try:
            raw_data, analysis_data, channels, preprocess_label = self.selected_analysis_data()
            markers = np.asarray(self.stim_markers, dtype=int)
            if markers.ndim != 2 or markers.shape[1] < 2 or markers.size == 0:
                raise RuntimeError("stimMarkers are empty or invalid.")

            tag = self.current_task_marker_tag()
            if tag is None:
                tag = self.selected_stim_tags(markers)[0]
            tag = int(tag)
            tag_markers = markers[markers[:, 1] == tag]
            if tag_markers.size == 0:
                raise RuntimeError(f"No markers found for tag {tag}.")

            sample_offset = self.marker_offset_samples()
            epochs, used_markers, t_ms = self.make_epochs(analysis_data, markers, tag=tag)
            corrected = self.baseline_correct_epochs(epochs, t_ms)
            first_epoch = corrected[0, :, :]
            first_marker_sample = int(used_markers[0, 0]) - sample_offset

            raw_first_epoch = None
            if self.analysis_view_mode_var.get() == "overlay":
                raw_epochs, _, _ = self.make_epochs(raw_data, markers, tag=tag)
                raw_corrected = self.baseline_correct_epochs(raw_epochs, t_ms)
                raw_first_epoch = raw_corrected[0, :, :]

            max_channels = min(4, len(channels))
            channel_indices = list(range(max_channels))
            if self.time is not None and len(self.time) == self.current_data().shape[0]:
                first_marker_x = (first_marker_sample + sample_offset) / self.fs
            else:
                first_marker_x = first_marker_sample / self.fs

            fig = Figure(figsize=(11.2, 2.25 * max_channels + 1.2), dpi=110)
            axes = fig.subplots(max_channels, 1, squeeze=False, sharex=True).ravel()
            t_sec = t_ms / 1000.0
            for ax_i, (ax, ch_i) in enumerate(zip(axes, channel_indices)):
                if raw_first_epoch is not None:
                    ax.plot(t_sec, raw_first_epoch[:, ch_i], color="0.62", linewidth=0.7, label="raw")
                ax.plot(t_sec, first_epoch[:, ch_i], color="black", linewidth=0.85, label="trial 1")
                ax.axvline(0, color="red", linestyle="--", linewidth=1.0)
                ax.set_title(f"ch{channels[ch_i]} trial 1", fontsize=9)
                ax.set_ylabel("mV", fontsize=8)
                ax.grid(alpha=0.2, linewidth=0.5)
                ax.tick_params(labelsize=8)
                if ax_i == 0 and raw_first_epoch is not None:
                    ax.legend(loc="upper right", fontsize=8)
            axes[-1].set_xlabel("Time to stimulation (second)", fontsize=9)

            fig.suptitle(
                f"First marker alignment check | tag {tag} | first marker={first_marker_x:.6f}s | {preprocess_label}",
                fontsize=11,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            dialog = tk.Toplevel(self)
            dialog.title(f"First marker window - tag {tag}")
            dialog.geometry("1150x820")
            dialog.transient(self.task_window if self.task_window is not None and self.task_window.winfo_exists() else self)
            canvas = FigureCanvasTkAgg(fig, master=dialog)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, dialog, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill="x", padx=10, pady=(8, 0))
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 4))

            def save_marker_response_figure():
                path = filedialog.asksaveasfilename(
                    parent=dialog,
                    title="Save first marker window figure",
                    defaultextension=".png",
                    initialfile=f"first_marker_window_tag_{tag}.png",
                    filetypes=[
                        ("PNG image", "*.png"),
                        ("PDF file", "*.pdf"),
                        ("SVG file", "*.svg"),
                        ("JPEG image", "*.jpg;*.jpeg"),
                        ("All files", "*.*"),
                    ],
                )
                if not path:
                    return
                try:
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    self.log(f"Saved first marker window figure: {path}")
                    messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=dialog)
                except Exception as exc:
                    self.log(traceback.format_exc())
                    messagebox.showerror("Save failed", str(exc), parent=dialog)

            info_text = (
                f"Tag {tag}; channels {', '.join(str(channels[i]) for i in channel_indices)}; "
                f"first marker at {first_marker_x:.3f} sec; "
                f"shown with the same epoch/baseline/display style as trial 1; "
                f"epoch {t_ms[0]:.1f}-{t_ms[-1]:.1f} ms."
            )
            button_bar = ttk.Frame(dialog)
            button_bar.pack(fill="x", padx=10, pady=(0, 10))
            ttk.Label(button_bar, text=info_text, foreground="#445").pack(side="left")
            ttk.Button(button_bar, text="Save figure", command=save_marker_response_figure).pack(side="right")
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("First marker window plot failed", str(exc))

    def format_table(self, headers: list[str], rows: list[list], widths: list[int]) -> list[str]:
        def fmt(value, width):
            if isinstance(value, float):
                if np.isnan(value):
                    text = "nan"
                elif abs(value) >= 1000 or (0 < abs(value) < 0.001):
                    text = f"{value:.2e}"
                else:
                    text = f"{value:.4g}"
            else:
                text = str(value)
            return text[:width].rjust(width)
        header = " ".join(str(h)[:w].rjust(w) for h, w in zip(headers, widths))
        sep = " ".join("-" * w for w in widths)
        lines = [header, sep]
        for row in rows:
            lines.append(" ".join(fmt(v, w) for v, w in zip(row, widths)))
        return lines

    def update_state_nav(self, plot_type: str | None):
        self.current_paged_plot = plot_type
        total_pages = 0
        if plot_type == "resting":
            self.state_nav_info_var.set(self.resting_page_var.get())
            if self.resting_cache:
                total_pages = int(np.ceil(len(self.resting_cache["channels"]) / self.resting_page_size()))
        elif plot_type == "tvep":
            self.state_nav_info_var.set(self.tvep_page_var.get())
            if self.tvep_cache:
                tag_results = self.tvep_cache.get("tag_results", [])
                if tag_results:
                    tag_result = tag_results[min(self.tvep_tag_index, len(tag_results) - 1)]
                    total_pages = int(np.ceil(len(tag_result["channels"]) / self.tvep_page_size()))
        elif plot_type == "stim_task":
            if self.stim_task_cache:
                tag_results = self.stim_task_cache.get("tag_results", [])
                if tag_results:
                    idx = min(self.stim_task_tag_index, len(tag_results) - 1)
                    result = tag_results[idx]
                    self.state_nav_info_var.set(
                        f"Task tag {result['tag']}: {idx + 1}/{len(tag_results)}"
                    )
                    total_pages = len(tag_results)
                else:
                    self.state_nav_info_var.set("")
            else:
                self.state_nav_info_var.set("")
        else:
            self.state_nav_info_var.set("")
        nav_state = "normal" if total_pages > 1 else "disabled"
        self.state_prev_button.configure(state=nav_state)
        self.state_next_button.configure(state=nav_state)
        combo = getattr(self, "tvep_tag_combo", None)
        if combo is not None and plot_type not in ("tvep", "stim_task"):
            combo.configure(state="disabled")

    def update_tvep_tag_choices(self):
        combo = getattr(self, "tvep_tag_combo", None)
        if combo is None:
            return
        cache = self.tvep_cache
        index_attr = "tvep_tag_index"
        suffix = "Hz"
        if self.current_paged_plot == "stim_task":
            cache = self.stim_task_cache
            index_attr = "stim_task_tag_index"
            suffix = "tag"
        if not cache or not cache.get("tag_results"):
            combo.configure(values=(), state="disabled")
            self.tvep_tag_var.set("")
            return
        labels = [f"{result['tag']} {suffix}" for result in cache["tag_results"]]
        index = min(max(getattr(self, index_attr), 0), len(labels) - 1)
        setattr(self, index_attr, index)
        combo.configure(values=labels, state="readonly")
        self.tvep_tag_var.set(labels[index])

    def on_tvep_tag_selected(self, event=None):
        label = self.tvep_tag_var.get()
        if self.current_paged_plot == "stim_task":
            if not self.stim_task_cache or not self.stim_task_cache.get("tag_results"):
                return
            labels = [f"{result['tag']} tag" for result in self.stim_task_cache["tag_results"]]
            if label in labels:
                self.stim_task_tag_index = labels.index(label)
                self.render_stim_task_result()
                if self.auto_plot_marker_alignment_var.get():
                    self.show_marker_response_figure()
            return
        if not self.tvep_cache or not self.tvep_cache.get("tag_results"):
            return
        labels = [f"{result['tag']} Hz" for result in self.tvep_cache["tag_results"]]
        if label in labels:
            self.tvep_tag_index = labels.index(label)
            self.tvep_page_index = 0
            self.render_tvep_page()
            if self.auto_plot_marker_alignment_var.get():
                self.show_marker_response_figure()

    def prev_state_page(self):
        if self.current_paged_plot == "resting":
            self.change_resting_page(-1)
        elif self.current_paged_plot == "tvep":
            self.change_tvep_page(-1)
        elif self.current_paged_plot == "stim_task":
            self.change_stim_task_result(-1)

    def next_state_page(self):
        if self.current_paged_plot == "resting":
            self.change_resting_page(1)
        elif self.current_paged_plot == "tvep":
            self.change_tvep_page(1)
        elif self.current_paged_plot == "stim_task":
            self.change_stim_task_result(1)

    def load_stim_timing(self):
        log_path = self.log_txt_path_var.get().strip()
        if not log_path:
            messagebox.showerror("Missing log.txt", "Select the experiment log.txt first.")
            return
        try:
            timing = self.parse_log_timing(Path(log_path))
            self.timing_info = timing
            markers = None
            csv_path = self.event_csv_path_var.get().strip()
            if csv_path:
                markers = self.build_stim_markers_from_csv(Path(csv_path), timing)
                self.stim_markers = markers
            delay = timing.get("stim_delay_sec")
            lines = [
                "Stimulus timing loaded.",
                "",
                f"log.txt: {log_path}",
                f"Recording ON:  {timing.get('rec_start_text', 'not found')}",
                f"Recording OFF: {timing.get('rec_end_text', 'not found')}",
                f"PROGRAM_START: {timing.get('stim_start_text', 'not found')}",
                f"PROGRAM_STOP:  {timing.get('stim_end_text', 'not found')}",
                f"Raw log delay PROGRAM_START - Recording_ON: {timing.get('stim_delay_raw_sec', np.nan):.6f} sec",
                f"BIN deltaT1 correction added: {timing.get('deltaT1_sec', 0.0) * 1000.0:.3f} ms",
                f"Corrected stim delay used for markers: {delay:.6f} sec" if delay is not None else "Stim delay: not available",
            ]
            if markers is not None:
                preview = markers[:10, :].astype(int)
                lines.extend([
                    "",
                    f"Event CSV: {csv_path}",
                    f"Built stimMarkers: {markers.shape[0]} events x {markers.shape[1]} columns",
                    "First markers [sample, tag]:",
                    np.array2string(preview, separator=", "),
                ])
            else:
                lines.extend([
                    "",
                    "No Event CSV loaded yet. log.txt timing is confirmed, but trial-level markers are not built.",
                ])
            self.write_state_text("\n".join(lines))
            self.show_task_analysis_window()
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Stim timing failed", str(exc))

    def parse_log_timing(self, path: Path) -> dict:
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
        result = {
            "rec_start_text": None,
            "rec_end_text": None,
            "stim_start_text": None,
            "stim_end_text": None,
        }
        for i, line in enumerate(lines):
            prev = lines[i - 1] if i > 0 else line
            if "Recording ON" in line:
                result["rec_start_text"] = self.extract_time_text(prev) or self.extract_time_text(line)
            elif "Recording OFF" in line:
                result["rec_end_text"] = self.extract_time_text(prev) or self.extract_time_text(line)
            elif "PROGRAM_START" in line:
                result["stim_start_text"] = self.extract_time_text(line)
            elif "PROGRAM_STOP" in line:
                result["stim_end_text"] = self.extract_time_text(line)
        if not result["rec_start_text"]:
            raise ValueError("Recording ON time was not found in log.txt.")
        if not result["stim_start_text"]:
            raise ValueError("PROGRAM_START time was not found in log.txt.")
        rec_sec = self.time_text_to_seconds(result["rec_start_text"])
        stim_sec = self.time_text_to_seconds(result["stim_start_text"])
        raw_delay = stim_sec - rec_sec
        if raw_delay < 0:
            raw_delay += 24 * 3600
        delta_t1 = float(getattr(self, "bin_delta_t1_sec", 0.0) or 0.0)
        delay = raw_delay + delta_t1
        result["rec_start_sec"] = rec_sec
        result["stim_start_sec"] = stim_sec
        result["stim_delay_raw_sec"] = raw_delay
        result["deltaT1_sec"] = delta_t1
        result["stim_delay_sec"] = delay
        return result

    def extract_time_text(self, text: str) -> str | None:
        matches = re.findall(r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?", text)
        return matches[-1] if matches else None

    def time_text_to_seconds(self, text: str) -> float:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError(f"Cannot parse time text: {text}")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    def format_clock_seconds(self, value: float) -> str:
        value = float(value) % (24 * 3600)
        hours = int(value // 3600)
        minutes = int((value % 3600) // 60)
        seconds = value % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:09.6f}"

    def build_stim_markers_from_csv(self, path: Path, timing: dict) -> np.ndarray:
        if pd is None:
            raise ImportError("pandas/openpyxl is required to read event CSV files.")
        try:
            table = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            table = pd.read_csv(path, encoding="gbk")
        mode = self.task_mode_var.get()
        if mode == "Flash":
            if table.shape[1] < 6:
                raise ValueError("Flash Event CSV must have at least 6 columns, matching the old MATLAB logic.")
            stim_mode = pd.to_numeric(table.iloc[:, 1], errors="coerce").to_numpy()
            stim_num = pd.to_numeric(table.iloc[:, 3], errors="coerce").fillna(0).astype(int).to_numpy()
            stim_type = pd.to_numeric(table.iloc[:, 1], errors="coerce").to_numpy()
            flash_offset_ms = pd.to_numeric(table.iloc[:, 5], errors="coerce").fillna(0).to_numpy()
            delay_sec = float(timing["stim_delay_sec"])
            samples = []
            tags = []
            for freq, count, tag, offset_ms in zip(stim_mode, stim_num, stim_type, flash_offset_ms):
                if not np.isfinite(freq) or freq <= 0 or count <= 0:
                    continue
                ipi = round(self.fs / freq)
                first_sample = round(self.fs * (delay_sec + offset_ms / 1000.0))
                for ix in range(count):
                    samples.append(first_sample + ipi * ix)
                    tags.append(int(tag) if np.isfinite(tag) else int(freq))
            if not samples:
                raise ValueError("No Flash stim markers could be built from the CSV.")
            markers = np.column_stack([samples, tags]).astype(int)
            return markers[np.argsort(markers[:, 0])]
        if mode == "Letter":
            if table.shape[1] < 15:
                raise ValueError("Letter Event CSV must have at least 15 columns, matching VEPAverager_experiment2_v3.")
            stim_mode = table.iloc[:, 1].astype(str).to_numpy()
            dot_offset_sec = pd.to_numeric(table.iloc[:, 12], errors="coerce").to_numpy()
            letter_offset_sec = pd.to_numeric(table.iloc[:, 13], errors="coerce").to_numpy()
            delay_sec = float(timing["stim_delay_sec"])

            tag_map = {
                "O": 6,
                "|": 4,
                "-": 7,
                "\u95ea\u70c1\u5149\u70b9": 5,
            }
            samples = []
            tags = []
            for mode_text, offset_sec in zip(stim_mode, letter_offset_sec):
                if not np.isfinite(offset_sec):
                    continue
                tag = tag_map.get(normalize_letter_stim_mode(mode_text), 100)
                samples.append(round(self.fs * (delay_sec + float(offset_sec))))
                tags.append(tag)
            for offset_sec in dot_offset_sec:
                if not np.isfinite(offset_sec):
                    continue
                samples.append(round(self.fs * (delay_sec + float(offset_sec))))
                tags.append(100)
            if not samples:
                raise ValueError("No Letter stim markers could be built from the CSV.")
            markers = np.column_stack([samples, tags]).astype(int)
            return markers[np.argsort(markers[:, 0])]
        raise ValueError(f"Unknown task mode: {mode}")

    def selected_segment(self) -> tuple[np.ndarray, list[int], np.ndarray]:
        raw, data, channels, label = self.selected_analysis_data()
        start = max(0, int(parse_float(self.state_start_var.get()) * self.fs))
        dur = max(1, int(parse_float(self.state_duration_var.get(), 5) * self.fs))
        end = min(data.shape[0], start + dur)
        if end <= start:
            raise RuntimeError("Selected state-analysis time range is empty.")
        if self.time is not None and len(self.time) == self.current_data().shape[0]:
            x = np.asarray(self.time[start:end]).ravel()
        else:
            x = np.arange(start, end) / self.fs
        return data[start:end, :], channels, x

    def selected_segment_pair(self) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, str]:
        raw, analysis, channels, label = self.selected_analysis_data()
        start = max(0, int(parse_float(self.state_start_var.get()) * self.fs))
        dur = max(1, int(parse_float(self.state_duration_var.get(), 5) * self.fs))
        end = min(analysis.shape[0], start + dur)
        if end <= start:
            raise RuntimeError("Selected state-analysis time range is empty.")
        if self.time is not None and len(self.time) == self.current_data().shape[0]:
            x = np.asarray(self.time[start:end]).ravel()
        else:
            x = np.arange(start, end) / self.fs
        return raw[start:end, :], analysis[start:end, :], channels, x, label

    def compute_psd(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = np.asarray(data, dtype=DATA_DTYPE)
        if data.ndim == 1:
            data = data[:, None]
        n = data.shape[0]
        if n < 4:
            raise RuntimeError("Not enough samples to compute PSD.")
        window = np.asarray(np.hanning(n), dtype=DATA_DTYPE)[:, None]
        demeaned = data - np.nanmean(data, axis=0, keepdims=True)
        if scifft is None:
            raise ImportError("scipy.fft is required for float32 PSD computation.")
        spec = scifft.rfft(np.asarray(demeaned * window, dtype=DATA_DTYPE), axis=0)
        freqs = np.asarray(scifft.rfftfreq(n, d=1.0 / self.fs), dtype=DATA_DTYPE)
        scale = self.fs * np.sum(window[:, 0] ** 2)
        psd = np.asarray((np.abs(spec) ** 2) / max(scale, np.finfo(DATA_DTYPE).eps), dtype=DATA_DTYPE)
        return freqs, psd

    def run_resting_analysis(self):
        try:
            raw_segment, segment, channels, x, preprocess_label = self.selected_segment_pair()
            rms = np.sqrt(np.nanmean(segment ** 2, axis=0))
            std = np.nanstd(segment, axis=0)
            freqs, psd = self.compute_psd(segment)
            high_mask = (freqs >= 80) & (freqs <= min(300, freqs[-1]))
            line_mask = (freqs >= 48) & (freqs <= 52)
            high_power = np.nanmean(psd[high_mask, :], axis=0) if np.any(high_mask) else np.full(len(channels), np.nan)
            line_power = np.nanmean(psd[line_mask, :], axis=0) if np.any(line_mask) else np.full(len(channels), np.nan)

            self.resting_cache = {
                "raw_segment": raw_segment,
                "segment": segment,
                "channels": channels,
                "x": x,
                "preprocess_label": preprocess_label,
                "rms": rms,
                "std": std,
                "high_power": high_power,
                "line_power": line_power,
            }
            self.resting_page_index = 0
            self.render_resting_page()

            order = np.argsort(high_power)[::-1]
            lines = [
                "Resting analysis",
                "",
                f"Preprocess/display: {preprocess_label}",
                f"Channels: {len(channels)}",
                f"Time range: {x[0]:.3f} - {x[-1]:.3f} sec",
                "",
                "Top channels by high-frequency noise",
            ]
            table_rows = []
            for idx in order[: min(40, len(channels))]:
                table_rows.append([channels[idx], rms[idx], std[idx], high_power[idx], line_power[idx]])
            lines.extend(self.format_table(
                ["ch", "RMS", "std", "80-300Hz", "48-52Hz"],
                table_rows,
                [5, 10, 10, 12, 12],
            ))
            self.write_state_text("\n".join(lines))
            self.show_task_analysis_window()
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Resting analysis failed", str(exc))

    def resting_page_size(self) -> int:
        return max(1, parse_int(self.resting_page_size_var.get(), 12))

    def render_resting_page(self):
        if not self.resting_cache:
            self.resting_page_var.set("Resting page: none")
            return
        cache = self.resting_cache
        channels = cache["channels"]
        total = len(channels)
        page_size = self.resting_page_size()
        total_pages = max(1, int(np.ceil(total / page_size)))
        self.resting_page_index = min(max(self.resting_page_index, 0), total_pages - 1)
        start = self.resting_page_index * page_size
        end = min(total, start + page_size)
        raw_segment = cache["raw_segment"]
        segment = cache["segment"]
        x = cache["x"]
        preprocess_label = cache["preprocess_label"]

        self.state_fig.clear()
        count = end - start
        cols = min(4, count)
        rows = int(np.ceil(count / cols))
        for local_i, ch_i in enumerate(range(start, end)):
            ax = self.state_fig.add_subplot(rows, cols, local_i + 1)
            if self.analysis_view_mode_var.get() == "overlay":
                ax.plot(x, raw_segment[:, ch_i], linewidth=0.45, color="0.65", label="raw")
                ax.plot(x, segment[:, ch_i], linewidth=0.6, color="tab:blue", label="filtered")
            else:
                ax.plot(x, segment[:, ch_i], linewidth=0.55)
            ax.set_title(f"ch{channels[ch_i]}", fontsize=9)
            ax.tick_params(labelsize=7)
            if local_i == 0 and self.analysis_view_mode_var.get() == "overlay":
                ax.legend(fontsize=7, loc="upper right")
        self.state_fig.suptitle(
            f"Resting selected-channel grid ({preprocess_label}) "
            f"page {self.resting_page_index + 1}/{total_pages}",
            fontsize=9,
        )
        self.state_fig.tight_layout()
        self.state_canvas.draw()
        self.resting_page_var.set(
            f"Resting page: {self.resting_page_index + 1}/{total_pages} "
            f"channels {start + 1}-{end} of {total}"
        )
        self.update_state_nav("resting")

    def change_resting_page(self, delta: int):
        if not self.resting_cache:
            messagebox.showinfo("No resting result", "Run Resting analysis first.")
            return
        self.resting_page_index += delta
        self.render_resting_page()

    def require_stim_markers(self) -> np.ndarray:
        if self.stim_markers is None:
            raise RuntimeError("stimMarkers are not built. Load log.txt and Event CSV first.")
        return np.asarray(self.stim_markers, dtype=int)

    def marker_offset_samples(self) -> int:
        if self.time is not None and len(self.time) == self.current_data().shape[0]:
            return int(round(float(np.asarray(self.time).ravel()[0]) * self.fs))
        return 0

    def make_epochs(self, channels_data: np.ndarray, markers: np.ndarray, tag: int | None = None):
        start_ms = parse_float(self.epoch_start_ms_var.get(), -500)
        end_ms = parse_float(self.epoch_end_ms_var.get(), 800)
        start_offset = int(round(start_ms * self.fs / 1000.0))
        end_offset = int(round(end_ms * self.fs / 1000.0))
        if end_offset <= start_offset:
            raise RuntimeError("Epoch end must be after epoch start.")
        offset = self.marker_offset_samples()
        use_markers = markers if tag is None else markers[markers[:, 1] == tag]
        max_trials = parse_int(self.trials_per_stim_var.get(), 0)
        epochs = []
        used = []
        for marker in use_markers:
            center = int(marker[0]) - offset
            s = center + start_offset
            e = center + end_offset
            if s < 0 or e > channels_data.shape[0]:
                continue
            epochs.append(channels_data[s:e, :])
            used.append(marker)
            if max_trials > 0 and len(epochs) >= max_trials:
                break
        if not epochs:
            raise RuntimeError("No valid epochs were found in the loaded data segment.")
        epochs = np.stack(epochs, axis=0)
        used = np.asarray(used, dtype=int)
        t_ms = np.arange(start_offset, end_offset) / self.fs * 1000.0
        return epochs, used, t_ms

    def parse_stimtags(self) -> list[int]:
        tags = []
        for part in self.stimtag_var.get().replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            tags.append(int(float(part)))
        return tags

    def selected_stim_tags(self, markers: np.ndarray) -> list[int]:
        available = [int(x) for x in np.unique(markers[:, 1])]
        requested = self.parse_stimtags()
        if requested:
            tags = [tag for tag in requested if tag in available]
        else:
            tags = available
        n_types = parse_int(self.n_stim_types_var.get(), len(tags))
        if n_types > 0:
            tags = tags[:n_types]
        if not tags:
            raise RuntimeError(f"No requested stimtag values are present in stimMarkers. Available: {available}")
        return tags

    def baseline_correct_epochs(self, epochs: np.ndarray, t_ms: np.ndarray) -> np.ndarray:
        b0 = parse_float(self.baseline_start_ms_var.get(), -200)
        b1 = parse_float(self.baseline_end_ms_var.get(), 0)
        mask = (t_ms >= b0) & (t_ms <= b1)
        if not np.any(mask):
            return epochs
        base = np.nanmean(epochs[:, mask, :], axis=1, keepdims=True)
        return epochs - base

    def task_response_aggregate(self, data: np.ndarray, axis=0) -> np.ndarray:
        method = self.task_response_aggregate_var.get().strip().lower()
        if method == "median":
            return np.nanmedian(data, axis=axis)
        return np.nanmean(data, axis=axis)

    def task_response_aggregate_label(self) -> str:
        method = self.task_response_aggregate_var.get().strip().lower()
        return "median" if method == "median" else "mean"

    def response_metrics(self, epochs: np.ndarray, t_ms: np.ndarray, channels: list[int]) -> list[dict]:
        corrected = self.baseline_correct_epochs(epochs, t_ms)
        mean_wave = self.task_response_aggregate(corrected, axis=0)
        r0 = parse_float(self.response_start_ms_var.get(), 0)
        r1 = parse_float(self.response_end_ms_var.get(), 300)
        b0 = parse_float(self.baseline_start_ms_var.get(), -200)
        b1 = parse_float(self.baseline_end_ms_var.get(), 0)
        resp_mask = (t_ms >= r0) & (t_ms <= r1)
        base_mask = (t_ms >= b0) & (t_ms <= b1)
        metrics = []
        for i, ch in enumerate(channels):
            wave = mean_wave[:, i]
            resp = wave[resp_mask]
            resp_t = t_ms[resp_mask]
            if resp.size == 0:
                continue
            peak_i = int(np.nanargmax(np.abs(resp)))
            peak_amp = float(resp[peak_i])
            peak_abs = float(abs(resp[peak_i]))
            peak_latency = float(resp_t[peak_i])
            base_vals = corrected[:, base_mask, i].ravel() if np.any(base_mask) else corrected[:, :, i].ravel()
            sigma = float(np.nanstd(base_vals))
            threshold = 3 * sigma
            onset = np.nan
            if threshold > 0:
                above = np.abs(resp) > threshold
                run = max(1, int(round(0.010 * self.fs)))
                for j in range(0, max(0, above.size - run + 1)):
                    if np.all(above[j:j + run]):
                        onset = float(resp_t[j])
                        break
            metrics.append({
                "channel": ch,
                "peak_amp": peak_amp,
                "peak_abs": peak_abs,
                "peak_latency_ms": peak_latency,
                "onset_ms": onset,
                "baseline_std": sigma,
                "significant": bool(np.isfinite(onset)),
            })
        return metrics

    def trial_robustness_metrics(self, corrected: np.ndarray, t_ms: np.ndarray) -> dict:
        trial_a = max(1, parse_int(self.compare_trial_a_var.get(), 30))
        trial_b = max(trial_a + 1, parse_int(self.compare_trial_b_var.get(), 60))
        valid_b = min(trial_b, corrected.shape[0])
        if corrected.shape[0] < trial_a:
            return {
                "available": corrected.shape[0],
                "trial_a": trial_a,
                "trial_b": trial_b,
                "valid_b": valid_b,
                "corr": np.nan,
                "peak_a": np.nan,
                "peak_b": np.nan,
                "change": np.nan,
                "status": f"need at least {trial_a} trials",
            }
        avg_a = self.task_response_aggregate(corrected[:trial_a, :, :], axis=(0, 2))
        avg_b = self.task_response_aggregate(corrected[:valid_b, :, :], axis=(0, 2))
        finite = np.isfinite(avg_a) & np.isfinite(avg_b)
        corr = float(np.corrcoef(avg_a[finite], avg_b[finite])[0, 1]) if finite.sum() > 2 else np.nan
        peak_a = float(np.nanmax(np.abs(avg_a)))
        peak_b = float(np.nanmax(np.abs(avg_b)))
        change = (peak_b - peak_a) / max(abs(peak_a), np.finfo(DATA_DTYPE).eps) * 100
        return {
            "available": corrected.shape[0],
            "trial_a": trial_a,
            "trial_b": trial_b,
            "valid_b": valid_b,
            "corr": corr,
            "peak_a": peak_a,
            "peak_b": peak_b,
            "change": change,
            "status": "ok",
        }

    def run_tvep_analysis(self):
        try:
            raw_data, data, channels, preprocess_label = self.selected_analysis_data()
            markers = self.require_stim_markers()
            tags = self.selected_stim_tags(markers)
            tag_results: list[dict] = []
            skipped: list[str] = []
            for tag in tags:
                tag = int(tag)
                try:
                    epochs, used, t_ms = self.make_epochs(data, markers, tag=tag)
                    corrected = self.baseline_correct_epochs(epochs, t_ms)
                    mean_wave = self.task_response_aggregate(corrected, axis=0)
                    metrics = self.response_metrics(epochs, t_ms, channels)
                    robustness = self.trial_robustness_metrics(corrected, t_ms)
                    event_lfp_snr_rows = []
                    event_spike_snr_rows = []
                    if getattr(self, "task_analysis_source", "lfp") == "lfp":
                        event_lfp_snr_rows = self.event_locked_lfp_snr_rows(epochs, t_ms, channels, tag)
                    elif getattr(self, "task_analysis_source", "lfp") == "spike":
                        event_spike_snr_rows = self.event_locked_spike_rate_snr_rows(data, used, t_ms, channels, tag)

                    raw_mean_wave = None
                    if self.analysis_view_mode_var.get() == "overlay":
                        raw_epochs, _, _ = self.make_epochs(raw_data, markers, tag=tag)
                        raw_corrected = self.baseline_correct_epochs(raw_epochs, t_ms)
                        raw_mean_wave = self.task_response_aggregate(raw_corrected, axis=0)
                    tag_results.append({
                        "tag": tag,
                        "mean_wave": mean_wave,
                        "raw_mean_wave": raw_mean_wave,
                        "channels": channels,
                        "t_sec": t_ms / 1000.0,
                        "t_ms": t_ms,
                        "trial_count": epochs.shape[0],
                        "metrics": metrics,
                        "robustness": robustness,
                        "event_lfp_snr_rows": event_lfp_snr_rows,
                        "event_spike_snr_rows": event_spike_snr_rows,
                    })
                except Exception as tag_exc:
                    skipped.append(f"{tag} Hz: {tag_exc}")
            if not tag_results:
                detail = "\n".join(skipped) if skipped else "No requested Flash frequencies produced valid epochs."
                raise RuntimeError(f"No valid Flash/TVEP frequency result was found.\n{detail}")

            self.tvep_cache = {
                "tag_results": tag_results,
                "preprocess_label": preprocess_label,
            }
            self.tvep_tag_index = 0
            self.tvep_page_index = 0
            self.update_tvep_tag_choices()
            self.render_tvep_page()

            lines = [
                "TVEP / Flash analysis",
                "",
                f"Preprocess/display: {preprocess_label}",
                f"Response aggregate: {self.task_response_aggregate_label()}",
                f"nStimTypes: {self.n_stim_types_var.get()}",
                f"trialsPerStim: {self.trials_per_stim_var.get()}",
                f"stimtag: {self.stimtag_var.get()}",
                f"Tags analyzed: {', '.join(str(result['tag']) for result in tag_results)}",
                f"Channels: {len(channels)}",
                f"Epoch: {tag_results[0]['t_ms'][0]:.1f} to {tag_results[0]['t_ms'][-1]:.1f} ms",
                "",
                "Frequency summary",
            ]
            summary_rows = []
            for result in tag_results:
                rob = result["robustness"]
                metrics = result["metrics"]
                top_peak = max((m["peak_abs"] for m in metrics), default=np.nan)
                event_rows = (
                    result.get("event_spike_snr_rows", [])
                    if getattr(self, "task_analysis_source", "lfp") == "spike"
                    else result.get("event_lfp_snr_rows", [])
                )
                event_snr_vals = np.asarray([
                    float(row["snr_db"]) for row in event_rows
                    if np.isfinite(float(row.get("snr_db", np.nan)))
                ], dtype=DATA_DTYPE)
                event_snr_med = float(np.nanmedian(event_snr_vals)) if event_snr_vals.size else np.nan
                summary_rows.append([
                    result["tag"],
                    result["trial_count"],
                    rob["trial_a"],
                    rob["valid_b"],
                    rob["corr"],
                    top_peak,
                    event_snr_med,
                    rob["status"],
                ])
            lines.extend(self.format_table(
                ["Hz", "trials", "A", "B", "corr", "top abs", "event SNR", "status"],
                summary_rows,
                [5, 8, 5, 5, 8, 10, 10, 18],
            ))
            if skipped:
                lines.extend(["", "Skipped requested frequencies"])
                lines.extend(skipped)
            first_result = tag_results[0]
            robustness = first_result["robustness"]
            metrics = first_result["metrics"]
            lines.extend([
                "",
                f"Current frequency detail: {first_result['tag']} Hz",
                "Trial-count robustness",
                f"Compare: first {robustness['trial_a']} vs first {robustness['valid_b']} trials",
                f"Status: {robustness['status']}",
                f"Corr: {robustness['corr']:.4g}",
                f"Peak abs A: {robustness['peak_a']:.4g}",
                f"Peak abs B: {robustness['peak_b']:.4g}",
                f"Peak change: {robustness['change']:.3f}%",
                "",
                "Top channels by absolute peak for current frequency",
            ])
            table_rows = []
            for m in sorted(metrics, key=lambda x: x["peak_abs"], reverse=True)[:60]:
                table_rows.append([
                    m["channel"],
                    m["peak_amp"],
                    m["peak_abs"],
                    m["peak_latency_ms"],
                    m["onset_ms"],
                    m["baseline_std"],
                    "yes" if m["significant"] else "no",
                ])
            lines.extend(self.format_table(
                ["ch", "peak", "abs", "lat ms", "onset", "base sd", "sig"],
                table_rows,
                [5, 10, 10, 8, 8, 10, 5],
            ))
            self.write_state_text("\n".join(lines))
            self.show_task_analysis_window()
            self.show_task_source_snr_figure("Flash_Task")
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("TVEP analysis failed", str(exc))

    def tvep_page_size(self) -> int:
        return max(1, parse_int(self.tvep_page_size_var.get(), 12))

    def render_tvep_page(self):
        if not self.tvep_cache:
            self.tvep_page_var.set("TVEP page: none")
            self.update_tvep_tag_choices()
            return
        cache = self.tvep_cache
        tag_results = cache.get("tag_results", [])
        if not tag_results:
            self.tvep_page_var.set("TVEP page: none")
            self.update_tvep_tag_choices()
            return
        self.tvep_tag_index = min(max(self.tvep_tag_index, 0), len(tag_results) - 1)
        tag_result = tag_results[self.tvep_tag_index]
        channels = tag_result["channels"]
        total = len(channels)
        page_size = self.tvep_page_size()
        total_pages = max(1, int(np.ceil(total / page_size)))
        self.tvep_page_index = min(max(self.tvep_page_index, 0), total_pages - 1)
        start = self.tvep_page_index * page_size
        end = min(total, start + page_size)
        count = end - start
        cols = min(4, count)
        rows = int(np.ceil(count / cols))
        t_sec = tag_result["t_sec"]
        mean_wave = tag_result["mean_wave"]
        raw_mean_wave = tag_result["raw_mean_wave"]

        self.state_fig.clear()
        for local_i, ch_i in enumerate(range(start, end)):
            ax = self.state_fig.add_subplot(rows, cols, local_i + 1)
            ax._tvep_channel_index = ch_i
            ax._tvep_channel_label = channels[ch_i]
            if raw_mean_wave is not None:
                ax.plot(t_sec, raw_mean_wave[:, ch_i], color="0.55", linewidth=0.9, label="raw")
            ax.plot(t_sec, mean_wave[:, ch_i], color="black", linewidth=1.1, label=self.task_response_aggregate_label())
            ax.axvline(0, color="red", linestyle="--", linewidth=1.8)
            ax.set_title(f"ch{channels[ch_i]}", fontsize=9)
            ax.tick_params(labelsize=7)
            if local_i % cols == 0:
                ax.set_ylabel("Amplitude (mV)", fontsize=8)
            if local_i >= count - cols:
                ax.set_xlabel("Time to stimulation (second)", fontsize=8)
            if local_i == 0 and raw_mean_wave is not None:
                ax.legend(fontsize=7, loc="upper right")
        self.state_fig.suptitle(
            f"Flash frequency = {tag_result['tag']} Hz | trials={tag_result['trial_count']} | "
            f"aggregate={self.task_response_aggregate_label()} | "
            f"freq {self.tvep_tag_index + 1}/{len(tag_results)} | page {self.tvep_page_index + 1}/{total_pages}",
            fontsize=9,
            color="red",
            fontweight="bold",
        )
        self.state_fig.tight_layout()
        self.state_canvas.draw()
        self.tvep_page_var.set(
            f"TVEP freq {tag_result['tag']} Hz: page {self.tvep_page_index + 1}/{total_pages} "
            f"channels {start + 1}-{end} of {total}"
        )
        self.update_tvep_tag_choices()
        self.update_state_nav("tvep")

    def on_state_figure_click(self, event):
        if event.button != 1 or event.inaxes is None:
            return
        if self.current_paged_plot != "tvep":
            return
        ch_i = getattr(event.inaxes, "_tvep_channel_index", None)
        if ch_i is None:
            return
        self.show_tvep_channel_trial_grid(int(ch_i))

    def show_tvep_channel_trial_grid(self, channel_index: int):
        if not self.tvep_cache or not self.tvep_cache.get("tag_results"):
            messagebox.showinfo("No TVEP result", "Run TVEP / Flash analysis first.")
            return
        if self.stim_markers is None:
            messagebox.showerror("No markers", "Load log.txt and Event CSV first to build stimMarkers.")
            return
        try:
            tag_results = self.tvep_cache["tag_results"]
            self.tvep_tag_index = min(max(self.tvep_tag_index, 0), len(tag_results) - 1)
            tag_result = tag_results[self.tvep_tag_index]
            cached_channels = tag_result["channels"]
            if channel_index < 0 or channel_index >= len(cached_channels):
                raise RuntimeError("Clicked channel is outside the current TVEP result.")
            channel_label = int(cached_channels[channel_index])

            raw_data, analysis_data, channels, preprocess_label = self.selected_analysis_data()
            if channel_label not in channels:
                raise RuntimeError(
                    f"ch{channel_label} is no longer in the selected channel set. "
                    "Re-run Flash_Task or restore the same selected channels."
                )
            plot_col = channels.index(channel_label)
            markers = self.require_stim_markers()
            tag = int(tag_result["tag"])
            epochs, used_markers, t_ms = self.make_epochs(analysis_data, markers, tag=tag)
            corrected = self.baseline_correct_epochs(epochs, t_ms)
            trial_count = corrected.shape[0]
            if trial_count == 0:
                raise RuntimeError(f"No valid trials found for tag {tag}.")

            raw_corrected = None
            if self.analysis_view_mode_var.get() == "overlay":
                raw_epochs, _, _ = self.make_epochs(raw_data, markers, tag=tag)
                raw_corrected = self.baseline_correct_epochs(raw_epochs, t_ms)

            cols = min(5, trial_count)
            rows = int(np.ceil(trial_count / cols))
            fig = Figure(figsize=(3.0 * cols, 1.75 * rows + 0.9), dpi=110)
            axes = fig.subplots(rows, cols, squeeze=False, sharex=True, sharey=False).ravel()
            t_sec = t_ms / 1000.0
            for trial_i in range(trial_count):
                ax = axes[trial_i]
                if raw_corrected is not None and trial_i < raw_corrected.shape[0]:
                    ax.plot(t_sec, raw_corrected[trial_i, :, plot_col], color="0.62", linewidth=0.7, label="raw")
                ax.plot(t_sec, corrected[trial_i, :, plot_col], color="black", linewidth=0.85, label="trial")
                ax.axvline(0, color="red", linestyle="--", linewidth=1.0)
                ax.set_title(f"trial {trial_i + 1}", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.2, linewidth=0.5)
                if trial_i % cols == 0:
                    ax.set_ylabel("mV", fontsize=7)
                if trial_i >= trial_count - cols:
                    ax.set_xlabel("s", fontsize=7)
                if trial_i == 0 and raw_corrected is not None:
                    ax.legend(fontsize=7, loc="upper right")
            for ax in axes[trial_count:]:
                ax.set_visible(False)
            fig.suptitle(
                f"ch{channel_label} single trials | Flash frequency {tag} Hz | "
                f"trials={trial_count} | {preprocess_label}",
                fontsize=10,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            dialog = tk.Toplevel(self)
            dialog.title(f"ch{channel_label} trials - {tag} Hz")
            dialog.geometry("1250x850")
            dialog.transient(self.task_window if self.task_window is not None and self.task_window.winfo_exists() else self)
            canvas = FigureCanvasTkAgg(fig, master=dialog)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, dialog, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill="x", padx=10, pady=(8, 0))
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 4))

            def save_trial_grid():
                path = filedialog.asksaveasfilename(
                    parent=dialog,
                    title="Save single-trial grid figure",
                    defaultextension=".png",
                    initialfile=f"ch{channel_label}_{tag}Hz_single_trials.png",
                    filetypes=[
                        ("PNG image", "*.png"),
                        ("PDF file", "*.pdf"),
                        ("SVG file", "*.svg"),
                        ("JPEG image", "*.jpg;*.jpeg"),
                        ("All files", "*.*"),
                    ],
                )
                if not path:
                    return
                try:
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    self.log(f"Saved single-trial grid figure: {path}")
                    messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=dialog)
                except Exception as exc:
                    self.log(traceback.format_exc())
                    messagebox.showerror("Save failed", str(exc), parent=dialog)

            info_text = (
                f"Click source: ch{channel_label}, tag {tag} Hz; "
                f"epoch {t_ms[0]:.1f}-{t_ms[-1]:.1f} ms; "
                "red dashed line is marker time."
            )
            button_bar = ttk.Frame(dialog)
            button_bar.pack(fill="x", padx=10, pady=(0, 10))
            ttk.Label(button_bar, text=info_text, foreground="#445").pack(side="left")
            ttk.Button(button_bar, text="Save figure", command=save_trial_grid).pack(side="right")
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Single-trial plot failed", str(exc))

    def change_tvep_page(self, delta: int):
        if not self.tvep_cache:
            messagebox.showinfo("No TVEP result", "Run TVEP / Flash analysis first.")
            return
        self.tvep_page_index += delta
        self.render_tvep_page()

    def marker_segment_for_tag(self, data: np.ndarray, markers: np.ndarray, tag: int) -> np.ndarray:
        tag_markers = markers[markers[:, 1] == tag]
        if tag_markers.size == 0:
            raise RuntimeError(f"No markers for tag {tag}.")
        offset = self.marker_offset_samples()
        start = max(0, int(tag_markers[0, 0]) - offset)
        end = min(data.shape[0], int(tag_markers[-1, 0]) - offset + int(round(self.fs)))
        if end <= start:
            raise RuntimeError(f"Tag {tag} is outside the loaded data segment.")
        return data[start:end, :]

    def run_ssvep_analysis(self):
        try:
            raw_data, data, channels, preprocess_label = self.selected_analysis_data()
            markers = self.require_stim_markers()
            target_freq = parse_float(self.stim_freq_var.get(), 10.0)
            target_tag = int(round(target_freq))
            available_tags = sorted({int(round(float(tag))) for tag in np.unique(markers[:, 1])})
            if target_tag not in available_tags:
                raise RuntimeError(
                    f"stim freq {target_freq:g} Hz was not found in stimMarkers. "
                    f"Available marker tags: {available_tags}"
                )
            seg = self.marker_segment_for_tag(data, markers, target_tag)
            if seg.shape[0] < int(self.fs * 0.5):
                raise RuntimeError(f"Tag {target_tag} segment is too short or outside the loaded data.")

            self.state_fig.clear()
            ax_psd = self.state_fig.add_subplot(2, 1, 1)
            report = [
                "SSVEP analysis",
                "",
                f"Preprocess/display: {preprocess_label}",
                f"Target frequency: {target_freq:g} Hz",
                f"Available marker tags: {available_tags}",
                "",
            ]
            freqs, psd = self.compute_psd(seg)
            mean_psd = np.nanmean(psd, axis=1)
            ax_psd.plot(freqs, 10 * np.log10(mean_psd + np.finfo(DATA_DTYPE).eps), linewidth=1.0, label=f"{target_freq:g} Hz")
            target_mask = np.abs(freqs - target_freq) <= 0.25
            neigh_mask = (np.abs(freqs - target_freq) > 0.5) & (np.abs(freqs - target_freq) <= 2.0)
            target_power = float(np.nanmean(mean_psd[target_mask])) if np.any(target_mask) else np.nan
            neigh_power = float(np.nanmean(mean_psd[neigh_mask])) if np.any(neigh_mask) else np.nan
            snr = 10 * np.log10((target_power + np.finfo(DATA_DTYPE).eps) / (neigh_power + np.finfo(DATA_DTYPE).eps))
            ax_psd.set_xlim(0, min(80, self.fs / 2))
            ax_psd.set_title(f"PSD for {target_freq:g} Hz SSVEP condition")
            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_ylabel("Power (dB)")
            ax_psd.legend(fontsize=8)

            ax_tf = self.state_fig.add_subplot(2, 1, 2)
            mean_signal = np.nanmean(seg, axis=1)
            nfft = min(512, max(64, len(mean_signal) // 8))
            noverlap = min(nfft // 2, nfft - 1)
            ax_tf.specgram(mean_signal, NFFT=nfft, Fs=self.fs, noverlap=noverlap, cmap="viridis")
            ax_tf.set_ylim(0, min(80, self.fs / 2))
            ax_tf.set_title(f"Spectrogram for {target_freq:g} Hz condition")
            ax_tf.set_xlabel("Time (sec)")
            ax_tf.set_ylabel("Frequency (Hz)")
            self.state_fig.tight_layout()
            self.state_canvas.draw()
            self.update_state_nav(None)
            report.append("Frequency-condition SNR")
            report.extend(self.format_table(
                ["tag", "target", "neighbor", "snr dB", "sec"],
                [[target_tag, target_power, neigh_power, snr, seg.shape[0] / self.fs]],
                [5, 11, 11, 9, 8],
            ))
            self.write_state_text("\n".join(report))
            self.show_task_analysis_window()
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("SSVEP analysis failed", str(exc))

    def run_stim_task_analysis(self):
        try:
            raw_data, data, channels, preprocess_label = self.selected_analysis_data()
            markers = self.require_stim_markers()
            tags = self.selected_stim_tags(markers)
            trial_a = max(1, parse_int(self.compare_trial_a_var.get(), 30))
            trial_b = max(trial_a + 1, parse_int(self.compare_trial_b_var.get(), 60))
            original_trials = self.trials_per_stim_var.get()
            tag_results: list[dict] = []
            skipped: list[str] = []
            self.trials_per_stim_var.set(str(trial_b))
            try:
                for tag in tags:
                    tag = int(tag)
                    try:
                        epochs, used, t_ms = self.make_epochs(data, markers, tag=tag)
                        corrected = self.baseline_correct_epochs(epochs, t_ms)
                        if corrected.shape[0] < trial_a:
                            raise RuntimeError(
                                f"Only {corrected.shape[0]} valid trials are available; need at least {trial_a}."
                            )
                        mean_wave = self.task_response_aggregate(corrected, axis=0)
                        raw_mean_wave = None
                        if self.analysis_view_mode_var.get() == "overlay":
                            raw_epochs, _, _ = self.make_epochs(raw_data, markers, tag=tag)
                            raw_corrected = self.baseline_correct_epochs(raw_epochs, t_ms)
                            raw_mean_wave = self.task_response_aggregate(raw_corrected, axis=0)
                        event_lfp_snr_rows = []
                        event_spike_snr_rows = []
                        if getattr(self, "task_analysis_source", "lfp") == "lfp":
                            event_lfp_snr_rows = self.event_locked_lfp_snr_rows(epochs, t_ms, channels, tag)
                        elif getattr(self, "task_analysis_source", "lfp") == "spike":
                            event_spike_snr_rows = self.event_locked_spike_rate_snr_rows(data, used, t_ms, channels, tag)
                        valid_b = min(trial_b, corrected.shape[0])
                        avg_a = self.task_response_aggregate(corrected[:trial_a, :, :], axis=(0, 2))
                        avg_b = self.task_response_aggregate(corrected[:valid_b, :, :], axis=(0, 2))
                        finite = np.isfinite(avg_a) & np.isfinite(avg_b)
                        corr = float(np.corrcoef(avg_a[finite], avg_b[finite])[0, 1]) if finite.sum() > 2 else np.nan
                        peak_a = float(np.nanmax(np.abs(avg_a)))
                        peak_b = float(np.nanmax(np.abs(avg_b)))
                        change = (peak_b - peak_a) / max(abs(peak_a), np.finfo(DATA_DTYPE).eps) * 100
                        tag_results.append({
                            "tag": tag,
                            "t_ms": t_ms,
                            "t_sec": t_ms / 1000.0,
                            "mean_wave": mean_wave,
                            "raw_mean_wave": raw_mean_wave,
                            "channels": channels,
                            "avg_a": avg_a,
                            "avg_b": avg_b,
                            "diff": avg_b - avg_a,
                            "trial_a": trial_a,
                            "trial_b": trial_b,
                            "valid_b": valid_b,
                            "valid_trials": corrected.shape[0],
                            "corr": corr,
                            "peak_a": peak_a,
                            "peak_b": peak_b,
                            "change": change,
                            "event_lfp_snr_rows": event_lfp_snr_rows,
                            "event_spike_snr_rows": event_spike_snr_rows,
                        })
                    except Exception as tag_exc:
                        skipped.append(f"{tag}: {tag_exc}")
            finally:
                self.trials_per_stim_var.set(original_trials)
            if not tag_results:
                detail = "\n".join(skipped) if skipped else "No requested tags produced valid task epochs."
                raise RuntimeError(f"No valid LettermodeData task result was found.\n{detail}")

            self.stim_task_cache = {
                "tag_results": tag_results,
                "preprocess_label": preprocess_label,
            }
            self.stim_task_tag_index = 0
            self.render_stim_task_result()

            lines = [
                "Stimulus / Task trial consistency",
                "",
                f"Preprocess/display: {preprocess_label}",
                f"Response aggregate: {self.task_response_aggregate_label()}",
                f"nStimTypes: {self.n_stim_types_var.get()}",
                f"trialsPerStim: {original_trials}",
                f"stimtag: {self.stimtag_var.get()}",
                f"Tags analyzed: {', '.join(str(result['tag']) for result in tag_results)}",
                "",
                "Tag summary",
            ]
            lines.extend(self.format_table(
                ["tag", "trials", "A", "B", "corr", "peak A", "peak B", "event SNR", "chg%"],
                [
                    [
                        result["tag"],
                        result["valid_trials"],
                        result["trial_a"],
                        result["valid_b"],
                        result["corr"],
                        result["peak_a"],
                        result["peak_b"],
                        (
                            float(np.nanmedian([
                                float(row["snr_db"]) for row in (
                                    result.get("event_spike_snr_rows", [])
                                    if getattr(self, "task_analysis_source", "lfp") == "spike"
                                    else result.get("event_lfp_snr_rows", [])
                                )
                                if np.isfinite(float(row.get("snr_db", np.nan)))
                            ]))
                            if (
                                result.get("event_spike_snr_rows", [])
                                if getattr(self, "task_analysis_source", "lfp") == "spike"
                                else result.get("event_lfp_snr_rows", [])
                            ) else np.nan
                        ),
                        result["change"],
                    ]
                    for result in tag_results
                ],
                [6, 8, 5, 5, 8, 10, 10, 10, 8],
            ))
            if skipped:
                lines.extend(["", "Skipped requested tags"])
                lines.extend(skipped)
            lines.extend([
                "",
                "Suggested first-pass consistency rule:",
                "corr > 0.8, peak_change_percent within +/-20%, latency visually stable.",
            ])
            self.write_state_text("\n".join(lines))
            self.show_task_analysis_window()
            self.show_task_source_snr_figure("LettermodeData")
            self.show_letter_average_response_window()
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Stimulus task analysis failed", str(exc))

    def render_stim_task_result(self):
        if not self.stim_task_cache or not self.stim_task_cache.get("tag_results"):
            self.update_state_nav(None)
            return
        results = self.stim_task_cache["tag_results"]
        self.stim_task_tag_index = min(max(self.stim_task_tag_index, 0), len(results) - 1)
        result = results[self.stim_task_tag_index]
        t_ms = result["t_ms"]
        self.state_fig.clear()
        ax1 = self.state_fig.add_subplot(2, 1, 1)
        aggregate_label = self.task_response_aggregate_label()
        ax1.plot(t_ms, result["avg_a"], label=f"{aggregate_label}: first {result['trial_a']} trials", linewidth=1.5)
        ax1.plot(t_ms, result["avg_b"], label=f"{aggregate_label}: first {result['valid_b']} trials", linewidth=1.5)
        ax1.axvline(0, color="k", linewidth=0.8)
        ax1.set_title(
            f"Trial-count consistency, tag {result['tag']} "
            f"({self.stim_task_tag_index + 1}/{len(results)})"
        )
        ax1.set_xlabel("Time (ms)")
        ax1.legend()
        ax2 = self.state_fig.add_subplot(2, 1, 2)
        ax2.plot(t_ms, result["diff"], color="tab:purple", linewidth=1.2)
        ax2.axhline(0, color="k", linewidth=0.8)
        ax2.axvline(0, color="k", linewidth=0.8)
        ax2.set_title(
            f"Difference: larger-trial {self.task_response_aggregate_label()} minus smaller-trial "
            f"{self.task_response_aggregate_label()} | "
            f"corr={result['corr']:.4g}, change={result['change']:.3f}%"
        )
        ax2.set_xlabel("Time (ms)")
        self.state_fig.tight_layout()
        self.state_canvas.draw()
        self.update_tvep_tag_choices()
        self.update_state_nav("stim_task")

    def read_behavior_detection_results(self, path: Path) -> tuple[float, float, list[dict]]:
        text = None
        for encoding in ("utf-8-sig", "gbk", "utf-16"):
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeError:
                continue
        if text is None:
            text = path.read_text(errors="ignore")

        frame_rate = 30.0
        recording_time_sec = np.nan
        events = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("recordingtime"):
                time_text = self.extract_time_text(stripped)
                if time_text:
                    recording_time_sec = self.time_text_to_seconds(time_text)
                continue
            if stripped.lower().startswith("framerate"):
                values = re.findall(r"[-+]?\d+(?:\.\d+)?", stripped)
                if values:
                    frame_rate = float(values[-1])
                continue
            if not re.match(r"^[-+]?\d+\s*,", stripped):
                continue
            fields = [field.strip() for field in stripped.split(",")]
            if len(fields) < 5:
                continue
            try:
                tag = int(float(fields[0]))
                start_frame = int(float(fields[1]))
                end_frame = int(float(fields[2]))
            except (TypeError, ValueError):
                continue
            animal_start = None
            if fields[3].upper() not in {"", "NA", "NAN", "NONE"}:
                try:
                    animal_start = int(float(fields[3]))
                except (TypeError, ValueError):
                    animal_start = None
            events.append({
                "tag": tag,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "animal_start_frame": animal_start,
                "validity": fields[4].upper(),
            })
        if not events:
            raise RuntimeError("No detection events were found in the selected results file.")
        if not np.isfinite(frame_rate) or frame_rate <= 0:
            raise RuntimeError("Detection results contain an invalid frame rate.")
        return float(frame_rate), float(recording_time_sec), events

    def match_behavior_events_to_markers(
        self,
        markers: np.ndarray,
        frame_rate: float,
        video_recording_time_sec: float,
        eeg_recording_time_sec: float,
        detection_events: list[dict],
    ) -> dict[int, dict]:
        letter_markers = [
            row for row in np.asarray(markers, dtype=int)
            if int(row[1]) in {4, 5, 6, 7}
        ]
        letter_markers.sort(key=lambda row: int(row[0]))
        if not letter_markers:
            return {}
        first_marker = letter_markers[0]
        first_marker_sec = int(first_marker[0]) / self.fs
        first_tag = int(first_marker[1])
        first_event_index = next(
            (
                index
                for index, event in sorted(
                    enumerate(detection_events),
                    key=lambda item: int(item[1].get("start_frame", 0)),
                )
                if int(event.get("tag", -1)) == first_tag
            ),
            None,
        )
        if first_event_index is None:
            raise RuntimeError(f"No detection event matches the first Letter tag {first_tag}.")

        def frame_time_sec(frame: int) -> float:
            # Detection files use 1-based frame numbers; frame 1 starts at 0 s.
            return max(0, int(frame) - 1) / frame_rate

        first_frame_sec = frame_time_sec(detection_events[first_event_index]["start_frame"])
        # Calibrate the video-frame timeline to EEG using the first corresponding trial.
        video_eeg_offset_sec = first_marker_sec - first_frame_sec

        def detection_eeg_time(frame: int) -> float:
            return video_eeg_offset_sec + frame_time_sec(frame)

        unused = set(range(len(detection_events)))
        matched = {}
        for marker_index, marker in enumerate(letter_markers):
            marker_sample = int(marker[0])
            marker_tag = int(marker[1])
            marker_sec = marker_sample / self.fs
            candidates = [
                index for index in unused
                if int(detection_events[index].get("tag", -1)) == marker_tag
                and abs(detection_eeg_time(detection_events[index]["start_frame"]) - marker_sec) <= 3.0
            ]
            if marker_index == 0:
                candidates = [first_event_index] if first_event_index in unused else []
            if not candidates:
                continue
            index = min(
                candidates,
                key=lambda item: abs(
                    detection_eeg_time(detection_events[item]["start_frame"]) - marker_sec
                ),
            )
            unused.remove(index)
            event = dict(detection_events[index])
            detection_start_video_sec = max(0, int(event["start_frame"]) - 1) / frame_rate
            detection_start_eeg_sec = detection_eeg_time(event["start_frame"])
            start_latency = detection_start_eeg_sec - marker_sec
            animal_latency = np.nan
            animal_frame = event.get("animal_start_frame")
            if animal_frame is not None and int(animal_frame) > int(event["start_frame"]):
                animal_latency = detection_eeg_time(animal_frame) - marker_sec
            matched[marker_sample] = {
                "tag": marker_tag,
                "marker_sec": marker_sec,
                "visual_start_latency_sec": start_latency,
                "animal_latency_sec": animal_latency,
                "validity": event.get("validity", ""),
                "start_frame": event["start_frame"],
                "animal_start_frame": animal_frame,
                "start_frame_video_sec": detection_start_video_sec,
                "start_frame_eeg_sec": detection_start_eeg_sec,
                "animal_start_video_sec": (
                    max(0, int(animal_frame) - 1) / frame_rate if animal_frame is not None else np.nan
                ),
                "animal_start_eeg_sec": (
                    detection_eeg_time(animal_frame) if animal_frame is not None else np.nan
                ),
                "video_eeg_offset_sec": video_eeg_offset_sec,
                "alignment_basis": "first_letter_trial",
            }
        return matched

    def show_long_behavior_linked_window(self):
        if not self.has_loaded_data():
            messagebox.showerror("No data", "Load parsed MAT data first.")
            return
        if self.task_mode_var.get() != "Letter":
            messagebox.showinfo(
                "Letter mode only",
                "The long behavior-linked window is designed for LettermodeData.",
            )
            return
        if self.stim_markers is None:
            messagebox.showerror("No markers", "Load log.txt and Event CSV first to build stimMarkers.")
            return
        detection_path = self.behavior_detection_path
        if detection_path is None or not detection_path.exists():
            selected = filedialog.askopenfilename(
                parent=self.task_window if self.task_window is not None and self.task_window.winfo_exists() else self,
                title="Select detection results file",
                filetypes=[
                    ("Detection results", "*.txt"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*"),
                ],
            )
            if not selected:
                return
            detection_path = Path(selected)
        try:
            frame_rate, detection_recording_time_sec, detection_events = self.read_behavior_detection_results(
                detection_path
            )
            matched = self.match_behavior_events_to_markers(
                self.require_stim_markers(),
                frame_rate,
                detection_recording_time_sec,
                np.nan,
                detection_events,
            )
            if not matched:
                raise RuntimeError(
                    "No detection events could be matched to Letter markers. "
                    "Check that the detection file belongs to the same recording block."
                )
            self.behavior_detection_path = detection_path
            self.behavior_detection_events = detection_events
            self.behavior_detection_frame_rate = frame_rate
            self.behavior_detection_recording_time_sec = detection_recording_time_sec
            first_match = next(iter(matched.values()))
            self.behavior_detection_video_eeg_offset_sec = float(
                first_match["video_eeg_offset_sec"]
            )
            raw_data, analysis_data, channels, preprocess_label = self.selected_analysis_data()
            markers = self.require_stim_markers()
            available_tags = sorted({int(row[1]) for row in markers if int(row[1]) in {4, 5, 6, 7}})
            if not available_tags:
                raise RuntimeError("No Letter tags 4, 5, 6, or 7 are available in stimMarkers.")

            dialog = tk.Toplevel(self)
            dialog.title("Letter long behavior-linked response")
            dialog.geometry("1250x850")
            dialog.transient(self.task_window if self.task_window is not None and self.task_window.winfo_exists() else self)

            controls = ttk.Frame(dialog)
            controls.pack(fill="x", padx=10, pady=(8, 4))
            ttk.Label(controls, text="tag").pack(side="left", padx=(0, 4))
            tag_var = tk.StringVar()
            labels = [f"{tag} tag" for tag in available_tags]
            tag_combo = ttk.Combobox(controls, textvariable=tag_var, values=labels, state="readonly", width=12)
            tag_combo.pack(side="left", padx=(0, 8))
            page_var = tk.StringVar(value="")
            ttk.Label(controls, textvariable=page_var, foreground="#245").pack(side="left", padx=8)
            ttk.Label(
                dialog,
                text=(
                    f"Anchor: Letter Display Time Offset | visual check: StartFrame | "
                    f"behavior reference: valid AnimalStart | {detection_path.name}\n"
                    f"Alignment basis: first Letter trial | frame rate={frame_rate:g} Hz | "
                    f"first-trial video-to-EEG offset={self.behavior_detection_video_eeg_offset_sec:+.6f}s"
                ),
                foreground="#445",
            ).pack(fill="x", padx=10, pady=(0, 4))

            fig = Figure(figsize=(10.5, 7.0), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=dialog)
            toolbar = NavigationToolbar2Tk(canvas, dialog, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill="x", padx=10, pady=(0, 4))
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 8))
            page_size = self.tvep_page_size()
            state = {"tag_index": 0, "page_index": 0}
            epoch_cache = {}

            def make_long_epochs(tag: int):
                if tag in epoch_cache:
                    return epoch_cache[tag]
                tag_markers = markers[markers[:, 1] == int(tag)]
                tag_markers = np.asarray(tag_markers, dtype=int)
                if tag_markers.size == 0:
                    raise RuntimeError(f"No Letter markers for tag {tag}.")
                tag_events = [matched.get(int(row[0])) for row in tag_markers]
                animal_latencies = [
                    float(item["animal_latency_sec"])
                    for item in tag_events
                    if item is not None
                    and item.get("validity") == "Y"
                    and np.isfinite(item.get("animal_latency_sec", np.nan))
                ]
                end_sec = max(6.0, max(animal_latencies, default=0.0) + 0.5)
                start_ms = -500.0
                end_ms = end_sec * 1000.0
                start_offset = int(round(start_ms * self.fs / 1000.0))
                end_offset = int(round(end_ms * self.fs / 1000.0))
                sample_offset = self.marker_offset_samples()
                epochs = []
                used_markers = []
                max_trials = parse_int(self.trials_per_stim_var.get(), 0)
                for marker in tag_markers:
                    center = int(marker[0]) - sample_offset
                    begin = center + start_offset
                    finish = center + end_offset
                    if begin < 0 or finish > analysis_data.shape[0]:
                        continue
                    epochs.append(analysis_data[begin:finish, :])
                    used_markers.append(marker)
                    if max_trials > 0 and len(epochs) >= max_trials:
                        break
                if not epochs:
                    raise RuntimeError(f"No complete long epochs are available for tag {tag}.")
                epochs = np.stack(epochs, axis=0)
                t_ms = np.arange(start_offset, end_offset, dtype=np.float64) / self.fs * 1000.0
                corrected = self.baseline_correct_epochs(epochs, t_ms)
                aggregate = self.task_response_aggregate(corrected, axis=0)
                q10, q90 = np.nanpercentile(corrected, [10, 90], axis=0)
                result = {
                    "tag": int(tag),
                    "t_sec": t_ms / 1000.0,
                    "corrected": corrected,
                    "aggregate": aggregate,
                    "q10": q10,
                    "q90": q90,
                    "used_markers": np.asarray(used_markers, dtype=int),
                    "end_sec": end_sec,
                    "animal_count": len(animal_latencies),
                }
                epoch_cache[tag] = result
                return result

            def show_long_trial_grid(result, channel_i: int, tag: int):
                corrected = result["corrected"][:, :, channel_i]
                trial_count = corrected.shape[0]
                cols = min(5, trial_count)
                rows = int(np.ceil(trial_count / cols))
                grid_fig = Figure(figsize=(3.2 * cols, 1.9 * rows + 1.2), dpi=105)
                axes = grid_fig.subplots(rows, cols, squeeze=False, sharex=True, sharey=False).ravel()
                t_sec = result["t_sec"]
                plot_step = max(1, int(np.ceil(len(t_sec) / 6000)))
                plot_slice = slice(None, None, plot_step)
                used_marker_samples = [int(row[0]) for row in result["used_markers"]]
                for trial_i in range(trial_count):
                    ax = axes[trial_i]
                    ax.plot(
                        t_sec[plot_slice],
                        corrected[trial_i, plot_slice],
                        color="black",
                        linewidth=0.7,
                    )
                    ax.axvline(
                        0,
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                        label="Letter onset" if trial_i == 0 else None,
                    )
                    event = matched.get(used_marker_samples[trial_i])
                    if event is not None:
                        visual_latency = float(event.get("visual_start_latency_sec", np.nan))
                        if np.isfinite(visual_latency) and t_sec[0] <= visual_latency <= t_sec[-1]:
                            ax.axvline(
                                visual_latency,
                                color="tab:green",
                                linestyle=":",
                                linewidth=1.8,
                                alpha=0.95,
                                label="StartFrame" if trial_i == 0 else None,
                            )
                        animal_latency = float(event.get("animal_latency_sec", np.nan))
                        if (
                            event.get("validity") == "Y"
                            and np.isfinite(animal_latency)
                            and t_sec[0] <= animal_latency <= t_sec[-1]
                        ):
                            ax.axvline(
                                animal_latency,
                                color="tab:purple",
                                linestyle="-.",
                                linewidth=2.0,
                                alpha=0.95,
                                label="valid AnimalStart" if trial_i == 0 else None,
                            )
                    ax.set_title(f"trial {trial_i + 1}", fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(alpha=0.2, linewidth=0.5)
                    if trial_i % cols == 0:
                        ax.set_ylabel("mV", fontsize=7)
                    if trial_i >= trial_count - cols:
                        ax.set_xlabel("Time from Letter onset (s)", fontsize=7)
                for ax in axes[trial_count:]:
                    ax.set_visible(False)
                grid_fig.suptitle(
                    f"Single trials | ch{channels[channel_i]} | tag {tag} | trials={trial_count}\n"
                    "red = Letter onset | green = StartFrame | purple = valid AnimalStart",
                    fontsize=10,
                    fontweight="bold",
                )
                grid_fig.tight_layout(rect=[0, 0, 1, 0.95])
                trial_dialog = tk.Toplevel(dialog)
                trial_dialog.title(f"ch{channels[channel_i]} single trials - tag {tag}")
                trial_dialog.geometry("1350x900")
                trial_dialog.transient(dialog)
                trial_canvas = FigureCanvasTkAgg(grid_fig, master=trial_dialog)
                trial_canvas.draw()
                trial_toolbar = NavigationToolbar2Tk(trial_canvas, trial_dialog, pack_toolbar=False)
                trial_toolbar.update()
                trial_toolbar.pack(fill="x", padx=10, pady=(8, 0))
                trial_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 4))

                def save_trial_grid():
                    initialfile = sanitize_filename_part(
                        f"letter_behavior_trials_ch{channels[channel_i]}_tag_{tag}"
                    ) or "letter_behavior_trials"
                    self.save_figure_with_dialog(
                        grid_fig,
                        trial_dialog,
                        "Save long behavior single-trial figure",
                        f"{initialfile}.png",
                        "long behavior single-trial figure",
                    )

                button_bar = ttk.Frame(trial_dialog)
                button_bar.pack(fill="x", padx=10, pady=(0, 10))
                ttk.Label(
                    button_bar,
                    text=(
                        "Each panel is one trial. Red: Letter onset; "
                        "green: StartFrame; purple: valid AnimalStart."
                    ),
                    foreground="#445",
                ).pack(side="left")
                ttk.Button(button_bar, text="Save figure", command=save_trial_grid).pack(side="right")

            def render():
                tag = available_tags[state["tag_index"]]
                result = make_long_epochs(tag)
                total = len(channels)
                total_pages = max(1, int(np.ceil(total / page_size)))
                state["page_index"] = min(max(state["page_index"], 0), total_pages - 1)
                start = state["page_index"] * page_size
                end = min(total, start + page_size)
                count = end - start
                cols = min(4, count)
                rows = int(np.ceil(count / cols))
                fig.clear()
                used_marker_samples = {int(row[0]) for row in result["used_markers"]}
                for local_i, channel_i in enumerate(range(start, end)):
                    ax = fig.add_subplot(rows, cols, local_i + 1)
                    ax._behavior_channel_index = channel_i
                    ax._behavior_channel_label = channels[channel_i]
                    corrected = result["corrected"][:, :, channel_i]
                    ax.fill_between(
                        result["t_sec"],
                        result["q10"][:, channel_i],
                        result["q90"][:, channel_i],
                        color="0.82",
                        alpha=0.7,
                        linewidth=0,
                        label="10-90% trials",
                    )
                    ax.plot(
                        result["t_sec"],
                        result["aggregate"][:, channel_i],
                        color="black",
                        linewidth=1.0,
                        label=self.task_response_aggregate_label(),
                    )
                    ax.axvline(
                        0,
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                        label="Letter onset" if local_i == 0 else None,
                    )
                    visual_lines = []
                    animal_lines = []
                    for marker_sample in used_marker_samples:
                        event = matched.get(marker_sample)
                        if event is None:
                            continue
                        visual_latency = float(event.get("visual_start_latency_sec", np.nan))
                        if np.isfinite(visual_latency) and result["t_sec"][0] <= visual_latency <= result["t_sec"][-1]:
                            visual_lines.append(visual_latency)
                        animal_latency = float(event.get("animal_latency_sec", np.nan))
                        if (
                            event.get("validity") == "Y"
                            and np.isfinite(animal_latency)
                            and result["t_sec"][0] <= animal_latency <= result["t_sec"][-1]
                        ):
                            animal_lines.append(animal_latency)
                    for line_i, line_x in enumerate(visual_lines):
                        ax.axvline(
                            line_x,
                            color="tab:green",
                            linestyle=":",
                            linewidth=1.5,
                            alpha=0.85,
                            label="StartFrame" if local_i == 0 and line_i == 0 else None,
                        )
                    for line_i, line_x in enumerate(animal_lines):
                        ax.axvline(
                            line_x,
                            color="tab:purple",
                            linestyle="-.",
                            linewidth=1.8,
                            alpha=0.9,
                            label="valid AnimalStart" if local_i == 0 and line_i == 0 else None,
                        )
                    ax.set_title(f"ch{channels[channel_i]}", fontsize=9)
                    ax.tick_params(labelsize=7)
                    ax.grid(alpha=0.2, linewidth=0.5)
                    if local_i % cols == 0:
                        ax.set_ylabel("mV", fontsize=8)
                    if local_i >= count - cols:
                        ax.set_xlabel("Time from Letter onset (s)", fontsize=8)
                    if local_i == 0:
                        ax.legend(fontsize=7, loc="upper right")
                fig.suptitle(
                    f"Long behavior-linked response | tag {tag} | trials={result['corrected'].shape[0]} | "
                    f"AnimalStart valid={result['animal_count']} | window=-0.5 to +{result['end_sec']:.2f}s",
                    fontsize=10,
                    fontweight="bold",
                )
                fig.text(
                    0.5,
                    0.012,
                    "red dashed = Letter onset | green dotted = StartFrame | purple dash-dot = valid AnimalStart | click a subplot for all trials",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#334",
                )
                fig.tight_layout(rect=[0, 0.035, 1, 0.94])
                canvas.draw()
                tag_var.set(labels[state["tag_index"]])
                page_var.set(
                    f"page {state['page_index'] + 1}/{total_pages}, channels {start + 1}-{end} of {total} | "
                    "red=Letter onset, green=StartFrame, purple=valid AnimalStart; click a subplot for single trials"
                )

            def change_page(delta: int):
                state["page_index"] += delta
                render()

            def on_tag_selected(_event=None):
                label = tag_var.get()
                if label in labels:
                    state["tag_index"] = labels.index(label)
                    state["page_index"] = 0
                    render()

            def on_long_behavior_click(event):
                if event.button != 1 or event.inaxes is None:
                    return
                channel_i = getattr(event.inaxes, "_behavior_channel_index", None)
                if channel_i is None:
                    return
                tag = available_tags[state["tag_index"]]
                show_long_trial_grid(make_long_epochs(tag), int(channel_i), tag)

            def show_behavior_timing_table():
                timing_dialog = tk.Toplevel(dialog)
                timing_dialog.title("Letter marker and detection timing table")
                timing_dialog.geometry("1320x620")
                timing_dialog.transient(dialog)

                letter_rows = [
                    row for row in np.asarray(markers, dtype=int)
                    if int(row[1]) in {4, 5, 6, 7}
                ]
                letter_rows.sort(key=lambda row: int(row[0]))
                tag_trial_numbers = {}
                table_rows = []
                visual_offsets = []
                animal_latencies = []
                for global_i, row in enumerate(letter_rows, start=1):
                    marker_sample = int(row[0])
                    tag = int(row[1])
                    tag_trial_numbers[tag] = tag_trial_numbers.get(tag, 0) + 1
                    event = matched.get(marker_sample)
                    marker_sec = marker_sample / self.fs
                    if event is None:
                        table_rows.append([
                            global_i,
                            tag_trial_numbers[tag],
                            tag,
                            marker_sample,
                            marker_sec,
                            "NA",
                            "NA",
                            "NA",
                            "NA",
                            "NA",
                            "NA",
                            "NA",
                            "unmatched",
                        ])
                        continue
                    visual_offset_ms = float(event.get("visual_start_latency_sec", np.nan)) * 1000.0
                    if np.isfinite(visual_offset_ms):
                        visual_offsets.append(visual_offset_ms)
                    animal_frame = event.get("animal_start_frame")
                    animal_latency_ms = float(event.get("animal_latency_sec", np.nan)) * 1000.0
                    if event.get("validity") == "Y" and np.isfinite(animal_latency_ms):
                        animal_latencies.append(animal_latency_ms)
                    status = "valid" if event.get("validity") == "Y" else "invalid"
                    if animal_frame is not None and int(animal_frame) <= int(event["start_frame"]):
                        status = f"{status}; AnimalStart before StartFrame"
                    table_rows.append([
                        global_i,
                        tag_trial_numbers[tag],
                        tag,
                        marker_sample,
                        marker_sec,
                        event.get("start_frame", "NA"),
                        event.get("start_frame_video_sec", np.nan),
                        event.get("start_frame_eeg_sec", np.nan),
                        visual_offset_ms,
                        event.get("validity", "NA"),
                        animal_frame if animal_frame is not None else "NA",
                        animal_latency_ms if np.isfinite(animal_latency_ms) else "NA",
                        status,
                    ])

                def summary(values):
                    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
                    if finite.size == 0:
                        return "n/a"
                    return (
                        f"n={finite.size}, mean={np.mean(finite):.2f} ms, "
                        f"SD={np.std(finite):.2f} ms, max|offset|={np.max(np.abs(finite)):.2f} ms"
                    )

                matched_count = sum(1 for row in table_rows if row[-1] != "unmatched")
                match_rate = matched_count / max(1, len(letter_rows))
                finite_visual = np.asarray(visual_offsets, dtype=float)
                if finite_visual.size < 3 or match_rate < 0.8:
                    alignment_hint = (
                        "Alignment check: INSUFFICIENT DATA - too few matched StartFrame values; "
                        "do not judge alignment yet."
                    )
                    alignment_color = "#9a6700"
                else:
                    mean_offset = float(np.mean(finite_visual))
                    sd_offset = float(np.std(finite_visual))
                    if sd_offset <= 50.0 and abs(mean_offset) <= 50.0:
                        alignment_hint = (
                            "Alignment check: GOOD - StartFrame is close to Letter marker "
                            "and stable across trials."
                        )
                        alignment_color = "#18794e"
                    elif sd_offset <= 50.0:
                        alignment_hint = (
                            f"Alignment check: STABLE WITH SYSTEMATIC OFFSET - trials are aligned "
                            f"but StartFrame is shifted by about {mean_offset:+.1f} ms."
                        )
                        alignment_color = "#9a6700"
                    else:
                        alignment_hint = (
                            f"Alignment check: CHECK JITTER - trial-to-trial StartFrame variation "
                            f"is large (SD {sd_offset:.1f} ms)."
                        )
                        alignment_color = "#b42318"

                ttk.Label(
                    timing_dialog,
                    text=(
                        f"Frame rate: {frame_rate:g} Hz | "
                        f"Letter markers: {len(letter_rows)} | matched: "
                        f"{matched_count}\n"
                        f"StartFrame offset relative to Letter marker: {summary(visual_offsets)}\n"
                        f"Valid AnimalStart latency relative to Letter marker: {summary(animal_latencies)}\n"
                        "Positive StartFrame offset means detection starts after the Letter marker; "
                        "negative means it starts before it."
                    ),
                    justify="left",
                    anchor="w",
                    foreground="#334",
                ).pack(fill="x", padx=10, pady=(8, 6))
                tk.Label(
                    timing_dialog,
                    text=alignment_hint,
                    anchor="w",
                    justify="left",
                    fg=alignment_color,
                    font=("Segoe UI", 10, "bold"),
                ).pack(fill="x", padx=10, pady=(0, 6))
                ttk.Label(
                    timing_dialog,
                    text=(
                        "This judgment uses Letter marker versus StartFrame only. "
                        "AnimalStart is shown as a behavior reference and is not used to decide visual alignment."
                    ),
                    foreground="#556",
                ).pack(fill="x", padx=10, pady=(0, 6))

                table_frame = ttk.Frame(timing_dialog)
                table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))
                table_frame.rowconfigure(0, weight=1)
                table_frame.columnconfigure(0, weight=1)
                columns = (
                    "global_trial",
                    "tag_trial",
                    "tag",
                    "marker_sample",
                    "marker_sec",
                    "start_frame",
                    "start_video_sec",
                    "start_eeg_sec",
                    "start_offset_ms",
                    "validity",
                    "animal_frame",
                    "animal_latency_ms",
                    "status",
                )
                headings = {
                    "global_trial": "trial",
                    "tag_trial": "tag trial",
                    "tag": "tag",
                    "marker_sample": "Letter marker sample",
                    "marker_sec": "Letter marker sec",
                    "start_frame": "StartFrame",
                    "start_video_sec": "StartFrame video sec",
                    "start_eeg_sec": "StartFrame EEG sec",
                    "start_offset_ms": "Start offset ms",
                    "validity": "Validity",
                    "animal_frame": "AnimalStart",
                    "animal_latency_ms": "Animal latency ms",
                    "status": "status",
                }
                tree = ttk.Treeview(table_frame, columns=columns, show="headings")
                for column in columns:
                    tree.heading(column, text=headings[column])
                    tree.column(column, width=105, minwidth=75, anchor="center", stretch=False)
                tree.column("global_trial", width=60)
                tree.column("tag_trial", width=70)
                tree.column("tag", width=55)
                tree.column("marker_sample", width=135)
                tree.column("marker_sec", width=105)
                tree.column("start_frame", width=90)
                tree.column("start_video_sec", width=120)
                tree.column("start_eeg_sec", width=115)
                tree.column("start_offset_ms", width=115)
                tree.column("validity", width=70)
                tree.column("animal_frame", width=95)
                tree.column("animal_latency_ms", width=125)
                tree.column("status", width=180)

                def display_value(value):
                    if isinstance(value, (float, np.floating)):
                        if not np.isfinite(value):
                            return "NA"
                        return f"{float(value):.4f}"
                    return str(value)

                for table_row in table_rows:
                    tree.insert("", "end", values=[display_value(value) for value in table_row])
                y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
                x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
                tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
                tree.grid(row=0, column=0, sticky="nsew")
                y_scroll.grid(row=0, column=1, sticky="ns")
                x_scroll.grid(row=1, column=0, sticky="ew")

            def save_behavior_figure():
                tag = available_tags[state["tag_index"]]
                initialfile = sanitize_filename_part(
                    f"letter_behavior_linked_tag_{tag}_page_{state['page_index'] + 1}"
                ) or "letter_behavior_linked_response"
                self.save_figure_with_dialog(
                    fig,
                    dialog,
                    "Save long behavior-linked response figure",
                    f"{initialfile}.png",
                    "long behavior-linked response figure",
                )

            ttk.Button(controls, text="< Prev page", command=lambda: change_page(-1)).pack(side="left", padx=(12, 4))
            ttk.Button(controls, text="Next page >", command=lambda: change_page(1)).pack(side="left", padx=4)
            ttk.Button(controls, text="Timing table", command=show_behavior_timing_table).pack(side="left", padx=(8, 4))
            ttk.Button(controls, text="Save figure", command=save_behavior_figure).pack(side="right", padx=4)
            canvas.mpl_connect("button_press_event", on_long_behavior_click)
            tag_combo.bind("<<ComboboxSelected>>", on_tag_selected)
            tag_combo.current(0)
            render()
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Long behavior window failed", str(exc))

    def show_letter_average_response_window(self):
        if not self.stim_task_cache or not self.stim_task_cache.get("tag_results"):
            return
        tag_results = self.stim_task_cache["tag_results"]
        valid_results = [
            result for result in tag_results
            if result.get("mean_wave") is not None and result.get("channels")
        ]
        if not valid_results:
            return

        dialog = tk.Toplevel(self)
        dialog.title("LettermodeData average response")
        dialog.geometry("1150x780")
        dialog.transient(self.task_window if self.task_window is not None and self.task_window.winfo_exists() else self)

        controls = ttk.Frame(dialog)
        controls.pack(fill="x", padx=10, pady=(8, 4))
        ttk.Label(controls, text="tag").pack(side="left", padx=(0, 4))
        tag_var = tk.StringVar()
        labels = [f"{result['tag']} tag" for result in valid_results]
        tag_combo = ttk.Combobox(controls, textvariable=tag_var, values=labels, state="readonly", width=12)
        tag_combo.pack(side="left", padx=(0, 10))
        page_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=page_var, foreground="#245").pack(side="left", padx=8)

        fig = Figure(figsize=(9.5, 6.2), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=dialog)
        toolbar = NavigationToolbar2Tk(canvas, dialog, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x", padx=10, pady=(0, 4))
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 8))

        state = {"tag_index": 0, "page_index": 0}

        def render():
            result = valid_results[state["tag_index"]]
            channels = result["channels"]
            mean_wave = result["mean_wave"]
            raw_mean_wave = result.get("raw_mean_wave")
            t_sec = result["t_sec"]
            page_size = self.tvep_page_size()
            total = len(channels)
            total_pages = max(1, int(np.ceil(total / page_size)))
            state["page_index"] = min(max(state["page_index"], 0), total_pages - 1)
            start = state["page_index"] * page_size
            end = min(total, start + page_size)
            count = end - start
            cols = min(4, count)
            rows = int(np.ceil(count / cols))

            fig.clear()
            for local_i, ch_i in enumerate(range(start, end)):
                ax = fig.add_subplot(rows, cols, local_i + 1)
                ax._letter_channel_index = ch_i
                ax._letter_channel_label = channels[ch_i]
                if raw_mean_wave is not None:
                    ax.plot(t_sec, raw_mean_wave[:, ch_i], color="0.55", linewidth=0.9, label="raw")
                ax.plot(t_sec, mean_wave[:, ch_i], color="black", linewidth=1.1, label=self.task_response_aggregate_label())
                ax.axvline(0, color="red", linestyle="--", linewidth=1.8)
                ax.set_title(f"ch{channels[ch_i]}", fontsize=9)
                ax.tick_params(labelsize=7)
                if local_i % cols == 0:
                    ax.set_ylabel("Amplitude (mV)", fontsize=8)
                if local_i >= count - cols:
                    ax.set_xlabel("Time to stimulation (second)", fontsize=8)
                if local_i == 0 and raw_mean_wave is not None:
                    ax.legend(fontsize=7, loc="upper right")
            fig.suptitle(
                f"LettermodeData tag = {result['tag']} | trials={result['valid_trials']} | "
                f"aggregate={self.task_response_aggregate_label()} | "
                f"tag {state['tag_index'] + 1}/{len(valid_results)} | page {state['page_index'] + 1}/{total_pages}",
                fontsize=10,
                color="red",
                fontweight="bold",
            )
            fig.tight_layout()
            canvas.draw()
            tag_var.set(labels[state["tag_index"]])
            page_var.set(f"page {state['page_index'] + 1}/{total_pages}, channels {start + 1}-{end} of {total}")

        def change_page(delta: int):
            state["page_index"] += delta
            render()

        def on_tag_selected(_event=None):
            label = tag_var.get()
            if label in labels:
                state["tag_index"] = labels.index(label)
                state["page_index"] = 0
                render()

        def on_letter_average_click(event):
            if event.button != 1 or event.inaxes is None:
                return
            ch_i = getattr(event.inaxes, "_letter_channel_index", None)
            if ch_i is None:
                return
            result = valid_results[state["tag_index"]]
            self.show_letter_channel_trial_grid(result, int(ch_i))

        canvas.mpl_connect("button_press_event", on_letter_average_click)
        ttk.Button(controls, text="< Prev page", command=lambda: change_page(-1)).pack(side="left", padx=(14, 4))
        ttk.Button(controls, text="Next page >", command=lambda: change_page(1)).pack(side="left", padx=4)

        def save_letter_average_figure():
            result = valid_results[state["tag_index"]]
            initialfile = sanitize_filename_part(
                f"letter_average_tag_{result['tag']}_page_{state['page_index'] + 1}"
            ) or "letter_average_response"
            self.save_figure_with_dialog(
                fig,
                dialog,
                "Save Letter average response figure",
                f"{initialfile}.png",
                "Letter average response figure",
            )

        ttk.Button(controls, text="Save figure", command=save_letter_average_figure).pack(side="right", padx=(8, 0))
        tag_combo.bind("<<ComboboxSelected>>", on_tag_selected)
        tag_combo.current(0)
        render()

    def show_letter_channel_trial_grid(self, tag_result: dict, channel_index: int):
        if self.stim_markers is None:
            messagebox.showerror("No markers", "Load log.txt and Event CSV first to build stimMarkers.")
            return
        try:
            cached_channels = tag_result.get("channels", [])
            if channel_index < 0 or channel_index >= len(cached_channels):
                raise RuntimeError("Clicked channel is outside the current Letter result.")
            channel_label = int(cached_channels[channel_index])

            raw_data, analysis_data, channels, preprocess_label = self.selected_analysis_data()
            if channel_label not in channels:
                raise RuntimeError(
                    f"ch{channel_label} is no longer in the selected channel set. "
                    "Re-run LettermodeData or restore the same selected channels."
                )
            plot_col = channels.index(channel_label)
            markers = self.require_stim_markers()
            tag = int(tag_result["tag"])
            epochs, used_markers, t_ms = self.make_epochs(analysis_data, markers, tag=tag)
            corrected = self.baseline_correct_epochs(epochs, t_ms)
            trial_count = corrected.shape[0]
            if trial_count == 0:
                raise RuntimeError(f"No valid trials found for tag {tag}.")

            raw_corrected = None
            if self.analysis_view_mode_var.get() == "overlay":
                raw_epochs, _, _ = self.make_epochs(raw_data, markers, tag=tag)
                raw_corrected = self.baseline_correct_epochs(raw_epochs, t_ms)

            cols = min(5, trial_count)
            rows = int(np.ceil(trial_count / cols))
            fig = Figure(figsize=(3.0 * cols, 1.75 * rows + 0.9), dpi=110)
            axes = fig.subplots(rows, cols, squeeze=False, sharex=True, sharey=False).ravel()
            t_sec = t_ms / 1000.0
            for trial_i in range(trial_count):
                ax = axes[trial_i]
                if raw_corrected is not None and trial_i < raw_corrected.shape[0]:
                    ax.plot(t_sec, raw_corrected[trial_i, :, plot_col], color="0.62", linewidth=0.7, label="raw")
                ax.plot(t_sec, corrected[trial_i, :, plot_col], color="black", linewidth=0.85, label="trial")
                ax.axvline(0, color="red", linestyle="--", linewidth=1.0)
                ax.set_title(f"trial {trial_i + 1}", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.2, linewidth=0.5)
                if trial_i % cols == 0:
                    ax.set_ylabel("mV", fontsize=7)
                if trial_i >= trial_count - cols:
                    ax.set_xlabel("s", fontsize=7)
                if trial_i == 0 and raw_corrected is not None:
                    ax.legend(fontsize=7, loc="upper right")
            for ax in axes[trial_count:]:
                ax.set_visible(False)
            fig.suptitle(
                f"ch{channel_label} single trials | LettermodeData tag {tag} | "
                f"trials={trial_count} | {preprocess_label}",
                fontsize=10,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            dialog = tk.Toplevel(self)
            dialog.title(f"ch{channel_label} Letter trials - tag {tag}")
            dialog.geometry("1250x850")
            dialog.transient(self.task_window if self.task_window is not None and self.task_window.winfo_exists() else self)
            canvas = FigureCanvasTkAgg(fig, master=dialog)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, dialog, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill="x", padx=10, pady=(8, 0))
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 4))

            def save_trial_grid():
                path = filedialog.asksaveasfilename(
                    parent=dialog,
                    title="Save Letter single-trial grid figure",
                    defaultextension=".png",
                    initialfile=f"ch{channel_label}_letter_tag_{tag}_single_trials.png",
                    filetypes=[
                        ("PNG image", "*.png"),
                        ("PDF file", "*.pdf"),
                        ("SVG file", "*.svg"),
                        ("JPEG image", "*.jpg;*.jpeg"),
                        ("All files", "*.*"),
                    ],
                )
                if not path:
                    return
                try:
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    self.log(f"Saved Letter single-trial grid figure: {path}")
                    messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=dialog)
                except Exception as exc:
                    self.log(traceback.format_exc())
                    messagebox.showerror("Save failed", str(exc), parent=dialog)

            info_text = (
                f"Click source: ch{channel_label}, Letter tag {tag}; "
                f"epoch {t_ms[0]:.1f}-{t_ms[-1]:.1f} ms; "
                "red dashed line is marker time."
            )
            button_bar = ttk.Frame(dialog)
            button_bar.pack(fill="x", padx=10, pady=(0, 10))
            ttk.Label(button_bar, text=info_text, foreground="#445").pack(side="left")
            ttk.Button(button_bar, text="Save figure", command=save_trial_grid).pack(side="right")
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Letter single-trial plot failed", str(exc))

    def change_stim_task_result(self, delta: int):
        if not self.stim_task_cache or not self.stim_task_cache.get("tag_results"):
            messagebox.showinfo("No task result", "Run LettermodeData analysis first.")
            return
        n = len(self.stim_task_cache["tag_results"])
        self.stim_task_tag_index = (self.stim_task_tag_index + delta) % n
        self.render_stim_task_result()

    def prepare_state_analysis(self, state_name: str):
        try:
            selected, channels = self.selected_data()
            duration = selected.shape[0] / self.fs if self.fs else float("nan")
            channel_text = ",".join(str(ch) for ch in channels[:80])
            if len(channels) > 80:
                channel_text += "..."
            if self.stim_markers is not None:
                marker_text = f"stimMarkers: {self.stim_markers.shape[0]} events\n\n"
            else:
                marker_text = "stimMarkers: not built\n\n"
            self.write_state_text(
                f"{state_name} analysis will use the selected channel set.\n\n"
                f"Selected channels: {len(channels)}\n"
                f"Channel IDs: {channel_text}\n"
                f"Data matrix: {selected.shape[0]} samples x {selected.shape[1]} channels\n"
                f"FS: {self.fs:g} Hz\n"
                f"Duration: {duration:.3f} sec\n\n"
                f"Timing loaded: {'yes' if self.timing_info else 'no'}\n"
                f"{marker_text}"
                "This page is now connected to the selected-channel pool. "
                "The next analysis modules can use selected_data() so each state runs only on these channels."
            )
            self.show_task_analysis_window()
        except Exception as exc:
            messagebox.showerror("No selected channels", str(exc))

    def batch_param_value(self, row: dict, *names: str, default=None):
        for name in names:
            key = name.strip().lower()
            if key in row and str(row[key]).strip() != "":
                return row[key]
        return default

    def snr_db_value(self, row: dict) -> float:
        value = row.get("snr_db", np.nan)
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("'"):
                value = value[1:].strip()
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def is_snr_row_healthy(self, row: dict) -> bool:
        return (
            (np.isfinite(self.snr_db_value(row)) or bool(row.get("snr_skipped")))
            and not str(row.get("detail", "")).startswith("Bad Channel")
        )

    def current_lfp_snr_settings(self) -> dict:
        return {
            "mode": self.snr_mode_var.get(),
            "threshold_db": self.snr_threshold_var.get(),
            "notch_50hz": int(self.notch_var.get()),
            "skip_snr_scoring": int(self.skip_lfp_snr_var.get()),
            "snr_filter": self.snr_highpass_var.get(),
            "snr_band_low": self.snr_band_low_var.get(),
            "snr_band_high": self.snr_band_high_var.get(),
            "hp_order": self.filter_highpass_order_var.get(),
            "lp_order": self.filter_lowpass_order_var.get(),
            "motion_artifact_ica": int(self.motion_artifact_enable_var.get()),
            "motion_ica_components": self.motion_ica_components_var.get(),
            "motion_ica_exclude": self.motion_ica_exclude_var.get(),
            "motion_ica_lfreq": self.motion_ica_lfreq_var.get(),
            "motion_ica_hfreq": self.motion_ica_hfreq_var.get(),
            "motion_ica_decim": self.motion_ica_decim_var.get(),
            "motion_ica_max_iter": self.motion_ica_max_iter_var.get(),
            "bad_check": int(self.bad_channel_check_var.get()),
            "bad_check_parallel": int(self.bad_check_parallel_var.get()),
            "bad_check_workers": self.bad_check_workers_var.get(),
            "flat_std": self.bad_flat_std_var.get(),
            "flat_ratio_pct": self.bad_flat_ratio_var.get(),
            "two_s_win_bad_check": int(self.bad_window_check_var.get()),
            "valid_win_db": self.bad_window_snr_threshold_var.get(),
            "flat_win_ptp": self.bad_window_ptp_threshold_var.get(),
            "signal_band_low": self.signal_band_low_var.get(),
            "signal_band_high": self.signal_band_high_var.get(),
            "noise_band_low": self.noise_band_low_var.get(),
            "noise_band_high": self.noise_band_high_var.get(),
            "stim_interval": self.stim_interval_var.get(),
            "stim_duration": self.stim_duration_var.get(),
            "stim_freq": self.stim_freq_var.get(),
            "harmonics": self.harmonics_var.get(),
            "neighbor_bins": self.neighbor_bins_var.get(),
            "fft_len_sec": self.fft_len_var.get(),
            "spike_filter_low": self.spike_filter_low_var.get(),
            "spike_filter_high": self.spike_filter_high_var.get(),
            "spike_threshold_factor": self.spike_threshold_factor_var.get(),
            "spike_window_sec": self.spike_window_sec_var.get(),
            "spike_step_ms": self.spike_step_ms_var.get(),
            "spike_refractory_ms": self.spike_refractory_ms_var.get(),
            "spike_pre_samples": self.spike_pre_samples_var.get(),
            "spike_post_samples": self.spike_post_samples_var.get(),
            "spike_min_count": self.spike_min_count_var.get(),
        }

    def lfp_snr_settings_from_row(self, row: dict, config_index: int | None = None) -> dict:
        settings = dict(self.current_lfp_snr_settings())

        def set_if_present(setting_key: str, *names: str, bool_value: bool = False):
            value = self.batch_param_value(row, *names)
            if value is None:
                return
            settings[setting_key] = int(parse_bool_like(value, False)) if bool_value else str(value)

        value = self.batch_param_value(row, "mode")
        if value is not None:
            settings["mode"] = str(value).strip().lower()
        set_if_present("threshold_db", "threshold", "threshold_db", "red_if_db")
        set_if_present("notch_50hz", "notch", "notch_50hz", "50hz", "50hz_notch", bool_value=True)
        set_if_present("skip_snr_scoring", "skip_snr_scoring", "skip_snr", "skip_lfp_snr", bool_value=True)
        value = self.batch_param_value(row, "snr_filter", "filter", "preprocess")
        if value is not None:
            text = str(value).strip().lower()
            aliases = {"none": "off", "0": "off", "hp0.5": "0.5", "hp1": "1.0"}
            settings["snr_filter"] = aliases.get(text, text)
        set_if_present("snr_band_low", "snr_band_low", "pre_band_low", "band_low")
        set_if_present("snr_band_high", "snr_band_high", "pre_band_high", "band_high")
        set_if_present("hp_order", "hp_order", "highpass_order")
        set_if_present("lp_order", "lp_order", "lowpass_order")
        set_if_present("motion_artifact_ica", "motion_artifact_ica", "motion_ica", "ica_motion", bool_value=True)
        set_if_present("motion_ica_components", "motion_ica_components", "ica_components")
        set_if_present("motion_ica_exclude", "motion_ica_exclude", "ica_exclude")
        set_if_present("motion_ica_lfreq", "motion_ica_lfreq", "ica_lfreq")
        set_if_present("motion_ica_hfreq", "motion_ica_hfreq", "ica_hfreq")
        set_if_present("motion_ica_decim", "motion_ica_decim", "ica_decim")
        set_if_present("motion_ica_max_iter", "motion_ica_max_iter", "ica_max_iter")
        set_if_present("bad_check", "bad_check", bool_value=True)
        set_if_present("bad_check_parallel", "bad_check_parallel", "parallel_bad_check", "parallel", bool_value=True)
        set_if_present("bad_check_workers", "bad_check_workers", "bad_workers", "workers")
        set_if_present("flat_std", "flat_std")
        set_if_present("flat_ratio_pct", "flat_ratio", "flat_ratio_pct")
        set_if_present("two_s_win_bad_check", "2s_win_bad_check", "two_s_win_bad_check", "window_bad_check", bool_value=True)
        set_if_present("valid_win_db", "valid_win_db", "window_snr_threshold")
        set_if_present("flat_win_ptp", "flat_win_ptp", "window_flat_ptp")
        set_if_present("signal_band_low", "signal_low", "signal_band_low", "sig_low")
        set_if_present("signal_band_high", "signal_high", "signal_band_high", "sig_high")
        set_if_present("noise_band_low", "noise_low", "noise_band_low")
        set_if_present("noise_band_high", "noise_high", "noise_band_high")
        set_if_present("stim_interval", "stim_interval")
        set_if_present("stim_duration", "stim_duration")
        set_if_present("stim_freq", "stim_freq")
        set_if_present("harmonics", "harmonics")
        set_if_present("neighbor_bins", "neighbor_bins", "n_neighbor")
        set_if_present("fft_len_sec", "fft_len_sec", "fft_len", "fft_length_sec")
        set_if_present("spike_filter_low", "spike_filter_low", "spike_low")
        set_if_present("spike_filter_high", "spike_filter_high", "spike_high")
        set_if_present("spike_threshold_factor", "spike_threshold_factor", "spike_threshold", "threshold_factor")
        set_if_present("spike_window_sec", "spike_window_sec", "spike_window")
        set_if_present("spike_step_ms", "spike_step_ms", "spike_step")
        set_if_present("spike_refractory_ms", "spike_refractory_ms", "spike_refractory")
        set_if_present("spike_pre_samples", "spike_pre_samples", "pre_samples")
        set_if_present("spike_post_samples", "spike_post_samples", "post_samples")
        set_if_present("spike_min_count", "spike_min_count", "min_spikes")
        run_spike = self.batch_param_value(row, "run_spike", "spike", "spike_snr", default="0")
        settings["run_spike"] = int(parse_bool_like(run_spike, False))
        raw_name = self.batch_param_value(
            row,
            "name",
            "config",
            "config_name",
            default=f"config_{config_index:03d}" if config_index is not None else "config",
        )
        config_name = sanitize_filename_part(raw_name) or (
            f"config_{config_index:03d}" if config_index is not None else "config"
        )
        if config_index is not None:
            config_name = f"{config_index:03d}_{config_name}"
            settings["config_index"] = config_index
        settings["config_name"] = config_name
        return settings

    def filter_orders_from_settings(self, settings: dict) -> tuple[int, int]:
        hp_order = max(1, parse_int(settings.get("hp_order", self.filter_highpass_order_var.get()), 3))
        lp_order = max(1, parse_int(settings.get("lp_order", self.filter_lowpass_order_var.get()), 5))
        return hp_order, lp_order

    def apply_highpass_filter_with_order(
        self,
        data: np.ndarray,
        cutoff: float,
        order: int,
        axis: int = 0,
    ) -> np.ndarray:
        if scisig is None:
            raise ImportError("scipy.signal is required for highpass filtering.")
        nyq = self.fs / 2.0
        if cutoff <= 0 or cutoff >= nyq:
            raise ValueError(f"Invalid highpass cutoff: {cutoff:g} Hz; Nyquist is {nyq:g} Hz.")
        sos = scisig.butter(max(1, int(order)), cutoff / nyq, btype="highpass", output="sos")
        return np.asarray(scisig.sosfiltfilt(sos, np.asarray(data, dtype=DATA_DTYPE), axis=axis), dtype=DATA_DTYPE)

    def apply_lowpass_filter_with_order(
        self,
        data: np.ndarray,
        cutoff: float,
        order: int,
        axis: int = 0,
    ) -> np.ndarray:
        if scisig is None:
            raise ImportError("scipy.signal is required for lowpass filtering.")
        nyq = self.fs / 2.0
        if cutoff <= 0 or cutoff >= nyq:
            raise ValueError(f"Invalid lowpass cutoff: {cutoff:g} Hz; Nyquist is {nyq:g} Hz.")
        sos = scisig.butter(max(1, int(order)), cutoff / nyq, btype="lowpass", output="sos")
        return np.asarray(scisig.sosfiltfilt(sos, np.asarray(data, dtype=DATA_DTYPE), axis=axis), dtype=DATA_DTYPE)

    def apply_bandpass_filter_with_settings(
        self,
        data: np.ndarray,
        low: float,
        high: float,
        settings: dict,
        axis: int = 0,
    ) -> np.ndarray:
        nyq = self.fs / 2.0
        if not (0 < low < high < nyq):
            raise ValueError(f"Invalid bandpass: require 0 < low < high < {nyq:g} Hz.")
        hp_order, lp_order = self.filter_orders_from_settings(settings)
        filtered = self.apply_lowpass_filter_with_order(data, high, lp_order, axis=axis)
        filtered = self.apply_highpass_filter_with_order(filtered, low, hp_order, axis=axis)
        return filtered

    def snr_preprocess_highpass_with_settings(self, data: np.ndarray, settings: dict) -> np.ndarray:
        value = str(settings.get("snr_filter", "off")).strip().lower()
        if value in {"", "off", "none", "0"}:
            return np.asarray(data, dtype=DATA_DTYPE)
        if scisig is None:
            raise ImportError("scipy.signal is required for SNR filtering.")
        arr = np.asarray(data, dtype=DATA_DTYPE)
        if value == "bandpass":
            low = parse_float(settings.get("snr_band_low", 0.5), 0.5)
            high = parse_float(settings.get("snr_band_high", 300.0), 300.0)
            return self.apply_bandpass_filter_with_settings(arr, low, high, settings, axis=0)
        hp_order, _ = self.filter_orders_from_settings(settings)
        return self.apply_highpass_filter_with_order(arr, float(value), hp_order, axis=0)

    def snr_filter_label_from_settings(self, settings: dict) -> str:
        value = str(settings.get("snr_filter", "off")).strip().lower()
        if value in {"", "off", "none", "0"}:
            return ""
        hp_order, lp_order = self.filter_orders_from_settings(settings)
        if value == "bandpass":
            return (
                f"bandpass {settings.get('snr_band_low')}-{settings.get('snr_band_high')}Hz, "
                f"HP order {hp_order}, LP order {lp_order}"
            )
        return f"highpass {value}Hz, order {hp_order}"

    def compute_2s_window_snr_values_with_settings(self, sig: np.ndarray, settings: dict) -> list[dict]:
        if self.fs <= 0:
            return []
        win_samples = int(round(2.0 * self.fs))
        if win_samples <= 0:
            return []
        n_windows = sig.size // win_samples
        if n_windows <= 0:
            return []
        signal_band = (
            parse_float(settings.get("signal_band_low", 30.0), 30.0),
            parse_float(settings.get("signal_band_high", 80.0), 80.0),
        )
        noise_band = (
            parse_float(settings.get("noise_band_low", 1.0), 1.0),
            parse_float(settings.get("noise_band_high", 200.0), 200.0),
        )
        valid_snr_threshold = parse_float(settings.get("valid_win_db", 1.0), 1.0)
        flat_ptp_threshold = parse_float(settings.get("flat_win_ptp", 0.01), 0.01)
        rows: list[dict] = []
        for win_idx in range(n_windows):
            start = win_idx * win_samples
            end = start + win_samples
            segment = np.asarray(sig[start:end], dtype=DATA_DTYPE)
            finite = np.isfinite(segment)
            if finite.sum() < max(32, int(0.5 * win_samples)):
                continue
            fill = float(np.nanmedian(segment[finite]))
            segment = np.where(finite, segment, fill)
            ptp_value = float(np.nanmax(segment) - np.nanmin(segment))
            diff_abs = np.abs(np.diff(segment))
            dead_straight_ratio = float(np.nanmean(diff_abs <= 1e-6)) if diff_abs.size else 1.0
            segment = segment - float(np.nanmedian(segment))
            if parse_bool_like(settings.get("notch_50hz"), False):
                try:
                    segment = apply_notch_filter(segment, self.fs, freq=50.0, q=30, harmonics=1)
                except Exception:
                    pass
            try:
                snr_db = compute_resting_snr(segment, self.fs, signal_band, noise_band)
            except Exception:
                continue
            if not np.isfinite(snr_db):
                continue
            if snr_db > valid_snr_threshold:
                status_code = 1
                status_label = "normal"
            elif dead_straight_ratio > 0.20 or ptp_value <= flat_ptp_threshold:
                status_code = 0
                status_label = "flat/no-signal"
            else:
                status_code = 2
                status_label = "motion artifact"
            rows.append({
                "window": win_idx + 1,
                "start_sec": start / self.fs,
                "end_sec": end / self.fs,
                "snr_db": float(snr_db),
                "ptp": ptp_value,
                "dead_straight_ratio": dead_straight_ratio,
                "status": status_code,
                "status_label": status_label,
                "valid": bool(status_code != 0),
            })
        return rows

    def snr_valid_window_ratio_with_settings(self, sig: np.ndarray, settings: dict) -> tuple[float | None, int, int]:
        rows = self.compute_2s_window_snr_values_with_settings(sig, settings)
        if not rows:
            n_windows = sig.size // max(1, int(round(2.0 * self.fs))) if self.fs > 0 else 0
            return None, 0, n_windows
        used_windows = len(rows)
        valid_windows = int(sum(1 for row in rows if row["status"] != 0))
        if used_windows <= 0:
            return None, 0, 0
        return valid_windows / used_windows, valid_windows, used_windows

    def highpass_for_artifact_check_with_settings(
        self,
        sig: np.ndarray,
        settings: dict,
        cutoff: float = 1.0,
    ) -> np.ndarray | None:
        if scisig is None or self.fs <= 0:
            return None
        finite = np.asarray(sig, dtype=DATA_DTYPE)
        finite_mask = np.isfinite(finite)
        if finite_mask.sum() < max(32, int(round(self.fs))):
            return None
        fill = float(np.nanmedian(finite[finite_mask]))
        work = np.where(finite_mask, finite, fill)
        nyq = self.fs / 2.0
        if cutoff <= 0 or cutoff >= nyq:
            return None
        try:
            hp_order, _ = self.filter_orders_from_settings(settings)
            return self.apply_highpass_filter_with_order(work, cutoff, hp_order, axis=0)
        except Exception:
            return None

    def high_frequency_artifact_reasons_with_settings(self, sig: np.ndarray, settings: dict) -> list[str]:
        hp = self.highpass_for_artifact_check_with_settings(sig, settings, cutoff=1.0)
        if hp is None or hp.size < max(32, int(round(self.fs))):
            return []
        try:
            hp = apply_notch_filter(hp, self.fs, freq=50.0, q=30, harmonics=1)
        except Exception:
            pass
        centered = hp - float(np.nanmedian(hp))
        hf = centered
        if scisig is not None and self.fs > 0:
            try:
                low = min(80.0, self.fs / 2.0 * 0.25)
                high = min(300.0, self.fs / 2.0 * 0.9)
                if 0 < low < high < self.fs / 2.0:
                    hf = self.apply_bandpass_filter_with_settings(centered, low, high, settings, axis=0)
            except Exception:
                hf = centered
        hf_abs = np.abs(hf - float(np.nanmedian(hf)))
        hf_mad = float(np.nanmedian(hf_abs) / 0.6745)
        if not np.isfinite(hf_mad) or hf_mad <= 0:
            return []
        z = hf_abs / hf_mad
        z_mask = z > 8.0
        z_ratio = float(np.nanmean(z_mask))
        longest_ms = 1000.0 * self.longest_true_run(z_mask) / self.fs
        if z_ratio >= 0.05 or longest_ms >= 100.0:
            return [
                f"sustained high-frequency artifact, z>8 ratio={z_ratio * 100:.2f}%, longest={longest_ms:.0f}ms"
            ]
        return []

    def detect_bad_snr_channels_with_settings(
        self,
        data: np.ndarray,
        settings: dict,
        progress_callback=None,
        progress_start: float = 0.0,
        progress_end: float = 45.0,
        label_prefix: str = "LFP SNR",
    ) -> tuple[list[int], dict[int, str], dict[int, str]]:
        if not parse_bool_like(settings.get("bad_check"), True):
            return list(range(data.shape[1])), {}, {}
        data = np.asarray(data, dtype=DATA_DTYPE)
        flat_std_thr = parse_float(settings.get("flat_std", 1e-4), 1e-4)
        flat_ratio_thr = parse_float(settings.get("flat_ratio_pct", 30.0), 30.0) / 100.0
        good_indices: list[int] = []
        bad_reasons: dict[int, str] = {}
        candidate_reasons: dict[int, str] = {}
        n_channels = max(1, data.shape[1])

        def check_channel(idx: int) -> tuple[int, int, str | None, str | None]:
            sig = np.asarray(data[:, idx], dtype=DATA_DTYPE)
            finite = sig[np.isfinite(sig)]
            ch = idx + 1
            if finite.size < 10:
                return idx, ch, "Bad Channel: too few finite samples", None
            sig_std = float(np.nanstd(finite))
            sig_range = float(np.nanmax(finite) - np.nanmin(finite))
            if sig_std <= flat_std_thr or sig_range <= flat_std_thr * 6:
                return idx, ch, f"Bad Channel: flat/dead, std={sig_std:.3g}", None
            quant_step = max(flat_std_thr, 1e-4)
            unique_count = int(np.unique(np.round(finite / quant_step)).size)
            unique_ratio = unique_count / finite.size
            if unique_count <= 8 or unique_ratio <= 5e-4:
                return idx, ch, f"Bad Channel: too few unique levels, unique={unique_count}", None
            flat_ratio = self.flat_time_ratio_with_spike_tolerance(sig, flat_std_thr)
            if flat_ratio_thr > 0 and flat_ratio >= flat_ratio_thr:
                return idx, ch, f"Bad Channel: flat time={flat_ratio * 100:.1f}%", None
            if parse_bool_like(settings.get("two_s_win_bad_check"), False):
                valid_ratio, valid_windows, used_windows = self.snr_valid_window_ratio_with_settings(sig, settings)
                if valid_ratio is not None and valid_ratio < 0.5:
                    valid_snr_threshold = parse_float(settings.get("valid_win_db", 1.0), 1.0)
                    return idx, ch, (
                        f"Bad Channel: 2s window SNR valid ratio={valid_ratio * 100:.1f}% "
                        f"({valid_windows}/{used_windows} windows status 1/2; "
                        f"normal if >{valid_snr_threshold:g}dB, artifact counted)"
                    ), None
            artifact_reasons = self.high_frequency_artifact_reasons_with_settings(sig, settings)
            if artifact_reasons:
                return idx, ch, None, "Bad Channel candidate: " + "; ".join(artifact_reasons)
            return idx, ch, None, None

        max_workers = self.bad_check_max_workers_from_settings(settings, data.shape[1])
        if max_workers <= 1:
            for idx in range(data.shape[1]):
                if progress_callback:
                    progress = progress_start + (progress_end - progress_start) * idx / n_channels
                    progress_callback(progress, f"{label_prefix}: checking bad channels {idx + 1}/{data.shape[1]}")
                result_idx, ch, bad_detail, candidate_detail = check_channel(idx)
                if bad_detail:
                    bad_reasons[ch] = bad_detail
                    continue
                if candidate_detail:
                    candidate_reasons[ch] = candidate_detail
                good_indices.append(result_idx)
        else:
            done_count = 0
            if progress_callback:
                progress_callback(
                    progress_start,
                    f"{label_prefix}: parallel bad check using {max_workers} workers for {data.shape[1]} channels",
                )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                pending = {executor.submit(check_channel, idx) for idx in range(data.shape[1])}
                last_heartbeat = time.perf_counter()
                while pending:
                    done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                    if not done:
                        if progress_callback and time.perf_counter() - last_heartbeat >= 0.5:
                            progress = progress_start + (progress_end - progress_start) * done_count / n_channels
                            progress_callback(
                                progress,
                                f"{label_prefix}: parallel bad check running {done_count}/{data.shape[1]} channels",
                            )
                            last_heartbeat = time.perf_counter()
                        continue
                    for future in done:
                        result_idx, ch, bad_detail, candidate_detail = future.result()
                        done_count += 1
                        if bad_detail:
                            bad_reasons[ch] = bad_detail
                        else:
                            if candidate_detail:
                                candidate_reasons[ch] = candidate_detail
                            good_indices.append(result_idx)
                    if progress_callback:
                        progress = progress_start + (progress_end - progress_start) * done_count / n_channels
                        progress_callback(
                            progress,
                            f"{label_prefix}: parallel bad check {done_count}/{data.shape[1]} channels",
                        )
        good_indices.sort()
        if progress_callback:
            progress_callback(progress_end, f"{label_prefix}: bad-channel check done ({data.shape[1]} channels)")
        return good_indices, bad_reasons, candidate_reasons

    def compute_lfp_snr_with_settings(
        self,
        raw_data: np.ndarray,
        settings: dict,
        progress=None,
        label_prefix: str = "LFP SNR",
        preprocessed_data: np.ndarray | None = None,
    ) -> dict:
        raw_data = np.asarray(raw_data, dtype=DATA_DTYPE)
        mode = str(settings.get("mode", "resting")).strip().lower()
        good_indices, bad_reasons, candidate_reasons = self.detect_bad_snr_channels_with_settings(
            raw_data,
            settings,
            progress_callback=progress,
            progress_start=0.0,
            progress_end=45.0,
            label_prefix=label_prefix,
        )
        stim_mask = None
        stim_interval = parse_float(settings.get("stim_interval", self.stim_interval_var.get())) if mode != "evoked" else None
        if mode == "evoked":
            if progress:
                progress(46.0, f"{label_prefix}: building stimulus mask")
            stim_mask = self.build_stim_mask_for_loaded_data(parse_float(settings.get("stim_duration", 0.1), 0.1))

        rows: list[dict] = []
        channel_indices = list(range(raw_data.shape[1]))
        if channel_indices:
            if progress:
                progress(50.0, f"{label_prefix}: preprocessing/filtering data")
            if preprocessed_data is not None and np.asarray(preprocessed_data).shape == raw_data.shape:
                data = np.asarray(preprocessed_data, dtype=DATA_DTYPE)[:, channel_indices]
            else:
                data = self.snr_preprocess_highpass_with_settings(raw_data[:, channel_indices], settings)
            signal_band = (
                parse_float(settings.get("signal_band_low", 1.0), 1.0),
                parse_float(settings.get("signal_band_high", 30.0), 30.0),
            )
            noise_band = (
                parse_float(settings.get("noise_band_low", 1.0), 1.0),
                parse_float(settings.get("noise_band_high", 200.0), 200.0),
            )
            filter_label = self.snr_filter_label_from_settings(settings)
            if mode == "resting":
                if progress:
                    progress(55.0, f"{label_prefix}: vectorized Welch PSD for {len(channel_indices)} channels")
                score_data = data
                if parse_bool_like(settings.get("notch_50hz"), False):
                    if preprocessed_data is None:
                        if progress:
                            progress(60.0, f"{label_prefix}: applying vectorized 50Hz notch")
                        score_data = self.apply_notch_filter_matrix(score_data, freq=50.0, q=30.0, harmonics=1, axis=0)
                if progress:
                    progress(70.0, f"{label_prefix}: integrating PSD bands")
                snr_values = self.compute_resting_snr_vectorized(score_data, signal_band, noise_band)
                for local_idx, source_idx in enumerate(channel_indices):
                    row = {
                        "channel": source_idx + 1,
                        "snr_db": float(snr_values[local_idx]),
                        "mode": mode,
                        "detail": f"{signal_band[0]}-{signal_band[1]}Hz / {noise_band[0]}-{noise_band[1]}Hz",
                    }
                    if filter_label:
                        row["detail"] = f"{filter_label}; {row['detail']}"
                    bad_detail = bad_reasons.get(int(row["channel"]))
                    if bad_detail:
                        row["detail"] = f"{bad_detail}; {row['detail']}"
                    candidate_detail = candidate_reasons.get(int(row["channel"]))
                    if candidate_detail:
                        row["detail"] = f"{candidate_detail}; {row['detail']}"
                    rows.append(row)
            else:
                total_to_score = max(1, len(channel_indices))
                for local_idx, source_idx in enumerate(channel_indices):
                    if progress:
                        value = 55.0 + 40.0 * local_idx / total_to_score
                        progress(value, f"{label_prefix}: scoring channels {local_idx + 1}/{len(channel_indices)}")
                    snr_rows = compute_batch_channel_snr(
                        data[:, local_idx:local_idx + 1],
                        self.fs,
                        mode=mode,
                        signal_band=signal_band,
                        noise_band=noise_band,
                        stim_interval=stim_interval,
                        stim_duration=parse_float(settings.get("stim_duration", 0.1), 0.1),
                        first_onset=None,
                        stim_freq=parse_float(settings.get("stim_freq", 10.0), 10.0),
                        harmonics=parse_int(settings.get("harmonics", 3), 3),
                        n_neighbor=parse_int(settings.get("neighbor_bins", 4), 4),
                        fft_length_sec=parse_float(settings.get("fft_len_sec", 2.0), 2.0),
                        notch=parse_bool_like(settings.get("notch_50hz"), False),
                        stim_mask=stim_mask,
                    )
                    if not snr_rows:
                        continue
                    row = dict(snr_rows[0])
                    row["channel"] = source_idx + 1
                    if filter_label:
                        row["detail"] = f"{filter_label}; {row['detail']}"
                    bad_detail = bad_reasons.get(int(row["channel"]))
                    if bad_detail:
                        row["detail"] = f"{bad_detail}; {row['detail']}"
                    candidate_detail = candidate_reasons.get(int(row["channel"]))
                    if candidate_detail:
                        row["detail"] = f"{candidate_detail}; {row['detail']}"
                    rows.append(row)
            if progress:
                progress(95.0, f"{label_prefix}: updating result table")

        rows = sorted(rows, key=lambda row: int(row["channel"]))
        finite_channels = {int(row["channel"]) for row in rows if np.isfinite(row["snr_db"])}
        healthy_channels = {int(row["channel"]) for row in rows if self.is_snr_row_healthy(row)}
        total_channels = raw_data.shape[1]
        bad_count = len(bad_reasons)
        candidate_count = len(candidate_reasons)
        bad_like_count = bad_count + candidate_count
        bad_pct = 100.0 * bad_like_count / total_channels if total_channels else 0.0
        return {
            "rows": rows,
            "mode": mode,
            "valid_channels": healthy_channels,
            "finite_channels": finite_channels,
            "healthy_channels": healthy_channels,
            "total_channels": total_channels,
            "valid_count": len(healthy_channels),
            "finite_count": len(finite_channels),
            "healthy_count": len(healthy_channels),
            "bad_count": bad_count,
            "candidate_count": candidate_count,
            "bad_like_count": bad_like_count,
            "bad_pct": bad_pct,
        }

    def apply_lfp_snr_param_row(self, row: dict):
        value = self.batch_param_value(row, "mode")
        if value is not None:
            self.snr_mode_var.set(str(value).strip().lower())
        value = self.batch_param_value(row, "threshold", "threshold_db", "red_if_db")
        if value is not None:
            self.snr_threshold_var.set(str(value))
        value = self.batch_param_value(row, "notch", "notch_50hz", "50hz", "50hz_notch")
        if value is not None:
            self.notch_var.set(parse_bool_like(value, self.notch_var.get()))
        value = self.batch_param_value(row, "skip_snr_scoring", "skip_snr", "skip_lfp_snr")
        if value is not None:
            self.skip_lfp_snr_var.set(parse_bool_like(value, self.skip_lfp_snr_var.get()))
        value = self.batch_param_value(row, "snr_filter", "filter", "preprocess")
        if value is not None:
            text = str(value).strip().lower()
            aliases = {"none": "off", "0": "off", "hp0.5": "0.5", "hp1": "1.0"}
            self.snr_highpass_var.set(aliases.get(text, text))
        value = self.batch_param_value(row, "snr_band_low", "pre_band_low", "band_low")
        if value is not None:
            self.snr_band_low_var.set(str(value))
        value = self.batch_param_value(row, "snr_band_high", "pre_band_high", "band_high")
        if value is not None:
            self.snr_band_high_var.set(str(value))
        value = self.batch_param_value(row, "hp_order", "highpass_order")
        if value is not None:
            self.filter_highpass_order_var.set(str(value))
        value = self.batch_param_value(row, "lp_order", "lowpass_order")
        if value is not None:
            self.filter_lowpass_order_var.set(str(value))
        value = self.batch_param_value(row, "motion_artifact_ica", "motion_ica", "ica_motion")
        if value is not None:
            self.motion_artifact_enable_var.set(parse_bool_like(value, self.motion_artifact_enable_var.get()))
            self.clear_motion_artifact_cache()
        value = self.batch_param_value(row, "motion_ica_components", "ica_components")
        if value is not None:
            self.motion_ica_components_var.set(str(value))
            self.clear_motion_artifact_cache()
        value = self.batch_param_value(row, "motion_ica_exclude", "ica_exclude")
        if value is not None:
            self.motion_ica_exclude_var.set(str(value))
            self.clear_motion_artifact_cache()
        value = self.batch_param_value(row, "motion_ica_lfreq", "ica_lfreq")
        if value is not None:
            self.motion_ica_lfreq_var.set(str(value))
            self.clear_motion_artifact_cache()
        value = self.batch_param_value(row, "motion_ica_hfreq", "ica_hfreq")
        if value is not None:
            self.motion_ica_hfreq_var.set(str(value))
            self.clear_motion_artifact_cache()
        value = self.batch_param_value(row, "motion_ica_decim", "ica_decim")
        if value is not None:
            self.motion_ica_decim_var.set(str(value))
            self.clear_motion_artifact_cache()
        value = self.batch_param_value(row, "motion_ica_max_iter", "ica_max_iter")
        if value is not None:
            self.motion_ica_max_iter_var.set(str(value))
            self.clear_motion_artifact_cache()
        value = self.batch_param_value(row, "bad_check")
        if value is not None:
            self.bad_channel_check_var.set(parse_bool_like(value, self.bad_channel_check_var.get()))
        value = self.batch_param_value(row, "bad_check_parallel", "parallel_bad_check", "parallel")
        if value is not None:
            self.bad_check_parallel_var.set(parse_bool_like(value, self.bad_check_parallel_var.get()))
        value = self.batch_param_value(row, "bad_check_workers", "bad_workers", "workers")
        if value is not None:
            self.bad_check_workers_var.set(str(value))
        value = self.batch_param_value(row, "flat_std")
        if value is not None:
            self.bad_flat_std_var.set(str(value))
        value = self.batch_param_value(row, "flat_ratio", "flat_ratio_pct")
        if value is not None:
            self.bad_flat_ratio_var.set(str(value))
        value = self.batch_param_value(row, "2s_win_bad_check", "two_s_win_bad_check", "window_bad_check")
        if value is not None:
            self.bad_window_check_var.set(parse_bool_like(value, self.bad_window_check_var.get()))
        value = self.batch_param_value(row, "valid_win_db", "window_snr_threshold")
        if value is not None:
            self.bad_window_snr_threshold_var.set(str(value))
        value = self.batch_param_value(row, "flat_win_ptp", "window_flat_ptp")
        if value is not None:
            self.bad_window_ptp_threshold_var.set(str(value))
        value = self.batch_param_value(row, "signal_low", "signal_band_low", "sig_low")
        if value is not None:
            self.signal_band_low_var.set(str(value))
        value = self.batch_param_value(row, "signal_high", "signal_band_high", "sig_high")
        if value is not None:
            self.signal_band_high_var.set(str(value))
        value = self.batch_param_value(row, "noise_low", "noise_band_low")
        if value is not None:
            self.noise_band_low_var.set(str(value))
        value = self.batch_param_value(row, "noise_high", "noise_band_high")
        if value is not None:
            self.noise_band_high_var.set(str(value))
        value = self.batch_param_value(row, "stim_interval")
        if value is not None:
            self.stim_interval_var.set(str(value))
        value = self.batch_param_value(row, "stim_duration")
        if value is not None:
            self.stim_duration_var.set(str(value))
        value = self.batch_param_value(row, "stim_freq")
        if value is not None:
            self.stim_freq_var.set(str(value))
        value = self.batch_param_value(row, "harmonics")
        if value is not None:
            self.harmonics_var.set(str(value))
        value = self.batch_param_value(row, "neighbor_bins", "n_neighbor")
        if value is not None:
            self.neighbor_bins_var.set(str(value))
        value = self.batch_param_value(row, "fft_len_sec", "fft_len", "fft_length_sec")
        if value is not None:
            self.fft_len_var.set(str(value))
        value = self.batch_param_value(row, "spike_filter_low", "spike_low")
        if value is not None:
            self.spike_filter_low_var.set(str(value))
        value = self.batch_param_value(row, "spike_filter_high", "spike_high")
        if value is not None:
            self.spike_filter_high_var.set(str(value))
        value = self.batch_param_value(row, "spike_threshold_factor", "spike_threshold", "threshold_factor")
        if value is not None:
            self.spike_threshold_factor_var.set(str(value))
        value = self.batch_param_value(row, "spike_window_sec", "spike_window")
        if value is not None:
            self.spike_window_sec_var.set(str(value))
        value = self.batch_param_value(row, "spike_step_ms", "spike_step")
        if value is not None:
            self.spike_step_ms_var.set(str(value))
        value = self.batch_param_value(row, "spike_refractory_ms", "spike_refractory")
        if value is not None:
            self.spike_refractory_ms_var.set(str(value))
        value = self.batch_param_value(row, "spike_pre_samples", "pre_samples")
        if value is not None:
            self.spike_pre_samples_var.set(str(value))
        value = self.batch_param_value(row, "spike_post_samples", "post_samples")
        if value is not None:
            self.spike_post_samples_var.set(str(value))
        value = self.batch_param_value(row, "spike_min_count", "min_spikes")
        if value is not None:
            self.spike_min_count_var.set(str(value))

    def csv_numeric_value(self, value, kind: str = "float"):
        if value is None:
            return ""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("'"):
                value = value[1:].strip()
            if not value:
                return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(number):
            return ""
        if kind == "int":
            return int(round(number))
        return float(number)

    def normalize_snr_export_row(self, row: dict, selected_channels: set[int] | None = None) -> dict:
        selected_channels = selected_channels if selected_channels is not None else self.selected_channels
        export_row = dict(row)
        channel = self.csv_numeric_value(row.get("channel"), kind="int")
        export_row["channel"] = channel
        export_row["selected"] = int(channel in selected_channels) if channel != "" else 0
        export_row["snr_db"] = self.csv_numeric_value(row.get("snr_db"), kind="float")
        if "column_index" in export_row:
            export_row["column_index"] = self.csv_numeric_value(row.get("column_index"), kind="int")
        return export_row

    def save_snr_rows_csv(self, path: Path, rows: list[dict], settings: dict | None = None):
        settings = settings or {}
        fieldnames = list(settings.keys())
        for name in ["selected", "channel", "snr_db", "mode", "detail"]:
            if name not in fieldnames:
                fieldnames.append(name)
        if any("column_index" in row for row in rows):
            fieldnames.insert(fieldnames.index("snr_db"), "column_index")
        export_rows = []
        for row in rows:
            export_row = dict(settings)
            export_row.update(self.normalize_snr_export_row(row))
            export_rows.append(export_row)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(export_rows)

    def save_snr_rows_csv_for_selection(
        self,
        path: Path,
        rows: list[dict],
        settings: dict | None = None,
        selected_channels: set[int] | None = None,
    ):
        settings = settings or {}
        selected_channels = selected_channels or set()
        fieldnames = list(settings.keys())
        for name in ["selected", "channel", "snr_db", "mode", "detail"]:
            if name not in fieldnames:
                fieldnames.append(name)
        if any("column_index" in row for row in rows) and "column_index" not in fieldnames:
            fieldnames.insert(fieldnames.index("snr_db"), "column_index")
        export_rows = []
        for row in rows:
            export_row = dict(settings)
            export_row.update(self.normalize_snr_export_row(row, selected_channels))
            export_rows.append(export_row)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(export_rows)

    def snr_summary_from_rows(self, rows: list[dict]) -> dict:
        finite_all = int(sum(1 for row in rows if np.isfinite(self.snr_db_value(row))))
        values = np.asarray([
            self.snr_db_value(row)
            for row in rows
            if self.is_snr_row_healthy(row) and np.isfinite(self.snr_db_value(row))
        ], dtype=DATA_DTYPE)
        n = int(values.size)
        masked_count = finite_all - n

        def count_pct(mask):
            count = int(np.count_nonzero(mask)) if n else 0
            return count, 100.0 * count / n if n else 0.0

        lt0_count, lt0_pct = count_pct(values < 0)
        mid_count, mid_pct = count_pct((values > 0) & (values < 3))
        gt4_count, gt4_pct = count_pct(values > 4)
        gt5_count, gt5_pct = count_pct(values > 5)
        gt6_count, gt6_pct = count_pct(values > 6)
        return {
            "finite_channels": finite_all,
            "healthy_channels": n,
            "masked_channels": masked_count,
            "median_snr_db": float(np.nanmedian(values)) if n else np.nan,
            "mean_snr_db": float(np.nanmean(values)) if n else np.nan,
            "snr_lt_0_count": lt0_count,
            "snr_lt_0_pct": lt0_pct,
            "snr_0_3_count": mid_count,
            "snr_0_3_pct": mid_pct,
            "snr_gt_4_count": gt4_count,
            "snr_gt_4_pct": gt4_pct,
            "snr_gt_5_count": gt5_count,
            "snr_gt_5_pct": gt5_pct,
            "snr_gt_6_count": gt6_count,
            "snr_gt_6_pct": gt6_pct,
        }

    def make_snr_boxplot_figure(self, rows: list[dict], settings: dict, title: str) -> Figure:
        values = np.asarray([
            self.snr_db_value(row)
            for row in rows
            if self.is_snr_row_healthy(row) and np.isfinite(self.snr_db_value(row))
        ], dtype=DATA_DTYPE)
        if values.size == 0:
            raise ValueError("No healthy SNR values are available to plot after masking bad channels.")
        summary = self.snr_summary_from_rows(rows)
        threshold = parse_float(settings.get("threshold_db", self.snr_threshold_var.get()), 5.0)
        fig = Figure(figsize=(8.6, 6.0), dpi=120)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.2)
        ax = fig.add_subplot(gs[0])
        stats_ax = fig.add_subplot(gs[1])
        ax.boxplot(
            [values],
            labels=["Healthy channels"],
            widths=0.35,
            patch_artist=True,
            boxprops={"facecolor": "#dbeafe", "edgecolor": "#1f77b4", "linewidth": 1.2},
            medianprops={"color": "#e15759", "linewidth": 1.5},
            whiskerprops={"color": "#666", "linewidth": 1.0},
            capprops={"color": "#666", "linewidth": 1.0},
            flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#666", "markersize": 4},
        )
        ax.axhline(y=threshold, color="#e15759", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("SNR (dB)", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        stats_ax.axis("off")
        text = (
            "Parameters\n"
            f"mode: {settings.get('mode', '')}\n"
            f"threshold: {threshold:g} dB\n"
            f"50Hz notch: {'on' if parse_bool_like(settings.get('notch_50hz'), False) else 'off'}\n"
            f"signal: {settings.get('signal_band_low')}-{settings.get('signal_band_high')} Hz\n"
            f"noise: {settings.get('noise_band_low')}-{settings.get('noise_band_high')} Hz\n"
            f"SNR filter: {settings.get('snr_filter')}\n\n"
            "SNR Summary\n"
            f"healthy channels: {summary['healthy_channels']}\n"
            f"masked finite channels: {summary['masked_channels']}\n"
            f"finite all channels: {summary['finite_channels']}\n"
            f"median: {summary['median_snr_db']:.3f} dB\n"
            f"mean: {summary['mean_snr_db']:.3f} dB\n\n"
            "Channel Count / Ratio\n"
            f"SNR < 0 dB: {summary['snr_lt_0_count']} ({summary['snr_lt_0_pct']:.1f}%)\n"
            f"0 < SNR < 3 dB: {summary['snr_0_3_count']} ({summary['snr_0_3_pct']:.1f}%)\n"
            f"SNR > 4 dB: {summary['snr_gt_4_count']} ({summary['snr_gt_4_pct']:.1f}%)\n"
            f"SNR > 5 dB: {summary['snr_gt_5_count']} ({summary['snr_gt_5_pct']:.1f}%)\n"
            f"SNR > 6 dB: {summary['snr_gt_6_count']} ({summary['snr_gt_6_pct']:.1f}%)"
        )
        stats_ax.text(
            0.02, 0.98, text, transform=stats_ax.transAxes,
            va="top", ha="left", fontsize=9.2, linespacing=1.3,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.95},
        )
        fig.tight_layout()
        return fig

    def run_lfp_snr_param_csv(self):
        csv_path = filedialog.askopenfilename(
            title="Select LFP SNR parameter CSV",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
        )
        if not csv_path:
            return
        out_dir = filedialog.askdirectory(title="Select output folder for LFP SNR results")
        if not out_dir:
            return
        out_root = Path(out_dir) / f"lfp_snr_param_scan_{time.strftime('%Y%m%d_%H%M%S')}"
        out_root.mkdir(parents=True, exist_ok=True)
        try:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                raw_rows = [
                    {str(k).strip().lower(): v for k, v in row.items() if k is not None}
                    for row in csv.DictReader(f)
                ]
            raw_rows = [row for row in raw_rows if any(str(v).strip() for v in row.values())]
            if not raw_rows:
                raise ValueError("Parameter CSV has no data rows.")
        except Exception as exc:
            messagebox.showerror("Read CSV failed", str(exc))
            return

        settings_list = [
            self.lfp_snr_settings_from_row(param_row, cfg_idx)
            for cfg_idx, param_row in enumerate(raw_rows, start=1)
        ]
        n_configs = len(settings_list)
        cpu_count = os.cpu_count() or 4
        default_workers = max(1, min(4, n_configs, cpu_count))
        max_workers = simpledialog.askinteger(
            "Parallel LFP SNR",
            "How many parameter sets should run in parallel?",
            initialvalue=default_workers,
            minvalue=1,
            maxvalue=max(1, min(n_configs, cpu_count)),
            parent=self,
        )
        if max_workers is None:
            max_workers = default_workers

        raw_data = np.asarray(self.current_data(), dtype=DATA_DTYPE)
        summary_rows: list[dict] = []
        progress_events: queue.Queue = queue.Queue()
        self.notebook.select(self.snr_tab)
        overall_start = time.perf_counter()

        def worker(settings: dict) -> dict:
            cfg_idx = parse_int(settings.get("config_index", 0), 0)
            config_name = str(settings.get("config_name", f"config_{cfg_idx:03d}"))

            def progress(value: float, phase: str):
                progress_events.put((cfg_idx, float(value), phase))

            result = self.compute_lfp_snr_with_settings(
                raw_data,
                settings,
                progress=progress,
                label_prefix=f"Param scan {cfg_idx}/{n_configs}",
            )
            rows = result["rows"]
            good_channels = [
                int(row["channel"])
                for row in rows
                if self.is_snr_row_healthy(row)
            ]
            lfp_dynamic_rows: list[dict] = []
            lfp_dynamic_status = ""
            if good_channels:
                try:
                    progress(97.0, f"Param scan {cfg_idx}/{n_configs}: computing 10s LFP dynamic SNR")
                    lfp_dynamic_data = raw_data[:, [ch - 1 for ch in good_channels]]
                    lfp_dynamic_rows = self.compute_lfp_dynamic_snr_rows_with_settings(
                        lfp_dynamic_data,
                        good_channels,
                        settings,
                        window_sec=10.0,
                    )
                    lfp_dynamic_status = "ok"
                except Exception as exc:
                    lfp_dynamic_status = f"skipped: {exc}"
            spike_results: list[dict] = []
            spike_dynamic_rows: list[dict] = []
            spike_dynamic_status = ""
            spike_params: dict = {}
            if parse_bool_like(settings.get("run_spike"), False) and good_channels:
                progress(98.0, f"Param scan {cfg_idx}/{n_configs}: computing Spike SNR on non-bad channels")
                spike_data = raw_data[:, [ch - 1 for ch in good_channels]]
                spike_params = self.parse_spike_params_from_settings(settings)
                spike_results = self.compute_spike_snr_for_channel_set(spike_data, good_channels, spike_params)
                try:
                    progress(99.0, f"Param scan {cfg_idx}/{n_configs}: computing 10s Spike dynamic SNR")
                    spike_dynamic_rows = self.compute_spike_dynamic_snr_rows(
                        spike_data,
                        good_channels,
                        spike_params,
                        window_sec=10.0,
                    )
                    spike_dynamic_status = "ok"
                except Exception as exc:
                    spike_dynamic_status = f"skipped: {exc}"
            progress(100.0, f"Param scan {cfg_idx}/{n_configs}: compute done")
            return {
                "settings": settings,
                "config_index": cfg_idx,
                "config_name": config_name,
                "result": result,
                "rows": rows,
                "good_channels": good_channels,
                "lfp_dynamic_rows": lfp_dynamic_rows,
                "lfp_dynamic_status": lfp_dynamic_status,
                "spike_results": spike_results,
                "spike_dynamic_rows": spike_dynamic_rows,
                "spike_dynamic_status": spike_dynamic_status,
                "spike_params": spike_params,
            }

        def drain_progress(done_count: int):
            latest = None
            while True:
                try:
                    latest = progress_events.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                cfg_idx, value, phase = latest
                overall = 100.0 * (done_count + value / 100.0) / max(1, n_configs)
            else:
                overall = 100.0 * done_count / max(1, n_configs)
                phase = f"Param scan running with {max_workers} workers"
            elapsed = time.perf_counter() - overall_start
            self.set_snr_progress(
                overall,
                f"{phase} | overall {overall:.0f}% | done {done_count}/{n_configs} | elapsed {elapsed:.1f}s",
            )

        futures = {}
        done_count = 0
        last_payload = None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for settings in settings_list:
                futures[executor.submit(worker, settings)] = settings
            pending = set(futures)
            while pending:
                drain_progress(done_count)
                finished = [future for future in pending if future.done()]
                if not finished:
                    try:
                        self.update()
                    except tk.TclError:
                        pass
                    time.sleep(0.08)
                    continue
                for future in finished:
                    pending.remove(future)
                    settings = futures[future]
                    config_name = str(settings.get("config_name", "config"))
                    config_dir = out_root / config_name
                    config_dir.mkdir(parents=True, exist_ok=True)
                    done_count += 1
                    try:
                        payload = future.result()
                        rows = payload["rows"]
                        result = payload["result"]
                        good_channels = payload["good_channels"]
                        result_csv = config_dir / f"{config_name}_snr_rows.csv"
                        self.save_snr_rows_csv_for_selection(result_csv, rows, settings, set(good_channels))

                        figure_path = ""
                        if any(np.isfinite(row["snr_db"]) for row in rows):
                            fig = self.make_snr_boxplot_figure(rows, settings, f"{config_name} SNR distribution")
                            figure_path = str(config_dir / f"{config_name}_snr_boxplot.png")
                            fig.savefig(figure_path, dpi=300, bbox_inches="tight")

                        spike_csv = ""
                        spike_fig = ""
                        lfp_dynamic_csv = ""
                        lfp_dynamic_fig = ""
                        lfp_dynamic_rows = payload["lfp_dynamic_rows"]
                        if lfp_dynamic_rows:
                            lfp_dynamic_csv = str(config_dir / f"{config_name}_lfp_dynamic_10s.csv")
                            lfp_dynamic_fig = str(config_dir / f"{config_name}_lfp_dynamic_10s.png")
                            self.save_dynamic_snr_rows_csv(Path(lfp_dynamic_csv), lfp_dynamic_rows)
                            self.make_dynamic_snr_figure(
                                lfp_dynamic_rows,
                                f"{config_name} LFP SNR 10s dynamic median",
                                "LFP median SNR (dB)",
                            ).savefig(lfp_dynamic_fig, dpi=300, bbox_inches="tight")

                        spike_valid = 0
                        spike_dynamic_csv = ""
                        spike_dynamic_fig = ""
                        spike_results = payload["spike_results"]
                        if spike_results:
                            spike_csv = str(config_dir / f"{config_name}_spike_snr_rows.csv")
                            self.save_spike_results_csv_rows(Path(spike_csv), spike_results)
                            spike_valid = int(sum(1 for row in spike_results if np.isfinite(row["snr_db"])))
                            if spike_valid:
                                spike_fig = str(config_dir / f"{config_name}_spike_snr_summary.png")
                                self.make_spike_snr_figure_for_results(
                                    spike_results,
                                    self.spike_figure_params_from_settings(settings),
                                ).savefig(spike_fig, dpi=300, bbox_inches="tight")
                            spike_dynamic_rows = payload["spike_dynamic_rows"]
                            if spike_dynamic_rows:
                                spike_dynamic_csv = str(config_dir / f"{config_name}_spike_dynamic_10s.csv")
                                spike_dynamic_fig = str(config_dir / f"{config_name}_spike_dynamic_10s.png")
                                self.save_dynamic_snr_rows_csv(Path(spike_dynamic_csv), spike_dynamic_rows)
                                self.make_dynamic_snr_figure(
                                    spike_dynamic_rows,
                                    f"{config_name} Spike SNR 10s dynamic median",
                                    "Spike median SNR (dB)",
                                ).savefig(spike_dynamic_fig, dpi=300, bbox_inches="tight")

                        summary = self.snr_summary_from_rows(rows)
                        summary_row = dict(settings)
                        summary_row.update(summary)
                        summary_row.update({
                            "status": "ok",
                            "result_csv": str(result_csv),
                            "figure_png": figure_path,
                            "total_channels": result["total_channels"],
                            "bad_count": result["bad_count"],
                            "candidate_count": result["candidate_count"],
                            "bad_like_count": result["bad_like_count"],
                            "bad_pct": result["bad_pct"],
                            "good_channels_for_spike": len(good_channels),
                            "lfp_dynamic_csv": lfp_dynamic_csv,
                            "lfp_dynamic_figure_png": lfp_dynamic_fig,
                            "lfp_dynamic_status": payload.get("lfp_dynamic_status", ""),
                            "run_spike": int(parse_bool_like(settings.get("run_spike"), False)),
                            "spike_csv": spike_csv,
                            "spike_figure_png": spike_fig,
                            "spike_valid_snr": spike_valid,
                            "spike_dynamic_csv": spike_dynamic_csv,
                            "spike_dynamic_figure_png": spike_dynamic_fig,
                            "spike_dynamic_status": payload.get("spike_dynamic_status", ""),
                        })
                        summary_rows.append(summary_row)
                        last_payload = payload
                        self.log(f"LFP param scan saved: {config_dir}")
                    except Exception as exc:
                        self.log(traceback.format_exc())
                        summary_rows.append({
                            "config_index": settings.get("config_index", ""),
                            "config_name": config_name,
                            "status": f"error: {exc}",
                        })
                try:
                    self.update()
                except tk.TclError:
                    pass

        if last_payload is not None:
            last_settings = last_payload["settings"]
            self.apply_lfp_snr_param_row(last_settings)
            self.last_snr_rows = last_payload["rows"]
            self.selected_channels = set(last_payload["good_channels"])
            self.refresh_snr_tree(threshold=parse_float(last_settings.get("threshold_db", 5), 5))
            self.update_selected_summary()
            if last_payload["spike_results"]:
                self.spike_results = last_payload["spike_results"]
                self.refresh_spike_tree()

        summary_path = out_root / "lfp_snr_param_scan_summary.csv"
        all_fields: list[str] = []
        for row in summary_rows:
            for key in row.keys():
                if key not in all_fields:
                    all_fields.append(key)
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        self.set_snr_progress(100.0, f"LFP SNR parameter scan done: {n_configs} configs -> {out_root}")
        messagebox.showinfo("LFP SNR scan done", f"Finished {n_configs} parameter sets.\n\nSaved to:\n{out_root}")

    def export_snr_csv(self):
        if not self.last_snr_rows:
            messagebox.showerror("No SNR rows", "Run LFP SNR first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            export_rows = []
            for row in self.last_snr_rows:
                export_rows.append(self.normalize_snr_export_row(row))
            fieldnames = ["selected", "channel", "snr_db", "mode", "detail"]
            if any("column_index" in row for row in self.last_snr_rows):
                fieldnames.insert(2, "column_index")
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(export_rows)
        self.log(f"Exported SNR CSV: {path}")

    def plot_snr_db_by_channel(self):
        if not self.last_snr_rows:
            messagebox.showerror("No SNR rows", "Run LFP SNR first.")
            return

        snrs = [
            self.snr_db_value(row)
            for row in self.last_snr_rows
            if self.is_snr_row_healthy(row) and np.isfinite(self.snr_db_value(row))
        ]
        if not snrs:
            messagebox.showwarning("No valid SNR", "No healthy SNR values are available after masking bad channels.")
            return

        values = np.asarray(snrs, dtype=DATA_DTYPE)
        row_modes = sorted({
            str(row.get("mode", "")).strip()
            for row in self.last_snr_rows
            if str(row.get("mode", "")).strip()
        })
        mode = row_modes[0] if len(row_modes) == 1 else (", ".join(row_modes[:3]) if row_modes else self.snr_mode_var.get())
        if len(row_modes) > 3:
            mode += "..."
        settings = dict(self.current_lfp_snr_settings())
        settings["mode"] = mode
        try:
            fig = self.make_snr_boxplot_figure(
                self.last_snr_rows,
                settings,
                f"{mode} SNR overall distribution",
            )
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("SNR boxplot failed", str(exc))
            return

        dialog = tk.Toplevel(self)
        dialog.title(f"SNR dB boxplot - {mode}")
        dialog.geometry("900x680")
        dialog.transient(self)

        plot_frame = ttk.Frame(dialog)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=(10, 4))
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        toolbar = NavigationToolbar2Tk(canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

        def save_snr_boxplot():
            default_name = f"{mode}_SNR_boxplot.png"
            path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Save SNR boxplot",
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[
                    ("PNG image", "*.png"),
                    ("PDF file", "*.pdf"),
                    ("SVG file", "*.svg"),
                    ("JPEG image", "*.jpg;*.jpeg"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return
            try:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                self.log(f"Saved SNR boxplot: {path}")
                messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=dialog)
            except Exception as exc:
                self.log(traceback.format_exc())
                messagebox.showerror("Save failed", str(exc), parent=dialog)

        button_bar = ttk.Frame(dialog)
        button_bar.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(button_bar, text="Save figure", command=save_snr_boxplot).pack(side="right")
        return

        threshold = parse_float(self.snr_threshold_var.get(), 5.0)
        summary = self.snr_summary_from_rows(self.last_snr_rows)

        dialog = tk.Toplevel(self)
        dialog.title(f"SNR dB boxplot - {mode}")
        dialog.geometry("860x620")
        dialog.transient(self)

        n_channels = values.size
        median_snr = float(np.nanmedian(values))

        def count_pct(mask: np.ndarray) -> tuple[int, float]:
            count = int(np.count_nonzero(mask))
            pct = 100.0 * count / n_channels if n_channels else 0.0
            return count, pct

        lt0_count, lt0_pct = count_pct(values < 0)
        between0_3_count, between0_3_pct = count_pct((values > 0) & (values < 3))
        gt4_count, gt4_pct = count_pct(values > 4)
        gt5_count, gt5_pct = count_pct(values > 5)
        gt6_count, gt6_pct = count_pct(values > 6)

        fig = Figure(figsize=(8.6, 6.0), dpi=100)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.18)
        ax = fig.add_subplot(gs[0])
        stats_ax = fig.add_subplot(gs[1])
        ax.boxplot(
            [values],
            labels=["Healthy channels"],
            widths=0.35,
            patch_artist=True,
            boxprops={"facecolor": "#dbeafe", "edgecolor": "#1f77b4", "linewidth": 1.2},
            medianprops={"color": "#e15759", "linewidth": 1.5},
            whiskerprops={"color": "#666", "linewidth": 1.0},
            capprops={"color": "#666", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "#666",
                "markersize": 4,
                "alpha": 0.85,
            },
        )
        ax.axhline(y=threshold, color="#e15759", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_title(f"{mode} SNR overall distribution", fontsize=12)
        ax.set_ylabel("SNR (dB)", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        stats_ax.axis("off")

        signal_line = f"signal: {self.signal_band_low_var.get()}-{self.signal_band_high_var.get()} Hz"
        noise_line = f"noise: {self.noise_band_low_var.get()}-{self.noise_band_high_var.get()} Hz"

        upper_text = (
            "Parameters\n"
            f"mode: {mode}\n"
            f"threshold: {threshold:g} dB\n"
            f"50Hz notch: {'on' if self.notch_var.get() else 'off'}"
        )
        red_text = f"{signal_line}\n{noise_line}"
        lower_text = (
            f"channels: {n_channels}\n\n"
            f"masked finite channels: {summary['masked_channels']}\n"
            f"finite all channels: {summary['finite_channels']}\n\n"
            "SNR Summary\n"
            f"median: {median_snr:.3f} dB\n\n"
            "Channel Count / Ratio\n"
            f"SNR < 0 dB: {lt0_count} ({lt0_pct:.1f}%)\n"
            f"0 < SNR < 3 dB: {between0_3_count} ({between0_3_pct:.1f}%)\n"
            f"SNR > 4 dB: {gt4_count} ({gt4_pct:.1f}%)\n"
            f"SNR > 5 dB: {gt5_count} ({gt5_pct:.1f}%)\n"
            f"SNR > 6 dB: {gt6_count} ({gt6_pct:.1f}%)"
        )

        text_kw = {
            "transform": stats_ax.transAxes,
            "fontsize": 9.5,
            "va": "top",
            "ha": "left",
            "linespacing": 1.35,
        }

        y_start = 0.98

        # Draw layered text blocks; red band info is positioned after measuring text height.
        t1 = stats_ax.text(0.02, y_start, upper_text, color="black", zorder=5, **text_kw)
        t2 = stats_ax.text(0.02, y_start, red_text, color="red", zorder=10, **text_kw)
        t3 = stats_ax.text(
            0.02,
            y_start,
            lower_text,
            color="black",
            **text_kw,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.95},
        )

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=dialog)

        # Draw once so text extents are available.
        canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = stats_ax.get_window_extent()

        # Place the red block under the upper black block.
        h1 = t1.get_window_extent(renderer=renderer).height / ax_bbox.height
        y2 = y_start - h1
        t2.set_position((0.02, y2))

        # Place the summary block under the red block.
        h2 = t2.get_window_extent(renderer=renderer).height / ax_bbox.height
        y3 = y2 - h2
        t3.set_position((0.02, y3))

        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 4))

        def save_snr_boxplot():
            default_name = f"{mode}_SNR_boxplot.png"
            path = filedialog.asksaveasfilename(
                parent=dialog,
                title="Save SNR boxplot",
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[
                    ("PNG image", "*.png"),
                    ("PDF file", "*.pdf"),
                    ("SVG file", "*.svg"),
                    ("JPEG image", "*.jpg;*.jpeg"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return
            try:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                self.log(f"Saved SNR boxplot: {path}")
                messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=dialog)
            except Exception as exc:
                self.log(traceback.format_exc())
                messagebox.showerror("Save failed", str(exc), parent=dialog)

        button_bar = ttk.Frame(dialog)
        button_bar.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(button_bar, text="Save figure", command=save_snr_boxplot).pack(side="right")

if __name__ == "__main__":
    app = IntegratedPipelineGUI()
    app.mainloop()




