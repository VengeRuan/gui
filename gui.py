"""Stable launcher and targeted UI fixes for the SD analysis GUI."""

from __future__ import annotations

import importlib.machinery
import os
import re
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path


_RUNTIME_DIR = Path(__file__).resolve().parent
# gui_core.pyc is loaded manually, so its sibling pure-Python dependencies are
# not discovered by Python/PyInstaller unless this directory is importable.
if str(_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DIR))

_CORE_PATH = _RUNTIME_DIR / "gui_core.pyc"
if not _CORE_PATH.is_file():
    raise RuntimeError(f"Missing GUI runtime: {_CORE_PATH}")

_core = importlib.machinery.SourcelessFileLoader("_sd_gui_core", str(_CORE_PATH)).load_module()


def _restore_excel_dependencies_after_pyarrow_block() -> None:
    """Restore Pandas in the compiled GUI after its optional PyArrow import fails.

    The application-control policy blocks PyArrow's native compute extension.
    Pandas and OpenPyXL themselves are installed and work without it, but the
    compiled module records ``pd = None`` when its first import happens before
    PyArrow is isolated.
    """
    # A failed first Pandas import leaves partially initialized ``pandas.*``
    # modules in sys.modules.  Remove only those incomplete import entries
    # before reloading Pandas with PyArrow explicitly disabled.
    for module_name in list(sys.modules):
        if (
            module_name == "pandas"
            or module_name.startswith("pandas.")
            or module_name == "seaborn"
            or module_name.startswith("seaborn.")
        ):
            sys.modules.pop(module_name, None)
    sys.modules["pyarrow"] = None
    sys.modules["pyarrow.compute"] = None
    try:
        import openpyxl
        import pandas as pd
    except ImportError:
        return
    _core.pd = pd
    _core.openpyxl = openpyxl


_restore_excel_dependencies_after_pyarrow_block()


def _walk_widgets(widget):
    for child in widget.winfo_children():
        yield child
        yield from _walk_widgets(child)


def _force_page_scroll_top(self, page_key: str) -> None:
    """Keep a dynamically expanded page at its top after Tk reflows it."""
    try:
        canvas = self.page_scroll_canvases.get(page_key)
        if canvas is None:
            return

        def reset_scroll() -> None:
            try:
                canvas.update_idletasks()
                bbox = canvas.bbox("all")
                if bbox:
                    canvas.configure(scrollregion=bbox)
                canvas.yview_moveto(0.0)
            except Exception:
                pass

        # The first callback handles the normal idle refresh; the delayed
        # callback handles the second geometry pass caused by the expanded
        # parameter card.
        self.after_idle(reset_scroll)
        self.after(120, reset_scroll)
    except Exception:
        pass


def _pin_static_page_to_top(self, page_key: str) -> None:
    """Make short (initial) pages behave like fixed, top-aligned panels."""
    try:
        canvas = self.page_scroll_canvases.get(page_key)
        if canvas is None:
            return
        canvas.update_idletasks()
        bbox = canvas.bbox("all")
        viewport_height = canvas.winfo_height()
        viewport_width = canvas.winfo_width()
        if not bbox or viewport_height <= 1 or viewport_width <= 1:
            return
        content_height = bbox[3] - bbox[1]
        if content_height <= viewport_height:
            # Match the scrollregion to the viewport.  This keeps a short
            # page pinned at y=0 and leaves no vertical range to scroll.
            canvas.configure(
                scrollregion=(0, 0, max(viewport_width, bbox[2]), viewport_height)
            )
            canvas.yview_moveto(0.0)
    except Exception:
        pass


_original_analysis_gui_init = _core.AnalysisGUI.__init__


def _analysis_gui_init_with_static_initial_pages(self, *args, **kwargs) -> None:
    _original_analysis_gui_init(self, *args, **kwargs)
    for page_key, canvas in getattr(self, "page_scroll_canvases", {}).items():
        def pin_on_resize(_event=None, key=page_key):
            self.after_idle(lambda: _pin_static_page_to_top(self, key))

        canvas.bind("<Configure>", pin_on_resize, add="+")
        self.after_idle(lambda key=page_key: _pin_static_page_to_top(self, key))
    # The first geometry pass can occur after idle on Windows/Tk.
    self.after(120, lambda: [
        _pin_static_page_to_top(self, key)
        for key in getattr(self, "page_scroll_canvases", {})
    ])


_core.AnalysisGUI.__init__ = _analysis_gui_init_with_static_initial_pages


def _remove_skip_snr_controls(widget) -> None:
    for child in list(_walk_widgets(widget)):
        try:
            if "skip snr" in str(child.cget("text")).lower():
                child.destroy()
        except Exception:
            pass


_original_analysis_build_snr = _core.AnalysisGUI._build_snr


def _build_snr_without_skip(self, parent) -> None:
    _original_analysis_build_snr(self, parent)
    self.snr_skip_var.set(False)
    _remove_skip_snr_controls(parent)


_core.AnalysisGUI._build_snr = _build_snr_without_skip


_original_toggle_preprocess_parameters = _core.AnalysisGUI._toggle_preprocess_parameters


def _toggle_preprocess_parameters_keep_top(self) -> None:
    _original_toggle_preprocess_parameters(self)
    try:
        self._refresh_page_scroll("preprocess", move_to_top=True)
    except Exception:
        pass
    _force_page_scroll_top(self, "preprocess")


_core.AnalysisGUI._toggle_preprocess_parameters = _toggle_preprocess_parameters_keep_top


_original_toggle_import_parameters = _core.AnalysisGUI._toggle_import_parameters


def _toggle_import_parameters_keep_top(self) -> None:
    _original_toggle_import_parameters(self)
    try:
        self._refresh_page_scroll("import", move_to_top=True)
    except Exception:
        pass
    _force_page_scroll_top(self, "import")


_core.AnalysisGUI._toggle_import_parameters = _toggle_import_parameters_keep_top


_original_embedded_build_snr = _core._EmbeddedPipelineGUI._build_snr_tab


def _build_embedded_snr_without_skip(self) -> None:
    _original_embedded_build_snr(self)
    self.skip_lfp_snr_var.set(False)
    _remove_skip_snr_controls(self.snr_tab)


_core._EmbeddedPipelineGUI._build_snr_tab = _build_embedded_snr_without_skip


_original_sync_inputs = _core.AnalysisGUI._sync_pipeline_inputs


def _sync_inputs_without_skip(self):
    self.snr_skip_var.set(False)
    pipeline = _original_sync_inputs(self)
    if pipeline is not None:
        pipeline.skip_lfp_snr_var.set(False)
    return pipeline


_core.AnalysisGUI._sync_pipeline_inputs = _sync_inputs_without_skip


_original_read_mat = _core._EmbeddedPipelineGUI.read_mat


def _read_mat_with_channel_axis(self, path: Path):
    """Normalize single-channel H5 data before the compiled loader uses it."""
    data, fs, time_vec, metadata = _original_read_mat(self, path)
    array = _core.np.asarray(data)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim != 2:
        raise ValueError(
            f"H5/MAT signal must be 1D or 2D, but the loaded shape is {array.shape}."
        )

    normalized_metadata = dict(metadata or {})
    normalized_metadata.setdefault("channel_count", int(array.shape[1]))
    return array, fs, time_vec, normalized_metadata


_core._EmbeddedPipelineGUI.read_mat = _read_mat_with_channel_axis


def _normalize_samples_by_channels_safely(data):
    """Normalize legacy channel/sample layouts without transposing short H5."""
    array = _core.np.asarray(data)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Signal data must be 1D or 2D, got shape {array.shape}.")

    common_channel_counts = {1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 384, 512, 520}
    if (
        array.shape[0] in common_channel_counts
        and array.shape[1] not in common_channel_counts
    ):
        array = array.T
    elif (
        array.shape[1] not in common_channel_counts
        and array.shape[0] <= 520
        and array.shape[1] > array.shape[0]
    ):
        array = array.T
    return _core.np.asarray(array, dtype=_core.DATA_DTYPE)


_core.normalize_samples_by_channels = _normalize_samples_by_channels_safely


def _ensure_data_millivolts_with_channel_axis(self, data, *args, **kwargs):
    """Keep the pipeline's canonical signal layout as samples x channels."""
    unit = kwargs.get("unit", args[0] if args else "")
    source_label = kwargs.get(
        "source_label",
        args[1] if len(args) > 1 else "data",
    )
    array = _core.np.asarray(data)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim != 2:
        raise ValueError(
            f"Signal data must be 1D or 2D, but the loaded shape is {array.shape}."
        )

    # The H5 reader has already established samples x channels.  Do not call
    # the compiled normalizer here: its legacy short-array heuristic can
    # transpose a valid (500 samples, 512 channels) H5 file.
    array = _core.np.asarray(array, dtype=_core.DATA_DTYPE)
    unit_text = _core.data_unit_text(unit)
    should_convert = unit_text in {"v", "volt", "volts"}
    if not should_convert and not unit_text:
        should_convert = _core.looks_like_legacy_volts(array)
    if should_convert:
        self.log(f"Converted {source_label} from V to mV.")
        return _core.np.asarray(array * _core.V_TO_MV, dtype=_core.DATA_DTYPE)
    return array


_core._EmbeddedPipelineGUI.ensure_data_millivolts = _ensure_data_millivolts_with_channel_axis


def _render_all_raw_preview_with_single_channel(self):
    """Render both matrix data and a one-channel H5 signal safely."""
    if not self.has_loaded_data():
        return

    # A lazily loaded HDF5 source must never be promoted to a dense matrix for
    # display.  Read only the selected time window and channel.
    if getattr(self, "_lazy_h5_info", None) is not None:
        return _render_lazy_h5_single_channel_preview(self)

    data = _core.np.asarray(self.current_data())
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"Preview data must be 1D or 2D, got shape {data.shape}.")

    processed_settings = getattr(self, "preprocessed_filter_settings", {})
    processed_mode = str(processed_settings.get("mode", "off")).strip().lower()
    if getattr(self, "preprocessed_data", None) is not None:
        source_label = "滤波后" if processed_mode != "off" else "重映射后（未滤波）"
    elif self.remapped_data is not None:
        source_label = "重映射后"
    else:
        source_label = "原始数据"
    if not self.all_raw_preview_channels:
        self.all_raw_preview_channels = list(range(1, data.shape[1] + 1))
    self.all_raw_preview_index = max(
        0,
        min(self.all_raw_preview_index, len(self.all_raw_preview_channels) - 1),
    )
    channel = self.all_raw_preview_channels[self.all_raw_preview_index]
    channel = max(1, min(int(channel), data.shape[1]))

    start = max(0, int(_core.parse_float(self.preview_start_var.get()) * self.fs))
    duration = max(1, int(_core.parse_float(self.preview_duration_var.get(), 5) * self.fs))
    end = min(data.shape[0], start + duration)
    if end <= start:
        start = 0
        end = min(data.shape[0], duration)

    x = _core.np.arange(start, end) / self.fs
    y = _core.np.asarray(data[start:end, channel - 1], dtype=_core.DATA_DTYPE).ravel()

    self.preview_fig.clear()
    axis = self.preview_fig.add_subplot(111)
    linewidth_var = getattr(self, "preview_linewidth_var", None)
    linewidth_text = linewidth_var.get() if linewidth_var is not None else "0.45"
    linewidth = max(0.1, _core.parse_float(linewidth_text, 0.45))
    axis.plot(x, y, linewidth=linewidth, alpha=0.95, antialiased=True, label=f"ch{channel}")
    axis.set_xlabel("Time (sec)")
    axis.set_ylabel("Voltage (mV)")
    axis.set_title(
        f"{source_label} channel preview - ch{channel} "
        f"({self.all_raw_preview_index + 1}/{len(self.all_raw_preview_channels)})"
    )
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(alpha=0.25)
    self.preview_fig.tight_layout()
    self.preview_canvas.draw()
    self.all_raw_preview_status_var.set(
        f"{source_label}全通道预览："
        f"ch{channel} ({self.all_raw_preview_index + 1}/{len(self.all_raw_preview_channels)})"
    )


_core._EmbeddedPipelineGUI.render_all_raw_preview = _render_all_raw_preview_with_single_channel


def _render_lazy_h5_single_channel_preview(self):
    info = self._lazy_h5_info
    rows, channels = info["shape"]
    fs = float(info["fs"])
    if not _core.np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid sampling rate: {fs!r}.")
    if not self.all_raw_preview_channels:
        self.all_raw_preview_channels = list(range(1, channels + 1))
    self.all_raw_preview_index = max(0, min(self.all_raw_preview_index, len(self.all_raw_preview_channels) - 1))
    channel = max(1, min(int(self.all_raw_preview_channels[self.all_raw_preview_index]), channels))
    start = max(0, int(_core.parse_float(self.preview_start_var.get(), 0.0) * fs))
    duration = max(1, int(_core.parse_float(self.preview_duration_var.get(), 5.0) * fs))
    end = min(rows, start + duration)
    if end <= start:
        start, end = 0, min(rows, duration)
    y = _lazy_h5_read_slice(self, start, end, channel - 1).ravel()
    x = info.get("time_offset", 0.0) + _core.np.arange(start, end, dtype=float) / fs
    self.preview_fig.clear()
    axis = self.preview_fig.add_subplot(111)
    linewidth_var = getattr(self, "preview_linewidth_var", None)
    linewidth = max(0.1, _core.parse_float(linewidth_var.get() if linewidth_var is not None else 0.45, 0.45))
    axis.plot(x, y, linewidth=linewidth, alpha=0.95, antialiased=True, label=f"ch{channel}")
    axis.set_xlabel("Time (sec)")
    axis.set_ylabel("Voltage (mV)")
    axis.set_title(f"原始数据通道预览 - ch{channel} ({self.all_raw_preview_index + 1}/{len(self.all_raw_preview_channels)})")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(alpha=0.25)
    self.preview_fig.tight_layout()
    self.preview_canvas.draw()
    self.all_raw_preview_status_var.set(
        f"原始数据（HDF5懒加载） ch{channel} ({self.all_raw_preview_index + 1}/{len(self.all_raw_preview_channels)}) "
        f"| {start / fs:.3f}-{end / fs:.3f} s"
    )


_original_start_all_raw_preview = _core._EmbeddedPipelineGUI.start_all_raw_preview


def _start_all_raw_preview_with_imported_channels(self):
    """Skip placeholder columns when only dispersed channel files were loaded."""
    imported_ids = sorted(
        int(channel)
        for channel in (getattr(self, "imported_channel_ids", set()) or set())
        if int(channel) >= 1
    )
    if not imported_ids:
        if getattr(self, "_lazy_h5_info", None) is not None:
            self.all_raw_preview_enabled = True
            self.update_all_raw_preview_button()
            channels = int(self._lazy_h5_info["shape"][1])
            self.all_raw_preview_channels = list(range(1, channels + 1))
            self.all_raw_preview_index = 0
            self.render_all_raw_preview()
            return
        return _original_start_all_raw_preview(self)
    if not self.has_loaded_data():
        return
    data = _core.np.asarray(self.current_data())
    if data.ndim == 1:
        data = data[:, None]
    self.all_raw_preview_enabled = True
    self.update_all_raw_preview_button()
    self.all_raw_preview_channels = [channel for channel in imported_ids if channel <= data.shape[1]]
    self.all_raw_preview_index = 0
    if self.all_raw_preview_channels:
        self.render_all_raw_preview()


_core._EmbeddedPipelineGUI.start_all_raw_preview = _start_all_raw_preview_with_imported_channels


_original_run_lfp_snr = _core._EmbeddedPipelineGUI.run_lfp_snr


def _run_lfp_snr_without_skip(self, on_done=None):
    self.skip_lfp_snr_var.set(False)
    return _original_run_lfp_snr(self, on_done=on_done)


_core._EmbeddedPipelineGUI.run_lfp_snr = _run_lfp_snr_without_skip


class _LfpSnrCancelled(Exception):
    """Internal cooperative-cancellation signal for an LFP SNR run."""


def _raise_if_lfp_snr_cancelled(pipeline) -> None:
    if getattr(pipeline, "_lfp_snr_cancel_requested", False):
        raise _LfpSnrCancelled()


def _detect_bad_snr_channels_cancellable(
    self,
    data,
    settings,
    progress_callback=None,
    progress_start: float = 0.0,
    progress_end: float = 45.0,
    label_prefix: str = "LFP SNR",
):
    """Bad-channel check that can stop pending parallel channel jobs."""
    if not _core.parse_bool_like(settings.get("bad_check"), True):
        return list(range(data.shape[1])), {}, {}

    data = _core.np.asarray(data, dtype=_core.DATA_DTYPE)
    flat_std_thr = _core.parse_float(settings.get("flat_std", 1e-4), 1e-4)
    flat_ratio_thr = _core.parse_float(settings.get("flat_ratio_pct", 30.0), 30.0) / 100.0
    good_indices, bad_reasons, candidate_reasons = [], {}, {}
    n_channels = max(1, data.shape[1])

    def check_channel(idx):
        sig = _core.np.asarray(data[:, idx], dtype=_core.DATA_DTYPE)
        finite = sig[_core.np.isfinite(sig)]
        ch = idx + 1
        if finite.size < 10:
            return idx, ch, "Bad Channel: too few finite samples", None
        sig_std = float(_core.np.nanstd(finite))
        sig_range = float(_core.np.nanmax(finite) - _core.np.nanmin(finite))
        if sig_std <= flat_std_thr or sig_range <= flat_std_thr * 6:
            return idx, ch, f"Bad Channel: flat/dead, std={sig_std:.3g}", None
        quant_step = max(flat_std_thr, 1e-4)
        unique_count = int(_core.np.unique(_core.np.round(finite / quant_step)).size)
        if unique_count <= 8 or unique_count / finite.size <= 5e-4:
            return idx, ch, f"Bad Channel: too few unique levels, unique={unique_count}", None
        flat_ratio = self.flat_time_ratio_with_spike_tolerance(sig, flat_std_thr)
        if flat_ratio_thr > 0 and flat_ratio >= flat_ratio_thr:
            return idx, ch, f"Bad Channel: flat time={flat_ratio * 100:.1f}%", None
        if _core.parse_bool_like(settings.get("two_s_win_bad_check"), False):
            valid_ratio, valid_windows, used_windows = self.snr_valid_window_ratio_with_settings(sig, settings)
            if valid_ratio is not None and valid_ratio < 0.5:
                threshold = _core.parse_float(settings.get("valid_win_db", 1.0), 1.0)
                return idx, ch, (
                    f"Bad Channel: 2s window SNR valid ratio={valid_ratio * 100:.1f}% "
                    f"({valid_windows}/{used_windows}; normal if >{threshold:g} dB)"
                ), None
        artifact_reasons = self.high_frequency_artifact_reasons_with_settings(sig, settings)
        if artifact_reasons:
            return idx, ch, None, "Bad Channel candidate: " + "; ".join(artifact_reasons)
        return idx, ch, None, None

    max_workers = self.bad_check_max_workers_from_settings(settings, data.shape[1])
    if max_workers <= 1:
        for idx in range(data.shape[1]):
            _raise_if_lfp_snr_cancelled(self)
            if progress_callback:
                value = progress_start + (progress_end - progress_start) * idx / n_channels
                progress_callback(value, f"{label_prefix}: checking bad channels {idx + 1}/{n_channels}")
            result_idx, ch, bad_detail, candidate_detail = check_channel(idx)
            if bad_detail:
                bad_reasons[ch] = bad_detail
            else:
                if candidate_detail:
                    candidate_reasons[ch] = candidate_detail
                good_indices.append(result_idx)
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        pending = {executor.submit(check_channel, idx) for idx in range(data.shape[1])}
        done_count = 0
        try:
            if progress_callback:
                progress_callback(progress_start, f"{label_prefix}: parallel bad check using {max_workers} workers for {n_channels} channels")
            while pending:
                _raise_if_lfp_snr_cancelled(self)
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
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
                    value = progress_start + (progress_end - progress_start) * done_count / n_channels
                    progress_callback(value, f"{label_prefix}: parallel bad check {done_count}/{n_channels} channels")
        except _LfpSnrCancelled:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    good_indices.sort()
    if progress_callback:
        progress_callback(progress_end, f"{label_prefix}: bad-channel check done ({n_channels} channels)")
    return good_indices, bad_reasons, candidate_reasons


_core._EmbeddedPipelineGUI.detect_bad_snr_channels_with_settings = _detect_bad_snr_channels_cancellable


def _run_lfp_snr_cancellable(self, on_done=None):
    """Run LFP SNR with a cooperative stop path and a reliable UI callback."""
    self.skip_lfp_snr_var.set(False)
    self._lfp_snr_cancel_requested = False
    start_time = time.perf_counter()

    def progress(value, phase):
        _raise_if_lfp_snr_cancelled(self)
        elapsed = time.perf_counter() - start_time
        message = f"{phase} | {value:.0f}% | elapsed {elapsed:.1f}s"
        if value > 1:
            message += f" | ETA {elapsed * (100.0 - value) / value:.1f}s"
        self.set_snr_progress(value, message)

    try:
        self.set_snr_progress(0, "LFP SNR: starting...")
        selected_for_update = self.get_selected_channels()
        partial_update = bool(self.last_snr_rows and selected_for_update)
        if partial_update:
            result = self.compute_lfp_snr_selected_channels(
                selected_for_update, progress=progress,
                label_prefix=f"LFP SNR selected {len(selected_for_update)}ch",
            )
            self.last_snr_rows = self.merge_lfp_snr_rows(self.last_snr_rows, result["rows"])
        else:
            result = self.compute_lfp_snr_current_settings(progress=progress, label_prefix="LFP SNR")
            self.last_snr_rows = result["rows"]
        _raise_if_lfp_snr_cancelled(self)
        threshold = _core.parse_float(self.snr_threshold_var.get(), 5)
        if partial_update:
            self.selected_channels.intersection_update(set(selected_for_update))
        else:
            self.selected_channels.intersection_update(result["valid_channels"])
        self.refresh_snr_tree(threshold=threshold)
        self.update_selected_summary()
        self.snr_status_var.set(
            f"SNR status: {result['total_channels']} channels | healthy {result['healthy_count']} | "
            f"Bad/Candidate {result['bad_like_count']} ({result['bad_pct']:.1f}%)"
        )
        self.set_snr_progress(100.0, self.snr_status_var.get())
    except _LfpSnrCancelled:
        self.snr_status_var.set("LFP SNR stopped; existing SNR results were kept unchanged.")
        self.set_snr_progress(0.0, self.snr_status_var.get())
        self.log("LFP SNR stopped by user.")
    except Exception as exc:
        self.set_snr_progress(0.0, "SNR status: failed")
        self.log(_core.traceback.format_exc())
        _core.messagebox.showerror("LFP SNR failed", str(exc))
    finally:
        self._lfp_snr_cancel_requested = False
        if callable(on_done):
            on_done()


_core._EmbeddedPipelineGUI.run_lfp_snr = _run_lfp_snr_cancellable


_original_current_data = _core._EmbeddedPipelineGUI.current_data


def _current_data_with_preprocess(self):
    """Expose the latest filtered data without destroying remapped_data."""
    preprocessed = getattr(self, "preprocessed_data", None)
    base = getattr(self, "remapped_data", None)
    if preprocessed is not None:
        if base is not None and getattr(preprocessed, "shape", None) == getattr(base, "shape", None):
            return preprocessed
        # A new file was loaded, or remapping has not completed yet; discard a
        # cache belonging to the old data.
        self.preprocessed_data = None
    if getattr(self, "_lazy_h5_info", None) is not None:
        # Computation paths still need a dense matrix.  Display paths below
        # bypass current_data() and read only their visible HDF5 slice.
        return _materialize_lazy_h5(self)
    return _original_current_data(self)


_core._EmbeddedPipelineGUI.current_data = _current_data_with_preprocess


_original_has_loaded_data = _core._EmbeddedPipelineGUI.has_loaded_data


def _has_loaded_data_with_lazy_h5(self) -> bool:
    return getattr(self, "_lazy_h5_info", None) is not None or _original_has_loaded_data(self)


_core._EmbeddedPipelineGUI.has_loaded_data = _has_loaded_data_with_lazy_h5


def _h5_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _lazy_h5_info_from_path(pipeline, path: Path):
    if _core.h5py is None or not _core.h5py.is_hdf5(str(path)):
        return None
    with pipeline.open_h5_readonly(path) as h5:
        for key in ("rawData512", "raw_data", "signal", "data", "eeg"):
            if key not in h5 or not isinstance(h5[key], _core.h5py.Dataset):
                continue
            dataset = h5[key]
            if dataset.ndim != 2 or _core.h5py.check_dtype(ref=dataset.dtype) is not None:
                continue
            if not _core.np.issubdtype(dataset.dtype, _core.np.number):
                continue
            fs_value = h5["FS"][()] if "FS" in h5 else h5.attrs.get("FS", h5.attrs.get("fs", 6490.0))
            fs = float(_core.np.asarray(fs_value).squeeze())
            unit_value = h5["data_unit"][()] if "data_unit" in h5 else h5.attrs.get("data_unit", "")
            unit = _h5_text(unit_value).strip().lower()
            scale = float(getattr(_core, "V_TO_MV", 1000.0)) if unit in {"v", "volt", "volts"} else 1.0
            time_offset = 0.0
            if "time" in h5 and h5["time"].shape[0]:
                time_offset = float(h5["time"][0])
            elif "t0" in h5:
                time_offset = float(_core.np.asarray(h5["t0"][()]).squeeze())
            def timing_value(key, default=None):
                return h5[key][()] if key in h5 else h5.attrs.get(key, default)
            # Match integrated_pipeline_gui.py: retain the original dt1 data,
            # use an explicitly declared ms deltaT1 when present, and accept
            # old processed H5 files that store only normalized seconds.
            delta_t1 = timing_value("deltaT1", None)
            delta_t2 = timing_value("deltaT2", _core.np.nan)
            delta_unit = timing_value("deltaT1_unit", "")
            if delta_t1 is None and "bin_delta_t1_sec" in h5.attrs:
                delta_t1 = h5.attrs["bin_delta_t1_sec"]
                delta_unit = "s"
                delta_t2 = h5.attrs.get("bin_delta_t2_sec", delta_t2)
            if delta_t1 is None and "deltaT1_ms" in h5.attrs:
                delta_t1 = h5.attrs["deltaT1_ms"]
                delta_unit = "ms"
            elif delta_t1 is not None:
                # Raw BIN deltaT1/deltaT2 fields are always milliseconds.
                # Ignore a malformed historical "s" label on such a field.
                delta_unit = "ms"
            meta = {
                "data_unit": unit,
                "dt1": timing_value("dt1", ""),
                "deltaT1": 0.0 if delta_t1 is None else delta_t1,
                "deltaT2": delta_t2,
                "deltaT1_unit": _h5_text(delta_unit),
                "dt1_date": _h5_text(timing_value("dt1_date", "")),
            }
            return {
                "path": str(path),
                "dataset": key,
                "shape": tuple(int(value) for value in dataset.shape),
                "dtype": _core.np.dtype(dataset.dtype),
                "chunks": dataset.chunks,
                "fs": fs,
                "scale": scale,
                "time_offset": time_offset,
                "meta": meta,
            }
    return None


def _lazy_h5_read_slice(pipeline, first: int, last: int, columns):
    info = getattr(pipeline, "_lazy_h5_info", None)
    if info is None:
        raise RuntimeError("No lazy HDF5 source is active.")
    rows, channel_count = info["shape"]
    first = max(0, min(int(first), rows))
    last = max(first, min(int(last), rows))
    if isinstance(columns, int):
        columns = max(0, min(int(columns), channel_count - 1))
    with pipeline.open_h5_readonly(Path(info["path"])) as h5:
        values = _core.np.asarray(h5[info["dataset"]][first:last, columns], dtype=_core.DATA_DTYPE)
    if info["scale"] != 1.0:
        values = _core.np.asarray(values * info["scale"], dtype=_core.DATA_DTYPE)
    return values


def _materialize_lazy_h5(pipeline, progress_callback=None):
    info = getattr(pipeline, "_lazy_h5_info", None)
    if info is None:
        return _original_current_data(pipeline)
    rows, channels = info["shape"]
    data = _core.np.empty((rows, channels), dtype=_core.DATA_DTYPE)
    def report(fraction: float) -> None:
        if callable(progress_callback):
            progress_callback(max(0.0, min(1.0, float(fraction))))

    with pipeline.open_h5_readonly(Path(info["path"])) as h5:
        dataset = h5[info["dataset"]]
        if dataset.chunks:
            selections = dataset.iter_chunks()
            chunk_rows = max(1, int(dataset.chunks[0]))
            total_chunks = max(1, int(_core.np.ceil(rows / chunk_rows)))
            for index, selection in enumerate(selections, start=1):
                data[selection] = dataset[selection]
                report(index / total_chunks)
        else:
            for index, first in enumerate(range(0, rows, 200_000), start=1):
                last = min(rows, first + 200_000)
                data[first:last] = dataset[first:last]
                report(last / max(1, rows))
    if info["scale"] != 1.0:
        data *= info["scale"]
    pipeline.raw_data = data
    pipeline._lazy_h5_info = None
    pipeline.time = None
    pipeline.log(f"Materialized lazy HDF5 for computation: {rows} x {channels}.")
    report(1.0)
    return data


_original_load_existing_mat = _core._EmbeddedPipelineGUI.load_existing_mat


def _clear_analysis_channel_lineage(pipeline) -> None:
    for attribute in (
        "imported_channel_ids",
        "alignment_selected_channels",
        "lfp_analysis_selected_channels",
        "spike_analysis_selected_channels",
    ):
        try:
            delattr(pipeline, attribute)
        except AttributeError:
            pass


def _load_existing_mat_clear_preprocessed(self):
    # A newly loaded file must never reuse filtered data from the previous
    # file, even when both files happen to have the same shape.
    self.preprocessed_data = None
    self.preprocessed_filter_settings = {}
    # Release disk-backed remap/filtered caches before a different source file
    # replaces the active data matrix.
    self.remapped_data = None
    self._remapped_memmap_path = None
    self._preprocessed_memmap_path = None
    cleanup = globals().get("_cleanup_remap_memmaps")
    if callable(cleanup):
        cleanup(self)
    _clear_analysis_channel_lineage(self)
    self.timing_info = None
    self.stim_markers = None
    path_text = str(self.mat_path_var.get()).strip()
    if path_text:
        try:
            lazy_info = _lazy_h5_info_from_path(self, Path(path_text))
        except Exception:
            lazy_info = None
        if lazy_info is not None:
            self.raw_data = None
            self.remapped_data = None
            self.remapped_channel_ids = None
            self.layout_channel_ids = None
            self.layout_grid = None
            self._lazy_h5_info = lazy_info
            self.fs = float(lazy_info["fs"])
            self.time = None
            self.mat_path = Path(path_text)
            meta = lazy_info["meta"]
            delta_t1 = float(_core.np.asarray(meta.get("deltaT1", 0.0)).squeeze())
            delta_t2 = float(_core.np.asarray(meta.get("deltaT2", _core.np.nan)).squeeze())
            unit = str(meta.get("deltaT1_unit", "")).strip().lower()
            # Historical H5 files written by the old parser stored raw BIN
            # milliseconds 1000x too large (161230 ms instead of 161 ms).
            if unit in {"ms", "millisecond", "milliseconds"} and abs(delta_t1) >= 10_000.0:
                delta_t1 /= 1000.0
                if _core.np.isfinite(delta_t2):
                    delta_t2 /= 1000.0
            if unit in {"ms", "millisecond", "milliseconds"} or (not unit and abs(delta_t1) >= 0.5):
                delta_t1 /= 1000.0
                delta_t2 /= 1000.0
            self.bin_delta_t1_sec = delta_t1 if _core.np.isfinite(delta_t1) else 0.0
            self.bin_delta_t2_sec = delta_t2
            if meta.get("dt1_date"):
                self.record_time_var.set(str(meta["dt1_date"]))
            rows, channels = lazy_info["shape"]
            self.log(
                f"Loaded HDF5 lazily: {path_text}; {rows} samples x {channels} channels; "
                f"chunks={lazy_info['chunks']}."
            )
            self.log(
                f"BIN timing deltaT1={self.bin_delta_t1_sec * 1000.0:.3f} ms; "
                f"deltaT2={self.bin_delta_t2_sec * 1000.0:.3f} ms"
            )
            update_status = getattr(self, "update_bin_timing_status", None)
            if callable(update_status):
                update_status()
            self.all_raw_preview_status_var.set("All raw preview: lazy HDF5 ready.")
            if self.all_raw_preview_enabled:
                self.start_all_raw_preview()
            self.notebook.select(self.data_tab)
            return None
    # Do not leave a previous lazy file attached when the normal loader is
    # used for a different source.
    self._lazy_h5_info = None
    result = _original_load_existing_mat(self)
    _normalize_loaded_timing_offsets_to_seconds(self)
    return result


_core._EmbeddedPipelineGUI.load_existing_mat = _load_existing_mat_clear_preprocessed


def _queue_batch_progress(pipeline, value: float, message: str) -> None:
    """Safely update the visible import progress bar from a worker thread."""
    owner = getattr(pipeline, "popup_parent", None)
    target = owner if owner is not None else pipeline

    def update() -> None:
        try:
            if hasattr(target, "import_progress_var"):
                target.import_progress_var.set(max(0.0, min(100.0, float(value))))
            if hasattr(target, "import_progress_status_var"):
                target.import_progress_status_var.set(message)
            if hasattr(target, "batch_status_var"):
                target.batch_status_var.set(message)
        except Exception:
            pass

    queue_ui = getattr(pipeline, "_queue_ui", None)
    if callable(queue_ui):
        queue_ui(update)
    else:
        try:
            pipeline.after(0, update)
        except Exception:
            update()


_original_read_timing_metadata = _core.read_timing_metadata


def _read_timing_metadata_with_correct_ms(path: Path) -> dict:
    """Correct the legacy BIN timing parser's extra 1000x millisecond scale."""
    timing = dict(_original_read_timing_metadata(path) or {})
    for key in ("deltaT1_ms", "deltaT2_ms"):
        try:
            value = float(timing.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        # The legacy parser divided the hardware tick difference by 216,
        # producing a value 1000 times too large for the declared ms unit.
        if _core.np.isfinite(value):
            timing[key] = value / 1000.0
    timing.setdefault("deltaT1_unit", "ms")
    timing.setdefault("deltaT_unit", "ms")
    return timing


def _normalize_loaded_timing_offsets_to_seconds(pipeline) -> None:
    """Normalize timing from legacy H5 files before alignment uses it."""
    try:
        delta_t1 = float(getattr(pipeline, "bin_delta_t1_sec", 0.0))
        delta_t2 = float(getattr(pipeline, "bin_delta_t2_sec", float("nan")))
    except (TypeError, ValueError):
        return
    # Legacy H5 values were millisecond values stored as seconds.  This path
    # only applies to the out-of-range representation (>= 0.5 sec); normal
    # BIN offsets are sub-second and are left untouched.
    if _core.np.isfinite(delta_t1) and abs(delta_t1) >= 0.5:
        pipeline.bin_delta_t1_sec = delta_t1 / 1000.0
        if _core.np.isfinite(delta_t2):
            pipeline.bin_delta_t2_sec = delta_t2 / 1000.0
        update_status = getattr(pipeline, "update_bin_timing_status", None)
        if callable(update_status):
            update_status()


_core.read_timing_metadata = _read_timing_metadata_with_correct_ms


_original_run_python_parser_for_batch = _core._EmbeddedPipelineGUI.run_python_parser_for_batch


def _run_python_parser_for_batch_with_progress(
    self,
    bin_path: Path,
    output_h5: Path,
    animal: int,
    block: int,
    record_date: str,
) -> Path:
    """Run one batch parser and forward create_h5's frame progress."""
    progress_callback = getattr(self, "_batch_progress_callback", None)
    if not callable(progress_callback):
        return _original_run_python_parser_for_batch(
            self, bin_path, output_h5, animal, block, record_date
        )

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(bin_path).resolve()
    layout = _core.inspect_bin(source_path)
    reader = _core.BinReader(layout, 0.0, 0.0, chunk_rows=4096)
    timing = _core.read_timing_metadata(source_path)
    metadata = {
        "source_bin": str(source_path),
        "date": record_date,
        "animal": int(animal),
        "block": int(block),
        "start_sec": 0.0,
        "duration_sec": reader.selected_duration_sec,
    }
    metadata.update(timing)
    _core.create_h5(
        output_h5,
        reader,
        metadata,
        compression_level=5,
        progress_callback=progress_callback,
    )
    return output_h5


_core._EmbeddedPipelineGUI.run_python_parser_for_batch = _run_python_parser_for_batch_with_progress


def _batch_parse_bins_with_progress(self):
    """Batch BIN import with file-level and frame-level progress reporting."""
    if _core.h5py is None:
        _core.messagebox.showerror("缺少依赖", "批量解析需要 h5py。")
        return

    input_dir_text = _core.filedialog.askdirectory(title="选择包含 BIN 文件的根目录")
    if not input_dir_text:
        return

    input_dir = Path(input_dir_text)
    output_text = self.output_dir_var.get().strip()
    output_root = Path(output_text or str(input_dir / "parsed_h5"))
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = []
    for path in sorted(input_dir.rglob("*.bin")):
        try:
            records.append(self.parse_batch_bin_filename(path))
        except Exception as exc:
            skipped.append(f"{path}: {exc}")

    if not records:
        detail = "\n".join(skipped[:10])
        _core.messagebox.showerror("没有有效 BIN", f"没有找到符合命名规则的 BIN 文件。\n{detail}")
        _queue_batch_progress(self, 0.0, "未找到有效 BIN 文件")
        return

    total = len(records)
    _queue_batch_progress(self, 0.0, f"批量解析：找到 {total} 个 BIN 文件")

    def worker():
        successes = []
        failures = list(skipped)

        for index, record in enumerate(records, start=1):
            bin_path = Path(record["path"])
            file_name = bin_path.name
            file_start = (index - 1) * 100.0 / total
            file_span = 100.0 / total
            parse_span = file_span * 0.75
            last_emit = [0.0]

            def report_frames(done, frame_total):
                now = time.perf_counter()
                fraction = (float(done) / float(frame_total)) if frame_total else 0.0
                value = file_start + parse_span * max(0.0, min(1.0, fraction))
                if done not in (0, frame_total) and now - last_emit[0] < 0.15:
                    return
                last_emit[0] = now
                _queue_batch_progress(
                    self,
                    value,
                    f"批量解析 {index}/{total}：{file_name}（{int(done)}/{int(frame_total)} 帧）",
                )

            try:
                _queue_batch_progress(self, file_start, f"批量解析 {index}/{total}：准备 {file_name}")
                self._batch_progress_callback = report_frames
                record_output_dir = output_root / record["date_compact"] / str(record["animal"])
                record_output_dir.mkdir(parents=True, exist_ok=True)
                final_h5 = record_output_dir / f"{record['stem']}.h5"

                existing_h5 = False
                if final_h5.exists():
                    try:
                        with _core.h5py.File(final_h5, "r") as h5:
                            existing_h5 = "rawData512" in h5 and "FS" in h5
                        if not existing_h5:
                            raise ValueError("missing rawData512 or FS")
                        self.log(f"Reusing existing total H5: {final_h5}")
                    except Exception as exc:
                        self.log(f"Existing total H5 is invalid; rebuilding: {final_h5}; {exc}")

                if not existing_h5:
                    self.run_python_parser_for_batch(
                        bin_path,
                        final_h5,
                        record["animal"],
                        record["block"],
                        record["date"],
                    )
                _queue_batch_progress(
                    self,
                    file_start + file_span,
                    f"批量解析 {index}/{total}：{file_name} 已生成完整 H5",
                )
                successes.append(f"{record['stem']}: total={final_h5}")
            except Exception as exc:
                failures.append(f"{bin_path}: {exc}")
                self.log(_core.traceback.format_exc())
                _queue_batch_progress(self, file_start + file_span, f"批量解析 {index}/{total}：{file_name} 失败")
            finally:
                self._batch_progress_callback = None

        summary = f"批量解析完成。成功：{len(successes)}；失败/跳过：{len(failures)}。"
        details = "\n".join(successes[:10])
        if failures:
            details += "\n\n失败/跳过：\n" + "\n".join(failures[:10])
        _queue_batch_progress(self, 100.0, summary)
        self._queue_ui(lambda: _core.messagebox.showinfo("批量解析", f"{summary}\n\n{details}"))

    _core.threading.Thread(target=worker, daemon=True).start()


_core._EmbeddedPipelineGUI.batch_parse_bins = _batch_parse_bins_with_progress


def _filter_mode_enabled(mode) -> bool:
    """Return whether a stored preprocessing filter mode enables filtering."""
    mode = str(mode or "off").strip().lower()
    return mode not in {"", "off", "none", "false", "0", "关闭", "未启用"}


def _preprocess_filter_enabled(pipeline) -> bool:
    """Return whether the current preprocessing settings request a filter."""
    try:
        return _filter_mode_enabled(pipeline.preprocess_filter_settings().get("mode"))
    except Exception:
        return False


def _main_preprocess_filter_enabled(self) -> bool:
    """Read the GUI's filter switch without triggering pipeline resync."""
    try:
        return _filter_mode_enabled(self.preprocess_filter_mode_var.get())
    except Exception:
        return False


def _set_preprocess_filter_widgets_visible(self, visible: bool) -> None:
    """Show the optional filter-stage widgets only when filtering is enabled."""
    status_variable = str(getattr(self, "preprocess_filter_status_var", ""))
    progress_variable = str(getattr(self, "preprocess_filter_progress_var", ""))
    if not status_variable or not progress_variable:
        return
    tracked = getattr(self, "_preprocess_filter_widget_layout", {})
    if visible:
        for widget, layout in list(tracked.items()):
            try:
                if layout[0] == "grid":
                    widget.grid()
                elif layout[0] == "pack":
                    widget.pack(**layout[1])
                elif layout[0] == "place":
                    widget.place(**layout[1])
            except Exception:
                pass
        self._preprocess_filter_widget_layout = {}
        return

    for widget in _walk_widgets(self):
        try:
            matches_status = str(widget.cget("textvariable")) == status_variable
        except Exception:
            matches_status = False
        try:
            matches_progress = str(widget.cget("variable")) == progress_variable
        except Exception:
            matches_progress = False
        if not (matches_status or matches_progress) or widget in tracked:
            continue
        try:
            manager = widget.winfo_manager()
            if manager == "grid":
                tracked[widget] = ("grid", None)
                widget.grid_remove()
            elif manager == "pack":
                tracked[widget] = ("pack", widget.pack_info())
                widget.pack_forget()
            elif manager == "place":
                tracked[widget] = ("place", widget.place_info())
                widget.place_forget()
        except Exception:
            continue
    self._preprocess_filter_widget_layout = tracked


def _update_preprocess_filter_idle_display(self) -> bool:
    """Reflect whether the optional filter stage has any work to do."""
    enabled = _main_preprocess_filter_enabled(self)
    _set_preprocess_filter_widgets_visible(self, enabled)
    if enabled:
        self.preprocess_filter_status_var.set("预处理滤波：等待运行")
        self.preprocess_filter_progress_var.set(0.0)
    else:
        self.preprocess_filter_status_var.set("预处理滤波：未启用（无需运行）")
        self.preprocess_filter_progress_var.set(100.0)
    return enabled


def _processed_overview_button_text(pipeline) -> str:
    settings = getattr(pipeline, "preprocessed_filter_settings", {}) or {}
    if _filter_mode_enabled(settings.get("mode")):
        return "打开滤波后全通道总览"
    return "打开重映射后全通道总览（未滤波）"


def _refresh_preprocess_action_label(self) -> bool:
    """Keep the primary action text aligned with the filter mode."""
    try:
        enabled = _main_preprocess_filter_enabled(self)
        text = "执行通道重映射并滤波" if enabled else "执行通道重映射"
        for child in _walk_widgets(self):
            try:
                if child.winfo_manager() != "pack":
                    continue
                if str(child.cget("text")) in {
                    "执行通道重映射",
                    "执行通道重映射并滤波",
                }:
                    child.configure(text=text)
            except Exception:
                continue
        return enabled
    except Exception:
        return False


_original_build_preprocess = _core.AnalysisGUI._build_preprocess


def _build_preprocess_with_filter_label(self, parent) -> None:
    _original_build_preprocess(self, parent)
    enabled = _refresh_preprocess_action_label(self)
    if not enabled:
        self.preprocess_status_var.set("预处理：等待运行（未启用滤波）")
    _update_preprocess_filter_idle_display(self)
    mode_var = getattr(self, "preprocess_filter_mode_var", None)
    if mode_var is not None and not getattr(self, "_preprocess_filter_mode_trace", None):
        self._preprocess_filter_mode_trace = mode_var.trace_add(
            "write",
            lambda *_: self.after_idle(
                lambda: (
                    _refresh_preprocess_action_label(self),
                    _update_preprocess_filter_idle_display(self),
                )
            ),
        )


_core.AnalysisGUI._build_preprocess = _build_preprocess_with_filter_label


def _run_preprocess_remap_then_filter(self) -> None:
    pipeline = self._sync_pipeline_inputs()
    if pipeline is None:
        return
    self.preprocess_filter_progress_var.set(0.0)
    self.preprocess_filter_status_var.set("正在通道重映射…")
    export_button = getattr(self, "_processed_channel_export_button", None)
    selector_button = getattr(self, "_processed_channel_selector_button", None)
    if export_button is not None:
        export_button.state(["disabled"])
    if selector_button is not None:
        selector_button.state(["disabled"])
    try:
        was_already_remapped = pipeline.remapped_data is not None
        if not was_already_remapped:
            pipeline.apply_remapping()

            # apply_remapping() historically reports errors/cancellation
            # through a dialog and returns None.  Only filter after it has
            # created a usable remapped base.
            if pipeline.remapped_data is None:
                self.preprocess_status_var.set("通道重映射未完成")
                self.preprocess_filter_status_var.set("预处理未完成")
                self.status_var.set("通道重映射已取消或失败，未执行滤波")
                return
        else:
            self.preprocess_filter_status_var.set("已使用现有重映射数据，正在滤波…")

        # A repeated run must start from the immutable remapped base.
        pipeline.preprocessed_data = None
        settings = pipeline.preprocess_filter_settings()
        filter_enabled = _preprocess_filter_enabled(pipeline)
        # Keep the remapped base data immutable.  Store the filtered result in
        # the normal processed-data cache when available; otherwise attach a
        # dedicated attribute used by this launcher.  Writing the filtered
        # result back to remapped_data causes repeated clicks to filter an
        # already-filtered signal.
        remapped = _core.np.asarray(pipeline.remapped_data, dtype=_core.DATA_DTYPE)
        if filter_enabled:
            filtered = _core.np.asarray(
                pipeline.apply_preprocess_filter(remapped, settings),
                dtype=_core.DATA_DTYPE,
            )
        else:
            # Filtering is opt-in.  Keep a separate processed copy so later
            # export/preview steps work without changing remapped_data.
            filtered = remapped.copy()
        pipeline.preprocessed_data = filtered
        pipeline.preprocessed_filter_settings = dict(settings)
        pipeline.clear_processed_filter_cache()
        _render_preprocessed_channel_preview(self, show_empty_state=False)
        if hasattr(pipeline, "plot_preview"):
            pipeline.plot_preview()
        if getattr(pipeline, "all_raw_preview_enabled", False) and hasattr(pipeline, "render_all_raw_preview"):
            pipeline.render_all_raw_preview()
        if filter_enabled:
            self.preprocess_status_var.set("通道重映射和滤波已完成")
            self.preprocess_filter_status_var.set("预处理滤波：已完成")
            self.preprocess_filter_progress_var.set(100.0)
            self.status_var.set("已按当前预处理滤波参数完成通道重映射和滤波")
        else:
            self.preprocess_status_var.set("通道重映射已完成（未启用滤波）")
            self.preprocess_filter_status_var.set("预处理滤波：未启用（重映射已完成）")
            self.preprocess_filter_progress_var.set(100.0)
            self.status_var.set("已完成通道重映射；当前未启用滤波")
        overview_button = getattr(self, "_filtered_overview_button", None)
        if overview_button is not None:
            overview_button.configure(text=_processed_overview_button_text(pipeline))
        if export_button is not None:
            export_button.state(["!disabled"])
        if selector_button is not None:
            selector_button.state(["!disabled"])
    except Exception as exc:
        self.preprocess_status_var.set("预处理失败")
        self.preprocess_filter_status_var.set("预处理失败")
        self.preprocess_filter_progress_var.set(0.0)
        self.status_var.set(f"预处理失败: {exc}")
        _core.messagebox.showerror("预处理失败", str(exc), parent=self)


_core.AnalysisGUI._run_preprocess = _run_preprocess_remap_then_filter


# Large remapping must not allocate several full copies on the Tk thread.
_REMAP_CHUNK_SAMPLES = 200_000
_FILTER_CHANNEL_CHUNK = 8
_REMAP_MEMMAP_THRESHOLD_BYTES = 1_500_000_000


def _remap_memmap_root() -> Path:
    root = Path(tempfile.gettempdir()) / "sd_gui_remap_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _allocate_remap_storage(pipeline, shape, dtype, label: str):
    item_bytes = int(_core.np.dtype(dtype).itemsize)
    total_bytes = int(shape[0]) * int(shape[1]) * item_bytes
    if total_bytes < _REMAP_MEMMAP_THRESHOLD_BYTES:
        return _core.np.empty(shape, dtype=dtype), None
    path = _remap_memmap_root() / (
        f"remap_{os.getpid()}_{id(pipeline)}_{time.time_ns()}_{label}.dat"
    )
    array = _core.np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    paths = getattr(pipeline, "_remap_memmap_paths", [])
    paths.append(str(path))
    pipeline._remap_memmap_paths = paths
    return array, path


def _cleanup_remap_memmaps(pipeline, keep=()) -> None:
    keep_paths = {str(Path(item)) for item in keep if item}
    paths = list(getattr(pipeline, "_remap_memmap_paths", []) or [])
    retained = []
    for item in paths:
        path = Path(item)
        if str(path) in keep_paths:
            retained.append(str(path))
            continue
        try:
            path.unlink(missing_ok=True)
        except Exception:
            retained.append(str(path))
    pipeline._remap_memmap_paths = retained


def _optimized_remap_array(pipeline, source_data, custom_map, progress=None):
    source = _core.np.asarray(source_data)
    if source.ndim != 2:
        raise ValueError(f"Remapping requires a 2D samples x channels matrix, got {source.shape}.")
    input_channels = int(source.shape[1])
    mapping = _core.np.asarray(custom_map, dtype=_core.np.int64).ravel()
    if mapping.size != input_channels:
        raise ValueError(
            f"Remap vector has {mapping.size} entries but the data has {input_channels} channels."
        )
    if _core.np.any(mapping < 1) or _core.np.unique(mapping).size != mapping.size:
        raise ValueError("Remap vector must contain unique positive channel positions.")
    output_channels = max(input_channels, int(mapping.max()))
    destinations = mapping - 1
    inverse = _core.np.full(output_channels, -1, dtype=_core.np.int64)
    inverse[destinations] = _core.np.arange(input_channels, dtype=_core.np.int64)
    target, target_path = _allocate_remap_storage(
        pipeline,
        (int(source.shape[0]), output_channels),
        source.dtype,
        "base",
    )
    valid_destinations = _core.np.flatnonzero(inverse >= 0)
    valid_sources = inverse[valid_destinations]
    complete_permutation = valid_destinations.size == output_channels
    total = max(1, int(source.shape[0]))
    for start in range(0, int(source.shape[0]), _REMAP_CHUNK_SAMPLES):
        if getattr(pipeline, "_remap_cancel_requested", False):
            raise InterruptedError("Remapping cancelled by user.")
        end = min(start + _REMAP_CHUNK_SAMPLES, int(source.shape[0]))
        source_chunk = source[start:end]
        target_chunk = target[start:end]
        if complete_permutation:
            _core.np.take(source_chunk, inverse, axis=1, out=target_chunk)
        else:
            target_chunk[...] = 0
            target_chunk[:, valid_destinations] = source_chunk[:, valid_sources]
        if callable(progress):
            progress(70.0 * end / total, f"重映射：{end:,}/{total:,} samples")
    if isinstance(target, _core.np.memmap):
        target.flush()
    return target, target_path, output_channels


def _chunked_preprocess_filter(pipeline, remapped, settings, filter_enabled, progress=None):
    data = _core.np.asarray(remapped)
    target, target_path = _allocate_remap_storage(
        pipeline,
        data.shape,
        data.dtype,
        "processed",
    )
    total_channels = int(data.shape[1])
    for col_start in range(0, total_channels, _FILTER_CHANNEL_CHUNK):
        if getattr(pipeline, "_remap_cancel_requested", False):
            raise InterruptedError("Preprocessing cancelled by user.")
        col_end = min(col_start + _FILTER_CHANNEL_CHUNK, total_channels)
        block = _core.np.asarray(data[:, col_start:col_end], dtype=_core.DATA_DTYPE)
        if not filter_enabled:
            target[:, col_start:col_end] = block
        else:
            finite_columns = _core.np.isfinite(block).any(axis=0)
            target[:, col_start:col_end] = _core.np.nan
            if finite_columns.any():
                filtered = pipeline.apply_preprocess_filter(
                    block[:, finite_columns], settings
                )
                target[:, col_start:col_end][:, finite_columns] = filtered
        if callable(progress):
            progress(
                70.0 + 28.0 * col_end / max(1, total_channels),
                f"滤波/缓存：{col_end}/{total_channels} channels",
            )
    if isinstance(target, _core.np.memmap):
        target.flush()
    return target, target_path


def _run_preprocess_remap_then_filter_async(self) -> None:
    pipeline = self._sync_pipeline_inputs()
    if pipeline is None or getattr(self, "_preprocess_job_running", False):
        return
    self.preprocess_filter_progress_var.set(0.0)
    self.preprocess_filter_status_var.set("正在准备预处理…")
    export_button = getattr(self, "_processed_channel_export_button", None)
    selector_button = getattr(self, "_processed_channel_selector_button", None)
    for button in (export_button, selector_button):
        if button is not None:
            try:
                button.state(["disabled"])
            except Exception:
                pass

    remapped_exists = pipeline.remapped_data is not None
    map_path = None
    channel_map_path = None
    if not remapped_exists:
        raw = getattr(pipeline, "raw_data", None)
        if raw is None and getattr(pipeline, "_lazy_h5_info", None) is None:
            self.status_var.set("没有可用于重映射的数据")
            for button in (export_button, selector_button):
                if button is not None:
                    try:
                        button.state(["!disabled"])
                    except Exception:
                        pass
            return
        path_text = str(pipeline.remap_path_var.get()).strip()
        if not path_text or not Path(path_text).exists():
            path_text = pipeline.prompt_for_remap_file(path_text, "选择通道重映射 Excel 文件")
            if not path_text:
                return
            pipeline.remap_path_var.set(path_text)
        map_path = Path(path_text)
        try:
            channel_map_path = Path(pipeline.find_channel_map_file(str(map_path)))
        except Exception as exc:
            self.status_var.set(f"找不到通道映射列：{exc}")
            _core.messagebox.showerror("重映射准备失败", str(exc), parent=self)
            return

    settings = pipeline.preprocess_filter_settings()
    filter_enabled = _preprocess_filter_enabled(pipeline)
    pipeline._remap_cancel_requested = False
    self._preprocess_job_running = True
    self.status_var.set("预处理正在后台运行；窗口仍可响应。")

    def progress(value, message):
        pipeline._queue_ui(
            lambda v=float(value), msg=str(message): (
                self.preprocess_filter_progress_var.set(v),
                self.preprocess_filter_status_var.set(msg),
            )
        )

    def worker():
        new_remapped = None
        new_processed = None
        new_paths = []
        try:
            if remapped_exists:
                remapped = pipeline.remapped_data
                remapped_path = None
                progress(10.0, "使用已有重映射数据…")
            else:
                progress(2.0, "正在读取通道映射…")
                source = pipeline.raw_data
                if source is None:
                    # Remapping really does require all samples.  Keep that
                    # unavoidable read off the GUI thread and stream it using
                    # the HDF5 dataset's native chunks.
                    progress(3.0, "正在按 HDF5 分块读取数据，用于重映射…")
                    source = _materialize_lazy_h5(
                        pipeline,
                        lambda fraction: progress(
                            3.0 + 27.0 * fraction,
                            f"正在按 HDF5 分块读取数据：{fraction * 100:.0f}%",
                        ),
                    )
                custom_map = pipeline.read_channel_vector(channel_map_path, source.shape[1])
                new_remapped, remapped_path, _ = _optimized_remap_array(
                    pipeline, source, custom_map, progress
                )
                remapped = new_remapped
                new_paths.append(remapped_path)
            progress(70.0, "重映射完成，正在分块生成预处理结果…")
            new_processed, processed_path = _chunked_preprocess_filter(
                pipeline, remapped, settings, filter_enabled, progress
            )
            new_paths.append(processed_path)

            layout_ids = pipeline.read_layout_remap(
                str(map_path), remapped.shape[1]
            ) if map_path is not None else getattr(pipeline, "layout_channel_ids", None)
            layout_grid = pipeline.read_layout_grid(
                str(map_path), remapped.shape[1]
            ) if map_path is not None else getattr(pipeline, "layout_grid", None)

            def finish():
                pipeline.remapped_data = remapped
                pipeline.preprocessed_data = new_processed
                pipeline.preprocessed_filter_settings = dict(settings)
                pipeline.remapped_channel_ids = _core.np.arange(1, remapped.shape[1] + 1)
                if layout_ids is not None:
                    pipeline.layout_channel_ids = layout_ids
                if layout_grid is not None:
                    pipeline.layout_grid = layout_grid
                pipeline.raw_data = None
                if remapped_path is not None:
                    pipeline._remapped_memmap_path = str(remapped_path)
                else:
                    pipeline._remapped_memmap_path = None
                if processed_path is not None:
                    pipeline._preprocessed_memmap_path = str(processed_path)
                else:
                    pipeline._preprocessed_memmap_path = None
                pipeline.clear_processed_filter_cache()
                pipeline._remap_cancel_requested = False
                self._preprocess_job_running = False
                self.preprocess_filter_progress_var.set(100.0)
                self.preprocess_filter_status_var.set(
                    "预处理完成：重映射 + 滤波" if filter_enabled else "预处理完成：重映射（未启用滤波）"
                )
                self.status_var.set("预处理完成，处理后预览和导出已更新。")
                _render_preprocessed_channel_preview(self, show_empty_state=False)
                if hasattr(pipeline, "plot_preview"):
                    pipeline.plot_preview()
                if getattr(pipeline, "all_raw_preview_enabled", False):
                    pipeline.render_all_raw_preview()
                for button in (export_button, selector_button):
                    if button is not None:
                        try:
                            button.state(["!disabled"])
                        except Exception:
                            pass
                # Remove superseded disk caches only after the new arrays are live.
                _cleanup_remap_memmaps(
                    pipeline,
                    keep=(
                        getattr(pipeline, "_remapped_memmap_path", None),
                        getattr(pipeline, "_preprocessed_memmap_path", None),
                    ),
                )

            pipeline._queue_ui(finish)
        except InterruptedError as exc:
            for path in new_paths:
                if path:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except Exception:
                        pass
            pipeline._queue_ui(lambda: _finish_preprocess_failure(self, pipeline, str(exc), export_button, selector_button))
        except Exception as exc:
            pipeline.log(_core.traceback.format_exc())
            for path in new_paths:
                if path:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except Exception:
                        pass
            pipeline._queue_ui(lambda error=str(exc): _finish_preprocess_failure(self, pipeline, error, export_button, selector_button))

    _core.threading.Thread(target=worker, daemon=True).start()


def _finish_preprocess_failure(self, pipeline, error, export_button, selector_button):
    self._preprocess_job_running = False
    pipeline._remap_cancel_requested = False
    self.preprocess_filter_progress_var.set(0.0)
    self.preprocess_filter_status_var.set("预处理失败")
    self.status_var.set(f"预处理失败：{error}")
    for button in (export_button, selector_button):
        if button is not None:
            try:
                button.state(["!disabled"])
            except Exception:
                pass
    _core.messagebox.showerror("预处理失败", str(error), parent=self)


_core.AnalysisGUI._run_preprocess = _run_preprocess_remap_then_filter_async


def _show_filtered_all_channel_overview(self) -> None:
    """Open the full-channel overview using the latest processed data."""
    pipeline = self._sync_pipeline_inputs()
    if pipeline is None:
        return
    try:
        if not pipeline.has_loaded_data():
            _core.messagebox.showerror(
                "没有数据",
                "请先完成 BIN 解析或读取已有 H5。",
                parent=self,
            )
            return
        if getattr(pipeline, "preprocessed_data", None) is None:
            self.status_var.set("请先执行预处理")
            _core.messagebox.showinfo(
                "尚未完成预处理",
                "请先执行第二页的通道重映射；如已启用滤波，系统会一并完成滤波。",
                parent=self,
            )
            return
        pipeline.popup_parent = self
        pipeline.open_all_raw_overview_popup()
        if _processed_overview_button_text(pipeline).endswith("（未滤波）"):
            self.status_var.set("已打开重映射后全通道总览（未滤波）")
        else:
            self.status_var.set("已打开滤波后全通道总览")
    except Exception as exc:
        self.status_var.set(f"打开滤波后总览失败：{exc}")
        _core.messagebox.showerror("打开滤波后总览失败", str(exc), parent=self)


_original_open_all_raw_overview_popup = _core._EmbeddedPipelineGUI.open_all_raw_overview_popup


_seaborn_module = None
_seaborn_import_attempted = False


def _get_seaborn():
    """Load Seaborn without the blocked optional PyArrow extension.

    Pandas can run without PyArrow.  This machine's application-control policy
    blocks ``pyarrow._compute``; marking that optional package unavailable
    before Pandas starts lets Seaborn load normally without changing policy or
    uninstalling packages.
    """
    global _seaborn_module, _seaborn_import_attempted
    if _seaborn_import_attempted:
        return _seaborn_module
    _seaborn_import_attempted = True
    if "pyarrow" not in sys.modules:
        sys.modules["pyarrow"] = None
        sys.modules["pyarrow.compute"] = None
    try:
        import seaborn as sns

        _seaborn_module = sns
    except ImportError:
        _seaborn_module = None
    return _seaborn_module


def _open_all_raw_overview_with_seaborn(self) -> None:
    """Show the 10x10 channel overview with Seaborn line plots.

    Seaborn draws on the existing Matplotlib/Tk canvas, so the current
    navigation, export, and Tk event-loop behaviour stay unchanged.
    """
    sns = _get_seaborn()

    if not self.has_loaded_data():
        _core.messagebox.showerror("No data", "Load parsed MAT data first.")
        return

    overview_data = getattr(self, "_overview_data_override", None)
    lazy_info = getattr(self, "_lazy_h5_info", None) if overview_data is None else None
    lazy_overview = lazy_info is not None
    try:
        if lazy_overview:
            # Keep only metadata here.  Individual overview pages below read
            # ``raw[first:last, page_channels]`` on demand.
            data = None
            row_count, total_channels = (int(value) for value in lazy_info["shape"])
            fs = float(lazy_info["fs"])
            lazy_time_offset = float(lazy_info.get("time_offset", 0.0))
        else:
            data = _core.np.asarray(
                self.current_data() if overview_data is None else overview_data,
                dtype=_core.DATA_DTYPE,
            )
            if data.ndim == 1:
                data = data[:, None]
            if data.ndim != 2:
                raise ValueError(f"Expected a samples x channels matrix, got shape {data.shape}.")
            row_count, total_channels = data.shape
            fs = float(self.fs)
            lazy_time_offset = 0.0
        if row_count <= 0 or total_channels <= 0:
            raise ValueError(f"The signal matrix is empty (shape {(row_count, total_channels)}).")
        if not _core.np.isfinite(fs) or fs <= 0:
            raise ValueError(f"Invalid sampling rate: {self.fs!r}.")
    except Exception as exc:
        _core.messagebox.showerror("Preview data unavailable", str(exc), parent=getattr(self, "popup_parent", None))
        return
    is_remapped = self.remapped_data is not None
    source_label = getattr(
        self,
        "_overview_source_label",
        "Remapped/FPC channel overview" if is_remapped else "Original raw channel overview",
    )
    overview_title = getattr(
        self,
        "_overview_title",
        ("Remapped channels" if is_remapped else "All raw channels") + " - 10x10 overview",
    )
    grid_rows, grid_cols = 10, 10
    channels_per_page = grid_rows * grid_cols
    total_channels = int(total_channels)
    total_pages = max(1, int(_core.np.ceil(total_channels / channels_per_page)))
    page_index = 0
    # ``render_overview_page`` and ``open_channel_detail`` are sibling
    # closures.  Keep the current visible range here instead of relying on
    # local variables from the renderer (which are out of scope in detail).
    overview_render_state = {
        "start": None,
        "end": None,
        "display_label": "data",
        "start_seconds": None,
        "duration_seconds": None,
        "linewidth": None,
    }
    initial_start = getattr(self, "_overview_start_seconds", None)
    if initial_start is None:
        variable = getattr(self, "preview_start_var", None)
        initial_start = variable.get() if variable is not None else 0.0
    initial_duration = getattr(self, "_overview_duration_seconds", None)
    if initial_duration is None:
        variable = getattr(self, "preview_duration_var", None)
        initial_duration = variable.get() if variable is not None else 5.0
    initial_linewidth = getattr(self, "_overview_linewidth", None)
    if initial_linewidth is None:
        variable = getattr(self, "preview_linewidth_var", None)
        initial_linewidth = variable.get() if variable is not None else 0.45
    overview_render_state.update(
        start_seconds=max(0.0, _core.parse_float(initial_start, 0.0)),
        duration_seconds=max(0.01, _core.parse_float(initial_duration, 5.0)),
        linewidth=min(0.7, max(0.1, _core.parse_float(initial_linewidth, 0.45))),
    )

    # The embedded pipeline is a controller object in the compiled runtime,
    # not always a Tk widget.  Anchor the overview to the actual main GUI so
    # the new window is created on-screen and stays above its owner.
    popup_parent = getattr(self, "popup_parent", None) or self
    popup = _core.tk.Toplevel(popup_parent)
    popup.title(overview_title)
    popup.geometry("1500x950")
    popup.minsize(1000, 700)
    popup.resizable(True, True)
    popup.transient(popup_parent)
    try:
        popup.state("zoomed")
    except _core.tk.TclError:
        pass

    nav_row = _core.ttk.Frame(popup)
    nav_row.pack(fill="x", padx=8, pady=(8, 0))
    status_var = _core.tk.StringVar()
    is_fullscreen = _core.tk.BooleanVar(value=False)
    plot_frame = _core.ttk.Frame(popup)
    plot_frame.pack(fill="both", expand=True)
    overview_canvas = None
    overview_toolbar = None

    def set_overview_fullscreen(enabled: bool) -> None:
        is_fullscreen.set(bool(enabled))
        popup.attributes("-fullscreen", bool(enabled))
        fullscreen_button.configure(text="Exit fullscreen" if enabled else "Fullscreen")
        if not enabled:
            try:
                popup.state("zoomed")
            except _core.tk.TclError:
                pass

    def render_overview_page() -> None:
        nonlocal overview_canvas, overview_toolbar
        start_seconds = overview_render_state["start_seconds"]
        duration_seconds = overview_render_state["duration_seconds"]
        start = max(0, int(float(start_seconds) * fs))
        duration = max(1, int(float(duration_seconds) * fs))
        end = min(row_count, start + duration)
        if end <= start:
            start, end = 0, min(row_count, duration)
        if end <= start:
            _core.messagebox.showerror(
                "No samples",
                "No samples are available for the selected start/duration.",
                parent=popup,
            )
            return

        page_start = page_index * channels_per_page
        page_end = min(total_channels, page_start + channels_per_page)
        sample_count = end - start
        sample_step = max(1, int(_core.np.ceil(sample_count / 1200)))
        sample_indices = _core.np.arange(start, end, sample_step)
        time_seconds = lazy_time_offset + sample_indices / fs
        if lazy_overview:
            # A 10x10 page needs at most 100 channels.  Do not read all
            # channels or the entire recording just to draw this page.
            full_segment = _lazy_h5_read_slice(self, start, end, slice(page_start, page_end))
            display_label = "raw HDF5 (lazy slice)"
        elif overview_data is not None:
            # The preprocessing viewer must show exactly the cached output of
            # remapping/filtering, without applying import-page display modes
            # a second time.
            full_segment = data[start:end, page_start:page_end]
            display_label = "processed data"
        else:
            try:
                full_segment, display_label = self.preview_display_data(data[start:end, page_start:page_end])
            except AttributeError:
                # Some compiled runtime versions do not expose this helper.
                # Preserve the original data rather than failing the overview.
                full_segment, display_label = data[start:end, page_start:page_end], "raw data"
            except Exception as exc:
                _core.messagebox.showerror("Preview display failed", str(exc), parent=popup)
                return
        segment = _core.np.asarray(full_segment[::sample_step, :], dtype=_core.DATA_DTYPE)
        if segment.ndim != 2 or segment.shape[0] == 0:
            _core.messagebox.showerror(
                "No displayable samples",
                f"The selected range produced an empty plotting segment: {segment.shape}.",
                parent=popup,
            )
            return
        overview_render_state.update(
            start=int(start),
            end=int(end),
            display_label=str(display_label),
        )

        figure = _core.Figure(figsize=(16.5, 9.1), dpi=100)
        axes = figure.subplots(grid_rows, grid_cols, squeeze=False, sharex=False, sharey=False).ravel()
        line_width = float(overview_render_state["linewidth"])

        # ``estimator=None`` and ``errorbar=None`` keep each channel as its
        # original sampled trace instead of asking Seaborn to aggregate data.
        style_context = sns.axes_style("whitegrid") if sns is not None else nullcontext()
        with style_context:
            for local_index, axis in enumerate(axes):
                channel_index = page_start + local_index
                if channel_index >= page_end:
                    axis.axis("off")
                    continue

                y_values = segment[:, local_index]
                axis._overview_channel_index = channel_index
                if sns is not None:
                    sns.lineplot(
                        x=time_seconds,
                        y=y_values,
                        ax=axis,
                        color="black",
                        linewidth=line_width,
                        estimator=None,
                        errorbar=None,
                        sort=False,
                        legend=False,
                    )
                else:
                    axis.plot(time_seconds, y_values, color="black", linewidth=line_width, antialiased=True)
                finite_y = y_values[_core.np.isfinite(y_values)]
                if finite_y.size:
                    y_low, y_high = float(_core.np.nanmin(finite_y)), float(_core.np.nanmax(finite_y))
                    padding = max(1.0, abs(y_low) * 0.05) if y_low == y_high else (y_high - y_low) * 0.08
                    axis.set_ylim(y_low - padding, y_high + padding)
                    axis.set_yticks([y_low - padding, y_high + padding])

                axis.set_title(f"ch{channel_index + 1}", fontsize=7.5, pad=2)
                axis.tick_params(axis="both", labelsize=4.5, length=2, pad=1)
                axis.set_xticklabels([])
                if local_index % grid_cols == 0:
                    axis.set_ylabel("mV", fontsize=5.5, labelpad=0)
                else:
                    axis.set_ylabel("")

        figure.suptitle(
            f"{source_label} | Seaborn | page {page_index + 1}/{total_pages} | "
            f"channels {page_start + 1}-{page_end} of {total_channels} | "
            f"{lazy_time_offset + start / fs:.3f}-{lazy_time_offset + end / fs:.3f} s | display: {display_label} | "
            "Y: auto scale/channel | click a panel for detail",
            fontsize=12,
        )
        figure.subplots_adjust(left=0.045, right=0.995, bottom=0.055, top=0.92, wspace=0.28, hspace=0.32)

        if overview_toolbar is not None:
            overview_toolbar.destroy()
        if overview_canvas is not None:
            overview_canvas.get_tk_widget().destroy()
        overview_canvas = _core.FigureCanvasTkAgg(figure, master=plot_frame)
        overview_toolbar = _core.NavigationToolbar2Tk(overview_canvas, plot_frame, pack_toolbar=False)
        overview_toolbar.update()
        overview_toolbar.pack(fill="x", padx=8, pady=(4, 0))
        overview_canvas.draw()
        overview_canvas.mpl_connect("button_press_event", on_overview_click)
        overview_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        status_var.set(
            f"{'Remapped' if is_remapped else 'Raw'} page {page_index + 1}/{total_pages}: "
            f"ch{page_start + 1}-ch{page_end}; {display_label}; "
            f"{sample_count} samples, plotted every {sample_step} sample(s)."
        )

    def open_channel_detail(channel_index: int) -> None:
        """Open a high-resolution, zoomable view for one overview channel."""
        channel_index = int(channel_index)
        if channel_index < 0 or channel_index >= total_channels:
            return
        start = overview_render_state["start"]
        end = overview_render_state["end"]
        if start is None or end is None or int(end) <= int(start):
            _core.messagebox.showerror(
                "Detail unavailable",
                "Render the channel overview again before opening a channel detail.",
                parent=popup,
            )
            return
        start, end = int(start), int(end)
        display_label = overview_render_state["display_label"]
        detail = _core.tk.Toplevel(popup_parent)
        detail.title(f"Channel {channel_index + 1} detail")
        detail.geometry("1280x760")
        detail.minsize(900, 560)
        detail.transient(popup)

        detail_nav = _core.ttk.Frame(detail)
        detail_nav.pack(fill="x", padx=10, pady=(8, 2))
        detail_status = _core.tk.StringVar()
        _core.ttk.Label(detail_nav, textvariable=detail_status, foreground="#245").pack(side="left", padx=10)
        detail_frame = _core.ttk.Frame(detail)
        detail_frame.pack(fill="both", expand=True)
        detail_figure = _core.Figure(figsize=(11, 5.8), dpi=130)
        detail_canvas = _core.FigureCanvasTkAgg(detail_figure, master=detail_frame)
        detail_toolbar = _core.NavigationToolbar2Tk(detail_canvas, detail_frame, pack_toolbar=False)
        detail_toolbar.update()
        detail_toolbar.pack(fill="x", padx=8, pady=(4, 0))
        detail_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        detail_channel = {"index": channel_index}

        def render_detail() -> None:
            current = detail_channel["index"]
            detail_figure.clear()
            detail_axis = detail_figure.add_subplot(111)
            try:
                values_full = (
                    _lazy_h5_read_slice(self, start, end, current).ravel()
                    if lazy_overview
                    else _core.np.asarray(data[start:end, current], dtype=_core.DATA_DTYPE).ravel()
                )
                if values_full.size == 0:
                    raise ValueError("当前时间范围没有可显示的样本。")
                detail_step = max(1, int(_core.np.ceil(values_full.size / 200000)))
                detail_values = values_full[::detail_step]
                detail_x = (
                    _core.np.arange(detail_values.size, dtype=float) * detail_step / fs
                    + lazy_time_offset + start / fs
                )
                # Detailed inspection uses Matplotlib directly: it is more
                # robust for a dense single trace and keeps every sampled
                # point in the requested time range (up to the safety cap).
                detail_axis.plot(
                    detail_x,
                    detail_values,
                    color="#1f77b4",
                    linewidth=0.7,
                    antialiased=True,
                )
                finite_values = detail_values[_core.np.isfinite(detail_values)]
                if finite_values.size == 0:
                    detail_axis.text(
                        0.5,
                        0.5,
                        "This channel contains no finite samples in the selected range.",
                        ha="center",
                        va="center",
                        transform=detail_axis.transAxes,
                    )
                detail_axis.set_xlabel("Time (sec)")
                detail_axis.set_ylabel("Voltage (mV)")
                detail_axis.set_title(f"Channel {current + 1} detail | {display_label}")
                detail_axis.grid(alpha=0.25)
                detail_figure.tight_layout()
                detail_canvas.draw()
                detail_status.set(
                    f"Channel {current + 1}/{total_channels} | "
                    f"{lazy_time_offset + start / fs:.3f}-{lazy_time_offset + end / fs:.3f} s | "
                    f"{values_full.size} samples, displayed every {detail_step} sample(s)"
                )
            except Exception as exc:
                detail_axis.clear()
                detail_axis.axis("off")
                detail_axis.text(
                    0.5,
                    0.5,
                    f"Unable to render channel {current + 1}:\n{exc}",
                    ha="center",
                    va="center",
                    transform=detail_axis.transAxes,
                )
                detail_canvas.draw()
                detail_status.set(f"Channel {current + 1} render failed: {exc}")

        def move_detail(delta: int) -> None:
            detail_channel["index"] = (detail_channel["index"] + delta) % total_channels
            render_detail()

        _core.ttk.Button(detail_nav, text="< Previous channel", command=lambda: move_detail(-1)).pack(side="right", padx=4)
        _core.ttk.Button(detail_nav, text="Next channel >", command=lambda: move_detail(1)).pack(side="right", padx=4)
        render_detail()
        detail.deiconify()
        detail.lift()
        detail.after_idle(detail.lift)

    def on_overview_click(event) -> None:
        if getattr(event, "button", None) != 1 or event.inaxes is None:
            return
        channel_index = getattr(event.inaxes, "_overview_channel_index", None)
        if channel_index is not None:
            open_channel_detail(channel_index)

    def change_overview_page(delta: int) -> None:
        nonlocal page_index
        page_index = (page_index + delta) % total_pages
        render_overview_page()

    _core.ttk.Button(nav_row, text="< Prev page", command=lambda: change_overview_page(-1)).pack(side="left", padx=(0, 4))
    _core.ttk.Button(nav_row, text="Next page >", command=lambda: change_overview_page(1)).pack(side="left", padx=4)
    fullscreen_button = _core.ttk.Button(
        nav_row,
        text="Fullscreen",
        command=lambda: set_overview_fullscreen(not is_fullscreen.get()),
    )
    fullscreen_button.pack(side="left", padx=(12, 4))
    _core.ttk.Label(nav_row, textvariable=status_var, foreground="#245").pack(side="left", padx=12)
    popup.bind("<Escape>", lambda _event: set_overview_fullscreen(False))
    render_overview_page()
    popup.deiconify()
    popup.lift()
    popup.after_idle(popup.lift)


_core._EmbeddedPipelineGUI.open_all_raw_overview_popup = _open_all_raw_overview_with_seaborn


def _processed_export_record(pipeline, source_path: Path) -> dict:
    """Infer the same date/animal/block naming used by batch BIN export."""
    stem = source_path.stem
    match = re.search(
        r"(?P<date>\d{8})[_-](?P<animal>\d+)[_-]block_?(?P<block>\d+)",
        stem,
        flags=re.IGNORECASE,
    )

    def safe_int(value, default=0):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default

    if match is not None:
        date_compact = match.group("date")
        animal = int(match.group("animal"))
        block = int(match.group("block"))
    else:
        record_time_var = getattr(pipeline, "record_time_var", None)
        record_time = record_time_var.get() if record_time_var is not None else ""
        date_digits = re.sub(r"\D", "", str(record_time).strip())
        date_compact = date_digits[:8] if len(date_digits) >= 8 else time.strftime("%Y%m%d")
        animal_var = getattr(pipeline, "animal_var", None)
        block_var = getattr(pipeline, "block_var", None)
        animal = safe_int(animal_var.get() if animal_var is not None else "", 0)
        block = safe_int(block_var.get() if block_var is not None else "", 0)

    date_text = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
    safe_stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem).strip(" ._") or "processed_data"
    return {
        "date_compact": date_compact,
        "date": date_text,
        "animal": animal,
        "block": block,
        "stem": safe_stem,
    }


def _save_processed_channel_h5_files(
    pipeline,
    output_dir: Path,
    data,
    time_vec,
    fs: float,
    metadata: dict,
    channel_ids: list[int] | None = None,
) -> int:
    """Write processed per-channel H5 files without legacy axis guessing."""
    if _core.h5py is None:
        raise ImportError("h5py is required to export channel HDF5 files.")

    array = _core.np.asarray(data, dtype=_core.DATA_DTYPE)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"处理后数据必须是二维数组，当前形状为 {array.shape}。")

    times = _core.np.asarray(time_vec, dtype=_core.np.float64).ravel()
    if times.size != array.shape[0]:
        times = _core.np.arange(array.shape[0], dtype=_core.np.float64) / float(fs)

    output_dir.mkdir(parents=True, exist_ok=True)
    date_token = re.sub(r"[^0-9]", "", str(metadata["date"]))
    file_prefix = (
        f"{date_token}_{int(metadata['animal'])}_block{int(metadata['block'])}"
    )
    compression = _core.hdf5_compression_kwargs()
    string_dtype = _core.h5py.string_dtype(encoding="utf-8")

    if channel_ids is None:
        channel_ids = list(range(1, array.shape[1] + 1))
    channel_ids = sorted({int(channel) for channel in channel_ids if 1 <= int(channel) <= array.shape[1]})
    if not channel_ids:
        raise ValueError("未选择任何可导出的处理后通道。")

    for channel_id in channel_ids:
        channel_index = channel_id - 1
        output_path = output_dir / f"{file_prefix}_ch{channel_id:03d}.h5"
        with _core.h5py.File(output_path, "w") as h5:
            h5.attrs["format"] = "SD_processed_channel_hdf5"
            h5.attrs["compression"] = _core.HDF5_COMPRESSION_LABEL
            h5.attrs["compression_filter"] = "blosc:lz4"
            h5.attrs["compression_level"] = 5
            h5.attrs["compression_shuffle"] = "bitshuffle"
            h5.attrs["data_stage"] = "preprocessed"
            h5.create_dataset("signal", data=array[:, channel_index], chunks=True, **compression)
            h5.create_dataset("time", data=times, chunks=True, **compression)
            h5.create_dataset("FS", data=_core.np.array(float(fs), dtype=_core.np.float64))
            h5.create_dataset("channel", data=_core.np.array(channel_id, dtype=_core.np.int32))
            h5.create_dataset("physical_channel_id", data=_core.np.array(channel_id, dtype=_core.np.int32))
            h5.create_dataset("column_index", data=_core.np.array(channel_index, dtype=_core.np.int32))
            h5.attrs["channel_mapping_version"] = 2
            h5.create_dataset("source_bin", data=str(metadata["source_bin"]), dtype=string_dtype)
            h5.create_dataset("date", data=str(metadata["date"]), dtype=string_dtype)
            h5.create_dataset("animal", data=_core.np.array(int(metadata["animal"]), dtype=_core.np.int32))
            h5.create_dataset("block", data=_core.np.array(int(metadata["block"]), dtype=_core.np.int32))
            h5.create_dataset("data_unit", data="mV", dtype=string_dtype)
            h5.create_dataset("data_stage", data="preprocessed", dtype=string_dtype)
            filter_settings = metadata.get("preprocess_filter_settings")
            if filter_settings:
                h5.attrs["preprocess_filter_settings"] = str(filter_settings)

    return len(channel_ids)


def _custom_channel_id(path: Path) -> int | None:
    """Read the persisted physical channel ID from a single-channel H5."""
    channel_number = None
    if _core.h5py is not None:
        try:
            with _core.h5py.File(path, "r") as h5:
                for key in ("physical_channel_id", "channel", "column_index"):
                    if key in h5:
                        channel_number = int(_core.np.asarray(h5[key]).squeeze())
                        break
                    if key in h5.attrs:
                        channel_number = int(_core.np.asarray(h5.attrs[key]).squeeze())
                        break
        except Exception:
            channel_number = None
    return channel_number


def _custom_channel_sort_key(path: Path):
    """Sort by the persisted physical ID, with chXXX filename priority."""
    channel_number = _custom_channel_id_with_filename(path)
    if channel_number is None:
        channel_number = 10**9
    return channel_number, path.name.lower()


def _custom_channel_id_with_filename(path: Path) -> int | None:
    """Return the physical ID, preferring the user-visible chXXX filename.

    A few files exported by older builds contain a stale ``channel`` dataset
    after remapping, while their generated filename still has the requested
    physical ID. The filename is therefore authoritative when it is explicit.
    """
    match = re.search(r"(?:^|[_-])ch(?:annel)?[_-]?(\d+)(?:\D|$)", path.stem, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return _custom_channel_id(path)


def _import_custom_channel_data(self, on_done=None, preserve_alignment: bool = False) -> None:
    """Import one total H5 or multiple single-channel H5 files for processing."""
    pipeline = self._sync_pipeline_inputs()
    if pipeline is None:
        return

    selected = _core.filedialog.askopenfilenames(
        title="选择通道数据 H5 文件（可多选）",
        filetypes=[
            ("H5 / HDF5 / MAT", "*.h5 *.hdf5 *.mat"),
            ("HDF5", "*.h5 *.hdf5"),
            ("MAT", "*.mat"),
            ("所有文件", "*.*"),
        ],
    )
    if not selected:
        return

    paths = sorted((Path(item) for item in selected), key=_custom_channel_sort_key)
    self.preprocess_filter_progress_var.set(0.0)
    self.preprocess_status_var.set(f"正在读取 {len(paths)} 个自定义通道文件…")
    self.status_var.set("正在导入自定义通道数据…")

    def worker():
        try:
            matrices = []
            channel_ids = []
            is_physical_channel_import = len(paths) > 1
            first_time = None
            sample_rate = None
            total_files = len(paths)

            for index, path in enumerate(paths, start=1):
                data, fs, time_vec, metadata = pipeline.read_mat(path)
                data = pipeline.ensure_data_millivolts(
                    data,
                    metadata.get("data_unit", ""),
                    source_label=f"自定义通道 {path.name}",
                )
                data = _core.np.asarray(data, dtype=_core.DATA_DTYPE)
                if data.ndim == 1:
                    data = data[:, None]
                if data.ndim != 2 or data.shape[1] < 1:
                    raise ValueError(f"{path.name} 的数据形状无效：{data.shape}")

                if sample_rate is None:
                    sample_rate = float(fs)
                elif abs(float(fs) - sample_rate) > max(1e-6, sample_rate * 1e-6):
                    raise ValueError(
                        f"采样率不一致：{path.name} 为 {float(fs):g} Hz，"
                        f"前一个文件为 {sample_rate:g} Hz。"
                    )

                if first_time is None and time_vec is not None:
                    first_time = _core.np.asarray(time_vec, dtype=_core.np.float64).ravel()

                if len(paths) > 1:
                    if data.shape[1] != 1:
                        raise ValueError(
                            f"多选导入时每个文件必须是单通道，{path.name} 是 {data.shape[1]} 通道。"
                        )
                    metadata_channel_id = _custom_channel_id(path)
                    channel_id = _custom_channel_id_with_filename(path)
                    if (
                        metadata_channel_id is not None
                        and channel_id is not None
                        and metadata_channel_id != channel_id
                    ):
                        pipeline.log(
                            f"Channel ID mismatch in {path.name}: H5 metadata="
                            f"{metadata_channel_id}, filename={channel_id}; using filename."
                        )
                    if channel_id is None:
                        channel_id = index
                        pipeline.log(
                            f"Warning: {path.name} has no persisted channel ID; "
                            f"using fallback physical channel {channel_id}."
                        )
                    if channel_id < 1 or channel_id > 520:
                        raise ValueError(f"{path.name} contains invalid channel ID {channel_id}.")
                    if channel_id in channel_ids:
                        raise ValueError(
                            f"Duplicate physical channel ID {channel_id}: {path.name}."
                        )
                    channel_ids.append(channel_id)
                    matrices.append(data[:, 0])
                else:
                    matrices.append(data)

                progress = index * 100.0 / total_files
                pipeline._queue_ui(
                    lambda value=progress, name=path.name, i=index, n=total_files: (
                        self.preprocess_filter_progress_var.set(value),
                        self.preprocess_status_var.set(f"正在读取自定义通道 {i}/{n}：{name}"),
                    )
                )

            if len(paths) == 1:
                source = _core.np.asarray(matrices[0], dtype=_core.DATA_DTYPE)
                single_channel_id = (
                    _custom_channel_id_with_filename(paths[0])
                    if source.ndim == 2 and source.shape[1] == 1
                    else None
                )
                if single_channel_id is not None:
                    if single_channel_id < 1 or single_channel_id > 520:
                        raise ValueError(
                            f"{paths[0].name} contains invalid channel ID {single_channel_id}."
                        )
                    combined = _core.np.full(
                        (source.shape[0], single_channel_id),
                        _core.np.nan,
                        dtype=_core.DATA_DTYPE,
                    )
                    combined[:, single_channel_id - 1] = source[:, 0]
                    channel_ids = [single_channel_id]
                    is_physical_channel_import = True
                else:
                    combined = source
                    channel_ids = list(range(1, combined.shape[1] + 1))
            else:
                sample_counts = {int(item.shape[0]) for item in matrices}
                if len(sample_counts) != 1:
                    detail = ", ".join(
                        f"{path.name}: {matrix.shape[0]}"
                        for path, matrix in zip(paths, matrices)
                    )
                    raise ValueError(f"通道文件的样本数不一致：{detail}")
                max_channel_id = max(channel_ids)
                combined = _core.np.full(
                    (next(iter(sample_counts)), max_channel_id),
                    _core.np.nan,
                    dtype=_core.DATA_DTYPE,
                )
                for channel_id, signal in zip(channel_ids, matrices):
                    combined[:, channel_id - 1] = signal

            if combined.ndim == 1:
                combined = combined[:, None]
            if first_time is None or first_time.size != combined.shape[0]:
                first_time = _core.np.arange(
                    combined.shape[0], dtype=_core.np.float64
                ) / float(sample_rate)

            def finish():
                # A selected chXXX file already carries a physical channel
                # number.  It was placed in that exact matrix column above;
                # applying the Excel map again would map it a second time.
                pipeline._lazy_h5_info = None
                pipeline.raw_data = None if is_physical_channel_import else combined
                pipeline.remapped_data = combined if is_physical_channel_import else None
                pipeline.remapped_channel_ids = (
                    _core.np.arange(1, combined.shape[1] + 1, dtype=int)
                    if is_physical_channel_import
                    else None
                )
                pipeline._remapped_memmap_path = None
                pipeline._preprocessed_memmap_path = None
                _cleanup_remap_memmaps(pipeline)
                pipeline.layout_channel_ids = None
                pipeline.layout_grid = None
                pipeline.preprocessed_data = None
                pipeline.preprocessed_filter_settings = {}
                if preserve_alignment:
                    # LFP/Spike may reload only the already processed subset
                    # from the same recording.  Keep its valid time anchor,
                    # while replacing the channel pool with the newly loaded
                    # physical IDs.
                    for attribute in (
                        "lfp_analysis_selected_channels",
                        "spike_analysis_selected_channels",
                    ):
                        try:
                            delattr(pipeline, attribute)
                        except AttributeError:
                            pass
                    if getattr(pipeline, "timing_info", None):
                        pipeline.alignment_selected_channels = set(channel_ids)
                    else:
                        try:
                            delattr(pipeline, "alignment_selected_channels")
                        except AttributeError:
                            pass
                else:
                    _clear_analysis_channel_lineage(pipeline)
                pipeline.imported_channel_ids = set(channel_ids)
                if not preserve_alignment:
                    pipeline.timing_info = None
                    pipeline.stim_markers = None
                pipeline.clear_processed_filter_cache()
                pipeline.clear_selected_channels(update_tree=False)
                pipeline.fs = float(sample_rate)
                pipeline.time = first_time
                pipeline.mat_path = paths[0]
                pipeline.mat_path_var.set(str(paths[0]))
                pipeline.bin_path_var.set("")
                pipeline.all_raw_preview_channels = []
                if getattr(pipeline, "all_raw_preview_enabled", False):
                    pipeline.start_all_raw_preview()
                preview_channel_var = getattr(self, "preprocessed_preview_channel_var", None)
                if is_physical_channel_import and channel_ids and preview_channel_var is not None:
                    # The default preview is channel 1.  For a reloaded ch500
                    # that column is intentionally empty, so jump directly to
                    # the first real imported physical channel.
                    preview_channel_var.set(str(min(channel_ids)))

                self.active_file_var.set(
                    str(paths[0]) if len(paths) == 1 else f"自定义通道数据（{len(paths)} 个文件）"
                )
                self.active_file_summary_var.set(
                    f"自定义通道数据：{len(paths)} 个文件，{combined.shape[1]} 个通道"
                )
                self.preprocess_filter_progress_var.set(100.0)
                self.preprocess_status_var.set(
                    f"自定义通道数据已导入：{combined.shape[0]} 点 × {combined.shape[1]} 通道"
                )
                self.status_var.set(
                    "自定义通道数据已导入，可以执行通道重映射；如需滤波请在参数中启用"
                )
                self._refresh_page_scroll("preprocess", move_to_top=True)
                export_button = getattr(self, "_processed_channel_export_button", None)
                if export_button is not None:
                    export_button.state(["disabled"])
                if callable(on_done):
                    try:
                        on_done()
                    except Exception:
                        pipeline.log(_core.traceback.format_exc())

            pipeline._queue_ui(finish)
        except Exception as exc:
            pipeline.log(_core.traceback.format_exc())

            def fail(error=str(exc)):
                self.preprocess_status_var.set("自定义通道导入失败")
                self.status_var.set(f"自定义通道导入失败：{error}")
                _core.messagebox.showerror("导入通道数据失败", error, parent=self)

            pipeline._queue_ui(fail)

    _core.threading.Thread(target=worker, daemon=True).start()


def _default_fpc_520_layout() -> object:
    """Return the 20x26 serpentine channel layout shown in the reference."""
    grid = _core.np.empty((20, 26), dtype=int)
    for pair_index in range(13):
        start = 520 - pair_index * 40
        grid[:, pair_index * 2] = _core.np.arange(start, start - 20, -1)
        grid[:, pair_index * 2 + 1] = _core.np.arange(start - 39, start - 19)
    return grid


def _processed_export_layout(pipeline, channel_count: int):
    """Return the requested physical FPC layout for processed-channel export."""
    # The 20x26 selection dialog is explicitly an FPC-position selector.
    # Some workbooks contain a valid but linear 1..520 layout grid; using it
    # here would silently replace the electrode order requested by the user.
    if channel_count == 520:
        return _default_fpc_520_layout()

    # For non-standard channel counts, retain a valid workbook layout when it
    # exists rather than inventing a 20x26 position map.
    layout = getattr(pipeline, "layout_grid", None)
    if layout is not None:
        candidate = _core.np.asarray(layout)
        if candidate.shape == (20, 26):
            numeric = _core.np.asarray(candidate, dtype=float)
            ids = numeric[_core.np.isfinite(numeric)].astype(int)
            if len(ids) == 520 and len(set(ids.tolist())) == 520 and ids.min() >= 1 and ids.max() <= channel_count:
                return numeric.astype(int)
    # Non-520 inputs retain a simple, predictable order rather than inventing
    # electrode positions that do not exist in the mapping file.
    columns = min(26, max(1, channel_count))
    rows = int(_core.np.ceil(channel_count / columns))
    grid = _core.np.zeros((rows, columns), dtype=int)
    grid.flat[:channel_count] = _core.np.arange(1, channel_count + 1)
    return grid


def _open_processed_channel_selector(
    self,
    for_alignment: bool = False,
    analysis_kind: str | None = None,
) -> None:
    """Choose FPC-layout channels for export, alignment, LFP, or Spike."""
    analysis_kind = str(analysis_kind or "").strip().lower()
    if for_alignment or analysis_kind:
        pipeline = self._sync_pipeline_inputs()
        if pipeline is None or not pipeline.has_loaded_data():
            _core.messagebox.showinfo("没有数据", "请先加载脑电数据后再选择通道。", parent=self)
            return
        lazy_selector_info = getattr(pipeline, "_lazy_h5_info", None)
        # The selector only needs the number of channels.  Use a tiny shape
        # placeholder for a lazy source instead of loading all samples.
        data = (
            _core.np.empty((1, int(lazy_selector_info["shape"][1])), dtype=_core.DATA_DTYPE)
            if lazy_selector_info is not None
            else _core.np.asarray(pipeline.current_data(), dtype=_core.DATA_DTYPE)
        )
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or not data.size:
            _core.messagebox.showerror("通道不可用", f"当前数据矩阵不可用：{data.shape}", parent=self)
            return
    else:
        pipeline, data = _preprocessed_preview_data(self)
        if data is None:
            _core.messagebox.showinfo("尚未完成预处理", "请先完成通道重映射后再选择导出通道。", parent=self)
            return

    # Always show the full physical 20x26 plate for <=520-channel EEG data.
    # Positions not inherited from the previous stage stay visible but are
    # disabled, which makes missing channels immediately obvious.
    layout = _default_fpc_520_layout() if data.shape[1] <= 520 else _processed_export_layout(pipeline, data.shape[1])
    grid_ids = {int(value) for value in layout.ravel() if int(value) > 0}
    persisted_loaded_ids = set(getattr(pipeline, "imported_channel_ids", set()) or set())
    if persisted_loaded_ids:
        loaded_ids = grid_ids.intersection(int(channel) for channel in persisted_loaded_ids)
    else:
        loaded_ids = {channel for channel in grid_ids if channel <= data.shape[1]}
    if analysis_kind:
        aligned_ids = set(getattr(pipeline, "alignment_selected_channels", set()) or set())
        if not aligned_ids or not getattr(pipeline, "timing_info", None):
            _core.messagebox.showinfo(
                "尚未完成时间对齐",
                "请先在“时间对齐”页计算并刷新对齐，然后选择本分析使用的通道。",
                parent=self,
            )
            return
        available_ids = loaded_ids.intersection(aligned_ids)
        if not available_ids:
            _core.messagebox.showinfo(
                "没有可继承通道",
                "时间对齐结果中没有与当前数据对应的可用通道。",
                parent=self,
            )
            return
        stored_ids = getattr(pipeline, f"{analysis_kind}_analysis_selected_channels", None)
    elif for_alignment:
        available_ids = loaded_ids
        stored_ids = getattr(pipeline, "alignment_selected_channels", None)
    else:
        available_ids = grid_ids
        stored_ids = getattr(self, "processed_export_channel_ids", None)
    selected_ids = set(stored_ids) if stored_ids is not None else set(available_ids)
    selected_ids.intersection_update(available_ids)
    if (for_alignment or analysis_kind) and not selected_ids:
        selected_ids = set(available_ids)

    dialog = _core.tk.Toplevel(self)
    if analysis_kind == "lfp":
        dialog_title = "选择用于 LFP 分析的对齐后通道"
    elif analysis_kind == "spike":
        dialog_title = "选择用于 Spike 分析的对齐后通道"
    elif for_alignment:
        dialog_title = "选择用于时间对齐的通道"
    else:
        dialog_title = "选择处理后导出通道"
    dialog.title(dialog_title)
    dialog.geometry("1260x800")
    dialog.minsize(900, 620)
    dialog.transient(self)
    dialog.grab_set()
    try:
        dialog.state("zoomed")
    except _core.tk.TclError:
        pass

    header = _core.ttk.Frame(dialog)
    header.pack(fill="x", padx=12, pady=(10, 6))
    summary_var = _core.tk.StringVar()
    _core.ttk.Label(
        header,
        text=(
            f"{dialog_title}：红色为可选通道，灰色为上一阶段未加载/未选择通道。"
            if for_alignment or analysis_kind
            else "按电极布局勾选处理后通道：行/列按钮可批量勾选；默认全选。"
        ),
    ).pack(side="left")
    _core.ttk.Label(header, textvariable=summary_var, foreground="#245").pack(side="right")

    body = _core.ttk.Frame(dialog)
    body.pack(fill="both", expand=True, padx=12, pady=4)
    body.grid_columnconfigure(0, minsize=52)
    variables = {
        channel_id: _core.tk.BooleanVar(value=channel_id in selected_ids)
        for channel_id in grid_ids
    }
    select_all_buttons = []

    def update_summary() -> None:
        count = sum(variable.get() for variable in variables.values())
        summary_var.set(
            f"已选择 {count}/{len(available_ids)} 个可用通道"
            + (f"；不可用 {len(grid_ids) - len(available_ids)}" if len(grid_ids) != len(available_ids) else "")
        )
        button_text = f"全不选 {len(available_ids)}" if count == len(available_ids) else f"全选 {len(available_ids)}"
        for button in select_all_buttons:
            try:
                button.configure(text=button_text)
            except _core.tk.TclError:
                pass

    def set_channels(channel_ids, selected: bool | None = None) -> None:
        relevant = [channel for channel in channel_ids if channel in available_ids]
        if not relevant:
            return
        if selected is None:
            selected = not all(variables[channel].get() for channel in relevant)
        for channel in relevant:
            variables[channel].set(selected)
        update_summary()

    select_all_button = _core.ttk.Button(body, text=f"全选 {len(available_ids)}", command=lambda: set_channels(available_ids))
    select_all_buttons.append(select_all_button)
    select_all_button.grid(
        row=0, column=0, sticky="ew", padx=2, pady=2
    )
    for column_index in range(layout.shape[1]):
        column_ids = [int(value) for value in layout[:, column_index] if int(value) > 0]
        _core.ttk.Button(
            body,
            text=f"列{column_index + 1}",
            command=lambda ids=column_ids: set_channels(ids),
            width=4,
        ).grid(row=0, column=column_index + 1, sticky="ew", padx=1, pady=2)

    for row_index in range(layout.shape[0]):
        row_ids = [int(value) for value in layout[row_index, :] if int(value) > 0]
        _core.ttk.Button(
            body,
            text=f"行{row_index + 1}",
            command=lambda ids=row_ids: set_channels(ids),
            width=5,
        ).grid(row=row_index + 1, column=0, sticky="ew", padx=2, pady=1)
        for column_index, raw_channel in enumerate(layout[row_index, :]):
            channel_id = int(raw_channel)
            if channel_id <= 0:
                continue
            is_available = channel_id in available_ids
            channel_check = _core.tk.Checkbutton(
                body,
                text=str(channel_id),
                variable=variables[channel_id],
                command=update_summary,
                padx=1,
                pady=0,
                anchor="w",
                fg="#C62828" if is_available and (for_alignment or analysis_kind) else "#222222",
                activeforeground="#B71C1C" if is_available else "#999999",
                disabledforeground="#A8A8A8",
                selectcolor="#FFE5E5" if is_available and (for_alignment or analysis_kind) else "white",
            )
            if not is_available:
                channel_check.configure(state="disabled")
            channel_check.grid(row=row_index + 1, column=column_index + 1, sticky="w", padx=1, pady=0)

    def apply_selection() -> None:
        chosen = {channel for channel, variable in variables.items() if variable.get()}
        if not chosen:
            _core.messagebox.showinfo("未选择通道", "请至少勾选一个通道。", parent=dialog)
            return
        if for_alignment:
            pipeline.selected_channels = chosen
            refresh_selection = getattr(pipeline, "update_selected_summary", None)
            if callable(refresh_selection):
                refresh_selection()
            try:
                self._mirror_pipeline_state(pipeline)
            except Exception:
                pass
            dialog.destroy()
            _original_refresh_alignment(self)
            if getattr(pipeline, "timing_info", None):
                pipeline.alignment_selected_channels = set(chosen)
            _update_analysis_file_banners(self, pipeline)
            return
        if analysis_kind:
            setattr(pipeline, f"{analysis_kind}_analysis_selected_channels", set(chosen))
            pipeline.selected_channels = set(chosen)
            refresh_selection = getattr(pipeline, "update_selected_summary", None)
            if callable(refresh_selection):
                refresh_selection()
            dialog.destroy()
            _update_analysis_file_banners(self, pipeline)
            if analysis_kind == "lfp":
                _start_lfp_snr_after_selection(self, pipeline)
            else:
                _original_run_spike(self)
            return
        self.processed_export_channel_ids = chosen
        if hasattr(self, "processed_export_selection_var"):
            self.processed_export_selection_var.set(f"已选 {len(chosen)}/{len(available_ids)} 个通道")
        self.status_var.set(f"已选择 {len(chosen)} 个处理后通道用于导出")
        dialog.destroy()
        _export_processed_channel_data(self)

    # Keep all actions in the same grid, directly below the final channel row.
    # This remains visible in both normal and maximized windows and cannot be
    # covered by the expanding 20x26 channel matrix.
    action_row = _core.ttk.Frame(body)
    action_row.grid(
        row=layout.shape[0] + 1,
        column=1,
        columnspan=layout.shape[1],
        sticky="ew",
        padx=2,
        pady=(8, 4),
    )
    footer_select_all_button = _core.ttk.Button(
        action_row,
        text=f"全选 {len(available_ids)}",
        command=lambda: set_channels(available_ids),
    )
    select_all_buttons.append(footer_select_all_button)
    footer_select_all_button.pack(side="left", padx=(0, 4))
    _core.ttk.Button(action_row, text="全不选", command=lambda: set_channels(available_ids, False)).pack(side="left", padx=4)
    _core.ttk.Button(
        action_row,
        text="反选",
        command=lambda: [variable.set(not variable.get()) for variable in variables.values()] and update_summary(),
    ).pack(side="left", padx=4)
    _core.ttk.Button(action_row, text="取消", command=dialog.destroy).pack(side="right", padx=4)
    _core.ttk.Button(
        action_row,
        text=(
            "使用所选通道并刷新对齐"
            if for_alignment
            else f"使用所选通道运行 {analysis_kind.upper()}" if analysis_kind
            else "导出所选通道"
        ),
        command=apply_selection,
    ).pack(side="right", padx=4)
    update_summary()


_original_refresh_alignment = _core.AnalysisGUI._refresh_alignment


def _refresh_alignment_with_channel_selector(self) -> None:
    """Select the LFP channels before starting marker/time alignment."""
    _open_processed_channel_selector(self, for_alignment=True)


_core.AnalysisGUI._refresh_alignment = _refresh_alignment_with_channel_selector


_original_run_spike = _core.AnalysisGUI._run_spike


def _run_spike_with_channel_selector(self) -> None:
    """Select an independent aligned-channel pool before Spike analysis."""
    _open_processed_channel_selector(self, analysis_kind="spike")


_core.AnalysisGUI._run_spike = _run_spike_with_channel_selector


def _activate_analysis_channel_pool(self, kind: str):
    """Restore the independent LFP/Spike channel set before follow-up actions."""
    pipeline = self._sync_pipeline_inputs()
    if pipeline is None:
        return None
    chosen = set(getattr(pipeline, f"{kind}_analysis_selected_channels", set()) or set())
    if chosen:
        pipeline.selected_channels = chosen
        refresh_selection = getattr(pipeline, "update_selected_summary", None)
        if callable(refresh_selection):
            refresh_selection()
    return pipeline


def _install_analysis_channel_activation() -> None:
    method_groups = {
        "lfp": (
            "_preview_lfp_selected",
            "_manual_lfp_review",
            "_open_lfp_view",
            "_export_lfp",
            "_export_filtered_selected_channels",
        ),
        "spike": (
            "_plot_spike",
            "_dynamic_spike",
            "_open_flash_spike",
            "_open_letter_spike",
            "_export_spike",
            "_export_spike_bundle",
        ),
    }
    for kind, method_names in method_groups.items():
        for method_name in method_names:
            original = getattr(_core.AnalysisGUI, method_name, None)
            if not callable(original):
                continue

            def activated(self, *args, _original=original, _kind=kind, **kwargs):
                _activate_analysis_channel_pool(self, _kind)
                return _original(self, *args, **kwargs)

            setattr(_core.AnalysisGUI, method_name, activated)


_install_analysis_channel_activation()


def _export_processed_channel_data(self) -> None:
    """Export the latest processed (remapped and optionally filtered) H5 files."""
    pipeline = self._sync_pipeline_inputs()
    if pipeline is None:
        return

    try:
        if not pipeline.has_loaded_data():
            _core.messagebox.showerror(
                "没有数据",
                "请先读取总 H5 或完成 BIN 导入。",
                parent=self,
            )
            return

        processed = getattr(pipeline, "preprocessed_data", None)
        if processed is None:
            self.status_var.set("请先执行预处理")
            _core.messagebox.showinfo(
                "尚未完成处理",
                "请先执行“通道重映射”；如已启用滤波，系统会一并完成滤波，然后即可导出。",
                parent=self,
            )
            return

        output_text = str(self.output_var.get()).strip()
        if not output_text:
            selected_dir = _core.filedialog.askdirectory(
                title="选择处理后通道数据的保存目录",
                parent=self,
            )
            if not selected_dir:
                self.status_var.set("已取消选择处理后通道数据的保存目录")
                return
            output_text = str(selected_dir).strip()
            self.output_var.set(output_text)

        data = _core.np.asarray(processed, dtype=_core.DATA_DTYPE)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2:
            raise ValueError(f"处理后数据必须是二维数组，当前形状为 {data.shape}。")
        selected_channel_ids = sorted(
            int(channel)
            for channel in getattr(self, "processed_export_channel_ids", set(range(1, data.shape[1] + 1)))
            if 1 <= int(channel) <= data.shape[1]
        )
        if not selected_channel_ids:
            _core.messagebox.showinfo("未选择通道", "请先在“选择导出通道”中至少勾选一个通道。", parent=self)
            return

        source_text = ""
        source_variables = [
            getattr(self, "active_file_var", None),
            getattr(self, "path_var", None),
            getattr(self, "parsed_h5_path_var", None),
            getattr(pipeline, "bin_path_var", None),
            getattr(pipeline, "mat_path_var", None),
        ]
        for variable in source_variables:
            if variable is not None:
                source_text = str(variable.get()).strip()
                if source_text:
                    break
        source_path = Path(source_text) if source_text else Path("processed_data")
        record = _processed_export_record(pipeline, source_path)

        output_root = Path(output_text).expanduser().resolve()
        channel_root = output_root / "channel_exports" / "processed"
        output_dir = (
            channel_root
            / record["date_compact"]
            / str(record["animal"])
            / record["stem"]
        )
        pipeline.channel_dir_var.set(str(channel_root))
        channel_folder_var = getattr(self, "channel_folder_var", None)
        if channel_folder_var is not None:
            channel_folder_var.set(str(channel_root))

        time_vec = getattr(pipeline, "time", None)
        if time_vec is None or len(time_vec) != data.shape[0]:
            time_vec = _core.np.arange(data.shape[0], dtype=_core.DATA_DTYPE) / float(pipeline.fs)
        else:
            time_vec = _core.np.asarray(time_vec, dtype=_core.np.float64).ravel()

        metadata = {
            "source_bin": str(source_path.resolve()),
            "date": record["date"],
            "animal": record["animal"],
            "block": record["block"],
            "data_unit": "mV",
            "data_stage": "preprocessed",
            "preprocess_filter_settings": getattr(
                pipeline, "preprocessed_filter_settings", {}
            ),
        }

        output_lock = getattr(pipeline, "_output_job_lock", None)
        if output_lock is not None and not output_lock.acquire(blocking=False):
            _core.messagebox.showinfo(
                "正在导出",
                "已有通道导出任务正在运行，请等待当前任务完成。",
                parent=self,
            )
            return
        pipeline._active_output_job = "处理后通道 H5 导出"
        self.preprocess_filter_progress_var.set(0.0)
        self.preprocess_status_var.set("正在导出处理后通道 H5…")
        self.status_var.set(f"正在导出 {len(selected_channel_ids)} 个处理后通道 H5…")

        def worker():
            try:
                count = _save_processed_channel_h5_files(
                    pipeline,
                    output_dir,
                    data,
                    time_vec,
                    float(pipeline.fs),
                    metadata,
                    selected_channel_ids,
                )

                def finish():
                    self.preprocess_filter_progress_var.set(100.0)
                    self.preprocess_status_var.set(f"处理后通道 H5 已导出：{count} 个")
                    self.status_var.set(f"处理后通道 H5 已导出到：{output_dir}")
                    _core.messagebox.showinfo(
                        "导出完成",
                        f"已导出 {count} 个处理后通道 H5：\n{output_dir}",
                        parent=self,
                    )

                pipeline._queue_ui(finish)
            except Exception as exc:
                pipeline.log(_core.traceback.format_exc())

                def fail(error=str(exc)):
                    self.preprocess_status_var.set("处理后通道导出失败")
                    self.status_var.set(f"处理后通道导出失败：{error}")
                    _core.messagebox.showerror("通道导出失败", error, parent=self)

                pipeline._queue_ui(fail)
            finally:
                pipeline._active_output_job = ""
                if output_lock is not None:
                    output_lock.release()

        _core.threading.Thread(target=worker, daemon=True).start()
    except Exception as exc:
        self.preprocess_status_var.set("处理后通道导出失败")
        self.status_var.set(f"处理后通道导出失败：{exc}")
        _core.messagebox.showerror("通道导出失败", str(exc), parent=self)


def _import_preprocess_h5(self) -> None:
    """Load one complete H5 directly as the preprocessing input."""
    path_text = _core.filedialog.askopenfilename(
        title="选择要进行预处理的完整 H5",
        filetypes=[
            ("H5/MAT files", "*.h5 *.hdf5 *.mat"),
            ("All files", "*.*"),
        ],
        parent=self,
    )
    if not path_text:
        return

    pipeline = self._sync_pipeline_inputs()
    if pipeline is None:
        return
    try:
        selected_path = Path(path_text).resolve()
        pipeline.bin_path_var.set("")
        pipeline.mat_path_var.set(str(selected_path))
        pipeline.raw_data = None
        pipeline.remapped_data = None
        pipeline.remapped_channel_ids = None
        pipeline.layout_channel_ids = None
        pipeline.layout_grid = None
        pipeline.preprocessed_data = None
        pipeline.preprocessed_filter_settings = {}
        pipeline.clear_processed_filter_cache()
        pipeline.clear_selected_channels(update_tree=False)
        pipeline.load_existing_mat()

        _render_preprocessed_channel_preview(self, show_empty_state=True)

        self.active_file_var.set(str(selected_path))
        if hasattr(self, "active_file_summary_var"):
            self.active_file_summary_var.set(selected_path.name)
        if hasattr(self, "parsed_h5_path_var"):
            self.parsed_h5_path_var.set(str(selected_path))
        if _main_preprocess_filter_enabled(self):
            self.preprocess_filter_progress_var.set(0.0)
            self.preprocess_filter_status_var.set("预处理滤波：等待运行")
        else:
            _update_preprocess_filter_idle_display(self)
        self.preprocess_status_var.set("预处理输入已加载")
        self.status_var.set(f"已加载预处理输入: {selected_path.name}")
        self._refresh_page_scroll("preprocess", move_to_top=True)
    except Exception as exc:
        self.status_var.set(f"加载预处理 H5 失败: {exc}")
        _core.messagebox.showerror("加载预处理 H5 失败", str(exc), parent=self)


_original_build_preprocess_with_overview = _core.AnalysisGUI._build_preprocess


def _build_preprocess_with_overview(self, parent) -> None:
    _original_build_preprocess_with_overview(self, parent)
    if getattr(self, "_filtered_overview_button", None) is not None:
        return
    for child in _walk_widgets(parent):
        try:
            label = str(child.cget("text"))
            if "滤波并批量导出全部通道" not in label:
                continue
            button = _core.ttk.Button(
                child.master,
                text=_processed_overview_button_text(self._sync_pipeline_inputs()),
                command=lambda owner=self: _show_filtered_all_channel_overview(owner),
            )
            button.pack(side="left", padx=(8, 4))
            self._filtered_overview_button = button
            break
        except Exception:
            continue


_core.AnalysisGUI._build_preprocess = _build_preprocess_with_overview


_original_build_preprocess_with_channel_export = _core.AnalysisGUI._build_preprocess


def _preprocessed_preview_data(self):
    """Return the post-remapping/post-filtering signal matrix, if available."""
    pipeline = self._sync_pipeline_inputs()
    if pipeline is None:
        return None, None
    data = getattr(pipeline, "preprocessed_data", None)
    if data is None:
        return pipeline, None
    array = _core.np.asarray(data, dtype=_core.DATA_DTYPE)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not array.size:
        return pipeline, None
    return pipeline, array


def _render_preprocessed_channel_preview(self, show_empty_state: bool = True) -> None:
    """Render one channel from the preprocessing result cache."""
    figure = getattr(self, "preprocessed_preview_fig", None)
    canvas = getattr(self, "preprocessed_preview_canvas", None)
    status_var = getattr(self, "preprocessed_preview_status_var", None)
    if figure is None or canvas is None:
        return

    pipeline, data = _preprocessed_preview_data(self)
    figure.clear()
    axis = figure.add_subplot(111)
    if data is None:
        axis.axis("off")
        axis.set_title("Preprocessed channel preview")
        if show_empty_state:
            axis.text(
                0.5,
                0.5,
                "Run channel remapping (and optional filtering) to view processed data here.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        canvas.draw()
        if status_var is not None:
            status_var.set("Preprocessed preview: waiting for preprocessing")
        return

    try:
        fs = float(pipeline.fs)
        if not _core.np.isfinite(fs) or fs <= 0:
            raise ValueError(f"Invalid sampling rate: {pipeline.fs!r}.")
        try:
            channel = int(float(self.preprocessed_preview_channel_var.get().strip()))
        except (TypeError, ValueError):
            channel = 1
        channel = max(1, min(channel, data.shape[1]))
        self.preprocessed_preview_channel_var.set(str(channel))
        start_seconds = max(0.0, _core.parse_float(self.preprocessed_preview_start_var.get(), 0.0))
        duration_seconds = max(0.01, _core.parse_float(self.preprocessed_preview_duration_var.get(), 5.0))
        start = min(data.shape[0], int(start_seconds * fs))
        end = min(data.shape[0], start + max(1, int(duration_seconds * fs)))
        if end <= start:
            start, end = 0, min(data.shape[0], max(1, int(duration_seconds * fs)))
        if end <= start:
            raise ValueError("The selected time range contains no samples.")

        sample_count = end - start
        sample_step = max(1, int(_core.np.ceil(sample_count / 20000)))
        sample_indices = _core.np.arange(start, end, sample_step)
        time_seconds = sample_indices / fs
        values = _core.np.asarray(data[start:end:sample_step, channel - 1], dtype=_core.DATA_DTYPE).ravel()
        if values.size == 0:
            raise ValueError("The selected channel has no samples in this range.")
        finite_count = int(_core.np.isfinite(values).sum())
        mode = str((getattr(pipeline, "preprocessed_filter_settings", {}) or {}).get("mode", "off"))
        stage = "Remapping + filtering" if _filter_mode_enabled(mode) else "Remapping (unfiltered)"
        if finite_count == 0:
            axis.axis("off")
            axis.set_title(f"Preprocessed ch{channel} | {stage}")
            axis.text(
                0.5,
                0.5,
                "This channel contains no finite samples in the selected range.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            canvas.draw()
            if status_var is not None:
                status_var.set(f"ch{channel}/{data.shape[1]} | no finite samples")
            return

        line_width = min(1.5, max(0.1, _core.parse_float(self.preprocessed_preview_linewidth_var.get(), 0.55)))
        sns = _get_seaborn()
        if sns is not None:
            with sns.axes_style("whitegrid"):
                sns.lineplot(
                    x=time_seconds,
                    y=values,
                    ax=axis,
                    color="#1f77b4",
                    linewidth=line_width,
                    estimator=None,
                    errorbar=None,
                    sort=False,
                    legend=False,
                )
        else:
            axis.plot(time_seconds, values, color="#1f77b4", linewidth=line_width, antialiased=True)

        axis.set_xlabel("Time (sec)")
        axis.set_ylabel("Voltage (mV)")
        axis.set_title(f"Preprocessed ch{channel} | {stage}")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        canvas.draw()
        if status_var is not None:
            status_var.set(
                f"ch{channel}/{data.shape[1]} | {start / fs:.3f}-{end / fs:.3f} s | "
                f"{sample_count} samples, every {sample_step} sample(s) | finite {finite_count}/{values.size}"
            )
    except Exception as exc:
        axis.clear()
        axis.axis("off")
        axis.set_title("Preprocessed channel preview unavailable")
        axis.text(
            0.5,
            0.5,
            f"Unable to render the processed channel:\n{exc}",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        canvas.draw()
        if status_var is not None:
            status_var.set(f"Preprocessed preview failed: {exc}")
        try:
            pipeline.log(_core.traceback.format_exc())
        except Exception:
            pass


def _change_preprocessed_preview_channel(self, delta: int) -> None:
    _, data = _preprocessed_preview_data(self)
    if data is None:
        _core.messagebox.showinfo("Preprocessing required", "Run channel remapping before opening processed data.", parent=self)
        return
    try:
        channel = int(float(self.preprocessed_preview_channel_var.get()))
    except (TypeError, ValueError):
        channel = 1
    self.preprocessed_preview_channel_var.set(str((channel - 1 + delta) % data.shape[1] + 1))
    _render_preprocessed_channel_preview(self)


def _show_preprocessed_all_channel_overview(self) -> None:
    """Reuse the import-page 10x10 overview with processed data as source."""
    pipeline, data = _preprocessed_preview_data(self)
    if data is None:
        _core.messagebox.showinfo("Preprocessing required", "Run channel remapping before opening processed data.", parent=self)
        return
    mode = str((getattr(pipeline, "preprocessed_filter_settings", {}) or {}).get("mode", "off"))
    stage = "Remapping + filtering" if _filter_mode_enabled(mode) else "Remapping (unfiltered)"
    pipeline.popup_parent = self
    pipeline._overview_data_override = data
    pipeline._overview_source_label = f"Preprocessed channel overview ({stage})"
    pipeline._overview_title = f"Preprocessed all-channel overview - {stage} - 10x10"
    pipeline._overview_start_seconds = max(0.0, _core.parse_float(self.preprocessed_preview_start_var.get(), 0.0))
    pipeline._overview_duration_seconds = max(0.01, _core.parse_float(self.preprocessed_preview_duration_var.get(), 5.0))
    pipeline._overview_linewidth = _core.parse_float(self.preprocessed_preview_linewidth_var.get(), 0.55)
    try:
        pipeline.open_all_raw_overview_popup()
        self.status_var.set(f"Opened preprocessed all-channel overview: {stage}")
    finally:
        for attribute in (
            "_overview_data_override",
            "_overview_source_label",
            "_overview_title",
            "_overview_start_seconds",
            "_overview_duration_seconds",
            "_overview_linewidth",
        ):
            try:
                delattr(pipeline, attribute)
            except AttributeError:
                pass


def _build_preprocessed_preview_panel(self, parent) -> None:
    """Fill the lower blank region of the preprocessing page with a viewer."""
    if getattr(self, "_preprocessed_preview_panel", None) is not None:
        return
    panel = _core.ttk.LabelFrame(parent, text="Preprocessed channel data viewer")
    # The compiled preprocessing page lays out its cards with ``grid``.
    # Use the next free grid row instead of mixing ``pack`` into that parent.
    grid_rows = []
    grid_columns = []
    for child in parent.winfo_children():
        try:
            info = child.grid_info()
            if not info:
                continue
            grid_rows.append(int(info.get("row", 0)))
            grid_columns.append(int(info.get("column", 0)) + int(info.get("columnspan", 1)))
        except Exception:
            continue
    panel_row = max(grid_rows, default=-1) + 1
    panel_columns = max(grid_columns, default=1)
    panel.grid(
        row=panel_row,
        column=0,
        columnspan=panel_columns,
        sticky="nsew",
        padx=10,
        pady=(6, 10),
    )
    parent.grid_rowconfigure(panel_row, weight=1)
    controls = _core.ttk.Frame(panel)
    controls.pack(fill="x", padx=8, pady=(8, 4))
    self.preprocessed_preview_channel_var = _core.tk.StringVar(value="1")
    self.preprocessed_preview_start_var = _core.tk.StringVar(value="0")
    self.preprocessed_preview_duration_var = _core.tk.StringVar(value="5")
    self.preprocessed_preview_linewidth_var = _core.tk.StringVar(value="0.55")
    self.preprocessed_preview_status_var = _core.tk.StringVar(value="Preprocessed preview: waiting for preprocessing")
    for label, variable, width in (
        ("Channel", self.preprocessed_preview_channel_var, 7),
        ("Start (s)", self.preprocessed_preview_start_var, 8),
        ("Duration (s)", self.preprocessed_preview_duration_var, 8),
        ("Line width", self.preprocessed_preview_linewidth_var, 6),
    ):
        _core.ttk.Label(controls, text=label).pack(side="left", padx=(0, 4))
        _core.ttk.Entry(controls, textvariable=variable, width=width).pack(side="left", padx=(0, 10))
    _core.ttk.Button(controls, text="View processed channel", command=lambda: _render_preprocessed_channel_preview(self)).pack(side="left", padx=4)
    _core.ttk.Button(controls, text="< Prev", command=lambda: _change_preprocessed_preview_channel(self, -1)).pack(side="left", padx=4)
    _core.ttk.Button(controls, text="Next >", command=lambda: _change_preprocessed_preview_channel(self, 1)).pack(side="left", padx=4)
    _core.ttk.Button(controls, text="Fullscreen overview (10x10)", command=lambda: _show_preprocessed_all_channel_overview(self)).pack(side="left", padx=(12, 4))
    _core.ttk.Label(panel, textvariable=self.preprocessed_preview_status_var, foreground="#245").pack(fill="x", padx=10, pady=(0, 4))
    figure_frame = _core.ttk.Frame(panel)
    figure_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
    self.preprocessed_preview_fig = _core.Figure(figsize=(9, 4.6), dpi=120)
    self.preprocessed_preview_canvas = _core.FigureCanvasTkAgg(self.preprocessed_preview_fig, master=figure_frame)
    self.preprocessed_preview_toolbar = _core.NavigationToolbar2Tk(self.preprocessed_preview_canvas, figure_frame, pack_toolbar=False)
    self.preprocessed_preview_toolbar.update()
    self.preprocessed_preview_toolbar.pack(fill="x", padx=2, pady=(0, 2))
    self.preprocessed_preview_canvas.get_tk_widget().pack(fill="both", expand=True)
    self._preprocessed_preview_panel = panel
    _render_preprocessed_channel_preview(self, show_empty_state=True)


def _build_preprocess_with_channel_export(self, parent) -> None:
    _original_build_preprocess_with_channel_export(self, parent)
    if getattr(self, "_custom_channel_import_button", None) is not None:
        return

    for child in _walk_widgets(parent):
        try:
            if str(child.cget("text")) not in {
                "执行通道重映射",
                "执行通道重映射并滤波",
            }:
                continue
            # The visible primary action is packed; the same text in the
            # detailed parameter card is grid-managed and is left untouched.
            if child.winfo_manager() != "pack":
                continue
            import_button = _core.ttk.Button(
                child.master,
                text="导入自定义通道数据",
                style="Soft.TButton",
                command=lambda owner=self: _import_custom_channel_data(owner),
            )
            import_button.pack(side="left", padx=(0, 4), before=child)
            self._custom_channel_import_button = import_button

            h5_button = _core.ttk.Button(
                child.master,
                text="导入完整 H5 做预处理",
                style="Soft.TButton",
                command=lambda owner=self: _import_preprocess_h5(owner),
            )
            h5_button.pack(side="left", padx=(0, 4), before=child)
            self._preprocess_h5_import_button = h5_button

            selector_button = _core.ttk.Button(
                child.master,
                text="选择导出通道（20×26）",
                style="Soft.TButton",
                command=lambda owner=self: _open_processed_channel_selector(owner),
            )
            selector_button.pack(side="left", padx=(8, 4), after=child)
            selector_button.state(["disabled"])
            self._processed_channel_selector_button = selector_button

            # Export is intentionally kept inside the channel-layout dialog.
            # The main preprocessing page only needs the one entry point,
            # avoiding two adjacent selection/export action buttons.
            self._processed_channel_export_button = None
            break
        except Exception:
            continue
    _build_preprocessed_preview_panel(self, parent)


_core.AnalysisGUI._build_preprocess = _build_preprocess_with_channel_export


_original_build_import = _core.AnalysisGUI._build_import


def _build_import_with_output_first(self, parent) -> None:
    """Put the shared output directory before any batch-import action."""
    _original_build_import(self, parent)
    # Keep the parsing-range terminology consistent with the Chinese UI while
    # retaining the BooleanVar and all range-selection logic unchanged.
    for child in _walk_widgets(parent):
        try:
            if str(child.cget("text")).strip().lower() == "full length":
                child.configure(text="全时间段")
        except Exception:
            continue
    if getattr(self, "_output_first_box", None) is not None:
        return

    output_var_name = str(getattr(self, "output_var", ""))
    batch_status_var_name = str(getattr(self, "batch_status_var", ""))
    output_entry = None
    batch_status_label = None

    for child in _walk_widgets(parent):
        try:
            textvariable = str(child.cget("textvariable"))
            if output_var_name and textvariable == output_var_name:
                output_entry = child
            if batch_status_var_name and textvariable == batch_status_var_name:
                batch_status_label = child
        except Exception:
            continue

    if output_entry is None or batch_status_label is None:
        return

    # The batch status label is inside the batch card body.  Its parent and
    # the card hierarchy are stable in gui_core, while avoiding hard-coded
    # widget names keeps this wrapper compatible with the compiled core.
    batch_body = batch_status_label.master
    batch_card = batch_body.master
    actions = batch_card.master

    # Folder selection belongs to the batch action itself.  Keeping a second
    # visible selector in this card only duplicates that workflow.
    for child in list(_walk_widgets(batch_card)):
        try:
            if str(child.cget("text")).strip() == "选择文件夹":
                child.destroy()
        except Exception:
            pass

    # Hide only the old output-directory label, entry, and browse button.
    # Keep the save-H5 button and progress display in the original output card.
    old_output_body = output_entry.master
    for child in list(old_output_body.winfo_children()):
        try:
            child_text = str(child.cget("text"))
        except Exception:
            child_text = ""
        if child is output_entry or child_text in ("输出目录", "选择目录"):
            try:
                child.grid_remove()
            except Exception:
                pass

    colors = getattr(_core, "COLORS", {})
    card_bg = colors.get("card", "white")
    line_color = colors.get("line", "#cfd4dc")
    ink_color = colors.get("ink", "#202124")
    muted_color = colors.get("muted", "#6b7280")

    output_box = _core.tk.Frame(
        actions,
        bg=card_bg,
        highlightbackground=line_color,
        highlightcolor=line_color,
        highlightthickness=1,
        bd=0,
    )
    output_box.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    output_box.grid_columnconfigure(1, weight=1)

    _core.tk.Label(
        output_box,
        text="输出目录（请先确认）",
        bg=card_bg,
        fg=ink_color,
        font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, sticky="w", padx=(14, 12), pady=(12, 2))

    output_entry_top = _core.tk.Entry(
        output_box,
        textvariable=self.output_var,
        width=48,
        relief="solid",
        bd=1,
        font=("Segoe UI", 9),
    )
    output_entry_top.grid(row=0, column=1, sticky="ew", padx=4, pady=(10, 2))

    _core.ttk.Button(
        output_box,
        text="选择目录",
        style="Soft.TButton",
        command=self._choose_output,
    ).grid(row=0, column=2, sticky="e", padx=(8, 14), pady=(10, 2))

    _core.tk.Label(
        output_box,
        text="批量解析生成的总 H5 和单通道 H5 都以此目录为根目录",
        bg=card_bg,
        fg=muted_color,
        font=("Segoe UI", 9),
        anchor="w",
    ).grid(row=1, column=1, columnspan=2, sticky="w", padx=4, pady=(0, 10))

    # Move the two action cards below the directory confirmation block.
    for child in actions.winfo_children():
        if child is output_box:
            continue
        try:
            info = child.grid_info()
            if str(info.get("row")) == "0":
                child.grid_configure(row=1)
        except Exception:
            pass
    actions.grid_rowconfigure(1, weight=1)
    self._output_first_box = output_box


_core.AnalysisGUI._build_import = _build_import_with_output_first


def _text_variable_value(owner, name: str) -> str:
    variable = getattr(owner, name, None)
    if variable is None:
        return ""
    try:
        return str(variable.get()).strip()
    except Exception:
        return ""


def _analysis_source_text(self, pipeline) -> str:
    for owner, name in (
        (self, "active_file_var"),
        (self, "parsed_h5_path_var"),
        (pipeline, "mat_path_var"),
        (pipeline, "bin_path_var"),
    ):
        value = _text_variable_value(owner, name)
        if value:
            return value
    mat_path = getattr(pipeline, "mat_path", None)
    return str(mat_path) if mat_path else "尚未加载脑电文件"


def _update_analysis_file_banners(self, pipeline=None) -> None:
    if pipeline is None:
        try:
            pipeline = self._sync_pipeline_inputs()
        except Exception:
            pipeline = None
    if pipeline is None:
        return
    source = _analysis_source_text(self, pipeline)
    aligned = set(getattr(pipeline, "alignment_selected_channels", set()) or set())
    timing_ready = bool(getattr(pipeline, "timing_info", None))
    alignment_state = f"已完成（{len(aligned)} 通道）" if timing_ready and aligned else "尚未完成"

    alignment_var = getattr(self, "alignment_file_banner_var", None)
    if alignment_var is not None:
        event_csv = _text_variable_value(self, "game_csv_var") or "未选择"
        detection = _text_variable_value(self, "detection_result_var") or "未选择"
        log_file = _text_variable_value(self, "alignment_log_var") or "未选择"
        alignment_var.set(
            f"当前脑电：{source}\n"
            f"对齐输入：Event CSV={event_csv}  |  视频检测={detection}  |  log.txt={log_file}  |  状态={alignment_state}"
        )

    for kind, label in (("lfp", "LFP"), ("spike", "Spike")):
        banner_var = getattr(self, f"{kind}_file_banner_var", None)
        if banner_var is None:
            continue
        chosen = set(getattr(pipeline, f"{kind}_analysis_selected_channels", set()) or set())
        banner_var.set(
            f"当前脑电：{source}\n"
            f"时间对齐：{alignment_state}  |  {label} 独立选择：{len(chosen)} 通道"
        )


def _install_analysis_file_banner(self, parent, kind: str, title: str) -> None:
    variable_name = f"{kind}_file_banner_var"
    if getattr(self, variable_name, None) is not None:
        return
    # Move the page's existing content wrapper down by one grid row and keep
    # the file/data lineage banner permanently visible at the top.
    for child in list(parent.winfo_children()):
        try:
            if child.winfo_manager() == "grid":
                info = child.grid_info()
                child.grid_configure(row=int(info.get("row", 0)) + 1)
        except Exception:
            continue
    banner_var = _core.tk.StringVar(value=f"{title}｜当前文件：等待加载")
    setattr(self, variable_name, banner_var)
    banner = _core.tk.Frame(
        parent,
        bg="#FFF0F0",
        highlightbackground="#C62828",
        highlightthickness=2,
    )
    banner.grid(row=0, column=0, sticky="ew", padx=22, pady=(8, 2))
    _core.tk.Label(
        banner,
        text=f"{title} 数据来源",
        bg="#C62828",
        fg="white",
        font=("Segoe UI", 10, "bold"),
        padx=10,
        pady=5,
    ).pack(side="left", fill="y")
    _core.tk.Label(
        banner,
        textvariable=banner_var,
        bg="#FFF0F0",
        fg="#7F0000",
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify="left",
        padx=12,
        pady=5,
        wraplength=1450,
    ).pack(side="left", fill="x", expand=True)
    try:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=0)
        parent.grid_rowconfigure(1, weight=1)
    except Exception:
        pass
    self.after_idle(lambda owner=self: _update_analysis_file_banners(owner))


_original_mirror_pipeline_state_for_banners = _core.AnalysisGUI._mirror_pipeline_state


def _mirror_pipeline_state_with_file_banners(self, pipeline) -> None:
    _original_mirror_pipeline_state_for_banners(self, pipeline)
    _update_analysis_file_banners(self, pipeline)


_core.AnalysisGUI._mirror_pipeline_state = _mirror_pipeline_state_with_file_banners


_original_build_alignment_with_banner = _core.AnalysisGUI._build_alignment


def _find_descendant_button_by_text(parent, button_text: str):
    pending = list(parent.winfo_children())
    while pending:
        widget = pending.pop(0)
        try:
            if isinstance(widget, _core.ttk.Button) and str(widget.cget("text")) == button_text:
                return widget
            pending.extend(widget.winfo_children())
        except Exception:
            continue
    return None


def _after_alignment_custom_channel_import(self) -> None:
    """Refresh the alignment page after dispersed processed channels load."""
    _update_analysis_file_banners(self)
    pipeline = self._sync_pipeline_inputs()
    if pipeline is not None:
        imported = sorted(getattr(pipeline, "imported_channel_ids", set()) or set())
        if imported:
            self.status_var.set(
                f"已为时间对齐导入 {len(imported)} 个自定义通道："
                f"{', '.join(str(channel) for channel in imported[:12])}"
                + (" …" if len(imported) > 12 else "")
            )


def _install_alignment_custom_channel_import_button(self, parent) -> None:
    calculate_button = _find_descendant_button_by_text(parent, "计算并刷新对齐")
    if calculate_button is None:
        return
    existing = getattr(self, "_alignment_custom_channel_import_button", None)
    if existing is not None:
        try:
            if existing.winfo_exists():
                return
        except Exception:
            pass

    grid_info = calculate_button.grid_info()
    button_parent = calculate_button.nametowidget(calculate_button.winfo_parent())
    row = int(grid_info.get("row", 7))
    column = int(grid_info.get("column", 0))
    calculate_button.grid_configure(columnspan=1, padx=(20, 6))
    import_button = _core.ttk.Button(
        button_parent,
        text="导入自定义通道数据",
        style="Soft.TButton",
        command=lambda owner=self: _import_custom_channel_data(
            owner,
            on_done=lambda: _after_alignment_custom_channel_import(owner),
        ),
    )
    import_button.grid(
        row=row,
        column=column + 1,
        sticky="w",
        padx=(6, 20),
        pady=(14, 20),
    )
    self._alignment_custom_channel_import_button = import_button


def _after_analysis_custom_channel_import(self, analysis_kind: str) -> None:
    """Show the imported subset immediately on its target analysis page."""
    _update_analysis_file_banners(self)
    pipeline = self._sync_pipeline_inputs()
    if pipeline is None:
        return
    imported = sorted(getattr(pipeline, "imported_channel_ids", set()) or set())
    if imported:
        title = "LFP" if analysis_kind == "lfp" else "Spike"
        next_step = (
            "可直接选择通道并运行分析。"
            if getattr(pipeline, "timing_info", None)
            else "请先完成时间对齐后再选择通道并运行分析。"
        )
        self.status_var.set(
            f"已为 {title} 分析导入 {len(imported)} 个处理后通道；"
            + next_step
        )


def _install_analysis_custom_channel_import_button(
    self,
    parent,
    run_button_text: str,
    analysis_kind: str,
) -> None:
    """Add per-page partial-channel import next to the analysis run button."""
    attr_name = f"_{analysis_kind}_custom_channel_import_button"
    existing = getattr(self, attr_name, None)
    if existing is not None:
        try:
            if existing.winfo_exists():
                return
        except Exception:
            pass
    run_button = _find_descendant_button_by_text(parent, run_button_text)
    if run_button is None:
        return
    button_parent = run_button.nametowidget(run_button.winfo_parent())
    import_button = _core.ttk.Button(
        button_parent,
        text="导入自定义通道数据",
        style="Soft.TButton",
        command=lambda owner=self, kind=analysis_kind: _import_custom_channel_data(
            owner,
            preserve_alignment=True,
            on_done=lambda: _after_analysis_custom_channel_import(owner, kind),
        ),
    )
    pack_info = run_button.pack_info()
    import_button.pack(
        side=pack_info.get("side", "left"),
        padx=(8, 0),
        pady=pack_info.get("pady", 0),
        before=run_button,
    )
    setattr(self, attr_name, import_button)


def _run_lfp_parameter_csv_batch(self) -> None:
    """Expose the existing multi-record LFP/Spike batch runner on the LFP page."""
    pipeline = self._sync_pipeline_inputs()
    if pipeline is None:
        return
    runner = getattr(pipeline, "run_lfp_snr_param_csv", None)
    if not callable(runner):
        _core.messagebox.showerror("批量处理不可用", "当前运行环境不支持参数 CSV 批量处理。", parent=self)
        return
    runner()


def _install_analysis_export_buttons(self, parent, run_button_text: str, analysis_kind: str) -> None:
    """Expose result exports beside each analysis page's primary action."""
    attr_name = f"_{analysis_kind}_result_export_button"
    existing = getattr(self, attr_name, None)
    if existing is not None:
        try:
            if existing.winfo_exists():
                return
        except Exception:
            pass
    run_button = _find_descendant_button_by_text(parent, run_button_text)
    if run_button is None:
        return
    button_parent = run_button.nametowidget(run_button.winfo_parent())
    if analysis_kind == "lfp":
        export_button = _core.ttk.Button(
            button_parent,
            text="导出 LFP 结果",
            command=lambda owner=self: owner._export_lfp(),
        )
        export_button.pack(side="left", padx=(8, 0))
        _core.ttk.Button(
            button_parent,
            text="参数 CSV 批量处理",
            style="Soft.TButton",
            command=lambda owner=self: _run_lfp_parameter_csv_batch(owner),
        ).pack(side="left", padx=(8, 0))
    else:
        export_button = _core.ttk.Button(
            button_parent,
            text="导出 Spike 结果",
            command=lambda owner=self: owner._export_spike_bundle(),
        )
        export_button.pack(side="left", padx=(8, 0))
        _core.ttk.Button(
            button_parent,
            text="Spike 参数 CSV 批量处理",
            style="Soft.TButton",
            command=lambda owner=self: _run_lfp_parameter_csv_batch(owner),
        ).pack(side="left", padx=(8, 0))
    setattr(self, attr_name, export_button)


def _build_alignment_with_file_banner(self, parent) -> None:
    _original_build_alignment_with_banner(self, parent)
    _install_alignment_custom_channel_import_button(self, parent)
    _install_analysis_file_banner(self, parent, "alignment", "时间对齐")


_core.AnalysisGUI._build_alignment = _build_alignment_with_file_banner


_original_choose_alignment_file_with_banner = _core.AnalysisGUI._choose_alignment_file


def _choose_alignment_file_with_banner(self, *args, **kwargs):
    result = _original_choose_alignment_file_with_banner(self, *args, **kwargs)
    _update_analysis_file_banners(self)
    return result


_core.AnalysisGUI._choose_alignment_file = _choose_alignment_file_with_banner


_original_build_spike_with_banner = _core.AnalysisGUI._build_flash


def _build_spike_with_file_banner(self, parent) -> None:
    _original_build_spike_with_banner(self, parent)
    _install_analysis_custom_channel_import_button(
        self,
        parent,
        "运行 Spike 检测 / SNR",
        "spike",
    )
    _install_analysis_export_buttons(
        self,
        parent,
        "运行 Spike 检测 / SNR",
        "spike",
    )
    _install_analysis_file_banner(self, parent, "spike", "Spike 分析")


_core.AnalysisGUI._build_flash = _build_spike_with_file_banner


def _restore_lfp_snr_button(self) -> None:
    self._lfp_snr_running = False
    button = getattr(self, "_lfp_snr_run_button", None)
    if button is not None:
        try:
            button.configure(text="运行 LFP SNR", state="normal")
        except Exception:
            pass


def _toggle_lfp_snr_run(self) -> None:
    pipeline = self._sync_pipeline_inputs()
    if pipeline is None:
        return
    button = getattr(self, "_lfp_snr_run_button", None)
    if getattr(self, "_lfp_snr_running", False):
        pipeline._lfp_snr_cancel_requested = True
        if button is not None:
            button.configure(text="正在停止…", state="disabled")
        self.status_var.set("正在停止 LFP SNR；等待当前通道检查返回…")
        return

    _open_processed_channel_selector(self, analysis_kind="lfp")


def _start_lfp_snr_after_selection(self, pipeline) -> None:
    """Start LFP only after its independent aligned-channel selection."""
    button = getattr(self, "_lfp_snr_run_button", None)
    self._lfp_snr_running = True
    pipeline._lfp_snr_cancel_requested = False
    if button is not None:
        button.configure(text="停止 LFP SNR", state="normal")
    self.status_var.set("LFP SNR 正在运行；再次点击可停止。")
    pipeline.run_lfp_snr(on_done=lambda owner=self: _restore_lfp_snr_button(owner))


_previous_build_snr_with_stop = _core.AnalysisGUI._build_snr


def _build_snr_with_stop_button(self, parent) -> None:
    _previous_build_snr_with_stop(self, parent)
    if getattr(self, "_lfp_snr_run_button", None) is None:
        for child in _walk_widgets(parent):
            try:
                label = str(child.cget("text"))
                if "LFP SNR" not in label or ("运行" not in label and "Run" not in label):
                    continue
                child.configure(command=lambda owner=self: _toggle_lfp_snr_run(owner))
                self._lfp_snr_run_button = child
                break
            except Exception:
                continue
    _install_analysis_custom_channel_import_button(
        self,
        parent,
        "运行 LFP SNR",
        "lfp",
    )
    _install_analysis_export_buttons(
        self,
        parent,
        "运行 LFP SNR",
        "lfp",
    )
    _install_analysis_file_banner(self, parent, "lfp", "LFP 分析")


_core.AnalysisGUI._build_snr = _build_snr_with_stop_button


AnalysisGUI = _core.AnalysisGUI


if __name__ == "__main__":
    # Qt is the primary control framework.  Keep the prior Tk implementation
    # available explicitly while the project transitions existing workflows.
    if "--legacy-tk" in sys.argv:
        app = AnalysisGUI()
        app.mainloop()
    else:
        from qt_gui import main as qt_main
        raise SystemExit(qt_main())
