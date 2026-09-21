#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Signal SNR Analysis GUI (with SSVEP dedicated mode)
Supports: .mat, .txt, .csv, .bin files.
Analysis modes:
  - Resting-state   : frequency band power ratio
  - Stimulus-evoked : time-segmented SNR
  - SSVEP           : narrow-band SNR at stimulus frequency and harmonics
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal as scisig, stats
from scipy.interpolate import interp1d
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import scipy.io as sio
    MAT_SUPPORT = True
except ImportError:
    MAT_SUPPORT = False


# ==================== Signal Processing Functions ====================
def integrate_trapezoid(y, x):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def apply_notch_filter(sig, fs, freq=50.0, q=30, harmonics=1):
    filtered = sig.copy()
    for k in range(1, harmonics + 1):
        w0 = k * freq / (fs / 2)
        if w0 >= 1.0:
            continue
        b, a = scisig.iirnotch(w0, q)
        filtered = scisig.filtfilt(b, a, filtered)
    return filtered


def load_txt_data(file_path, stim_interval=None, stim_duration=None, first_onset=0.5):
    data = np.loadtxt(file_path)
    t = data[:, 0]
    sig = data[:, 1]
    if data.shape[1] >= 3:
        stim_mask = data[:, 2].astype(bool)
    elif stim_interval and stim_duration:
        stim_mask = np.zeros(len(t), dtype=bool)
        start = first_onset
        while start < t[-1]:
            stim_mask[(t >= start) & (t < start + stim_duration)] = True
            start += stim_interval
    else:
        stim_mask = np.zeros(len(t), dtype=bool)
    return t, sig, stim_mask


def load_bin_data(file_path, fs, dtype='float32', offset=0,
                  stim_interval=None, stim_duration=None, first_onset=0.5):
    sig = np.fromfile(file_path, dtype=dtype, offset=offset)
    n = len(sig)
    t = np.arange(n) / fs
    if stim_interval and stim_duration:
        stim_mask = np.zeros(n, dtype=bool)
        start = first_onset
        while start < t[-1]:
            stim_mask[(t >= start) & (t < start + stim_duration)] = True
            start += stim_interval
    else:
        stim_mask = np.zeros(n, dtype=bool)
    return t, sig, stim_mask


def load_mat_data(file_path, fs=None, var_signal='signal', var_time='time',
                  var_fs='FS', var_stim='stim_mask',
                  stim_interval=None, stim_duration=None, first_onset=0.5):
    if not MAT_SUPPORT:
        raise ImportError("scipy not installed")
    mat = sio.loadmat(file_path)
    sig = mat[var_signal].flatten()
    n = len(sig)
    if var_time in mat:
        t = mat[var_time].flatten()
        if len(t) != n:
            raise ValueError("Time length mismatch")
        if len(t) > 1:
            fs = 1 / (t[1] - t[0])
    else:
        if var_fs in mat:
            fs = float(mat[var_fs].flatten()[0])
        if fs is None:
            raise ValueError("Sampling rate unknown")
        t = np.arange(n) / fs
    if var_stim in mat:
        stim_mask = mat[var_stim].flatten().astype(bool)
    elif stim_interval and stim_duration:
        stim_mask = np.zeros(n, dtype=bool)
        start = first_onset
        while start < t[-1]:
            stim_mask[(t >= start) & (t < start + stim_duration)] = True
            start += stim_interval
    else:
        stim_mask = np.zeros(n, dtype=bool)
    return t, sig, stim_mask


def compute_snr_segmented(sig, stim_mask, method='power'):
    mask = np.asarray(stim_mask, bool)
    noise = ~mask
    if method == 'power':
        p_stim = np.mean(sig[mask] ** 2)
        p_noise = np.mean(sig[noise] ** 2)
        return np.inf if p_noise == 0 else 10 * np.log10(p_stim / p_noise)
    elif method == 'rms':
        rms_stim = np.sqrt(np.mean(sig[mask] ** 2))
        rms_noise = np.sqrt(np.mean(sig[noise] ** 2))
        return np.inf if rms_noise == 0 else 20 * np.log10(rms_stim / rms_noise)


def compute_trial_snrs(sig, onsets, stim_dur, fs, pre_stim=0.2):
    snrs = []
    for onset in onsets:
        istart = int(onset * fs)
        iend = int((onset + stim_dur) * fs)
        if iend > len(sig):
            continue
        stim_seg = sig[istart:iend]
        nstart = int((onset - pre_stim) * fs) if pre_stim > 0 else istart - len(stim_seg)
        if nstart < 0:
            continue
        noise_seg = sig[nstart:istart]
        if len(noise_seg) == 0:
            continue
        ps = np.mean(stim_seg ** 2)
        pn = np.mean(noise_seg ** 2)
        snrs.append(np.inf if pn == 0 else 10 * np.log10(ps / pn))
    return np.array(snrs)


def compute_resting_snr(sig, fs, signal_band, noise_band):
    freqs, psd = scisig.welch(sig, fs, nperseg=1024)
    mask_s = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    mask_n = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
    ps = integrate_trapezoid(psd[mask_s], freqs[mask_s])
    pn = integrate_trapezoid(psd[mask_n], freqs[mask_n]) - ps
    if pn <= 0:
        return np.inf
    return 10 * np.log10(ps / pn)


RESTING_MULTIBAND_DEFINITIONS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 80.0),
    ("high_gamma", 80.0, 200.0),
)


def _integrate_masked_psd(freqs, psd, low, high, excluded=None):
    """Integrate disjoint PSD segments without bridging excluded line gaps."""
    mask = (freqs >= float(low)) & (freqs <= float(high))
    if excluded is not None:
        mask &= ~np.asarray(excluded, dtype=bool)
    indices = np.flatnonzero(mask)
    if indices.size < 2:
        return np.nan, 0.0
    split_at = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, split_at)
    power = 0.0
    bandwidth = 0.0
    for group in groups:
        if group.size < 2:
            continue
        power += float(integrate_trapezoid(psd[group], freqs[group]))
        bandwidth += float(freqs[group[-1]] - freqs[group[0]])
    return (power if bandwidth > 0 else np.nan), bandwidth


def _line_contamination_from_psd(
    freqs, psd, fs, *, line_frequency=50.0, harmonics=3,
    max_frequency=200.0, line_half_width=1.0,
    reference_low=2.0, reference_high=4.0,
):
    """Return per-harmonic residual line metrics from one Welch spectrum."""
    total_power, _ = _integrate_masked_psd(freqs, psd, 1.0, max_frequency)
    records = {}
    line_powers = []
    nyquist = float(fs) / 2.0
    for harmonic in range(1, max(1, int(harmonics)) + 1):
        center = float(line_frequency) * harmonic
        key = f"line_{int(round(center))}_hz"
        if center - line_half_width < 1.0 or center + line_half_width > min(max_frequency, nyquist):
            records[key] = {"power": np.nan, "peak_db": np.nan, "fraction": np.nan, "available": False}
            continue
        line_power, line_bandwidth = _integrate_masked_psd(
            freqs, psd, center - line_half_width, center + line_half_width
        )
        left, left_width = _integrate_masked_psd(
            freqs, psd, center - reference_high, center - reference_low
        )
        right, right_width = _integrate_masked_psd(
            freqs, psd, center + reference_low, center + reference_high
        )
        reference_power = np.nansum([left, right])
        reference_width = left_width + right_width
        line_density = line_power / line_bandwidth if line_bandwidth > 0 else np.nan
        reference_density = reference_power / reference_width if reference_width > 0 else np.nan
        peak_db = 10.0 * np.log10(line_density / reference_density) if (
            np.isfinite(line_density) and np.isfinite(reference_density) and reference_density > 0 and line_density > 0
        ) else np.nan
        fraction = line_power / total_power if np.isfinite(line_power) and np.isfinite(total_power) and total_power > 0 else np.nan
        records[key] = {"power": float(line_power), "peak_db": float(peak_db), "fraction": float(fraction), "available": True}
        if np.isfinite(line_power):
            line_powers.append(line_power)
    records["line_max_db"] = float(np.nanmax([item["peak_db"] for item in records.values() if isinstance(item, dict) and np.isfinite(item["peak_db"])])) if any(
        isinstance(item, dict) and np.isfinite(item["peak_db"]) for item in records.values()
    ) else np.nan
    records["line_power_fraction"] = float(np.nansum(line_powers) / total_power) if line_powers and np.isfinite(total_power) and total_power > 0 else np.nan
    return records


def compute_resting_multiband_snr(
    sig, fs, *, target_frequency_resolution=0.5,
    line_frequency=50.0, line_harmonics=3, line_guard_hz=1.0,
    max_frequency=200.0,
):
    """Compute resting-state band SNRs with explicit line-frequency masks.

    The ratio remains ``P_band / (P_1_200 - P_band)``.  Both powers use the
    same frequency mask, excluding mains harmonics and their guard bands.
    This keeps the requested definition while preventing line peaks or gaps
    from being counted as neural band power.
    """
    values = np.asarray(sig, dtype=float).reshape(-1)
    if values.size < 8 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("静息态多频段 SNR 需要足够的有限采样点和有效采样率。")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    nyquist = float(fs) / 2.0
    max_frequency = float(max_frequency)
    if max_frequency <= 1.0:
        raise ValueError("静息态 SNR 总频段上限必须大于 1 Hz。")
    nperseg = min(values.size, max(8, int(np.ceil(float(fs) / max(float(target_frequency_resolution), 0.01)))))
    noverlap = min(nperseg - 1, nperseg // 2)
    freqs, psd = scisig.welch(values, fs=float(fs), nperseg=nperseg, noverlap=noverlap, window="hann")
    frequency_resolution = float(freqs[1] - freqs[0]) if freqs.size > 1 else np.nan
    effective_max = min(max_frequency, nyquist)
    line_half_width = max(float(line_guard_hz), 2.0 * frequency_resolution)
    excluded = np.zeros(freqs.shape, dtype=bool)
    for harmonic in range(1, max(1, int(line_harmonics)) + 1):
        center = float(line_frequency) * harmonic
        excluded |= np.abs(freqs - center) <= line_half_width
    total_power, total_bandwidth = _integrate_masked_psd(freqs, psd, 1.0, effective_max, excluded)
    bands = {}
    for name, low, high in RESTING_MULTIBAND_DEFINITIONS:
        available = high <= effective_max + frequency_resolution * 0.25
        band_power, bandwidth = _integrate_masked_psd(freqs, psd, low, high, excluded) if available else (np.nan, 0.0)
        noise_power = total_power - band_power if np.isfinite(total_power) and np.isfinite(band_power) else np.nan
        snr_db = 10.0 * np.log10(band_power / noise_power) if (
            np.isfinite(band_power) and np.isfinite(noise_power) and band_power > 0 and noise_power > 0
        ) else np.nan
        bands[name] = {
            "low_hz": low, "high_hz": high, "band_power": float(band_power),
            "noise_power": float(noise_power), "snr_db": float(snr_db),
            "effective_bandwidth_hz": float(bandwidth), "available": bool(available),
        }
    return {
        "bands": bands,
        "total_power_1_200": float(total_power),
        "effective_total_bandwidth_hz": float(total_bandwidth),
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "frequency_resolution_hz": frequency_resolution,
        "line_guard_hz": line_half_width,
        "line_metrics": _line_contamination_from_psd(
            freqs, psd, fs, line_frequency=line_frequency,
            harmonics=line_harmonics, max_frequency=max_frequency,
            line_half_width=line_half_width,
        ),
    }


def compute_ssvep_snr(sig, fs, stim_freq, harmonics=3, n_neighbor=4,
                       fft_length_sec=2.0, window='hann'):
    nfft = int(fft_length_sec * fs)
    if len(sig) < nfft:
        nfft = len(sig)
    freqs, psd = scisig.welch(sig, fs, nperseg=nfft, noverlap=nfft // 2, window=window)
    freq_res = freqs[1] - freqs[0]

    snrs = {}
    for k in range(1, harmonics + 1):
        target = k * stim_freq
        idx = np.argmin(np.abs(freqs - target))
        p_sig = psd[idx]
        left = max(0, idx - n_neighbor)
        right = min(len(freqs), idx + n_neighbor + 1)
        noise_idxs = np.r_[left:idx, idx+1:right]
        if len(noise_idxs) == 0:
            snrs[f'{target:.1f} Hz'] = np.inf
            continue
        p_noise = np.mean(psd[noise_idxs])
        snr_db = 10 * np.log10(p_sig / p_noise) if p_noise > 0 else np.inf
        snrs[f'{target:.1f} Hz'] = snr_db
    valid_values = [v for v in snrs.values() if v != np.inf]
    overall = np.mean(valid_values) if valid_values else np.inf
    return snrs, overall


def compute_batch_channel_snr(data, fs, mode='resting',
                              signal_band=(30, 80), noise_band=(1, 200),
                              stim_interval=None, stim_duration=None, first_onset=0.5,
                              stim_freq=10.0, harmonics=3, n_neighbor=4,
                              fft_length_sec=2.0, notch=False, stim_mask=None,
                              notch_freq=50.0, notch_q=30.0, notch_harmonics=1):
    """Compute one SNR score per channel for the integrated pipeline GUI.

    Parameters
    ----------
    data : ndarray
        Neural data as samples x channels.
    fs : float
        Sampling rate.
    mode : {'resting', 'evoked', 'ssvep'}
        SNR definition to use.

    Returns
    -------
    list[dict]
        Each item contains channel, snr_db, and optional detail fields.
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[0] < data.shape[1] and data.shape[0] <= 520:
        data = data.T

    n_samples, n_channels = data.shape
    t = np.arange(n_samples) / fs
    if stim_mask is not None:
        stim_mask = np.asarray(stim_mask, dtype=bool)
        if stim_mask.shape[0] != n_samples:
            raise ValueError("stim_mask length must match the number of samples.")
    if mode == 'evoked':
        if stim_mask is None:
            if stim_interval is None or stim_duration is None:
                raise ValueError("Evoked batch SNR requires stim_mask or stim_interval/stim_duration.")
            stim_mask = np.zeros(n_samples, dtype=bool)
            start = first_onset
            while start < t[-1]:
                stim_mask[(t >= start) & (t < start + stim_duration)] = True
                start += stim_interval
        if not stim_mask.any() or stim_mask.all():
            raise ValueError("Generated stimulus mask is empty or covers all samples.")

    rows = []
    for ch in range(n_channels):
        sig = np.asarray(data[:, ch], dtype=float)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        if notch:
            sig = apply_notch_filter(sig, fs, notch_freq, notch_q, notch_harmonics)

        if mode == 'resting':
            snr_db = compute_resting_snr(sig, fs, signal_band, noise_band)
            detail = f"{signal_band[0]}-{signal_band[1]}Hz / {noise_band[0]}-{noise_band[1]}Hz"
        elif mode == 'evoked':
            snr_db = compute_snr_segmented(sig, stim_mask, method='power')
            detail = "stimMarkers mask" if stim_interval is None else f"stim {stim_duration}s every {stim_interval}s"
        elif mode == 'ssvep':
            snrs, snr_db = compute_ssvep_snr(
                sig, fs, stim_freq, harmonics=harmonics,
                n_neighbor=n_neighbor, fft_length_sec=fft_length_sec)
            detail = ", ".join(f"{k}:{v:.2f}" for k, v in snrs.items())
        else:
            raise ValueError("mode must be 'resting', 'evoked', or 'ssvep'")

        rows.append({
            'channel': ch + 1,
            'snr_db': float(snr_db),
            'mode': mode,
            'detail': detail,
        })
    return rows


# ==================== GUI Application ====================
class SSVEPAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        root.title("SSVEP & Neural SNR Analyzer")
        root.geometry("1300x850")

        self.t_ch1 = self.t_ch2 = None
        self.sig_ch1 = self.sig_ch2 = None
        self.mask_ch1 = self.mask_ch2 = None
        self.fs = None

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        self.control_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.control_frame, text='Controls')
        self._build_control_panel()

        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text='Results')
        self.results_text = tk.Text(self.results_frame, wrap='word', height=12)
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)

        self.plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_frame, text='Plots')
        self.canvas_frame = ttk.Frame(self.plots_frame)
        self.canvas_frame.pack(fill='both', expand=True)

        self.status_var = tk.StringVar()
        status = tk.Label(root, textvariable=self.status_var, bd=1,
                          relief=tk.SUNKEN, anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()

    # ---------- 控件构建 ----------
    def _build_control_panel(self):
        canvas = tk.Canvas(self.control_frame)
        scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # File Selection
        file_frame = ttk.LabelFrame(scroll_frame, text="Data Files")
        file_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(file_frame, text="Channel 1:").grid(row=0, column=0, sticky='w')
        self.file1_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file1_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self._browse_file(1)).grid(row=0, column=2)
        ttk.Button(file_frame, text="Detect Vars", command=lambda: self._detect_mat_vars(1)).grid(row=0, column=3)

        ttk.Label(file_frame, text="Channel 2:").grid(row=1, column=0, sticky='w')
        self.file2_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file2_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self._browse_file(2)).grid(row=1, column=2)
        ttk.Button(file_frame, text="Detect Vars", command=lambda: self._detect_mat_vars(2)).grid(row=1, column=3)

        ttk.Label(file_frame, text="Sampling Rate (Hz, if not in file):").grid(row=2, column=0, sticky='w')
        self.fs_var = tk.StringVar(value="6490")
        ttk.Entry(file_frame, textvariable=self.fs_var, width=10).grid(row=2, column=1, sticky='w', padx=5)

        # .mat variable names
        mat_frame = ttk.LabelFrame(scroll_frame, text="MAT File Variables (overrides)")
        mat_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(mat_frame, text="Signal:").grid(row=0, column=0, sticky='w')
        self.var_signal = tk.StringVar(value="signal")
        ttk.Entry(mat_frame, textvariable=self.var_signal, width=12).grid(row=0, column=1, padx=2)
        ttk.Label(mat_frame, text="Time:").grid(row=0, column=2, sticky='w')
        self.var_time = tk.StringVar(value="time")
        ttk.Entry(mat_frame, textvariable=self.var_time, width=12).grid(row=0, column=3, padx=2)
        ttk.Label(mat_frame, text="FS:").grid(row=0, column=4, sticky='w')
        self.var_fs = tk.StringVar(value="FS")
        ttk.Entry(mat_frame, textvariable=self.var_fs, width=12).grid(row=0, column=5, padx=2)
        ttk.Label(mat_frame, text="Stim Mask:").grid(row=1, column=0, sticky='w')
        self.var_stim = tk.StringVar(value="stim_mask")
        ttk.Entry(mat_frame, textvariable=self.var_stim, width=12).grid(row=1, column=1, padx=2)

        # Analysis Mode
        mode_frame = ttk.LabelFrame(scroll_frame, text="Analysis Mode")
        mode_frame.pack(fill='x', padx=5, pady=5)
        self.analysis_mode = tk.StringVar(value="ssvep")
        ttk.Radiobutton(mode_frame, text="SSVEP (narrow-band SNR)", variable=self.analysis_mode,
                        value="ssvep", command=self._update_mode).grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(mode_frame, text="Resting-state (frequency band SNR)", variable=self.analysis_mode,
                        value="resting", command=self._update_mode).grid(row=1, column=0, sticky='w')
        ttk.Radiobutton(mode_frame, text="Stimulus-evoked (segment SNR)", variable=self.analysis_mode,
                        value="evoked", command=self._update_mode).grid(row=2, column=0, sticky='w')

        # Resting bands
        self.band_frame = ttk.LabelFrame(scroll_frame, text="Frequency Bands (Hz) - Resting")
        self.band_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(self.band_frame, text="Signal low:").grid(row=0, column=0, sticky='w')
        self.sig_low_var = tk.StringVar(value="30")
        self.sig_low_entry = ttk.Entry(self.band_frame, textvariable=self.sig_low_var, width=8)
        self.sig_low_entry.grid(row=0, column=1, padx=2)
        ttk.Label(self.band_frame, text="high:").grid(row=0, column=2, sticky='w')
        self.sig_high_var = tk.StringVar(value="80")
        self.sig_high_entry = ttk.Entry(self.band_frame, textvariable=self.sig_high_var, width=8)
        self.sig_high_entry.grid(row=0, column=3, padx=2)
        ttk.Label(self.band_frame, text="Noise low:").grid(row=1, column=0, sticky='w')
        self.noise_low_var = tk.StringVar(value="1")
        self.noise_low_entry = ttk.Entry(self.band_frame, textvariable=self.noise_low_var, width=8)
        self.noise_low_entry.grid(row=1, column=1, padx=2)
        ttk.Label(self.band_frame, text="high:").grid(row=1, column=2, sticky='w')
        self.noise_high_var = tk.StringVar(value="200")
        self.noise_high_entry = ttk.Entry(self.band_frame, textvariable=self.noise_high_var, width=8)
        self.noise_high_entry.grid(row=1, column=3, padx=2)

        # SSVEP parameters
        self.ssvep_frame = ttk.LabelFrame(scroll_frame, text="SSVEP Parameters")
        self.ssvep_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(self.ssvep_frame, text="Stim. Freq (Hz):").grid(row=0, column=0, sticky='w')
        self.stim_freq_var = tk.StringVar(value="15")
        self.stim_freq_entry = ttk.Entry(self.ssvep_frame, textvariable=self.stim_freq_var, width=8)
        self.stim_freq_entry.grid(row=0, column=1, padx=2)
        ttk.Label(self.ssvep_frame, text="Harmonics:").grid(row=0, column=2, sticky='w')
        self.harmonics_var = tk.StringVar(value="3")
        self.harmonics_entry = ttk.Entry(self.ssvep_frame, textvariable=self.harmonics_var, width=5)
        self.harmonics_entry.grid(row=0, column=3, padx=2)
        ttk.Label(self.ssvep_frame, text="Neighbor bins:").grid(row=0, column=4, sticky='w')
        self.n_neighbor_var = tk.StringVar(value="4")
        self.n_neighbor_entry = ttk.Entry(self.ssvep_frame, textvariable=self.n_neighbor_var, width=5)
        self.n_neighbor_entry.grid(row=0, column=5, padx=2)
        ttk.Label(self.ssvep_frame, text="FFT length (s):").grid(row=1, column=0, sticky='w')
        self.fft_len_var = tk.StringVar(value="2.0")
        self.fft_len_entry = ttk.Entry(self.ssvep_frame, textvariable=self.fft_len_var, width=8)
        self.fft_len_entry.grid(row=1, column=1, padx=2)

        # Stimulus parameters (evoked)
        self.stim_frame = ttk.LabelFrame(scroll_frame, text="Stimulus Parameters (for evoked)")
        self.stim_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(self.stim_frame, text="Interval (s):").grid(row=0, column=0, sticky='w')
        self.stim_interval_var = tk.StringVar(value="2.0")
        self.stim_interval_entry = ttk.Entry(self.stim_frame, textvariable=self.stim_interval_var, width=8)
        self.stim_interval_entry.grid(row=0, column=1, padx=2)
        ttk.Label(self.stim_frame, text="Duration (s):").grid(row=0, column=2, sticky='w')
        self.stim_duration_var = tk.StringVar(value="0.5")
        self.stim_duration_entry = ttk.Entry(self.stim_frame, textvariable=self.stim_duration_var, width=8)
        self.stim_duration_entry.grid(row=0, column=3, padx=2)
        ttk.Label(self.stim_frame, text="First onset (s):").grid(row=1, column=0, sticky='w')
        self.first_onset_var = tk.StringVar(value="0.5")
        self.first_onset_entry = ttk.Entry(self.stim_frame, textvariable=self.first_onset_var, width=8)
        self.first_onset_entry.grid(row=1, column=1, padx=2)

        # Notch filter
        notch_frame = ttk.LabelFrame(scroll_frame, text="Notch Filter")
        notch_frame.pack(fill='x', padx=5, pady=5)
        self.use_notch = tk.BooleanVar(value=False)
        ttk.Checkbutton(notch_frame, text="Enable", variable=self.use_notch,
                        command=self._update_notch).grid(row=0, column=0, sticky='w')
        ttk.Label(notch_frame, text="Freq (Hz):").grid(row=0, column=1, sticky='w')
        self.notch_freq_var = tk.StringVar(value="50")
        self.notch_freq_entry = ttk.Entry(notch_frame, textvariable=self.notch_freq_var, width=8)
        self.notch_freq_entry.grid(row=0, column=2, padx=2)
        ttk.Label(notch_frame, text="Q:").grid(row=0, column=3, sticky='w')
        self.notch_q_var = tk.StringVar(value="30")
        self.notch_q_entry = ttk.Entry(notch_frame, textvariable=self.notch_q_var, width=8)
        self.notch_q_entry.grid(row=0, column=4, padx=2)
        ttk.Label(notch_frame, text="Harmonics:").grid(row=0, column=5, sticky='w')
        self.notch_harm_var = tk.StringVar(value="1")
        self.notch_harm_entry = ttk.Entry(notch_frame, textvariable=self.notch_harm_var, width=4)
        self.notch_harm_entry.grid(row=0, column=6, padx=2)

        ttk.Button(scroll_frame, text="Run Analysis", command=self.run_analysis).pack(pady=10)

        self._update_mode()
        self._update_notch()

    def _browse_file(self, num):
        fname = filedialog.askopenfilename(
            filetypes=[("All supported", "*.mat;*.txt;*.csv;*.bin"),
                       ("MAT files", "*.mat"),
                       ("Text files", "*.txt;*.csv"),
                       ("Binary files", "*.bin")])
        if fname:
            if num == 1:
                self.file1_var.set(fname)
            else:
                self.file2_var.set(fname)

    def _detect_mat_vars(self, num):
        fname = self.file1_var.get() if num == 1 else self.file2_var.get()
        if not fname or not fname.endswith('.mat'):
            messagebox.showwarning("Warning", "Please select a .mat file first.")
            return
        if not MAT_SUPPORT:
            messagebox.showerror("Error", "scipy.io is required to read .mat files.")
            return
        try:
            mat = sio.loadmat(fname)
            vars_list = [k for k in mat.keys() if not k.startswith('__')]
            msg = "Variables found:\n" + "\n".join(vars_list)
            messagebox.showinfo("MAT File Variables", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _update_mode(self):
        mode = self.analysis_mode.get()
        rb_state = 'normal' if mode == 'resting' else 'disabled'
        for entry in [self.sig_low_entry, self.sig_high_entry, self.noise_low_entry, self.noise_high_entry]:
            entry.configure(state=rb_state)
        ss_state = 'normal' if mode == 'ssvep' else 'disabled'
        for entry in [self.stim_freq_entry, self.harmonics_entry, self.n_neighbor_entry, self.fft_len_entry]:
            entry.configure(state=ss_state)
        ev_state = 'normal' if mode == 'evoked' else 'disabled'
        for entry in [self.stim_interval_entry, self.stim_duration_entry, self.first_onset_entry]:
            entry.configure(state=ev_state)

    def _update_notch(self):
        state = 'normal' if self.use_notch.get() else 'disabled'
        for entry in [self.notch_freq_entry, self.notch_q_entry, self.notch_harm_entry]:
            entry.configure(state=state)

    def _load_data(self, file_path, var_override, stim_interval, stim_duration, first_onset):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.txt', '.csv']:
            return load_txt_data(file_path, stim_interval=stim_interval,
                                 stim_duration=stim_duration, first_onset=first_onset)
        elif ext == '.bin':
            dtype = 'float32'
            offset = 0
            return load_bin_data(file_path, fs=self.fs_int, dtype=dtype, offset=offset,
                                 stim_interval=stim_interval, stim_duration=stim_duration,
                                 first_onset=first_onset)
        elif ext == '.mat':
            return load_mat_data(file_path, fs=self.fs_int,
                                 var_signal=var_override.get('signal', 'signal'),
                                 var_time=var_override.get('time', 'time'),
                                 var_fs=var_override.get('fs', 'FS'),
                                 var_stim=var_override.get('stim', 'stim_mask'),
                                 stim_interval=stim_interval,
                                 stim_duration=stim_duration,
                                 first_onset=first_onset)
        else:
            raise ValueError("Unsupported file type")

    def run_analysis(self):
        file1 = self.file1_var.get().strip()
        file2 = self.file2_var.get().strip()
        if not file1 or not file2:
            messagebox.showerror("Error", "Please select both channel files.")
            return

        fs_str = self.fs_var.get().strip()
        self.fs_int = int(fs_str) if fs_str else None
        mode = self.analysis_mode.get()

        signal_band = noise_band = None
        stim_interval = stim_duration = first_onset = None
        stim_freq = harmonics = n_neighbor = fft_len = None

        if mode == 'resting':
            try:
                signal_band = (float(self.sig_low_var.get()), float(self.sig_high_var.get()))
                noise_band = (float(self.noise_low_var.get()), float(self.noise_high_var.get()))
            except:
                messagebox.showerror("Error", "Invalid frequency band values.")
                return
        elif mode == 'evoked':
            try:
                stim_interval = float(self.stim_interval_var.get())
                stim_duration = float(self.stim_duration_var.get())
                first_onset = float(self.first_onset_var.get())
            except:
                messagebox.showerror("Error", "Invalid stimulus timing parameters.")
                return
        elif mode == 'ssvep':
            try:
                stim_freq = float(self.stim_freq_var.get())
                harmonics = int(self.harmonics_var.get())
                n_neighbor = int(self.n_neighbor_var.get())
                fft_len = float(self.fft_len_var.get())
            except:
                messagebox.showerror("Error", "Invalid SSVEP parameters.")
                return

        use_notch = self.use_notch.get()
        if use_notch:
            try:
                notch_freq = float(self.notch_freq_var.get())
                notch_q = float(self.notch_q_var.get())
                notch_harm = int(self.notch_harm_var.get())
            except:
                messagebox.showerror("Error", "Invalid notch filter parameters.")
                return

        var_override = {
            'signal': self.var_signal.get(),
            'time': self.var_time.get(),
            'fs': self.var_fs.get(),
            'stim': self.var_stim.get()
        }

        try:
            self.status_var.set("Loading data...")
            self.root.update()
            self.t_ch1, self.sig_ch1, self.mask_ch1 = self._load_data(
                file1, var_override, stim_interval, stim_duration, first_onset)
            self.t_ch2, self.sig_ch2, self.mask_ch2 = self._load_data(
                file2, var_override, stim_interval, stim_duration, first_onset)

            if not np.array_equal(self.t_ch1, self.t_ch2):
                f_sig = interp1d(self.t_ch2, self.sig_ch2, kind='linear', fill_value='extrapolate')
                self.sig_ch2 = f_sig(self.t_ch1)
                f_mask = interp1d(self.t_ch2, self.mask_ch2.astype(float), kind='nearest', fill_value='extrapolate')
                self.mask_ch2 = f_mask(self.t_ch1) > 0.5
                self.t_ch2 = self.t_ch1
            t = self.t_ch1
            stim_mask = self.mask_ch1 & self.mask_ch2

            if len(t) > 1:
                self.fs = 1 / (t[1] - t[0])
            elif self.fs_int:
                self.fs = self.fs_int
            else:
                raise ValueError("Cannot determine sampling rate.")

            if use_notch:
                self.status_var.set("Applying notch filter...")
                self.root.update()
                self.sig_ch1 = apply_notch_filter(self.sig_ch1, self.fs, notch_freq, notch_q, notch_harm)
                self.sig_ch2 = apply_notch_filter(self.sig_ch2, self.fs, notch_freq, notch_q, notch_harm)

            self.status_var.set("Computing SNR...")
            self.root.update()
            results_str = ""

            if mode == 'resting':
                snr1 = compute_resting_snr(self.sig_ch1, self.fs, signal_band, noise_band)
                snr2 = compute_resting_snr(self.sig_ch2, self.fs, signal_band, noise_band)
                results_str += f"Resting SNR ({signal_band[0]}-{signal_band[1]} Hz / {noise_band[0]}-{noise_band[1]} Hz)\n"
                results_str += f"Channel 1: {snr1:.2f} dB\nChannel 2: {snr2:.2f} dB\n"
                self._plot_resting(t)

            elif mode == 'evoked':
                onsets = np.where(np.diff(stim_mask.astype(int)) == 1)[0] / self.fs
                if stim_mask[0]:
                    onsets = np.insert(onsets, 0, 0.0)
                if len(onsets) == 0:
                    messagebox.showerror("Error", "No stimulus onsets found.")
                    return
                snr1_global = compute_snr_segmented(self.sig_ch1, stim_mask, method='power')
                snr2_global = compute_snr_segmented(self.sig_ch2, stim_mask, method='power')
                snrs1_trial = compute_trial_snrs(self.sig_ch1, onsets, stim_duration, self.fs)
                snrs2_trial = compute_trial_snrs(self.sig_ch2, onsets, stim_duration, self.fs)
                results_str += "Evoked SNR (power ratio)\n"
                results_str += f"Channel 1 global: {snr1_global:.2f} dB\nChannel 2 global: {snr2_global:.2f} dB\n"
                results_str += f"Channel 1 trial mean ± std: {np.mean(snrs1_trial):.2f} ± {np.std(snrs1_trial):.2f} dB\n"
                results_str += f"Channel 2 trial mean ± std: {np.mean(snrs2_trial):.2f} ± {np.std(snrs2_trial):.2f} dB\n"
                self._plot_evoked(t, stim_mask, snrs1_trial, snrs2_trial)

            elif mode == 'ssvep':
                snrs1, overall1 = compute_ssvep_snr(self.sig_ch1, self.fs, stim_freq,
                                                    harmonics=harmonics, n_neighbor=n_neighbor, fft_length_sec=fft_len)
                snrs2, overall2 = compute_ssvep_snr(self.sig_ch2, self.fs, stim_freq,
                                                    harmonics=harmonics, n_neighbor=n_neighbor, fft_length_sec=fft_len)
                results_str += f"SSVEP SNR @ {stim_freq} Hz (harmonics={harmonics}, neighbor bins={n_neighbor})\n"
                results_str += "Channel 1:\n"
                for f, s in snrs1.items():
                    results_str += f"  {f}: {s:.2f} dB\n"
                results_str += f"  Mean: {overall1:.2f} dB\n"
                results_str += "Channel 2:\n"
                for f, s in snrs2.items():
                    results_str += f"  {f}: {s:.2f} dB\n"
                results_str += f"  Mean: {overall2:.2f} dB\n"
                self._plot_ssvep(snrs1, snrs2, stim_freq, t)

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_str)
            self.status_var.set("Analysis complete.")
            self.notebook.select(self.results_frame)

        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            self.status_var.set("Error occurred.")

    def _clear_plot(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

    def _embed_figure(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.notebook.select(self.plots_frame)

    def _plot_resting(self, t):
        self._clear_plot()
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        limit = min(2000, len(t))
        axs[0].plot(t[:limit], self.sig_ch1[:limit], label='Ch 1', alpha=0.7)
        axs[0].plot(t[:limit], self.sig_ch2[:limit], label='Ch 2', alpha=0.7)
        axs[0].set_title("Signal Comparison")
        axs[0].legend()
        f1, P1 = scisig.welch(self.sig_ch1, self.fs, nperseg=1024)
        f2, P2 = scisig.welch(self.sig_ch2, self.fs, nperseg=1024)
        axs[1].semilogy(f1, P1, label='Ch 1')
        axs[1].semilogy(f2, P2, label='Ch 2')
        axs[1].set_title("PSD")
        axs[1].legend()
        plt.tight_layout()
        self._embed_figure(fig)

    def _plot_evoked(self, t, mask, snrs1, snrs2):
        self._clear_plot()
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        show_len = min(2000, len(t))
        ax = axs[0, 0]
        ax.plot(t[:show_len], self.sig_ch1[:show_len], label='Ch 1')
        ax.plot(t[:show_len], self.sig_ch2[:show_len], label='Ch 2')
        for i in range(show_len - 1):
            if mask[i]:
                ax.axvspan(t[i], t[i+1], color='yellow', alpha=0.3, lw=0)
        ax.set_title("Signal with stimulus shading")
        ax.legend()
        axs[0, 1].boxplot([snrs1, snrs2], labels=['Ch 1', 'Ch 2'])
        axs[0, 1].set_title("Trial SNR Comparison")
        if np.sum(mask) > 256:
            f1, P1 = scisig.welch(self.sig_ch1[mask], self.fs, nperseg=256)
            f2, P2 = scisig.welch(self.sig_ch2[mask], self.fs, nperseg=256)
            axs[1, 0].semilogy(f1, P1, label='Ch 1')
            axs[1, 0].semilogy(f2, P2, label='Ch 2')
            axs[1, 0].set_title("PSD during stimulus")
            axs[1, 0].legend()
        if np.sum(~mask) > 256:
            f1, P1 = scisig.welch(self.sig_ch1[~mask], self.fs, nperseg=256)
            f2, P2 = scisig.welch(self.sig_ch2[~mask], self.fs, nperseg=256)
            axs[1, 1].semilogy(f1, P1, label='Ch 1')
            axs[1, 1].semilogy(f2, P2, label='Ch 2')
            axs[1, 1].set_title("PSD during silence")
            axs[1, 1].legend()
        plt.tight_layout()
        self._embed_figure(fig)

    def _plot_ssvep(self, snrs1, snrs2, stim_freq, t):
        self._clear_plot()
        freqs = list(snrs1.keys())
        ch1_vals = [snrs1[f] for f in freqs]
        ch2_vals = [snrs2[f] for f in freqs]
        x = np.arange(len(freqs))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width/2, ch1_vals, width, label='Channel 1', color='#1f77b4')
        bars2 = ax.bar(x + width/2, ch2_vals, width, label='Channel 2', color='#ff7f0e')
        ax.set_ylabel('SNR (dB)')
        ax.set_title(f'SSVEP Harmonic SNRs (fundamental = {stim_freq} Hz)')
        ax.set_xticks(x)
        ax.set_xticklabels(freqs, rotation=45)
        ax.legend()
        for bar in bars1 + bars2:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        self._embed_figure(fig)


if __name__ == "__main__":
    root = tk.Tk()
    app = SSVEPAnalyzerGUI(root)
    root.mainloop()
