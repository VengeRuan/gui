"""Disposable, centred localization curves. Never modify analysis waveforms."""
import numpy as np
from scipy.signal import savgol_filter


def localization_config(settings, fs):
    mode = settings.get("localization_mode", "raw")
    if mode not in ("raw", "trend"):
        raise ValueError("未知定位方式，请选择原波形极值或去毛刺趋势定位。")
    config = {"localization_mode": mode,
              "despike_window_ms": float(settings.get("despike_window_ms", 5)),
              "despike_mad_threshold": float(settings.get("despike_mad_threshold", 6)),
              "trend_window_ms": float(settings.get("trend_window_ms", 20)),
              "trend_polyorder": 2}
    if mode == "raw":
        return config
    if not all(np.isfinite(config[key]) for key in ("despike_window_ms", "despike_mad_threshold", "trend_window_ms")):
        raise ValueError("定位参数必须是有限数值。")
    if config["despike_window_ms"] < 0 or config["trend_window_ms"] <= 0 or config["despike_mad_threshold"] <= 0:
        raise ValueError("去毛刺窗不能为负，趋势窗及 MAD 倍数必须大于 0。")
    def odd_window(ms):
        size = max(3, int(round(ms * fs / 1000)))
        return size if size % 2 else size + 1
    config["despike_window_samples"] = odd_window(config["despike_window_ms"]) if config["despike_window_ms"] else 1
    config["trend_window_samples"] = odd_window(config["trend_window_ms"])
    return config


def trial_localization(read, trial, rows, fs, settings, check=lambda: None, reserved_bytes=0):
    """Use a reproducible padded slice, independent of the chosen peak/view.

    Returns loaded, despiked and signed-trend arrays on exactly the same sample
    grid. Missing values are interpolated only for filtering and then masked;
    they can never become candidate alignment samples or replacement raw data.
    """
    config = localization_config(settings, fs)
    half_median = config["despike_window_samples"] // 2
    half_smooth = config["trend_window_samples"] // 2
    pre = int(round(settings["pre_ms"] * fs / 1000))
    post = int(round(settings["post_ms"] * fs / 1000))
    first = max(0, trial["search_first"] - pre - half_median - half_smooth)
    last = min(rows, trial["search_last"] + post + half_median + half_smooth)
    # Bound both sliding-window workspaces and retained result memory before IO.
    window = config["despike_window_samples"]
    block_cells = 262144
    if max(window, config["trend_window_samples"]) > block_cells:
        raise ValueError("定位窗口过长，请缩短去毛刺窗或趋势窗。")
    workspace = (last - first + 2 * window) * 64 + block_cells * 32
    if workspace + reserved_bytes > int(settings.get("memory_budget_bytes", 256 * 1024 * 1024)):
        raise ValueError("定位曲线和已保留结果超过测试内存预算，请减少通道、Trial、搜索或截取长度。")
    check()
    raw = np.asarray(read(first, last), dtype=np.float32).reshape(-1)
    finite = np.isfinite(raw)
    if not finite.any():
        raise ValueError("定位上下文中没有有效数值。")
    work = np.asarray(raw, dtype=np.float64).copy()
    if not finite.all():
        positions = np.arange(work.size)
        work[~finite] = np.interp(positions[~finite], positions[finite], work[finite])
    check()
    cleaned = work.copy()
    replaced = np.zeros(work.size, dtype=bool)
    if window > 1:
        padded = np.pad(work, half_median, mode="reflect" if work.size > 1 else "edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window)
        block_size = max(1, block_cells // window)
        floor = max(np.finfo(float).eps * max(1.0, float(np.max(np.abs(work)))) * 16, np.finfo(float).tiny)
        for start in range(0, work.size, block_size):
            check()
            stop = min(work.size, start + block_size)
            block = windows[start:stop]
            median = np.median(block, axis=1)
            mad = np.median(np.abs(block - median[:, None]), axis=1)
            spikes = np.abs(work[start:stop] - median) > config["despike_mad_threshold"] * np.maximum(1.4826 * mad, floor)
            cleaned[start:stop] = np.where(spikes, median, work[start:stop])
            replaced[start:stop] = spikes & finite[start:stop]
    check()
    trend = savgol_filter(cleaned, config["trend_window_samples"], config["trend_polyorder"], mode="mirror")
    check()
    cleaned[~finite] = np.nan
    trend[~finite] = np.nan
    return {"first": first, "last": last, "raw": raw,
            "despiked": cleaned.astype(np.float32), "trend": trend.astype(np.float32),
            "despike_mask": replaced, "config": config}
