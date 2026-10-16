"""Experimental AnimalStart peak/trough alignment, independent of task metrics.

Source.read returns mV. Nothing in this module rewrites source data, changes
task markers, or filters trials by waveform similarity.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import uuid

import numpy as np

from animal_peak_plotting import PLOT_STYLE, draw_trial_detail
from animal_peak_localization import localization_config, trial_localization


class AnalysisCancelled(Exception):
    pass


def _check(cancel):
    if cancel is not None and cancel():
        raise AnalysisCancelled("已取消测试操作")


def source_identity(source):
    meta = source.metadata
    return (id(source), id(getattr(source, "data", None)), str(meta.path),
            str(meta.dataset), int(meta.rows), float(meta.fs), float(meta.time_offset),
            tuple(int(x) for x in meta.channel_ids))


def _condition_channel(source, column, settings, cancel=None):
    """Mirror optional task conditioning, one channel at a time, never in place."""
    processing = settings.get("processing") or {}
    enabled = any(processing.get(k) for k in ("analysis_notch", "analysis_bandpass", "smooth", "zscore"))
    if not enabled:
        return None
    meta = source.metadata
    if int(meta.rows) * 4 * 8 > int(settings["memory_budget_bytes"]):
        raise ValueError("额外分析处理需要读取整条通道，超过测试内存预算。请关闭额外分析处理，使用已经预处理的数据。")
    _check(cancel)
    values = np.array(source.read(0, meta.rows, [column]), dtype=np.float32, copy=True).reshape(-1, 1)
    _check(cancel)
    if processing.get("analysis_notch") or processing.get("analysis_bandpass"):
        from qt_data_model import filter_array
        values, cache = filter_array(
            values, meta.fs, "bandpass" if processing.get("analysis_bandpass") else "off",
            float(processing.get("analysis_band_low", .5)), float(processing.get("analysis_band_high", 300)),
            notch=bool(processing.get("analysis_notch")),
        )
        _check(cancel)
        if cache is not None:
            raise ValueError("测试分析不支持额外处理产生的磁盘缓存；请先在预处理页完成处理。")
    values = values[:, 0]
    if processing.get("smooth"):
        width = max(1, int(round(float(processing.get("smooth_window_sec", .02)) * meta.fs)))
        if width > 1:
            if width > values.size:
                raise ValueError("平滑窗不能长于整条记录。")
            values = np.convolve(values, np.ones(width, dtype=np.float32) / width, mode="same")
            _check(cancel)
    if processing.get("zscore"):
        deviation = np.nanstd(values)
        values = (values - np.nanmean(values)) / (deviation if deviation > 0 else 1.0)
    _check(cancel)
    return values


def _read(source, column, first, last, conditioned=None):
    values = conditioned[first:last] if conditioned is not None else source.read(first, last, [column])
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size != last - first:
        raise ValueError(f"数据读取长度异常：期望 {last-first}，实际 {values.size}")
    return values


def _summarize(waves, cancel=None):
    """Equal-trial mean/SEM and leave-one-out shape correlation; no rejection."""
    finite = np.isfinite(waves)
    count = np.sum(finite, axis=0)
    total = np.sum(np.where(finite, waves, 0), axis=0, dtype=np.float64)
    mean = np.divide(total, count, out=np.full(waves.shape[1], np.nan), where=count > 0)
    squared = np.sum(np.where(finite, (waves - mean) ** 2, 0), axis=0)
    variance = np.divide(squared, count - 1, out=np.full(mean.size, np.nan), where=count > 1)
    sem = np.sqrt(np.divide(variance, count, out=np.full(mean.size, np.nan), where=count > 1))
    correlations = []
    for index, wave in enumerate(waves):
        _check(cancel)
        others_count = count - finite[index]
        others_total = total - np.where(finite[index], wave, 0)
        template = np.divide(others_total, others_count, out=np.full(mean.size, np.nan), where=others_count > 0)
        valid = np.isfinite(wave) & np.isfinite(template)
        if valid.sum() >= 3 and np.std(wave[valid]) > 0 and np.std(template[valid]) > 0:
            correlations.append(float(np.corrcoef(wave[valid], template[valid])[0, 1]))
        else:
            correlations.append(np.nan)
    return mean.astype(np.float32), sem.astype(np.float32), count, correlations


def _animal_exclusion(event, latency):
    """Explain the first failed validity/timing condition without changing it."""
    validity = event.get("validity")
    if validity != "Y":
        if validity in (None, ""):
            return "validity_missing", "validity 为空；视频检测 TXT 第5列未提供有效性标记，要求 Y"
        return "validity_not_y", f"validity 不是 Y；实际值={validity!r}，要求 Y（视频检测 TXT 第5列）"
    if np.isfinite(latency):
        return "", ""
    animal = event.get("animal_start_frame")
    start = event.get("start_frame")
    if event.get("animal_start_status") == "unparseable":
        return "animal_frame_unparseable", f"AnimalStart 帧号无法解析；TXT 第4列原值={event.get('animal_start_raw')!r}"
    if animal is None:
        raw = event.get("animal_start_raw", "未保留原值")
        return "animal_frame_missing", f"AnimalStart 帧号缺失；TXT 第4列原值={raw!r}，没有可用于对齐的帧号"
    if start is not None:
        if animal == start:
            return "animal_frame_equal_start", f"AnimalStart 帧号等于 StartFrame；AnimalStartFrame={animal}，StartFrame={start}，当前要求 AnimalStartFrame > StartFrame"
        if animal < start:
            return "animal_frame_before_start", f"AnimalStart 帧号早于 StartFrame；AnimalStartFrame={animal}，StartFrame={start}，当前要求 AnimalStartFrame > StartFrame"
    return "animal_latency_nonfinite", f"AnimalStart 对齐延迟不是有限数值；animal_latency_sec={event.get('animal_latency_sec')!r}，AnimalStartFrame={animal}，StartFrame={start}"


def analyze_animal_peaks(source, markers, matches, settings, progress=None, cancel=None):
    """Find one maximum AND minimum around AnimalStart in each eligible trial.

    Eligibility is tied to the existing Letter epoch. Excerpts may extend
    beyond that epoch (including beyond AnimalStart), but never beyond source
    bounds. Extremum ties choose the point nearest AnimalStart, then latest.
    """
    _check(cancel)
    meta = source.metadata
    initial_identity = source_identity(source)
    fs = float(meta.fs)
    settings = dict(settings)
    settings.setdefault("memory_budget_bytes", 256 * 1024 * 1024)
    settings.setdefault("search_after_ms", 0.0)
    settings.update(localization_config(settings, fs))
    trend_mode = settings["localization_mode"] == "trend"
    if trend_mode:
        # Localization is disposable. Average only the loaded source, never
        # optional task conditioning, spike replacements or smoothing output.
        settings["processing"] = {}
    settings["average_waveform"] = "loaded_source_baseline_corrected" if trend_mode else "legacy_task_conditioned_baseline_corrected"
    for key in ("search_ms", "search_after_ms", "pre_ms", "post_ms", "epoch_start_ms", "epoch_end_ms", "baseline_start_ms", "baseline_end_ms"):
        if not np.isfinite(float(settings[key])):
            raise ValueError(f"参数 {key} 必须为有限数值")
    if settings["search_ms"] <= 0 or min(settings["search_after_ms"], settings["pre_ms"], settings["post_ms"]) < 0:
        raise ValueError("AnimalStart 前搜索长度必须大于 0，后搜索长度及峰前/峰后截取长度不能为负。")
    search_samples = max(1, int(round(settings["search_ms"] * fs / 1000)))
    search_after_samples = int(round(settings["search_after_ms"] * fs / 1000))
    pre = int(round(settings["pre_ms"] * fs / 1000))
    post = int(round(settings["post_ms"] * fs / 1000))
    if pre + post + 1 < 3:
        raise ValueError("参与平均的片段至少需要 3 个采样点。")
    epoch_first = int(round(settings["epoch_start_ms"] * fs / 1000))
    epoch_last = int(round(settings["epoch_end_ms"] * fs / 1000))
    if epoch_last <= epoch_first:
        raise ValueError("Epoch 终点必须大于起点。")
    base_first = max(epoch_first, int(np.ceil(settings["baseline_start_ms"] * fs / 1000 - 1e-8)))
    base_last = min(epoch_last, int(np.floor(settings["baseline_end_ms"] * fs / 1000 + 1e-8)) + 1)
    if base_last <= base_first:
        raise ValueError("现有 Letter 基线窗没有采样点，请先调整任务基线参数。")
    lookup = {int(ch): index for index, ch in enumerate(meta.channel_ids)}
    channel_ids = list(dict.fromkeys(int(ch) for ch in settings["channel_ids"]))
    if not channel_ids or any(ch not in lookup for ch in channel_ids):
        raise ValueError("测试通道必须存在于当前 LFP 数据源中。")
    marker_array = np.asarray(markers, dtype=np.int64)
    if marker_array.ndim != 2 or marker_array.shape[1] != 2:
        raise ValueError("Letter marker 格式应为 sample、Tag 两列。")
    letter = sorted((int(sample), int(tag)) for sample, tag in marker_array if int(tag) in (4, 5, 6, 7))
    if not letter:
        raise ValueError("没有 Letter Tag 4/5/6/7 的对齐 marker。")
    wanted = set(int(tag) for tag in settings.get("tags", []) or (4, 5, 6, 7))
    limit = max(0, int(settings.get("trial_limit", 0)))
    offset = int(round(float(meta.time_offset) * fs))
    tag_counts = defaultdict(int)
    eligible, excluded = [], []
    for trial_id, (sample, tag) in enumerate(letter, start=1):
        _check(cancel)
        tag_counts[tag] += 1
        if tag not in wanted or (limit and tag_counts[tag] > limit):
            continue
        row = {"trial_id": trial_id, "tag_trial_id": tag_counts[tag], "tag": tag,
               "marker_sample": sample, "marker_sec": sample / fs}
        event = matches.get(sample)
        reason = ""
        reason_code = ""
        try:
            latency = float(event.get("animal_latency_sec", np.nan)) if event else np.nan
        except (TypeError, ValueError):
            latency = np.nan
        first, last = sample - offset + epoch_first, sample - offset + epoch_last
        animal_global = int(round(sample + latency * fs)) if np.isfinite(latency) else -1
        animal = animal_global - offset
        row.update(animal_sec=(sample / fs + latency), animal_sample=animal_global,
                   animal_latency_ms=latency * 1000, animal_local_sample=animal)
        row.update({name: event.get(name, "") if event is not None else "" for name in (
            "validity", "start_frame", "animal_start_frame", "animal_start_raw", "animal_start_status",
            "validity_raw", "detection_line_number",
        )})
        epoch_range = f"Epoch=[{epoch_first / fs * 1000:g}, {epoch_last / fs * 1000:g}) ms（相对 Letter）"
        if event is None:
            reason_code = "behavior_unmatched"
            reason = f"无对应行为记录；marker_sample={sample}，Tag={tag}，对齐结果中没有该 marker 的视频匹配项"
        else:
            reason_code, reason = _animal_exclusion(event, latency)
        if not reason:
            if first < 0:
                reason_code = "epoch_before_data"
                reason = f"原 Letter Epoch 起点早于当前数据段；Epoch 本地样本范围=[{first}, {last})，数据范围=[0, {meta.rows})"
            elif last > meta.rows:
                reason_code = "epoch_after_data"
                reason = f"原 Letter Epoch 终点超过当前数据段；Epoch 本地样本范围=[{first}, {last})，数据范围=[0, {meta.rows})"
            elif animal < first:
                reason_code = "animal_before_epoch"
                reason = f"AnimalStart 早于原 Letter Epoch 起点；AnimalStart 相对 Letter={latency * 1000:.3f} ms，{epoch_range}"
            elif animal >= last:
                reason_code = "animal_at_epoch_end" if animal == last else "animal_after_epoch"
                description = "正好位于" if animal == last else "晚于"
                reason = f"AnimalStart {description}原 Letter Epoch 终点；AnimalStart 相对 Letter={latency * 1000:.3f} ms，{epoch_range}，该时刻不在 Epoch 内"
            elif animal - search_samples < first:
                reason_code = "search_before_epoch"
                reason = f"原 Letter Epoch 未覆盖完整搜索范围：搜索起点早于 Epoch 起点；搜索起点相对 Letter={(animal - search_samples - (sample - offset)) / fs * 1000:.3f} ms，{epoch_range}"
            elif animal + search_after_samples > last:
                reason_code = "search_after_epoch"
                reason = f"原 Letter Epoch 未覆盖完整搜索范围：搜索终点超过 Epoch 终点；搜索终点相对 Letter={(animal + search_after_samples - (sample - offset)) / fs * 1000:.3f} ms，{epoch_range}"
        row["reason_code"] = reason_code
        row.update(epoch_first=first, epoch_last=last, search_first=animal - search_samples,
                   search_last=animal + search_after_samples)
        if reason:
            excluded.append({**row, "channel": "", "branch": "both", "status": "excluded", "reason": reason})
        else:
            eligible.append(row)
    if not eligible:
        # Preserve diagnostics even when no trial qualifies.
        return {"settings": settings, "groups": [], "records": excluded, "eligible_trials": [],
                "source_identity": source_identity(source), "source": _source_metadata(meta), "fs": fs,
                "units": "z" if (settings.get("processing") or {}).get("zscore") else "mV"}
    estimated = len(channel_ids) * len(eligible) * (3 if trend_mode else 2) * (pre + post + 1) * 4
    if estimated * 2 > settings["memory_budget_bytes"]:
        raise ValueError(f"预计峰谷结果及计算工作区需要约 {estimated*2/1024**2:.0f} MB，超过测试预算。请减少通道/Trial或截取长度。")
    records, groups = list(excluded), []
    animal_groups, animal_records = [], []
    total = len(channel_ids) * len(eligible)
    completed = 0
    for channel in channel_ids:
        _check(cancel)
        column = lookup[channel]
        conditioned = _condition_channel(source, column, settings, cancel)
        def read(first, last):
            _check(cancel)
            if source_identity(source) != initial_identity:
                raise ValueError("计算期间数据源已改变，请重新打开测试窗口。")
            values = _read(source, column, first, last, conditioned)
            _check(cancel)
            if source_identity(source) != initial_identity:
                raise ValueError("读取期间数据源已改变，请重新打开测试窗口。")
            return values
        buckets = defaultdict(list)
        animal_buckets = defaultdict(list)
        for trial in eligible:
            _check(cancel)
            center = trial["marker_sample"] - offset
            base = read(center + base_first, center + base_last)
            finite_base = base[np.isfinite(base)]
            search = read(trial["search_first"], trial["search_last"])
            valid = np.isfinite(search)
            if not finite_base.size or not valid.any():
                if not finite_base.size:
                    code = "baseline_no_finite"
                    detail = f"基线窗没有有效数值；有限采样点=0/{base.size}，本地样本范围=[{center + base_first}, {center + base_last})"
                else:
                    code = "search_no_finite"
                    detail = f"搜索窗没有有效数值；有限采样点=0/{search.size}，本地样本范围=[{trial['search_first']}, {trial['search_last']})"
                records.append({**trial, "channel": channel, "branch": "both", "status": "excluded",
                                "reason": detail, "reason_code": code})
            else:
                correction = float(np.mean(finite_base))
                localization = None
                locating = search
                if trend_mode:
                    localization = trial_localization(read, trial, meta.rows, fs, settings,
                                                      lambda: _check(cancel), reserved_bytes=estimated * 2)
                    first_index = trial["search_first"] - localization["first"]
                    last_index = trial["search_last"] - localization["first"]
                    locating = localization["trend"][first_index:last_index]
                    replaced_count = int(localization["despike_mask"][first_index:last_index].sum())
                    reference_first = trial["animal_local_sample"] - pre
                    reference_last = trial["animal_local_sample"] + post + 1
                    reference_row = {**trial, "channel": channel, "column": column,
                                     "branch": "animal", "baseline_correction": correction,
                                     "excerpt_first": reference_first, "excerpt_last": reference_last}
                    if reference_first < 0 or reference_last > meta.rows:
                        reference_row.update(status="excluded", reason="AnimalStart 对照截取片段超出源数据边界",
                                             excerpt_first=reference_first, excerpt_last=reference_last)
                    else:
                        reference_wave = read(reference_first, reference_last).copy() - correction
                        reference_wave[~np.isfinite(reference_wave)] = np.nan
                        reference_row.update(status="included", reason="")
                        animal_buckets[trial["tag"]].append((reference_row, reference_wave))
                    animal_records.append(reference_row)
                locating_valid = np.isfinite(locating) & valid
                if not locating_valid.any():
                    records.append({**trial, "channel": channel, "branch": "both", "status": "excluded",
                                    "reason_code": "localization_no_finite", "reason": "定位趋势线在搜索区没有有效数值"})
                    continue
                flat = bool(np.max(locating[locating_valid]) == np.min(locating[locating_valid]))
                for branch, extreme in (("peak", np.max(locating[locating_valid])), ("trough", np.min(locating[locating_valid]))):
                    _check(cancel)
                    candidates = np.flatnonzero(locating_valid & (locating == extreme))
                    distance = np.abs(trial["search_first"] + candidates - trial["animal_local_sample"])
                    index = int(candidates[distance == distance.min()][-1])
                    position = trial["search_first"] + index
                    left, right = position - pre, position + post + 1
                    flags = []
                    if flat:
                        flags.append("平直片段，极值不代表明确生理波")
                    if index in (0, search.size - 1):
                        flags.append("搜索边界极值")
                    if not valid.all():
                        flags.append("搜索区含非有限值，未作为极值")
                    row = {**trial, "channel": channel, "column": column, "branch": branch,
                           "extreme_local_sample": position, "extreme_sample": position + offset,
                           "extreme_sec": (position + offset) / fs,
                           "extreme_relative_ms": (position - trial["animal_local_sample"]) / fs * 1000,
                           "amplitude": float(search[index] - correction), "baseline_correction": correction,
                           "localization_mode": settings["localization_mode"],
                           "localization_amplitude": float(extreme - correction),
                           "excerpt_first": left, "excerpt_last": right,
                           "crosses_animal_start": right > trial["animal_local_sample"],
                           "flags": "; ".join(flags), "similarity_r": np.nan}
                    if trend_mode:
                        raw_extreme = np.max(search[valid]) if branch == "peak" else np.min(search[valid])
                        raw_candidates = np.flatnonzero(valid & (search == raw_extreme))
                        raw_distances = np.abs(trial["search_first"] + raw_candidates - trial["animal_local_sample"])
                        raw_index = int(raw_candidates[raw_distances == raw_distances.min()][-1])
                        row.update(despike_count_search=replaced_count,
                                   localization_context_first=localization["first"], localization_context_last=localization["last"],
                                   despike_window_samples=settings["despike_window_samples"], trend_window_samples=settings["trend_window_samples"],
                                   raw_extreme_relative_ms=(trial["search_first"] + raw_index - trial["animal_local_sample"]) / fs * 1000,
                                   raw_extreme_amplitude=float(raw_extreme - correction))
                    if left < 0:
                        row.update(status="excluded", reason_code="excerpt_before_data",
                                   reason=f"完整截取片段超出源数据边界：截取起点早于数据起点；片段本地样本范围=[{left}, {right})，数据范围=[0, {meta.rows})，仅排除本峰谷分支的平均")
                    elif right > meta.rows:
                        row.update(status="excluded", reason_code="excerpt_after_data",
                                   reason=f"完整截取片段超出源数据边界：截取终点超过数据终点；片段本地样本范围=[{left}, {right})，数据范围=[0, {meta.rows})，仅排除本峰谷分支的平均")
                    else:
                        wave = read(left, right).copy() - correction
                        wave[~np.isfinite(wave)] = np.nan
                        row.update(status="included", reason="")
                        buckets[(trial["tag"], branch)].append((row, wave))
                    records.append(row)
            completed += 1
            if progress:
                progress(100 * completed / total, f"测试峰谷：ch{channel}，Trial {trial['trial_id']} / Tag {trial['tag']}")
        for (tag, branch), entries in sorted(buckets.items()):
            _check(cancel)
            waves = np.stack([entry[1] for entry in entries])
            mean, sem, count, correlations = _summarize(waves, cancel)
            for (row, _), correlation in zip(entries, correlations):
                row["similarity_r"] = correlation
            groups.append({"channel": channel, "tag": tag, "branch": branch,
                           "time_ms": np.arange(-pre, post + 1, dtype=float) / fs * 1000,
                           "waves": waves, "mean": mean, "sem": sem, "valid_count": count,
                           "rows": [entry[0] for entry in entries]})
        for tag, entries in sorted(animal_buckets.items()):
            waves = np.stack([entry[1] for entry in entries])
            mean, sem, count, _ = _summarize(waves, cancel)
            animal_groups.append({"channel": channel, "tag": tag, "branch": "animal",
                                  "time_ms": np.arange(-pre, post + 1, dtype=float) / fs * 1000,
                                  "waves": waves, "mean": mean, "sem": sem, "valid_count": count,
                                  "rows": [entry[0] for entry in entries]})
        del conditioned, buckets, animal_buckets
    _check(cancel)
    return {"settings": settings, "groups": groups, "records": records, "eligible_trials": eligible,
            "animal_reference_groups": animal_groups, "animal_reference_records": animal_records,
            "source_identity": source_identity(source), "source": _source_metadata(meta), "fs": fs,
            "units": "z" if (settings.get("processing") or {}).get("zscore") else "mV"}


def _source_metadata(meta):
    return {"path": str(meta.path), "dataset": str(meta.dataset), "rows": int(meta.rows),
            "fs": float(meta.fs), "time_offset": float(meta.time_offset),
            "channel_ids": [int(ch) for ch in meta.channel_ids],
            "provenance": getattr(meta, "provenance", {})}


def trial_detail_trace(source, output, rows, conditioned=None):
    """Return only the search/excerpt union; no extra context is loaded."""
    if source_identity(source) != output["source_identity"]:
        raise ValueError("原数据源已改变，不能使用旧的峰谷结果读取详情；请重新计算。")
    first = max(0, min(min(row["search_first"], row["excerpt_first"]) for row in rows))
    last = min(source.metadata.rows, max(max(row["search_last"], row["excerpt_last"]) for row in rows))
    if conditioned is None and any((output["settings"].get("processing") or {}).get(k) for k in ("analysis_notch", "analysis_bandpass", "smooth", "zscore")):
        conditioned = _condition_channel(source, rows[0]["column"], output["settings"])
    values = _read(source, rows[0]["column"], first, last, conditioned) - rows[0]["baseline_correction"]
    if source_identity(source) != output["source_identity"]:
        raise ValueError("详情读取期间数据源已改变，请重新计算。")
    time_ms = (np.arange(first, last) - rows[0]["animal_local_sample"]) / output["fs"] * 1000
    return time_ms, values


def trial_detail_localization(source, output, rows, cancel=None):
    """Reproduce the exact padded locator used for selection, not a new zoom."""
    if output["settings"].get("localization_mode", "raw") != "trend":
        return None
    if source_identity(source) != output["source_identity"]:
        raise ValueError("原数据源已改变，请重新计算定位结果。")
    row = rows[0]
    localization = trial_localization(
        lambda first, last: _read(source, row["column"], first, last), row,
        source.metadata.rows, output["fs"], output["settings"], lambda: _check(cancel),
    )
    if source_identity(source) != output["source_identity"]:
        raise ValueError("定位曲线读取期间数据源已改变，请重新计算。")
    first = max(0, min(min(item["search_first"], item["excerpt_first"]) for item in rows))
    last = min(source.metadata.rows, max(max(item["search_last"], item["excerpt_last"]) for item in rows))
    start, stop = first - localization["first"], last - localization["first"]
    correction = row["baseline_correction"]
    return {"time_ms": (np.arange(first, last) - row["animal_local_sample"]) / output["fs"] * 1000,
            "despiked": localization["despiked"][start:stop] - correction,
            "trend": localization["trend"][start:stop] - correction,
            "despike_mask": localization["despike_mask"][start:stop]}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def export_animal_peak_results(output, source, directory, individual_png=True, progress=None, cancel=None):
    """Save high-resolution figures, every trial page, audit CSV and numeric data.

    A new directory prevents overwrites. Cancellation leaves clearly marked,
    recoverable partial output, never deleting recordings or existing exports.
    """
    _check(cancel)
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_pdf import PdfPages

    if source_identity(source) != output["source_identity"]:
        raise ValueError("数据源已改变，请重新计算后导出。")
    folder = Path(directory) / ("animal_peak_test_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "incomplete", "settings": output["settings"], "source": output["source"],
                "units": output["units"], "meaning": "Forced extremum-aligned means; similarity is descriptive, never an inclusion threshold."}
    def save_manifest():
        (folder / "parameters.json").write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    save_manifest()
    unit = "μV" if output["units"] == "mV" else "z"
    scale = 1000 if output["units"] == "mV" else 1
    records = output["records"]
    keys = sorted({key for row in records for key in row})
    with (folder / "trial_details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    if output.get("animal_reference_records"):
        reference_records = output["animal_reference_records"]
        with (folder / "animalstart_reference_details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in reference_records for key in row}))
            writer.writeheader(); writer.writerows(reference_records)
    pairs = defaultdict(dict)
    for group in output["groups"]:
        pairs[(group["channel"], group["tag"])][group["branch"]] = group
    for group in output.get("animal_reference_groups", []):
        pairs.setdefault((group["channel"], group["tag"]), {})
    detail_sets = defaultdict(list)
    for row in records:
        if "extreme_local_sample" in row:
            detail_sets[(row["channel"], row["tag"], row["trial_id"])].append(row)
    count = len(pairs) + len(detail_sets)
    done = 0
    try:
        with rc_context(PLOT_STYLE), PdfPages(folder / "all_trial_details.pdf") as pdf:
            _check(cancel)
            overview = Figure(figsize=(12, 8), layout="constrained"); FigureCanvasAgg(overview)
            axis = overview.subplots(); axis.set_axis_off()
            reasons = defaultdict(int)
            for row in records:
                if row["status"] == "excluded":
                    reasons[row["reason"].split("；", 1)[0]] += 1
            configuration = output["settings"]
            channel_label = str(configuration["channel_ids"][:20])
            if len(configuration["channel_ids"]) > 20:
                channel_label += f" … 共 {len(configuration['channel_ids'])} 通道（完整列表见参数文件）"
            report = [
                "AnimalStart 最大峰 / 最小谷对齐平均：测试报告",
                f"文件：{output['source']['path']}", f"数据集：{output['source']['dataset']} | FS={output['fs']:g} Hz | 存储单位={output['units']}",
                f"通道：{channel_label} | Tag：{configuration.get('tags') or '全部 Letter Tag'}",
                f"原 Letter Epoch：{configuration['epoch_start_ms']:g}～{configuration['epoch_end_ms']:g} ms",
                f"原 Letter 基线：{configuration['baseline_start_ms']:g}～{configuration['baseline_end_ms']:g} ms",
                f"AnimalStart 前/后搜索：{configuration['search_ms']:g}/{configuration.get('search_after_ms', 0):g} ms | 极值前/后截取：{configuration['pre_ms']:g}/{configuration['post_ms']:g} ms",
                (f"定位方式：去毛刺＋趋势线；去毛刺窗={configuration.get('despike_window_ms')} ms，MAD={configuration.get('despike_mad_threshold')}，平滑窗={configuration.get('trend_window_ms')} ms；只定位，平均载入波形。"
                 if configuration.get("localization_mode") == "trend" else "定位方式：原波形极值（旧方式）"),
                f"有效 AnimalStart Trial：{len(output['eligible_trials'])} | 峰谷纳入记录：{sum(r['status']=='included' for r in records)}",
                "每个有效 trial 都选最大值和最小值；同值取更接近 AnimalStart 的采样点，等距离取较晚点。",
                "正负分支独立平均，不反转、不归一化；相似度仅报告，不筛选。",
                "平均图零点=选中极值；逐 trial 详情零点=AnimalStart。",
                "排除原因（包含原 trial 排除及通道/分支不可计算记录）：",
                *[f"  {reason}：{number}" for reason, number in reasons.items()],
                "完整参数与处理历史见 parameters.json；所有 trial 编号、极值及原因见 trial_details.csv。",
                "强制极值对齐可使背景波形形成平均峰；本结果不能单独作为稳定 VEP 的证据。",
            ]
            axis.text(.01, .99, "\n\n".join(report), va="top", ha="left", fontsize=9, wrap=True, transform=axis.transAxes)
            overview.savefig(folder / "overview.png", dpi=220)
            overview.savefig(folder / "overview.pdf")
            pdf.savefig(overview); overview.clear()
            for (channel, tag), branches in sorted(pairs.items()):
                _check(cancel)
                fig = Figure(figsize=(12, 8), layout="constrained"); FigureCanvasAgg(fig)
                axes = fig.subplots(2, 2)
                fig.suptitle(f"{Path(output['source']['path']).name}\nch{channel} | Tag {tag} | 最大峰 / 最小谷对齐平均（相似度不筛选）", fontsize=12)
                for column, branch in enumerate(("peak", "trough")):
                    group = branches.get(branch)
                    title = "最大峰" if branch == "peak" else "最小谷"
                    if group is None:
                        axes[0, column].set_title(title + "：无完整截取片段")
                        continue
                    t = group["time_ms"]
                    axes[0, column].plot(t, group["waves"].T * scale, color="0.6", alpha=.22, linewidth=.5)
                    axes[0, column].plot(t, group["mean"] * scale, color="#b32025", linewidth=1.8, label="全部片段等权平均")
                    axes[0, column].fill_between(t, (group["mean"]-group["sem"])*scale,
                                                 (group["mean"]+group["sem"])*scale, color="#b32025", alpha=.15, label="±SEM")
                    axes[0, column].axvline(0, color="black", linestyle="--", linewidth=.8)
                    axes[0, column].set(title=f"{title}对齐 | n={len(group['rows'])}", xlabel="相对选中极值（ms）", ylabel=unit)
                    axes[0, column].legend(fontsize=8)
                    latency = [row["extreme_relative_ms"] for row in group["rows"]]
                    axes[1, column].hist(latency, bins=min(20, max(1, len(latency))), color="#4177a3", alpha=.8)
                    axes[1, column].set(xlabel="极值相对 AnimalStart（ms）", ylabel="Trial 数", title="保留原始峰时/谷时分布")
                    name = f"ch{channel}_tag{tag}_{branch}"
                    np.savez_compressed(folder / (name + ".npz"), time_ms=t, trials=group["waves"], mean=group["mean"],
                                        sem=group["sem"], valid_count=group["valid_count"],
                                        trial_ids=np.asarray([row["trial_id"] for row in group["rows"]]),
                                        tag_trial_ids=np.asarray([row["tag_trial_id"] for row in group["rows"]]),
                                        animal_sec=np.asarray([row["animal_sec"] for row in group["rows"]]),
                                        extrema_relative_ms=np.asarray(latency),
                                        similarity_r=np.asarray([row["similarity_r"] for row in group["rows"]]),
                                        units=output["units"], channel=channel, tag=tag)
                reference = next((item for item in output.get("animal_reference_groups", [])
                                  if item["channel"] == channel and item["tag"] == tag), None)
                if reference is not None:
                    ref_fig = Figure(figsize=(10, 5), layout="constrained"); FigureCanvasAgg(ref_fig)
                    ref_axis = ref_fig.subplots()
                    ref_axis.plot(reference["time_ms"], reference["waves"].T * scale, color=".6", alpha=.22, linewidth=.5)
                    ref_axis.plot(reference["time_ms"], reference["mean"] * scale, color="#2679b5", label="未去毛刺的载入波形等权平均")
                    ref_axis.fill_between(reference["time_ms"], (reference["mean"] - reference["sem"]) * scale,
                                          (reference["mean"] + reference["sem"]) * scale, color="#2679b5", alpha=.15, label="±SEM")
                    ref_axis.axvline(0, color="#9467bd", linestyle="--")
                    ref_axis.set(title=f"ch{channel} | Tag {tag} | AnimalStart 直接对齐对照 | n={len(reference['rows'])}",
                                 xlabel="相对 AnimalStart（ms）", ylabel=unit)
                    ref_axis.legend(); ref_axis.grid(alpha=.15)
                    ref_name = f"ch{channel}_tag{tag}_animalstart_reference"
                    ref_fig.savefig(folder / (ref_name + ".png"), dpi=220)
                    ref_fig.savefig(folder / (ref_name + ".pdf")); pdf.savefig(ref_fig); ref_fig.clear()
                    np.savez_compressed(folder / (ref_name + ".npz"), time_ms=reference["time_ms"],
                                        trials=reference["waves"], mean=reference["mean"], sem=reference["sem"],
                                        valid_count=reference["valid_count"], units=output["units"], channel=channel, tag=tag,
                                        trial_ids=np.asarray([row["trial_id"] for row in reference["rows"]]))
                fig.savefig(folder / f"ch{channel}_tag{tag}_summary.png", dpi=220)
                fig.savefig(folder / f"ch{channel}_tag{tag}_summary.pdf")
                pdf.savefig(fig); fig.clear()
                done += 1
                if progress:
                    progress(done / max(1, count) * 100, f"导出平均图：ch{channel} / Tag {tag}")
            page = None
            axes = None
            previous_channel = None
            conditioned = None
            page_index = 0
            trace_pair = None
            trace_parts = defaultdict(int)
            numeric_traces = []
            numeric_bytes = 0

            def save_trace_part():
                nonlocal numeric_traces, numeric_bytes
                if not numeric_traces:
                    return
                trace_parts[trace_pair] += 1
                offsets = np.r_[0, np.cumsum([item[2].size for item in numeric_traces])]
                payload = dict(time_ms=np.concatenate([item[1] for item in numeric_traces]),
                               values=np.concatenate([item[2] for item in numeric_traces]), offsets=offsets,
                               trial_ids=np.asarray([item[0] for item in numeric_traces]), units=output["units"],
                               channel=trace_pair[0], tag=trace_pair[1])
                if numeric_traces[0][3] is not None:
                    payload.update(trend_values=np.concatenate([item[3]["trend"] for item in numeric_traces]),
                                   despiked_values=np.concatenate([item[3]["despiked"] for item in numeric_traces]),
                                   despike_mask=np.concatenate([item[3]["despike_mask"] for item in numeric_traces]))
                np.savez_compressed(folder / f"ch{trace_pair[0]}_tag{trace_pair[1]}_trial_traces_part{trace_parts[trace_pair]:03d}.npz", **payload)
                numeric_traces = []; numeric_bytes = 0

            for detail_index, ((channel, tag, trial_id), rows) in enumerate(sorted(detail_sets.items())):
                _check(cancel)
                if trace_pair != (channel, tag):
                    save_trace_part(); trace_pair = (channel, tag)
                if previous_channel != channel:
                    conditioned = _condition_channel(source, rows[0]["column"], output["settings"], cancel)
                    previous_channel = channel
                time_ms, values = trial_detail_trace(source, output, rows, conditioned)
                localization = trial_detail_localization(source, output, rows, cancel)
                numeric_traces.append((trial_id, time_ms.astype(np.float32), values, localization))
                numeric_bytes += time_ms.size * 4 + values.nbytes
                if localization is not None:
                    numeric_bytes += sum(localization[key].nbytes for key in ("time_ms", "despiked", "trend", "despike_mask"))
                if numeric_bytes >= 16 * 1024 * 1024:
                    save_trace_part()
                if page_index == 0:
                    page = Figure(figsize=(12, 10), layout="constrained"); FigureCanvasAgg(page)
                    axes = page.subplots(2, 2).ravel()
                    page.suptitle(f"{Path(output['source']['path']).name} | 逐 Trial 找波与截取详情", fontsize=12)

                draw_trial_detail(axes[page_index], output, rows, time_ms, values, localization)
                if individual_png:
                    single = Figure(figsize=(10, 5), layout="constrained"); FigureCanvasAgg(single)
                    single.suptitle(Path(output["source"]["path"]).name, fontsize=10)
                    draw_trial_detail(single.subplots(), output, rows, time_ms, values, localization)
                    single.savefig(folder / f"ch{channel}_tag{tag}_trial{trial_id:05d}_detail.png", dpi=220)
                    single.clear()
                page_index += 1
                if page_index == 4 or detail_index == len(detail_sets)-1:
                    for unused in axes[page_index:]:
                        unused.set_visible(False)
                    pdf.savefig(page); page.clear(); page_index = 0
                done += 1
                if progress:
                    progress(done / max(1, count) * 100, f"导出详情：ch{channel} / Tag {tag} / Trial {trial_id}")
            save_trace_part()
        _check(cancel)
        manifest["status"] = "complete"
        manifest["record_count"] = len(records)
        manifest["detail_trial_channel_count"] = len(detail_sets)
        save_manifest()
    except Exception as exc:
        manifest["status"] = "cancelled" if isinstance(exc, AnalysisCancelled) else "failed"
        manifest["error"] = str(exc)
        save_manifest()
        error_type = AnalysisCancelled if isinstance(exc, AnalysisCancelled) else RuntimeError
        raise error_type(f"{exc}\n已保留部分输出（未删除）：{folder}") from exc
    return folder
