"""Reversible full-record channel filtering for the isolated peak experiment.

Disk-backed forward/backward SOS passes have one record-edge padding, not
one padding per Trial/chunk. Original H5 and unselected channels stay intact.
"""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import tempfile
import weakref

import numpy as np
from scipy import signal

from animal_peak_analysis import _check, _condition_channel, _read, _summarize, source_identity
from animal_peak_localization import trial_localization


def filter_config(config, fs):
    mode = str(config.get("mode", "notch"))
    if mode not in ("notch", "highpass", "lowpass", "bandpass"):
        raise ValueError("请选择陷波、高通、低通或带通。")
    cfg = {"mode": mode, "frequency_hz": float(config.get("frequency_hz", 1)),
           "q": float(config.get("q", 30)), "low_hz": float(config.get("low_hz", 1)),
           "high_hz": float(config.get("high_hz", 300)), "order": int(config.get("order", 3))}
    if not all(np.isfinite(v) for k, v in cfg.items() if k != "mode") or not 1 <= cfg["order"] <= 10:
        raise ValueError("滤波参数须为有限数值，阶数须在 1～10。")
    nyquist = float(fs) / 2
    if mode == "notch":
        if not 0 < cfg["frequency_hz"] < nyquist or cfg["q"] <= 0:
            raise ValueError(f"陷波频率须在 0～{nyquist:g} Hz 之间，Q 必须大于 0。")
        b, a = signal.iirnotch(cfg["frequency_hz"], cfg["q"], fs=fs)
        sos = signal.tf2sos(b, a)
    else:
        cutoff = cfg["low_hz"] if mode == "highpass" else cfg["high_hz"]
        if mode == "bandpass":
            if not 0 < cfg["low_hz"] < cfg["high_hz"] < nyquist:
                raise ValueError(f"带通要求 0 < 下限 < 上限 < {nyquist:g} Hz。")
            cutoff = [cfg["low_hz"], cfg["high_hz"]]
        elif not 0 < cutoff < nyquist:
            raise ValueError(f"截止频率须在 0～{nyquist:g} Hz 之间。")
        sos = signal.butter(cfg["order"], cutoff, btype=mode, fs=fs, output="sos")
    return cfg, sos


def _remove_private_cache(directory):
    path = Path(directory).resolve()
    # Only the exact directory created for this cache can ever be removed.
    if not path.name.startswith(".animal_peak_secondary_") or path == path.parent:
        raise ValueError("拒绝清理非二次滤波专用目录。")
    if path.exists():
        shutil.rmtree(path)


class _CacheDirectory:
    def __init__(self, directory):
        self.name = tempfile.mkdtemp(prefix=".animal_peak_secondary_", dir=directory)
        self._cleanup = weakref.finalize(self, _remove_private_cache, self.name)

    def cleanup(self):
        self._cleanup()


class _ChannelCache:
    def __init__(self, directory, rows, config):
        self.owner = _CacheDirectory(directory)
        self.path = Path(self.owner.name) / "channel.dat"
        self.rows, self.config = rows, dict(config)
        self.stats = {"short_finite_spans_unchanged": 0, "nonfinite_samples": 0}

    def __del__(self):
        owner = getattr(self, "owner", None)
        if owner is not None:
            owner.cleanup()


class SecondaryFilteredSource:
    """Same lazy read API; each cache is released when its last owner goes away."""
    def __init__(self, base, channels):
        self.base = base
        self.channels = dict(channels)  # actual channel IDs -> independently owned caches

    @property
    def metadata(self):
        return self.base.metadata

    @property
    def loaded(self):
        return self.base.loaded

    @property
    def data(self):
        return getattr(self.base, "data", None)

    def read(self, first, last, columns, step=1):
        meta = self.metadata
        first = max(0, min(int(first), meta.rows)); last = max(first, min(int(last), meta.rows))
        step = max(1, int(step))
        scalar = np.isscalar(columns)
        indexes = np.arange(meta.channels)[columns] if isinstance(columns, slice) else np.atleast_1d(columns)
        result = np.empty((len(range(first, last, step)), len(indexes)), dtype=np.float32)
        for index, column in enumerate(indexes):
            column = int(column)
            cache = self.channels.get(int(meta.channel_ids[column]))
            if cache is None:
                result[:, index] = np.asarray(self.base.read(first, last, [column], step=step)).reshape(-1)
            else:
                mapped = np.memmap(cache.path, dtype=np.float32, mode="r", shape=(meta.rows,))
                try:
                    result[:, index] = mapped[first:last:step]
                finally:
                    mapped._mmap.close()
        return result[:, 0] if scalar else result

    def read_many(self, ranges, columns, step=1):
        return [self.read(a, b, columns, step) for a, b in ranges]

    def state(self):
        return [{"channel": channel, **cache.config, **cache.stats,
                 "scope": "full_loaded_record", "sample_count": int(self.metadata.rows),
                 "duration_sec": self.metadata.rows / self.metadata.fs,
                 "method": "zero_phase_sos_forward_backward_odd_padding"}
                for channel, cache in sorted(self.channels.items())]


def _finite_runs(mapped, chunk, cancel):
    start = None
    for left in range(0, len(mapped), chunk):
        _check(cancel)
        finite = np.isfinite(mapped[left:left+chunk])
        if start is not None and not finite[0]:
            yield start, left
            start = None
        edges = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)))
        for a, b in edges.reshape(-1, 2):
            a, b = int(a)+left, int(b)+left
            if start is not None and a != left:
                yield start, left
                start = None
            if start is None:
                start = a
            if b < left + len(finite):
                yield start, b
                start = None
    if start is not None:
        yield start, len(mapped)


def _zero_phase_span(mapped, first, last, sos, scratch, chunk, cancel, progress=None):
    # Match scipy.signal.sosfiltfilt's default odd pad and steady-state zi.
    taps = 2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    edge = int(3 * taps)
    size = last-first
    if size <= edge:
        return False
    total = size + 2*edge
    forward = np.memmap(scratch, dtype=np.float64, mode="w+", shape=(total,))
    try:
        left_pad = 2*float(mapped[first]) - np.asarray(mapped[first+1:first+edge+1], dtype=np.float64)[::-1]
        right_pad = 2*float(mapped[last-1]) - np.asarray(mapped[last-edge-1:last-1], dtype=np.float64)[::-1]
        zi = signal.sosfilt_zi(sos) * left_pad[0]
        at = 0
        def forward_block(block):
            nonlocal zi, at
            _check(cancel)
            values, zi = signal.sosfilt(sos, np.asarray(block, dtype=np.float64), zi=zi)
            forward[at:at+len(values)] = values
            at += len(values)
            if progress:
                progress(.5*at/total)
        forward_block(left_pad)
        for lo in range(first, last, chunk):
            forward_block(mapped[lo:min(last, lo+chunk)])
        forward_block(right_pad)
        zi = signal.sosfilt_zi(sos) * forward[-1]
        for end in range(total, 0, -chunk):
            _check(cancel)
            begin = max(0, end-chunk)
            values, zi = signal.sosfilt(sos, np.asarray(forward[begin:end][::-1]), zi=zi)
            lo, hi = max(begin, edge), min(end, edge+size)
            if hi > lo:
                block = values[::-1][lo-begin:hi-begin]
                if not np.isfinite(block).all():
                    raise ValueError("滤波产生非有限数值；未提交结果。")
                mapped[first+lo-edge:first+hi-edge] = block
            if progress:
                progress(.5+.5*(total-begin)/total)
    finally:
        forward._mmap.close()
        scratch.unlink(missing_ok=True)
    return True


def _build_channel(base, column, config, sos, directory, chunk, progress, cancel):
    rows = base.metadata.rows
    parent = directory or tempfile.gettempdir()
    if shutil.disk_usage(parent).free < rows * 12 + 10*1024**2:
        raise ValueError("临时磁盘空间不足，无法安全完成整条通道二次滤波。")
    cache = _ChannelCache(directory, rows, config)
    mapped = np.memmap(cache.path, dtype=np.float32, mode="w+", shape=(rows,))
    success = False
    try:
        for first in range(0, rows, chunk):
            _check(cancel)
            last = min(rows, first+chunk)
            values = np.asarray(base.read(first, last, [column]), dtype=np.float32).reshape(-1)
            mapped[first:last] = values
            cache.stats["nonfinite_samples"] += int((~np.isfinite(values)).sum())
            if progress:
                progress(.2*last/max(1, rows))
        filtered = 0
        for first, last in _finite_runs(mapped, chunk, cancel):
            def span_progress(fraction):
                if progress:
                    progress(.2+.8*(first+fraction*(last-first))/max(1, rows))
            if _zero_phase_span(mapped, first, last, sos, Path(cache.owner.name)/"forward.dat", chunk, cancel, span_progress):
                filtered += 1
            else:
                cache.stats["short_finite_spans_unchanged"] += 1
            if progress:
                progress(.2 + .8*last/max(1, rows))
        if not filtered:
            raise ValueError("整条通道没有足够长的连续有效数据，无法完成双向滤波；未提交结果。")
        mapped.flush()
        success = True
    finally:
        mapped._mmap.close()
        if not success:
            cache.owner.cleanup()
    return cache


def _refresh_at_anchors(source, output, affected, progress, cancel):
    """Rebaseline and rebuild selected channels without detecting any new anchor."""
    settings, fs = output["settings"], float(output["fs"])
    records = [dict(row) for row in output["records"]]
    references = [dict(row) for row in output.get("animal_reference_records", [])]
    active = [row for row in records if row.get("channel") in affected and "extreme_local_sample" in row]
    active += [row for row in references if row.get("channel") in affected]
    retained = sum(g["waves"].nbytes for g in output["groups"] + output.get("animal_reference_groups", []))
    samples = int(round(settings["pre_ms"]*fs/1000)) + int(round(settings["post_ms"]*fs/1000)) + 1
    if retained*2 + len(active)*samples*48 > int(settings.get("memory_budget_bytes", 256*1024**2)):
        raise ValueError("更新滤波后的平均超过测试内存预算；原结果保持不变。")
    offset = int(round(source.metadata.time_offset*fs))
    base_first = max(int(round(settings["epoch_start_ms"]*fs/1000)), int(np.ceil(settings["baseline_start_ms"]*fs/1000-1e-8)))
    base_last = min(int(round(settings["epoch_end_ms"]*fs/1000)), int(np.floor(settings["baseline_end_ms"]*fs/1000+1e-8))+1)
    buckets, by_channel = defaultdict(list), defaultdict(list)
    for row in active:
        by_channel[row["channel"]].append(row)
    completed = 0
    for channel, entries in sorted(by_channel.items()):
        column = entries[0]["column"]
        conditioned = _condition_channel(source, column, settings, cancel)
        def read(a, b):
            _check(cancel)
            return _read(source, column, a, b, conditioned)
        trials = {}
        for row in entries:
            _check(cancel)
            key = row["trial_id"]
            if key not in trials:
                center = row["marker_sample"]-offset
                baseline = read(center+base_first, center+base_last)
                baseline = baseline[np.isfinite(baseline)]
                if not len(baseline):
                    raise ValueError(f"ch{channel} Trial {key} 滤波后基线无有效数值。")
                correction = float(np.mean(baseline))
                locator = (trial_localization(read, row, source.metadata.rows, fs, settings, lambda: _check(cancel))
                           if settings.get("localization_mode") == "trend" else None)
                trials[key] = correction, locator
            correction, locator = trials[key]
            row["baseline_correction"] = correction
            if "extreme_local_sample" in row:
                position = row["extreme_local_sample"]
                row["amplitude"] = float(read(position, position+1)[0] - correction)
                row["localization_amplitude"] = float((locator["trend"][position-locator["first"]]
                                                       if locator else read(position, position+1)[0]) - correction)
                search = read(row["search_first"], row["search_last"])
                if locator:
                    lo = row["search_first"]-locator["first"]
                    hi = row["search_last"]-locator["first"]
                    row["despike_count_search"] = int(locator["despike_mask"][lo:hi].sum())
                    extreme = np.nanmax(search) if row["branch"] == "peak" else np.nanmin(search)
                    candidates = np.flatnonzero(np.isfinite(search) & (search == extreme))
                    distances = abs(row["search_first"]+candidates-row["animal_local_sample"])
                    index = candidates[distances == distances.min()][-1]
                    row["raw_extreme_relative_ms"] = (row["search_first"]+index-row["animal_local_sample"])/fs*1000
                    row["raw_extreme_amplitude"] = float(extreme-correction)
                if "automatic_alignment" in row:
                    automatic = dict(row["automatic_alignment"])
                    pos = automatic["extreme_local_sample"]
                    automatic.update(baseline_correction=correction, amplitude=float(read(pos, pos+1)[0]-correction),
                                     localization_amplitude=float((locator["trend"][pos-locator["first"]]
                                                                  if locator else read(pos, pos+1)[0])-correction))
                    row["automatic_alignment"] = automatic
            if row["status"] == "included":
                wave = read(row["excerpt_first"], row["excerpt_last"]).copy()-correction
                wave[~np.isfinite(wave)] = np.nan
                buckets[(channel, row["tag"], row["branch"])].append((row, wave))
            completed += 1
            if progress:
                progress(80+19*completed/max(1,len(active)), f"保留点位更新滤波后平均：{completed}/{len(active)}")
        del conditioned, trials
    key_of = lambda row: (row.get("channel"), row.get("tag"), row.get("trial_id"), row.get("branch"))
    rows_by_key = {key_of(row): row for row in records + references}
    groups = [{**group, "rows": [rows_by_key[key_of(row)] for row in group["rows"]]}
              for group in output["groups"] if group["channel"] not in affected]
    animals = [{**group, "rows": [rows_by_key[key_of(row)] for row in group["rows"]]}
               for group in output.get("animal_reference_groups", []) if group["channel"] not in affected]
    for (channel, tag, branch), entries in sorted(buckets.items()):
        waves = np.stack([wave for _, wave in entries])
        mean, sem, count, correlations = _summarize(waves, cancel)
        if branch != "animal":
            for (row, _), correlation in zip(entries, correlations):
                row["similarity_r"] = correlation
        group = {"channel": channel, "tag": tag, "branch": branch, "waves": waves,
                 "time_ms": np.arange(-int(round(settings["pre_ms"]*fs/1000)), int(round(settings["post_ms"]*fs/1000))+1)/fs*1000,
                 "rows": [row for row, _ in entries], "mean": mean, "sem": sem, "valid_count": count}
        (animals if branch == "animal" else groups).append(group)
    result = {**output, "records": records, "animal_reference_records": references,
              "groups": sorted(groups, key=lambda g: (g["channel"],g["tag"],g["branch"])),
              "animal_reference_groups": sorted(animals, key=lambda g: (g["channel"],g["tag"])),
              "source_identity": source_identity(source)}
    if output.get("judgment_similarity"):
        result["judgment_similarity"] = {**output["judgment_similarity"], "stale": True}
    from animal_peak_letter_vep import invalidate
    return invalidate(result, "二次滤波或恢复改变了信号，请刷新LetterStart VEP")


def apply_secondary_filter(source, output, channel_ids, config=None, progress=None, cancel=None,
                           cache_dir=None, chunk_samples=262144):
    """Transactional apply/restore. None restores from the original loaded source."""
    _check(cancel)
    original_identity = source_identity(source)
    if original_identity != output["source_identity"]:
        raise ValueError("数据源已改变，请重新提取。")
    base = source.base if isinstance(source, SecondaryFilteredSource) else source
    channels = dict(source.channels) if isinstance(source, SecondaryFilteredSource) else {}
    selected = sorted({int(channel) for channel in channel_ids})
    allowed = set(map(int, output["settings"]["channel_ids"]))
    if not selected or not set(selected) <= allowed:
        raise ValueError("二次滤波通道必须属于本次已提取的通道。")
    cfg, sos = filter_config(config, base.metadata.fs) if config is not None else (None, None)
    previous = source.state() if isinstance(source, SecondaryFilteredSource) else []
    chunk = max(32, min(int(chunk_samples), int(output["settings"].get("memory_budget_bytes",256*1024**2))//128))
    directory = cache_dir
    if directory is None and Path(base.metadata.path).is_file() and os.access(Path(base.metadata.path).parent, os.W_OK):
        directory = Path(base.metadata.path).parent
    created = []
    try:
        for number, channel in enumerate(selected):
            _check(cancel)
            if cfg is None:
                channels.pop(channel, None)
            else:
                column = base.metadata.channel_ids.index(channel)
                def report(fraction):
                    if progress:
                        progress(79*(number+fraction)/len(selected), f"整段二次滤波：ch{channel}（读取、正向及反向计算）")
                cache = _build_channel(base, column, cfg, sos, directory, chunk, report, cancel)
                created.append(cache)
                channels[channel] = cache
        candidate = SecondaryFilteredSource(base, channels) if channels else base
        result = _refresh_at_anchors(candidate, output, set(selected), progress, cancel)
        _check(cancel)
        if source_identity(source) != original_identity:
            raise ValueError("滤波期间原数据源已改变；未提交结果。")
        state = candidate.state() if isinstance(candidate, SecondaryFilteredSource) else []
        result["secondary_filter_state"] = state
        result["secondary_filter_history"] = [*output.get("secondary_filter_history", []), {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "action": "restore_loaded_source" if cfg is None else "apply_from_loaded_source",
            "channel_ids": selected, "config": cfg, "previous_state": previous, "new_state": state,
            "meaning": "Full loaded record; existing anchors/Trial choices retained, baselines and means recalculated; original source untouched.",
        }]
        if progress:
            progress(100, "整段二次滤波/恢复已完成，点位保留，平均已更新")
        _check(cancel)
        return candidate, result
    except Exception:
        # Only new private caches lose references; shared, committed caches
        # remain owned by the previous source and are never deleted here.
        channels.clear()
        for cache in created:
            cache.owner.cleanup()
        raise
