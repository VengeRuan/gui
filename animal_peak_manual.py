"""Transactional edits of trial alignment anchors; source arrays stay untouched."""
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

from animal_peak_analysis import _check, _condition_channel, _read, _summarize, source_identity
from animal_peak_localization import trial_localization


def _set_excerpt_bounds(row, position, pre, post, source_rows):
    first, last = position-pre, position+post+1
    row.update(excerpt_first=first, excerpt_last=last,
               crosses_animal_start=last > row["animal_local_sample"], similarity_r=np.nan)
    if first < 0 or last > source_rows:
        row.update(status="excluded", reason_code="excerpt_before_data" if first < 0 else "excerpt_after_data",
                   reason=f"完整截取片段超出源数据边界：本地样本范围=[{first}, {last})，数据范围=[0, {source_rows})；保留定位点，仅排除本分支平均，可缩短截取窗后恢复")
    else:
        row.update(status="included", reason="", reason_code="")


def adjust_animal_peak(source, output, channel, tag, trial_id, branch,
                       relative_ms=None, restore=False, progress=None, cancel=None):
    """Return a new result only after all reads and recomputation succeed.

    No extrema snapping, no new filtering, no trial rejection. Invalid edits
    raise without modifying the last successful result. Other branches and
    AnimalStart references retain their numeric arrays.
    """
    check = lambda: _check(cancel)
    check()
    if source_identity(source) != output["source_identity"]:
        raise ValueError("数据源已改变，请重新提取后再人工调整。")
    if branch not in ("peak", "trough"):
        raise ValueError("请选择峰分支或谷分支。")
    key = (int(channel), int(tag), int(trial_id), branch)
    def record_key(row):
        return row.get("channel"), row.get("tag"), row.get("trial_id"), row.get("branch")
    original = next((row for row in output["records"] if record_key(row) == key
                     and "extreme_local_sample" in row), None)
    if original is None:
        raise ValueError("该 Trial 分支没有可调整的定位点；缺失 AnimalStart 的 Trial 不能通过手动选点纳入。")
    automatic = dict(original.get("automatic_alignment", original))
    fs, meta, settings = float(output["fs"]), source.metadata, output["settings"]
    if restore:
        position = int(automatic["extreme_local_sample"])
    else:
        if relative_ms is None or not np.isfinite(float(relative_ms)):
            raise ValueError("人工位置必须是相对 AnimalStart 的有限毫秒数。")
        position = original["animal_local_sample"] + int(round(float(relative_ms) * fs / 1000))
    if not original["search_first"] <= position < original["search_last"]:
        low = (original["search_first"] - original["animal_local_sample"]) / fs * 1000
        high = (original["search_last"] - original["animal_local_sample"]) / fs * 1000
        raise ValueError(f"人工位置必须在本次搜索范围 [{low:g}, {high:g}) ms 内；未更改原结果。")
    pre = int(round(settings["pre_ms"] * fs / 1000))
    post = int(round(settings["post_ms"] * fs / 1000))
    left, right = position - pre, position + post + 1
    if not restore and (left < 0 or right > meta.rows):
        raise ValueError(f"该位置的完整截取片段 [{left}, {right}) 超出源数据 [0, {meta.rows})；未更改原结果。")
    if progress:
        progress(5, f"人工调整：ch{channel} / Tag {tag} / Trial {trial_id} / {branch}")
    conditioned = _condition_channel(source, original["column"], settings, cancel)
    def read(first, last):
        check()
        values = _read(source, original["column"], first, last, conditioned)
        check()
        if source_identity(source) != output["source_identity"]:
            raise ValueError("人工调整期间数据源已改变，请重新提取。")
        return values
    updated = dict(automatic if restore else original)
    if restore:
        # The automatic anchor is immutable, but its old excerpt length is not:
        # restoring after a window resize must use the CURRENT window settings.
        _set_excerpt_bounds(updated, position, pre, post, meta.rows)
    if not restore:
        raw_value = read(position, position + 1)[0]
        if not np.isfinite(raw_value):
            raise ValueError("该采样点在原波形中不是有效数值，请重新选择位置。")
        locating_value = raw_value
        if settings.get("localization_mode", "raw") == "trend":
            locator = trial_localization(read, original, meta.rows, fs, settings, check)
            locating_value = locator["trend"][position - locator["first"]]
            if not np.isfinite(locating_value):
                raise ValueError("该位置的趋势值无效，请重新选择位置。")
        offset = int(round(float(meta.time_offset) * fs))
        flags = [item for item in original.get("flags", "").split("; ")
                 if item and item not in ("搜索边界极值", "人工定位（不要求局部极值）")]
        flags.append("人工定位（不要求局部极值）")
        updated.update(extreme_local_sample=position, extreme_sample=position + offset,
                       extreme_sec=(position + offset) / fs,
                       extreme_relative_ms=(position - original["animal_local_sample"]) / fs * 1000,
                       amplitude=float(raw_value - original["baseline_correction"]),
                       localization_amplitude=float(locating_value - original["baseline_correction"]),
                       excerpt_first=left, excerpt_last=right,
                       crosses_animal_start=right > original["animal_local_sample"],
                       flags="; ".join(flags), status="included", reason="", reason_code="")
    updated.update(automatic_alignment=automatic,
                   automatic_relative_ms=automatic["extreme_relative_ms"],
                   automatic_sample=automatic["extreme_sample"],
                   alignment_origin="automatic" if restore else "manual",
                   manual_edit_count=int(original.get("manual_edit_count", 0)) + 1)
    # Only dictionaries are copied globally; unchanged numerical arrays are
    # shared read-only. Correlations of affected peers are updated on new rows.
    records = [dict(updated if record_key(row) == key else row) for row in output["records"]]
    rows_by_key = {record_key(row): row for row in records}
    group_key = (int(channel), int(tag), branch)
    previous = next((group for group in output["groups"]
                     if (group["channel"], group["tag"], group["branch"]) == group_key), None)
    selected = [row for row in records if row.get("status") == "included"
                and (row.get("channel"), row.get("tag"), row.get("branch")) == group_key]
    groups = [{**group, "rows": [rows_by_key[record_key(row)] for row in group["rows"]]}
              for group in output["groups"]
              if (group["channel"], group["tag"], group["branch"]) != group_key]
    if selected:
        samples = pre + post + 1
        retained = sum(group["waves"].nbytes for group in output["groups"] + output.get("animal_reference_groups", []))
        if retained + len(selected) * samples * 48 > int(settings.get("memory_budget_bytes", 256 * 1024 * 1024)):
            raise ValueError("重新生成平均所需内存超过测试预算；原结果及人工记录保持不变。")
        old_waves = {row["trial_id"]: (row, wave) for row, wave in zip(previous["rows"], previous["waves"])} if previous else {}
        waves = np.empty((len(selected), samples), dtype=np.float32)
        for index, row in enumerate(selected):
            check()
            old = old_waves.get(row["trial_id"])
            if record_key(row) != key and old is not None:
                waves[index] = old[1]
            else:
                wave = read(row["excerpt_first"], row["excerpt_last"]).copy() - row["baseline_correction"]
                wave[~np.isfinite(wave)] = np.nan
                waves[index] = wave
            if progress:
                progress(10 + 60 * (index + 1) / len(selected), f"更新平均：{index + 1}/{len(selected)} 个 Trial")
        mean, sem, count, correlations = _summarize(waves, cancel)
        for row, correlation in zip(selected, correlations):
            row["similarity_r"] = correlation
        groups.append({"channel": int(channel), "tag": int(tag), "branch": branch,
                       "time_ms": np.arange(-pre, post + 1, dtype=float) / fs * 1000,
                       "waves": waves, "mean": mean, "sem": sem, "valid_count": count, "rows": selected})
    groups.sort(key=lambda group: (group["channel"], group["tag"], group["branch"]))
    history = list(output.get("manual_alignment_history", []))
    history.append({"revision": len(history) + 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "channel": int(channel), "tag": int(tag), "trial_id": int(trial_id), "branch": branch,
                    "action": "restore_automatic" if restore else "set_manual",
                    "requested_relative_ms": None if restore else float(relative_ms),
                    "previous_relative_ms": original["extreme_relative_ms"],
                    "new_relative_ms": updated["extreme_relative_ms"],
                    "automatic_relative_ms": automatic["extreme_relative_ms"],
                    "new_global_sample": updated["extreme_sample"]})
    check()
    if source_identity(source) != output["source_identity"]:
        raise ValueError("数据源已改变；未提交本次人工调整。")
    if progress:
        progress(100, "人工点位及平均响应已更新")
    check()
    result = {**output, "records": records, "groups": groups, "manual_alignment_history": history}
    if output.get("judgment_similarity"):
        result["judgment_similarity"] = {**output["judgment_similarity"], "stale": True}
    return result


def resize_animal_peak_windows(source, output, pre_ms, post_ms, progress=None, cancel=None):
    """Re-extract every existing anchor, never redetect or change manual points.

    Both peak/trough averages and AnimalStart reference windows use the new
    lengths. Invalid boundary excerpts retain their anchors and explicit reason;
    shortening later can recover them. Commit only after successful completion.
    """
    check = lambda: _check(cancel)
    check()
    if source_identity(source) != output["source_identity"]:
        raise ValueError("数据源已改变，不能使用旧点位刷新截取窗。")
    pre_ms, post_ms = float(pre_ms), float(post_ms)
    if not np.isfinite(pre_ms) or not np.isfinite(post_ms) or min(pre_ms, post_ms) < 0:
        raise ValueError("特征点前/后截取长度必须是非负有限数值。")
    fs = float(output["fs"])
    pre, post = int(round(pre_ms*fs/1000)), int(round(post_ms*fs/1000))
    samples = pre+post+1
    if samples < 3:
        raise ValueError("参与平均的片段至少需要 3 个采样点。")
    settings = {**output["settings"], "pre_ms": pre_ms, "post_ms": post_ms}
    records = [dict(row) for row in output["records"]]
    references = [dict(row) for row in output.get("animal_reference_records", [])]
    anchored = [row for row in records if "extreme_local_sample" in row]
    entries = anchored + references
    retained = sum(g["waves"].nbytes*2 for g in output["groups"] + output.get("animal_reference_groups", []))
    for group in output.get("judgment_similarity", {}).get("groups", []):
        retained += sum(group[key].nbytes for key in ("matrix", "counts", "coverage", "reason_codes"))
        retained += sum(trial["trend"].nbytes for trial in group["trials"] if "trend" in trial)
    if retained + len(entries)*samples*48 > int(settings.get("memory_budget_bytes", 256*1024*1024)):
        raise ValueError("新截取窗及平均计算超过测试内存预算，请缩短截取长度或减少通道；原结果及人工点位未修改。")
    by_column = defaultdict(list)
    for row in entries:
        position = row.get("extreme_local_sample", row["animal_local_sample"])
        _set_excerpt_bounds(row, int(position), pre, post, source.metadata.rows)
        by_column[int(row["column"])].append(row)
    buckets = defaultdict(list)
    completed = 0
    for column, rows in sorted(by_column.items()):
        check()
        conditioned = _condition_channel(source, column, settings, cancel)
        for row in rows:
            check()
            if source_identity(source) != output["source_identity"]:
                raise ValueError("刷新期间数据源已改变，未提交截取窗修改。")
            if row["status"] == "included":
                wave = _read(source, column, row["excerpt_first"], row["excerpt_last"], conditioned).copy()
                check()
                wave -= row["baseline_correction"]
                wave[~np.isfinite(wave)] = np.nan
                buckets[(row["channel"], row["tag"], row["branch"])].append((row, wave))
            completed += 1
            if progress:
                progress(85*completed/max(1,len(entries)), f"保留点位刷新截取：{completed}/{len(entries)}")
        del conditioned
    groups, animal_groups = [], []
    for number, ((channel, tag, branch), items) in enumerate(sorted(buckets.items())):
        check()
        waves = np.stack([wave for _, wave in items])
        mean, sem, count, correlations = _summarize(waves, cancel)
        if branch != "animal":
            for (row, _), correlation in zip(items, correlations):
                row["similarity_r"] = correlation
        group = {"channel": channel, "tag": tag, "branch": branch,
                 "time_ms": np.arange(-pre, post+1, dtype=float)/fs*1000,
                 "waves": waves, "mean": mean, "sem": sem, "valid_count": count,
                 "rows": [row for row, _ in items]}
        (animal_groups if branch == "animal" else groups).append(group)
        if progress:
            progress(85+14*(number+1)/max(1,len(buckets)), f"更新平均：ch{channel} / Tag {tag} / {branch}")
    check()
    if source_identity(source) != output["source_identity"]:
        raise ValueError("刷新期间数据源已改变，未提交截取窗修改。")
    history = [*output.get("window_resize_history", []), {
        "revision": len(output.get("window_resize_history", []))+1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "previous_pre_ms": output["settings"]["pre_ms"], "previous_post_ms": output["settings"]["post_ms"],
        "pre_ms": pre_ms, "post_ms": post_ms, "pre_samples": pre, "post_samples": post,
        "meaning": "Existing automatic/manual anchors retained; no peak detection or baseline change.",
    }]
    result = {**output, "settings": settings, "records": records, "groups": groups,
              "animal_reference_records": references, "animal_reference_groups": animal_groups,
              "window_resize_history": history}
    if output.get("judgment_similarity"):
        result["judgment_similarity"] = {**output["judgment_similarity"], "stale": True}
    check()
    if progress:
        progress(100, "已保留全部点位，按新截取长度更新平均")
    check()
    return result
