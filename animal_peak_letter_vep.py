"""Event-aligned averages with shared Trial choices; never use peak anchors."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv

import numpy as np

from animal_peak_analysis import _animal_exclusion, _check, _condition_channel, _read, _summarize, source_identity


ALIGNMENTS = {"letter": "LetterStart", "dot": "Dot", "animal": "AnimalStart"}
EVENT_COLORS = {"dot": "#555555", "letter": "#c62828", "animal": "#9467bd"}
EVENT_MARKER_HINT = "细线：Dot灰 / Letter红 / Animal紫"
DEFAULTS = dict(alignment="letter", start_ms=0., end_ms=6000., baseline_enabled=True,
                baseline_start_ms=-1000., baseline_end_ms=0.)


def parameters(values=None):
    result = {**DEFAULTS, **(values or {})}
    mode = str(result["alignment"]).lower()
    if mode not in ALIGNMENTS:
        raise ValueError("对齐方式须为LetterStart、Dot或AnimalStart。")
    result["alignment"] = mode
    if mode == "animal":
        if "start_ms" not in (values or {}): result["start_ms"] = -6000.
        if "end_ms" not in (values or {}): result["end_ms"] = 0.
    for key in ("start_ms", "end_ms", "baseline_start_ms", "baseline_end_ms"):
        result[key] = float(result[key])
        if not np.isfinite(result[key]):
            raise ValueError("事件对齐VEP时间参数必须为有限数值。")
    result["baseline_enabled"] = bool(result["baseline_enabled"])
    if result["start_ms"] >= result["end_ms"]:
        raise ValueError("事件对齐VEP截取起点必须小于终点。")
    if mode == "animal" and result["end_ms"] > 0:
        raise ValueError("AnimalStart模式只看事件前的数据，截取终点须≤0 ms。")
    if result["baseline_enabled"] and not result["baseline_start_ms"] < result["baseline_end_ms"] <= 0:
        raise ValueError("事件对齐VEP基线须满足：起点 < 终点 ≤ 0 ms。")
    return result


def baseline_reference(config):
    return "Dot" if config.get("alignment", "letter") == "dot" else "LetterStart"


def configurations(state):
    """Only export modes actually refreshed; support older Letter-only states."""
    return state.get("configurations") or {state["parameters"].get("alignment", "letter"): parameters(state["parameters"])}


def new_state(markers, matches):
    return dict(parameters=parameters(), markers=np.asarray(markers, dtype=int).copy(),
                matches={int(key): dict(value) for key, value in matches.items()},
                selections=[], history=[], revision=0, cache=None, stale=False)


def available_pairs(output, state):
    wanted = set(output["settings"].get("tags") or (4, 5, 6, 7))
    tags = {int(tag) for _, tag in state["markers"] if int(tag) in wanted & {4, 5, 6, 7}}
    return [(int(channel), tag) for channel in output["settings"]["channel_ids"] for tag in sorted(tags)]


def selected_ids(state, pair):
    entry = next((entry for entry in state["selections"] if (entry["channel"], entry["tag"]) == tuple(pair)), None)
    return None if entry is None else tuple(entry["trial_ids"])


def trial_catalog(source, output, state, pair, config=None):
    """Check the chosen anchor independently when AnimalStart is unavailable."""
    cfg = parameters(config or state["parameters"])
    mode = cfg["alignment"]; name = ALIGNMENTS[mode]
    channel, wanted_tag = map(int, pair)
    if (channel, wanted_tag) not in available_pairs(output, state):
        raise ValueError("该通道/Tag不属于当前已完成测试的通道及Letter范围。")
    meta, settings = source.metadata, output["settings"]
    fs = float(meta.fs); offset = round(meta.time_offset*fs)
    column = list(map(int, meta.channel_ids)).index(channel)
    first, last = round(cfg["start_ms"]*fs/1000), round(cfg["end_ms"]*fs/1000)+1
    base_first = int(np.ceil(cfg["baseline_start_ms"]*fs/1000-1e-8))
    base_last = int(np.floor(cfg["baseline_end_ms"]*fs/1000+1e-8))+1
    if last-first < 2 or (cfg["baseline_enabled"] and base_last <= base_first):
        raise ValueError("当前采样率下，VEP或基线时间窗过短。")
    chosen = selected_ids(state, pair)
    counts, rows = defaultdict(int), []
    markers = sorted((int(sample), int(tag)) for sample, tag in state["markers"] if int(tag) in {4,5,6,7})
    limit = int(settings.get("trial_limit", 0))
    for trial_id, (sample, tag) in enumerate(markers, 1):
        counts[tag] += 1
        if tag != wanted_tag or (limit and counts[tag] > limit):
            continue
        event = state["matches"].get(sample)
        try:
            latency = float(event.get("animal_latency_sec", np.nan)) if event else np.nan
        except (ValueError, TypeError):
            latency = np.nan
        code, reason = ("behavior_unmatched", "无对应视频行为记录，无法确认有效AnimalStart") if event is None else _animal_exclusion(event, latency)
        if not reason:
            animal = round(sample+latency*fs)-offset
            if not settings["epoch_start_ms"] <= latency*1000 < settings["epoch_end_ms"]:
                code, reason = "animal_outside_epoch", "AnimalStart不在原Letter Epoch内"
            elif not 0 <= animal < meta.rows:
                code, reason = "animal_outside_source", "AnimalStart超出当前源数据范围"
        animal_valid = not bool(reason)
        animal_code, animal_reason = code, reason
        validity = str(event.get("validity") or "").strip().upper() if event else ""
        animal_missing = (not np.isfinite(latency) and validity in {"", "Y", "NA", "NAN", "NONE"}
                          and code not in {"animal_frame_equal_start", "animal_frame_before_start"})
        selection_eligible = animal_valid or animal_missing
        if mode != "animal" and animal_missing:
            code, reason = "", ""  # Letter/Dot have their own independent anchors.
        anchor_sample = sample
        anchor_event_sec = sample/fs
        if mode == "animal":
            anchor_event_sec = sample/fs+latency
            anchor_sample = round(sample+latency*fs) if np.isfinite(latency) else None
        elif mode == "dot":
            try: dot_sec = float(event.get("tag100_start_eeg_sec", np.nan)) if event else np.nan
            except (TypeError, ValueError): dot_sec = np.nan
            valid_dot = bool(event and event.get("tag100_source") == "csv" and event.get("tag100_status") == "csv" and np.isfinite(dot_sec))
            anchor_sample = round(dot_sec*fs) if valid_dot else None
            anchor_event_sec = dot_sec if valid_dot else np.nan
            if not reason and not valid_dot:
                code, reason = "dot_time_missing", "CSV Dot开始时间缺失或无效（M列），不使用TXT Tag100替代"
        baseline_sample = anchor_sample if mode == "dot" else sample
        center = anchor_sample-offset if anchor_sample is not None else None
        baseline_center = baseline_sample-offset if baseline_sample is not None else None
        row = dict(trial_id=trial_id, tag_trial_id=counts[tag], channel=channel, tag=tag, column=column,
                   marker_sample=sample, marker_sec=sample/fs, animal_latency_ms=latency*1000,
                   alignment=mode, alignment_sample=anchor_sample, alignment_event_sec=anchor_event_sec,
                   alignment_sec=anchor_sample/fs if anchor_sample is not None else np.nan,
                   baseline_reference=baseline_reference(cfg), baseline_anchor_sample=baseline_sample,
                   animal_valid=animal_valid, animal_missing=animal_missing,
                   animal_reason_code=animal_code, animal_reason=animal_reason,
                   selection_eligible=selection_eligible,
                   average_selected=selection_eligible and (chosen is None or trial_id in chosen),
                   first=center+first if center is not None else None, last=center+last if center is not None else None,
                   baseline_first=baseline_center+base_first if baseline_center is not None else None,
                   baseline_last=baseline_center+base_last if baseline_center is not None else None,
                   baseline_correction=0., reason_code=code, reason=reason)
        if event:
            row.update({key: value for key, value in event.items() if key.startswith("tag100_")})
        if not reason and not (0 <= row["first"] < row["last"] <= meta.rows):
            row.update(reason_code="letter_window_outside_source", reason=f"{name} VEP截取窗超出源数据；不补零、不截短")
        if not row["reason"] and cfg["baseline_enabled"] and not (0 <= row["baseline_first"] < row["baseline_last"] <= meta.rows):
            row.update(reason_code="letter_baseline_outside_source", reason=f"{baseline_reference(cfg)} VEP基线窗超出源数据；不使用不完整基线")
        row["status"] = "excluded" if row["reason"] else "included"
        rows.append(row)
    return rows, np.arange(first, last, dtype=np.float64)/fs*1000


def selection_eligible(row):
    """Shared choices survive switching to a mode whose anchor is missing."""
    return row.get("selection_eligible", row.get("animal_valid", False))


def trial_status_text(row):
    if row["reason"]:
        return row["reason"]
    if row.get("animal_missing"):
        return f"AnimalStart缺失：{row['animal_reason']}；本模式仍可计算"
    return "可用"


def animal_plot_unavailable(group):
    """Missing anchors are not a drawable zero; keep legacy explicit-empty previews."""
    rows = group["records"]
    return (group["parameters"].get("alignment") == "animal"
            and not any(r["animal_valid"] and r["average_selected"] for r in rows)
            and (any(r["average_selected"] for r in rows) or not any(r["animal_valid"] for r in rows)))


def set_selection(state, group, trial_ids):
    pair = (group["channel"], group["tag"])
    available = {row["trial_id"] for row in group["records"] if selection_eligible(row)}
    chosen = available if trial_ids is None else set(map(int, trial_ids))
    if chosen-available:
        raise ValueError("选择包含不属于当前通道/Tag或明确无效的Trial。")
    value = None if chosen == available else tuple(sorted(chosen))
    if value == selected_ids(state, pair):
        return state
    entries = [entry for entry in state["selections"] if (entry["channel"], entry["tag"]) != pair]
    if value is not None:
        entries.append(dict(channel=pair[0], tag=pair[1], trial_ids=list(value)))
    return {**state, "selections": entries, "cache": None, "revision": state["revision"]+1,
            "history": [*state["history"], dict(action="trial_selection", channel=pair[0], tag=pair[1],
                trial_ids=None if value is None else list(value), timestamp_utc=datetime.now(timezone.utc).isoformat())]}


def calculate_group(source, output, state, pair, config=None, progress=None, cancel=None):
    _check(cancel)
    identity = source_identity(source)
    if identity != output["source_identity"]:
        raise ValueError("数据源已改变，请重新提取后计算LetterStart VEP。")
    cfg = parameters(config or state["parameters"])
    records, time_ms = trial_catalog(source, output, state, pair, cfg)
    available = [row for row in records if row["status"] == "included"]
    budget = int(output["settings"].get("memory_budget_bytes", 256*1024**2))
    retained = sum(group["waves"].nbytes for group in output["groups"]+output.get("animal_reference_groups", []))
    if state.get("cache") is not None:
        retained += state["cache"]["waves"].nbytes
    processing = output["settings"].get("processing") or {}
    conditioning_bytes = source.metadata.rows*32 if any(processing.get(key) for key in ("analysis_notch", "analysis_bandpass", "smooth", "zscore")) else 0
    if retained + len(available)*len(time_ms)*4*8 + len(time_ms)*64 + conditioning_bytes > budget:
        raise ValueError("LetterStart VEP超过内存预算；请缩短截取窗或减少Trial数量后重新提取。原结果未改变。")
    # Same loaded/secondary-filtered waveform and committed task conditioning,
    # never the disposable localization trend and never a peak anchor.
    column = list(map(int, source.metadata.channel_ids)).index(int(pair[0]))
    conditioned = _condition_channel(source, column, {**output["settings"], "memory_budget_bytes": budget}, cancel) if available else None
    waves = np.empty((len(available), len(time_ms)), dtype=np.float32)
    kept = []
    for index, row in enumerate(available):
        _check(cancel)
        correction = 0.
        if cfg["baseline_enabled"]:
            baseline = _read(source, column, row["baseline_first"], row["baseline_last"], conditioned)
            if not np.isfinite(baseline).all():
                row.update(status="excluded", reason_code="baseline_nonfinite", reason=f"{baseline_reference(cfg)}基线含NaN/Inf；不使用不完整基线")
                continue
            correction = float(np.mean(baseline, dtype=np.float64))
        wave = _read(source, column, row["first"], row["last"], conditioned)
        if not np.isfinite(wave).all():
            row.update(status="excluded", reason_code="wave_nonfinite", reason=f"{ALIGNMENTS[cfg['alignment']]}截取波形含NaN/Inf；不补值")
            continue
        row["baseline_correction"] = correction
        waves[len(kept)] = wave-correction
        kept.append(row)
        if progress:
            progress((index+1)/max(1,len(available))*95, f"{ALIGNMENTS[cfg['alignment']]} VEP：ch{pair[0]} / Tag {pair[1]} / 原Trial {row['trial_id']}")
    # Do not keep the backing allocation for excluded/non-finite Trials alive.
    waves = waves[:len(kept)].copy() if len(kept) < len(available) else waves
    indices = [index for index, row in enumerate(kept) if row["average_selected"]]
    if indices:
        mean, sem, count, _ = _summarize(waves[indices], cancel)
    else:
        mean = np.full(len(time_ms), np.nan, dtype=np.float32); sem = mean.copy()
        count = np.zeros(len(time_ms), dtype=np.int64)
    for row in records:
        row["average_included"] = row["status"] == "included" and row["average_selected"]
    _check(cancel)
    if source_identity(source) != identity:
        raise ValueError("计算期间数据源改变；未提交LetterStart VEP结果。")
    return dict(channel=int(pair[0]), tag=int(pair[1]), parameters=cfg, source_identity=identity,
                records=records, rows=kept, time_ms=time_ms, waves=waves, selected_indices=indices,
                mean=mean, sem=sem, valid_count=count)


def refresh(source, output, markers, matches, pair, config, trial_ids="unchanged", progress=None, cancel=None):
    state = output.get("letter_vep") or new_state(markers, matches)
    saved_modes = configurations(state) if output.get("letter_vep") else {}
    cfg = parameters(config)
    if trial_ids != "unchanged":
        rows, _ = trial_catalog(source, output, state, pair, cfg)
        state = set_selection(state, dict(channel=pair[0], tag=pair[1], records=rows), trial_ids)
    group = calculate_group(source, output, state, pair, cfg, progress, cancel)
    revision = state["revision"]+1
    modes = {**saved_modes, cfg["alignment"]: cfg}
    state = {**state, "parameters": cfg, "cache": group, "stale": False, "revision": revision,
             "configurations": modes,
             "history": [*state["history"], dict(action="refresh", parameters=cfg, channel=pair[0], tag=pair[1],
                 timestamp_utc=datetime.now(timezone.utc).isoformat())]}
    return {**output, "letter_vep": state}


def invalidate(output, reason):
    if output.get("letter_vep") is None:
        return output
    return {**output, "letter_vep": {**output["letter_vep"], "cache": None, "stale": True, "stale_reason": reason}}


def manifest(state):
    if not state:
        return {"status": "not_calculated"}
    return dict(status="calculated", parameters=state["parameters"], selections=state["selections"],
                configurations=configurations(state), history=state["history"], revision=state["revision"],
                alignment="letter=CSV N corrected marker; dot=CSV same-row M corrected time (never TXT Tag100); animal=matched AnimalStart. Anchors rounded to source samples, no extrema/manual-anchor alignment.",
                baseline="Dot alignment: relative to Dot. Letter and AnimalStart alignment: relative to LetterStart. Default -1000..0 ms, optional; each saved mode records its applied offsets.",
                eligibility="Animal mode requires valid AnimalStart inside the original Letter Epoch and source. Missing AnimalStart alone does not exclude Letter/Dot modes; explicit invalidity and known bad Animal timing remain excluded. Each mode requires its own anchor, complete source and optional baseline windows; no fabricated anchors, peak/search/similarity requirement.",
                waveform="Current loaded / secondary-filtered source with committed task conditioning; no localization trend; optional per-Trial baseline subtraction",
                selection="One channel/Tag Trial list SHARED by all three event alignments; unspecified=all eligible, explicit empty=none. Independent of peak/trough and legacy AnimalStart reference averages.",
                event_time_markers="One thin line per actually averaged Trial and other event within the displayed sample-time window. Letter=marker, Dot=CSV M only, Animal=matched latency; all rounded to source samples relative to the current alignment sample. The aligned event has one shared zero line. No changes to waves, selection, baseline or mean.",
                export="All successfully refreshed modes and all channel/Tag groups; shared selection applied to every mode at export. Legacy letter_start_vep directory/key retained for compatibility; mode-specific file names.")


def reference_span_ms(source, output, pair, trial_id=None):
    """Display metadata only: no signal read, conditioning, or mean changes."""
    rows = [row for row in output["records"] if (row.get("channel"), row.get("tag")) == tuple(pair)
            and "extreme_local_sample" in row]
    if rows:
        chosen = trial_id if any(row["trial_id"] == trial_id for row in rows) else min(row["trial_id"] for row in rows)
        rows = [row for row in rows if row["trial_id"] == chosen]
        try:
            from animal_peak_analysis import _trial_detail_bounds
            first, last = _trial_detail_bounds(source, output, rows)
            span = max(1., (last-first-1)/float(output["fs"])*1000)
            # Same approximate 5% right / 10% two-sided margin as trial plots.
            from animal_peak_plotting import trial_display_mode
            margin = (1. if trial_display_mode(output) == "dot_animal" else
                      1.05 if output.get("display_settings", {}).get("letter_before_ms") is not None else 1.10)
            return span*margin, chosen
        except (KeyError, ValueError):
            pass
    settings = output["settings"]
    return max(1., float(settings["epoch_end_ms"])-float(settings["epoch_start_ms"])), None


def compact_plot_size(duration_ms, reference_ms, cell_width, cell_height, padding=85., minimum_width=260.):
    """Match the horizontal footprint of a trial segment, within readable bounds.

    Units may be logical pixels or inches, provided all size inputs agree.
    The width never exceeds one trial cell; no data is sliced here.
    """
    cell_width = max(1., float(cell_width))
    target = padding+max(1e-6, cell_width-padding)*max(0., float(duration_ms))/max(1e-6, float(reference_ms))
    return min(cell_width, max(minimum_width, target)), float(cell_height)


def event_time_markers(group, fs):
    """Display-only timestamps shared by the Qt plot and saved figures.

    Sample differences deliberately cancel the loaded source's time offset.
    Only selected, successfully read waves can contribute markers; absent
    CSV Dot metadata suppresses that marker, never the Trial's other events.
    """
    mode = group["parameters"].get("alignment", "letter")
    low, high = float(group["time_ms"][0]), float(group["time_ms"][-1])
    fs = float(fs)
    markers = []
    for index in group["selected_indices"]:
        row = group["rows"][index]
        if row.get("status", "included") != "included" or not row.get("average_included", True):
            continue
        try:
            letter_sample = float(row["marker_sample"])
            if not np.isfinite(letter_sample):
                continue
            samples = {"letter": round(letter_sample)}
            try:
                latency = float(row.get("animal_latency_ms", np.nan))/1000
                if row.get("animal_valid", True) and np.isfinite(latency):
                    samples["animal"] = round(letter_sample+latency*fs)
            except (TypeError, ValueError):
                pass
            if row.get("tag100_source") == "csv" and row.get("tag100_status") == "csv":
                try:
                    dot_sec = float(row["tag100_start_eeg_sec"])
                    if np.isfinite(dot_sec):
                        samples["dot"] = round(dot_sec*fs)
                except (KeyError, TypeError, ValueError):
                    pass
            anchor = row.get("alignment_sample", samples.get(mode))
            if anchor is None or not np.isfinite(anchor):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        for event in ("dot", "letter", "animal"):
            if event == mode or event not in samples:
                continue
            relative_ms = (samples[event]-anchor)/fs*1000
            if low <= relative_ms <= high:
                markers.append(dict(event=event, trial_id=row["trial_id"],
                                    relative_ms=float(relative_ms), sample=samples[event]))
    return markers


def draw_group(axis, output, group):
    from animal_peak_plotting import apply_waveform_y_limits, waveform_display
    scale, unit = waveform_display(output)
    t, indices = group["time_ms"], group["selected_indices"]
    if indices:
        axis.plot(t, group["waves"][indices].T*scale, color=".6", alpha=.22, linewidth=.5)
        axis.plot(t, group["mean"]*scale, color="#b32025", linewidth=1.8)
    cfg = group["parameters"]
    mode = cfg.get("alignment", "letter")
    for marker in event_time_markers(group, output["fs"]):
        line = axis.axvline(marker["relative_ms"], color=EVENT_COLORS[marker["event"]],
                           linewidth=.6, linestyle="--", alpha=.55, zorder=1.5)
        line._vep_event_marker = marker
        line.set_gid(f"trial_event:{marker['trial_id']}:{marker['event']}")
    zero = axis.axvline(0, color=EVENT_COLORS[mode], linewidth=1, zorder=1.6)
    zero.set_gid("event_alignment_zero")
    name = ALIGNMENTS[cfg["alignment"]]
    baseline = f"基线相对{baseline_reference(cfg)} {cfg['baseline_start_ms']:g}～{cfg['baseline_end_ms']:g} ms" if cfg["baseline_enabled"] else "未做基线校正"
    axis.set(title=f"ch{group['channel']} | Tag {group['tag']} | n={len(indices)}\n{name}对齐 VEP" + ("（无所选可用Trial）" if not indices else "") + f"\n{baseline}\n{EVENT_MARKER_HINT}",
             xlabel=f"相对 {name}（ms）", ylabel=unit, xlim=(t[0], t[-1]))
    axis.title.set_fontsize(8)
    axis.grid(alpha=.15)
    display = output.get("display_settings", {})
    apply_waveform_y_limits(axis, {**output, "display_settings": display.get("event_vep_y_axis", display)})


def export_results(source, output, folder, combined_pdf=None, progress=None, cancel=None):
    """Stream one channel/Tag at a time; export every group, not just the viewer."""
    state = output.get("letter_vep")
    if not state:
        return
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_pdf import PdfPages
    from animal_peak_plotting import PLOT_STYLE
    folder = Path(folder)/"letter_start_vep"; folder.mkdir()
    pairs = available_pairs(output, state)
    jobs = [(pair, config) for config in configurations(state).values() for pair in pairs]
    with (folder/"trial_details.csv").open("w", newline="", encoding="utf-8-sig") as handle, rc_context(PLOT_STYLE), PdfPages(folder/"all_letter_start_vep.pdf") as pdf:
        writer = None
        for index, (pair, config) in enumerate(jobs):
            _check(cancel)
            group = calculate_group(source, output, state, pair, config, cancel=cancel)
            # Fixed columns across groups, preserving all source-row annotations.
            if writer is None:
                fields = sorted(set().union(*(row.keys() for row in group["records"]))) if group["records"] else []
                # Other groups can contain optional Tag100 provenance not present here.
                for event in state["matches"].values():
                    fields = sorted(set(fields) | {key for key in event if key.startswith("tag100_")})
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            writer.writerows(group["records"])
            mode = group["parameters"]["alignment"]
            name = f"ch{pair[0]}_tag{pair[1]}_{mode}_start"
            np.savez_compressed(folder/(name+".npz"), time_ms=group["time_ms"], trials=group["waves"],
                mean=group["mean"], sem=group["sem"], valid_count=group["valid_count"],
                trial_ids=np.asarray([row["trial_id"] for row in group["rows"]], dtype=int),
                tag_trial_ids=np.asarray([row["tag_trial_id"] for row in group["rows"]], dtype=int),
                marker_samples=np.asarray([row["marker_sample"] for row in group["rows"]], dtype=np.int64),
                alignment=mode, alignment_samples=np.asarray([row["alignment_sample"] for row in group["rows"]], dtype=np.int64),
                alignment_event_sec=np.asarray([row["alignment_event_sec"] for row in group["rows"]]),
                baseline_reference=baseline_reference(group["parameters"]),
                baseline_anchor_samples=np.asarray([row["baseline_anchor_sample"] for row in group["rows"]], dtype=np.int64),
                selected_trial_ids=np.asarray([group["rows"][i]["trial_id"] for i in group["selected_indices"]], dtype=int),
                baseline_corrections=np.asarray([row["baseline_correction"] for row in group["rows"]]),
                channel=pair[0], tag=pair[1], units=output["units"])
            with (folder/(name+"_average.csv")).open("w", newline="", encoding="utf-8-sig") as average:
                csv_writer = csv.writer(average); csv_writer.writerow(["time_ms", "mean", "sem", "valid_count", "unit", "channel", "tag", "alignment"])
                for n, values in enumerate(zip(group["time_ms"], group["mean"], group["sem"], group["valid_count"])):
                    if n % 1024 == 0: _check(cancel)
                    csv_writer.writerow([*values, output["units"], *pair, mode])
            if mode == "animal" and not group["selected_indices"]:
                # Keep the audit data/reasons, but do not save an empty Animal
                # figure or add a placeholder page to either combined PDF.
                if progress:
                    progress((index+1)/max(1,len(jobs))*100, f"跳过AnimalStart图：ch{pair[0]} / Tag {pair[1]}（无有效AnimalStart）")
                del group
                continue
            span, _ = reference_span_ms(source, output, pair)
            size = compact_plot_size(np.ptp(group["time_ms"]), span, 6, 5, padding=.85, minimum_width=2.8)
            plot_display = output.get("display_settings", {}).get("event_vep_plot", {})
            if plot_display.get("width_mode") == "manual":
                # Qt logical pixels use a 96 px/in reference, independent of
                # monitor scaling. Keep the last applied frame aspect ratio.
                size = (float(plot_display["width_px"])/96, float(plot_display["height_px"])/96)
            fig = Figure(figsize=size, layout="constrained"); FigureCanvasAgg(fig)
            from textwrap import fill
            fig.suptitle(fill(Path(output["source"]["path"]).name, width=max(24,int(size[0]*9))), fontsize=8)
            draw_group(fig.subplots(), output, group)
            fig.savefig(folder/(name+".png"), dpi=220); fig.savefig(folder/(name+".pdf")); pdf.savefig(fig)
            if combined_pdf is not None: combined_pdf.savefig(fig)
            fig.clear()
            if progress: progress((index+1)/max(1,len(jobs))*100, f"导出{ALIGNMENTS[mode]} VEP：ch{pair[0]} / Tag {pair[1]}")
            del group  # Release this group's long Trial windows before reading the next.
    _check(cancel)
