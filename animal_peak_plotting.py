"""Shared trial-detail presentation for the Qt viewer and PDF/PNG export.

This module only draws supplied traces and extrema; it performs no analysis.
"""
import numpy as np


PLOT_STYLE = {
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "pdf.fonttype": 42,
}


def waveform_display(output):
    """Match preprocessing's numeric display, without a voltage conversion.

    The μV label is the requested preview convention, not a recalibration of
    the source or of the numerical CSV/NPZ units.
    """
    return 1, "μV" if output["units"] == "mV" else "z"


def letter_display_before_ms(output):
    value = output.get("display_settings", {}).get("letter_before_ms")
    if value is None:
        return None  # Compatibility with results without the new display option.
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Letter 开始前显示长度必须为非负有限数值。")
    return value


def trial_display_mode(output):
    """Old saved results retain their original Letter-based display."""
    mode = output.get("display_settings", {}).get("trial_range_mode", "letter")
    if mode not in {"letter", "dot_animal"}:
        raise ValueError("未知的 Trial 显示范围方式。")
    return mode


def draw_tag100_interval(axis, event, limits, *, origin_sec=0, time_scale=1):
    """Shared display-only cue shading for Letter seconds / AnimalStart ms.

    Clip to the existing visible interval so annotations never enlarge the
    view or read additional data. Both Qt and saved figures use this mapping.
    """
    if event.get("tag100_status") != "csv" or event.get("tag100_source") != "csv":
        return None
    try:
        start = (float(event["tag100_start_latency_sec"]) - origin_sec) * time_scale
        end = (float(event["tag100_end_latency_sec"]) - origin_sec) * time_scale
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([start, end]).all() or start >= end:
        return None
    left, right = sorted(limits)
    lo, hi = max(left, start), min(right, end)
    if lo >= hi:
        return None
    region = axis.axvspan(lo, hi, color="black", alpha=.18, zorder=1.5)
    region._tag100_interval = (start, end)
    if hasattr(region, "set_gid"):
        region.set_gid("tag100_interval")
    if hasattr(region, "setZValue"):
        region.setZValue(-5)  # PyQtGraph background, never conceal the curve.
    return region


def waveform_y_limits(output):
    """Display coordinates only (μV or z); never scale/clip stored samples."""
    display = output.get("display_settings", {})
    if display.get("y_axis_mode") != "fixed":
        return None
    low, high = float(display["y_min"]), float(display["y_max"])
    _scale, unit = waveform_display(output)
    if not np.isfinite([low, high]).all() or low >= high or display.get("y_unit") != unit:
        raise ValueError("固定 Y 轴上下限或显示单位无效。")
    return low, high


def apply_waveform_y_limits(axis, output):
    limits = waveform_y_limits(output)
    if limits is not None:
        axis.set_ylim(*limits, auto=False)
        # Keep vertical pan/zoom/Home from changing the applied comparison scale.
        axis.callbacks.connect("ylim_changed", lambda ax: ax.set_ylim(*limits, emit=False, auto=False))


def judgment_info(output, row):
    result = output.get("judgment_similarity")
    if not result or result.get("stale"):
        return None, None, None
    group = next((g for g in result["groups"] if (g["channel"], g["tag"]) == (row["channel"], row["tag"])), None)
    if group is None:
        return None, None, None
    trial = next((t for t in group["trials"] if t["trial_id"] == row["trial_id"]), None)
    summary = next((t for t in group["trial_summaries"] if t["trial_id"] == row["trial_id"]), None)
    return group, trial, summary


def draw_similarity_heatmap(figure, group):
    """Shared UI/report matrix with explicit original Trial labels and CJK fonts."""
    from matplotlib import rc_context
    with rc_context(PLOT_STYLE):
        figure.clear(); axis = figure.subplots()
        image = axis.imshow(group["matrix"], vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
        count = len(group["trials"])
        step = max(1, count//20); labels = [str(t["trial_id"]) for t in group["trials"]]
        axis.set_xticks(range(0, count, step), labels[::step], rotation=90)
        axis.set_yticks(range(0, count, step), labels[::step])
        axis.set(xlabel="原 Trial", ylabel="原 Trial", title=f"ch{group['channel']} / Tag {group['tag']} | 趋势 r；空白=不可比较/自身")
        figure.colorbar(image, ax=axis)


def draw_trial_detail(axis, output, rows, time_ms, values, localization=None, display_mode="overlay"):
    """Use identical labels, units, bounds and annotations in both views."""
    scale, unit = waveform_display(output)
    settings = output["settings"]
    first = rows[0]
    strict_crop = trial_display_mode(output) == "dot_animal"
    search_start_ms = (first["search_first"] - first["animal_local_sample"]) / output["fs"] * 1000
    search_end_ms = (first["search_last"] - first["animal_local_sample"]) / output["fs"] * 1000
    similarity_group, judgment, similarity = judgment_info(output, first)
    show_raw = localization is None or display_mode in ("raw", "overlay")
    visible_values = []
    if show_raw:
        axis.plot(time_ms, values * scale, color="#30343a", linewidth=.7,
                  label="载入波形（参与平均）" if localization is not None else None)
        visible_values.append(np.asarray(values))
    if localization is not None and display_mode in ("trend", "overlay", "despiked"):
        key = "despiked" if display_mode == "despiked" else "trend"
        axis.plot(localization["time_ms"], localization[key] * scale,
                  color="#2679b5", linewidth=1.4,
                  label="仅去毛刺曲线（不用于平均）" if key == "despiked" else "定位趋势线（不用于平均）")
        visible_values.append(localization[key])
    axis.axvspan(search_start_ms, search_end_ms,
                 color="#8cb8d8", alpha=.15, label="搜索区")
    axis.axvline(0, color="#9467bd", linestyle="--", linewidth=.9, label="AnimalStart")
    for row in rows:
        color = "#d44436" if row["branch"] == "peak" else "#16885b"
        label = "峰" if row["branch"] == "peak" else "谷"
        manual = row.get("alignment_origin") == "manual"
        if manual:
            label += "[人工]"
            axis.axvline(row["automatic_relative_ms"], color=".5", linestyle="--", linewidth=.6, alpha=.6)
        position = row["extreme_relative_ms"]
        marker_value = row["amplitude"]
        if localization is not None:
            if not show_raw:
                key = "despiked" if display_mode == "despiked" else "trend"
                nearest = int(np.argmin(np.abs(localization["time_ms"] - position)))
                marker_value = localization[key][nearest]
            axis.axvline(position, color=color, linestyle=":", linewidth=.7, alpha=.7)
            if display_mode == "overlay":
                axis.scatter([position], [row["localization_amplitude"] * scale], marker="D",
                             facecolors="none", edgecolors=color, s=25, zorder=4)
                if not strict_crop or time_ms[0] <= position <= time_ms[-1]:
                    visible_values.append(np.asarray([row["localization_amplitude"]]))
        axis.scatter([position], [marker_value * scale], color=color, s=22, zorder=3)
        if not strict_crop or time_ms[0] <= position <= time_ms[-1]:
            visible_values.append(np.asarray([marker_value]))
        axis.axvspan(position - settings["pre_ms"], position + settings["post_ms"],
                     color=color, alpha=.07)
        axis.plot([], [], color=color,
                  label=f"{label}: {position:.2f} ms, {row['amplitude']*scale:.3g} {unit}"
                  + (f", 旧原波形模板r={row['similarity_r']:.2f}" if not output.get("judgment_similarity") else ""))
    similarity_label = ""
    if similarity is not None:
        ratio = similarity["similar_fraction"]
        ratio_text = f"{ratio*100:.1f}%" if np.isfinite(ratio) else "不可比较"
        best = (f"最高r={similarity['best_r']:.2f} → 原Trial {similarity['best_trial_id']}"
                if np.isfinite(similarity["best_r"]) else "无可比较伙伴")
        similarity_label = f"\n判断窗趋势：相似伙伴 {similarity['similar_count']}/{similarity['comparable_count']}={ratio_text}；{best}"
        if judgment and "window_start_ms" in judgment:
            axis.axvspan(judgment["window_start_ms"], judgment["window_end_ms"],
                        facecolor="#efb642", alpha=.13, edgecolor="#a87000", label="响应判断窗（趋势相似度）")
        if similarity.get("reason"):
            similarity_label += "；" + similarity["reason"]
    elif output.get("judgment_similarity"):
        similarity_label = "\n判断窗相似度待重算（不显示旧分数）"
    flags = "; ".join(dict.fromkeys(filter(None, (row["flags"] or row["reason"] for row in rows))))
    if first.get("tag100_status") not in {None, "csv"}:
        flags = "; ".join(filter(None, (flags, first.get("tag100_message", "CSV Dot 时间缺失或无效；不画黑区"))))
    axis.set_title(
        f"ch{first['channel']} | Tag {first['tag']} | 原 Trial {first['trial_id']} | Tag 内 {first['tag_trial_id']}\n"
        f"AnimalStart={first['animal_sec']:.6f} s | 相对 Letter 开始={first['animal_latency_ms']:+.3f} ms\n"
        + (f"趋势定位；搜索区替换尖刺 {first.get('despike_count_search', 0)} 点；" if localization is not None else "")
        + (f"人工定位 {sum(row.get('alignment_origin') == 'manual' for row in rows)} 点；"
           if any(row.get("alignment_origin") == "manual" for row in rows) else "")
        + (flags or '完整片段；无相似度筛选') + similarity_label,
        fontsize=9, wrap=True,
    )
    axis.set(xlabel="相对 AnimalStart（ms）", ylabel=unit)
    # Explicit bounds are derived only from real data/known intervals, never
    # a graphics overlay's self-referential bounding rectangle.
    left = min(float(time_ms[0]), search_start_ms,
               *(row["extreme_relative_ms"] - settings["pre_ms"] for row in rows))
    right = max(float(time_ms[-1]), search_end_ms,
                *(row["extreme_relative_ms"] + settings["post_ms"] for row in rows))
    if judgment and "window_start_ms" in judgment:
        left = min(left, judgment["window_start_ms"]); right = max(right, judgment["window_end_ms"])
    margin = max((right - left) * .05, .001)
    if strict_crop:
        # Strict display crop; search, manual anchors and overlays never
        # enlarge this interval beyond the Dot or AnimalStart endpoints.
        axis.set_xlim(float(time_ms[0]), float(time_ms[-1]))
    elif letter_display_before_ms(output) is not None:
        # The trace has already been read from Letter minus the display-only
        # duration (or the available source start). Do not pull the view back
        # to a different search/excerpt boundary or add a blank left margin.
        left = float(time_ms[0])
        right = max(right, float(time_ms[-1]))
        axis.set_xlim(left, right + margin)
    else:
        axis.set_xlim(left - margin, right + margin)
    finite = np.concatenate([np.asarray(item).reshape(-1) for item in visible_values])
    finite = finite[np.isfinite(finite)] * scale
    if finite.size:
        bottom, top = float(finite.min()), float(finite.max())
        padding = max((top - bottom) * .05, abs(top) * .01, 1e-9)
        axis.set_ylim(bottom - padding, top + padding)
    apply_waveform_y_limits(axis, output)
    # Letter onset is t=0 of this Trial, not the earlier Epoch crop boundary.
    # Add it after establishing data bounds so an off-screen event never
    # expands the view. Update visibility when the user zooms/pans as well.
    letter_ms = -float(first["animal_latency_ms"])
    if np.isfinite(letter_ms):
        letter_line = axis.axvline(letter_ms, color="#c62828", linestyle="-",
                                  linewidth=1.2, label="Letter 开始", zorder=4)
        letter_line.set_gid("letter_start")

        def update_letter_visibility(current_axis):
            low, high = sorted(current_axis.get_xlim())
            letter_line.set_visible(low <= letter_ms <= high)

        update_letter_visibility(axis)
        axis.callbacks.connect("xlim_changed", update_letter_visibility)
    draw_tag100_interval(axis, first, axis.get_xlim(),
                         origin_sec=first["animal_latency_ms"]/1000, time_scale=1000)
    # No in-axes legend: it obscures trial waveforms in both viewer and exports.
    legend = axis.get_legend()
    if legend is not None:
        legend.remove()
    axis.grid(alpha=.15)
