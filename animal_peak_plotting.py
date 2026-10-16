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


def draw_trial_detail(axis, output, rows, time_ms, values, localization=None, display_mode="overlay"):
    """Use identical labels, units, bounds and annotations in both views."""
    scale = 1000 if output["units"] == "mV" else 1
    unit = "μV" if output["units"] == "mV" else "z"
    settings = output["settings"]
    first = rows[0]
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
    axis.axvspan(-settings["search_ms"], settings.get("search_after_ms", 0),
                 color="#8cb8d8", alpha=.15, label="搜索区")
    axis.axvline(0, color="#9467bd", linestyle="--", linewidth=.9, label="AnimalStart")
    for row in rows:
        color = "#d44436" if row["branch"] == "peak" else "#16885b"
        label = "峰" if row["branch"] == "peak" else "谷"
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
                visible_values.append(np.asarray([row["localization_amplitude"]]))
        axis.scatter([position], [marker_value * scale], color=color, s=22, zorder=3)
        visible_values.append(np.asarray([marker_value]))
        axis.axvspan(position - settings["pre_ms"], position + settings["post_ms"],
                     color=color, alpha=.07)
        axis.plot([], [], color=color,
                  label=f"{label}: {position:.2f} ms, {row['amplitude']*scale:.3g} {unit}, r={row['similarity_r']:.2f}")
    flags = "; ".join(dict.fromkeys(filter(None, (row["flags"] or row["reason"] for row in rows))))
    axis.set_title(
        f"ch{first['channel']} | Tag {first['tag']} | 原 Trial {first['trial_id']} | Tag 内 {first['tag_trial_id']}\n"
        f"AnimalStart={first['animal_sec']:.6f} s\n"
        + (f"趋势定位；搜索区替换尖刺 {first.get('despike_count_search', 0)} 点；" if localization is not None else "")
        + (flags or '完整片段；无相似度筛选'),
        fontsize=9, wrap=True,
    )
    axis.set(xlabel="相对 AnimalStart（ms）", ylabel=unit)
    # Explicit bounds are derived only from real data/known intervals, never
    # a graphics overlay's self-referential bounding rectangle.
    left = min(float(time_ms[0]), -settings["search_ms"],
               *(row["extreme_relative_ms"] - settings["pre_ms"] for row in rows))
    right = max(float(time_ms[-1]), settings.get("search_after_ms", 0),
                *(row["extreme_relative_ms"] + settings["post_ms"] for row in rows))
    margin = max((right - left) * .05, .001)
    axis.set_xlim(left - margin, right + margin)
    finite = np.concatenate([np.asarray(item).reshape(-1) for item in visible_values])
    finite = finite[np.isfinite(finite)] * scale
    if finite.size:
        bottom, top = float(finite.min()), float(finite.max())
        padding = max((top - bottom) * .05, abs(top) * .01, 1e-9)
        axis.set_ylim(bottom - padding, top + padding)
    axis.legend(fontsize=7, loc="best")
    axis.grid(alpha=.15)
