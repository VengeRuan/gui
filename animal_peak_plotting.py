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


def draw_trial_detail(axis, output, rows, time_ms, values):
    """Use identical labels, units, bounds and annotations in both views."""
    scale = 1000 if output["units"] == "mV" else 1
    unit = "μV" if output["units"] == "mV" else "z"
    settings = output["settings"]
    first = rows[0]
    axis.plot(time_ms, values * scale, color="#30343a", linewidth=.7)
    axis.axvspan(-settings["search_ms"], settings.get("search_after_ms", 0),
                 color="#8cb8d8", alpha=.15, label="搜索区")
    axis.axvline(0, color="#9467bd", linestyle="--", linewidth=.9, label="AnimalStart")
    for row in rows:
        color = "#d44436" if row["branch"] == "peak" else "#16885b"
        label = "峰" if row["branch"] == "peak" else "谷"
        position = row["extreme_relative_ms"]
        axis.scatter([position], [row["amplitude"] * scale], color=color, s=22, zorder=3)
        axis.axvspan(position - settings["pre_ms"], position + settings["post_ms"],
                     color=color, alpha=.07)
        axis.plot([], [], color=color,
                  label=f"{label}: {position:.2f} ms, {row['amplitude']*scale:.3g} {unit}, r={row['similarity_r']:.2f}")
    flags = "; ".join(dict.fromkeys(filter(None, (row["flags"] or row["reason"] for row in rows))))
    axis.set_title(
        f"ch{first['channel']} | Tag {first['tag']} | 原 Trial {first['trial_id']} | Tag 内 {first['tag_trial_id']}\n"
        f"AnimalStart={first['animal_sec']:.6f} s\n{flags or '完整片段；无相似度筛选'}",
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
    finite = np.asarray(values)[np.isfinite(values)] * scale
    amplitudes = [row["amplitude"] * scale for row in rows if np.isfinite(row["amplitude"])]
    if finite.size:
        bottom = min(float(finite.min()), *amplitudes) if amplitudes else float(finite.min())
        top = max(float(finite.max()), *amplitudes) if amplitudes else float(finite.max())
        padding = max((top - bottom) * .05, abs(top) * .01, 1e-9)
        axis.set_ylim(bottom - padding, top + padding)
    axis.legend(fontsize=7, loc="best")
    axis.grid(alpha=.15)
