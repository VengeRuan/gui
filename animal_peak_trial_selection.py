"""Explicit, reversible Trial choices for averages only; never delete records."""
from datetime import datetime, timezone

import numpy as np


def average_trial_ids(output, channel, tag):
    """None means default-all; an explicit empty tuple means average nothing."""
    entry = next((entry for entry in output.get("average_trial_selections", [])
                  if (entry["channel"], entry["tag"]) == (channel, tag)), None)
    return None if entry is None else tuple(entry["trial_ids"])


def set_average_trial_selection(output, channel, tag, trial_ids):
    available = {int(row["trial_id"]) for row in output["records"]
                 if (row.get("channel"), row.get("tag")) == (channel, tag)
                 and "extreme_local_sample" in row}
    if not available:
        raise ValueError("本组没有已定位的 Trial，无法设置叠加平均选择。")
    chosen = available if trial_ids is None else {int(trial) for trial in trial_ids}
    if chosen - available:
        raise ValueError("选择包含不属于当前通道/Tag的 Trial："+",".join(map(str, sorted(chosen-available))))
    ids = None if chosen == available else tuple(sorted(chosen))
    previous = average_trial_ids(output, channel, tag)
    if ids == previous:
        return output
    entries = [entry for entry in output.get("average_trial_selections", [])
               if (entry["channel"], entry["tag"]) != (channel, tag)]
    if ids is not None:
        entries.append({"channel": int(channel), "tag": int(tag), "trial_ids": list(ids)})
    history = [*output.get("average_selection_history", []), {
        "revision": len(output.get("average_selection_history", []))+1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "channel": int(channel), "tag": int(tag),
        "previous_trial_ids": None if previous is None else list(previous),
        "trial_ids": None if ids is None else list(ids), "default_all": ids is None,
    }]
    return {**output, "average_trial_selections": entries, "average_selection_history": history}


def selected_average_group(output, group, cancel=None):
    """Derive a selected average from committed waveforms, never modify base data."""
    if group is None:
        return None
    ids = average_trial_ids(output, group["channel"], group["tag"])
    if ids is None:
        return group
    from animal_peak_analysis import _summarize, _check
    _check(cancel)
    wanted = set(ids)
    indices = [i for i, row in enumerate(group["rows"]) if row["trial_id"] in wanted]
    waves = group["waves"][indices]
    if indices:
        mean, sem, count, _ = _summarize(waves, cancel)
    else:
        mean = np.full(group["mean"].shape, np.nan, dtype=np.float32)
        sem = mean.copy(); count = np.zeros(mean.shape, dtype=np.int64)
    return {**group, "rows": [group["rows"][i] for i in indices], "waves": waves,
            "mean": mean, "sem": sem, "valid_count": count,
            "available_trial_count": len(group["rows"]), "average_selection_applied": True}
