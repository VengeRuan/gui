"""Descriptive trend-window reproducibility; never filter or change peak averages."""
from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path

import numpy as np

from animal_peak_analysis import _check, _read, _json_safe, source_identity
from animal_peak_localization import trial_localization


def similarity_config(parameters=None):
    values = {"before_ms": 50., "after_ms": 50., "r_threshold": .7,
              "required_fraction": .6, "min_window_coverage": .8,
              "min_reference_coverage": .8, **(parameters or {})}
    values = {key: float(values[key]) for key in (
        "before_ms", "after_ms", "r_threshold", "required_fraction",
        "min_window_coverage", "min_reference_coverage")}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("判断窗相似度参数必须为有限数值。")
    if min(values["before_ms"], values["after_ms"]) < 0:
        raise ValueError("判断窗前后延伸不能为负。")
    if not 0 <= values["r_threshold"] <= 1:
        raise ValueError("形状相似度阈值必须在 0～1 之间。")
    if any(not 0 < values[key] <= 1 for key in (
            "required_fraction", "min_window_coverage", "min_reference_coverage")):
        raise ValueError("比例参数必须大于 0 且不超过 100%。")
    return values


REASONS = {0: "", 1: "判断窗无效（见逐 Trial 原因）", 2: "共同有限采样点不足 3 个",
           3: "共同有效数据未达到两窗各自的最低覆盖率", 4: "共同片段为平直曲线，相关系数不可定义"}


def compare_windows(first, second, min_coverage=.8):
    """Align earlier anchors at zero, preserving milliseconds; no shift/stretch."""
    count = min(len(first), len(second))
    left, right = np.asarray(first[:count], dtype=float), np.asarray(second[:count], dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    n = int(valid.sum())
    coverage = n / max(len(first), len(second), 1)
    if n < 3:
        return np.nan, n, coverage, 2
    if coverage < min_coverage:
        return np.nan, n, coverage, 3
    left, right = left[valid], right[valid]
    left -= np.mean(left); right -= np.mean(right)
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    if not np.isfinite(norm) or norm == 0:
        return np.nan, n, coverage, 4
    return float(np.clip(np.dot(left, right) / norm, -1., 1.)), n, coverage, 0


def summarize_group(group, parameters, reference_trial=None):
    """One-trial vs others and reference support are intentionally different."""
    config = similarity_config(parameters)
    matrix = group["matrix"]
    trials = group["trials"]
    summaries = []
    for index, trial in enumerate(trials):
        others = [j for j in range(len(trials)) if j != index and np.isfinite(matrix[index, j])]
        ranked = sorted(others, key=lambda j: (-matrix[index, j], trials[j]["trial_id"]))
        matched = [j for j in others if matrix[index, j] >= config["r_threshold"]]
        summaries.append({"trial_id": trial["trial_id"], "tag_trial_id": trial["tag_trial_id"],
                          "comparable_count": len(others), "similar_count": len(matched),
                          "similar_fraction": len(matched) / len(others) if others else np.nan,
                          "median_r": float(np.median(matrix[index, others])) if others else np.nan,
                          "best_trial_id": trials[ranked[0]]["trial_id"] if ranked else None,
                          "best_r": float(matrix[index, ranked[0]]) if ranked else np.nan,
                          "reason": trial.get("reason", "")})
    candidates = [i for i, trial in enumerate(trials) if not trial.get("reason")]
    chosen = next((i for i in candidates if trials[i]["trial_id"] == reference_trial), None)
    manual = chosen is not None
    if chosen is None and candidates:
        chosen = max(candidates, key=lambda i: (summaries[i]["similar_count"],
                     summaries[i]["median_r"] if np.isfinite(summaries[i]["median_r"]) else -2,
                     -trials[i]["trial_id"]))
    comparable = summaries[chosen]["comparable_count"] if chosen is not None else 0
    similar = summaries[chosen]["similar_count"] if chosen is not None else 0
    fraction = similar / comparable if comparable else np.nan
    reference_coverage = comparable / (len(trials) - 1) if len(trials) > 1 else 0.
    members = ([chosen] + [j for j in range(len(trials)) if j != chosen and
                np.isfinite(matrix[chosen, j]) and matrix[chosen, j] >= config["r_threshold"]]
               if chosen is not None else [])
    internal = [matrix[i, j] for order, i in enumerate(members) for j in members[order+1:]
                if np.isfinite(matrix[i, j])]
    return {**group, "trial_summaries": summaries, "reference_mode": "manual" if manual else "auto",
            "reference_trial_id": trials[chosen]["trial_id"] if chosen is not None else None,
            "comparable_count": comparable, "similar_count": similar,
            "reference_fraction": fraction, "reference_coverage": reference_coverage,
            "reference_threshold_met": bool(comparable and fraction >= config["required_fraction"]
                                             and reference_coverage >= config["min_reference_coverage"]),
            "matched_trial_ids": [trials[i]["trial_id"] for i in members],
            "matched_group_fraction": len(members)/len(trials) if trials else np.nan,
            "internal_pair_count": len(internal),
            "internal_median_r": float(np.median(internal)) if internal else np.nan,
            "internal_similar_pair_fraction": float(np.mean(np.asarray(internal) >= config["r_threshold"])) if internal else np.nan,
            "internal_comparison_coverage": len(internal) / (len(members)*(len(members)-1)/2) if len(members) > 1 else 0.}


def calculate_similarity(source, output, parameters=None, progress=None, cancel=None):
    """Return a new result, cache only bounded windows/matrices, no source writes."""
    config = similarity_config(parameters)
    _check(cancel)
    if source_identity(source) != output["source_identity"]:
        raise ValueError("数据源已改变，请重新提取峰谷。")
    if output["settings"].get("localization_mode") != "trend":
        raise ValueError("请先选择“去毛刺＋趋势线定位”并完成峰谷提取，再计算判断窗相似度。")
    fs = output["fs"]
    pre, post = (int(round(config[key] * fs / 1000)) for key in ("before_ms", "after_ms"))
    revision = len(output.get("manual_alignment_history", []))
    old = output.get("judgment_similarity", {})
    cache_valid = (old.get("anchor_revision") == revision and old.get("pre_samples") == pre
                   and old.get("window_resize_revision", 0) == len(output.get("window_resize_history", []))
                   and old.get("post_samples") == post and old.get("source_identity") == source_identity(source))
    cached = {(g["channel"], g["tag"]): g for g in old.get("groups", [])} if cache_valid else {}
    by_pair = defaultdict(lambda: defaultdict(list))
    global_excluded = [row for row in output["records"] if row.get("channel") == ""]
    for row in output["records"]:
        if row.get("channel") != "":
            by_pair[(int(row["channel"]), int(row["tag"]))][row["trial_id"]].append(row)
    for channel in output["settings"]["channel_ids"]:
        for row in global_excluded:
            by_pair[(int(channel), int(row["tag"]))][row["trial_id"]].append(row)
    budget = int(output["settings"].get("memory_budget_bytes", 256*1024**2))
    retained = sum(group["waves"].nbytes for group in output["groups"])
    retained += sum(group["waves"].nbytes for group in output.get("animal_reference_groups", []))
    # Old results remain recoverable until successful completion.
    retained += sum(g["matrix"].nbytes + g["counts"].nbytes + g["coverage"].nbytes + g["reason_codes"].nbytes
                    + sum(t.get("trend", np.empty(0)).nbytes for t in g["trials"]) for g in old.get("groups", []))
    groups = []
    for number, ((channel, tag), trial_rows) in enumerate(sorted(by_pair.items())):
        _check(cancel)
        prior = cached.get((channel, tag))
        if progress:
            progress(number/max(1, len(by_pair))*100, f"判断窗：ch{channel} / Tag {tag}")
        if prior is not None:
            trials = prior["trials"]
        else:
            trials = []
            for trial_id, rows in sorted(trial_rows.items()):
                _check(cancel)
                trial = {"trial_id": trial_id, "tag_trial_id": rows[0]["tag_trial_id"], "reason": ""}
                anchors = {row["branch"]: row for row in rows if "extreme_local_sample" in row}
                if not all(branch in anchors for branch in ("peak", "trough")):
                    trial["reason"] = "缺少峰/谷确认点：" + "; ".join(dict.fromkeys(row.get("reason", "未定位") for row in rows))
                else:
                    left, right = sorted(anchors[b]["extreme_local_sample"] for b in ("peak", "trough"))
                    row = anchors["peak"]
                    first, last = left-pre, right+post+1
                    trial.update(anchor_local_sample=left, first=first, last=last,
                                 window_start_ms=(first-row["animal_local_sample"])/fs*1000,
                                 window_end_ms=(last-1-row["animal_local_sample"])/fs*1000,
                                 peak_ms=anchors["peak"]["extreme_relative_ms"], trough_ms=anchors["trough"]["extreme_relative_ms"],
                                 peak_origin=anchors["peak"].get("alignment_origin", "automatic"),
                                 trough_origin=anchors["trough"].get("alignment_origin", "automatic"))
                    if first < 0 or last > source.metadata.rows:
                        trial["reason"] = f"判断窗超出源数据：[{first}, {last})，源数据=[0, {source.metadata.rows})"
                    elif last-first < 3:
                        trial["reason"] = "判断窗不足 3 个采样点"
                    else:
                        if retained + (last-first)*8 > budget:
                            raise ValueError("判断窗趋势及相似度结果超过测试内存预算，请减少通道/Trial或判断窗长度。")
                        # Expand the ORIGINAL locator's padded interval, never shrink it:
                        # filtering boundaries remain identical wherever already displayed.
                        context = {**row, "search_first": min(row["search_first"], first),
                                   "search_last": max(row["search_last"], last)}
                        locator = trial_localization(lambda a, b: _read(source, row["column"], a, b),
                            context, source.metadata.rows, fs, output["settings"], lambda: _check(cancel), retained)
                        trend = locator["trend"][first-locator["first"]:last-locator["first"]].copy()
                        trend -= row["baseline_correction"]
                        trial["trend"] = trend
                        retained += trend.nbytes
                        valid = trend[np.isfinite(trend)]
                        if valid.size < 3:
                            trial["reason"] = "判断窗有限采样点不足 3 个"
                        elif np.ptp(valid) == 0:
                            trial["reason"] = "判断窗趋势为平直曲线，相关系数不可定义"
                trials.append(trial)
        n = len(trials)
        if retained + n*n*17 > budget:
            raise ValueError("Trial 两两相似度矩阵超过测试内存预算，请减少通道或Trial。")
        matrix = np.full((n, n), np.nan)
        counts = np.zeros((n, n), dtype=np.int32)
        coverage = np.zeros((n, n), dtype=np.float32)
        codes = np.ones((n, n), dtype=np.uint8)
        retained += n*n*17
        for i, trial in enumerate(trials):
            _check(cancel)
            for j in range(i+1, n):
                _check(cancel)
                if trial.get("reason") or trials[j].get("reason"):
                    continue
                r, count, cover, code = compare_windows(trial["trend"], trials[j]["trend"], config["min_window_coverage"])
                matrix[i, j] = matrix[j, i] = r
                counts[i, j] = counts[j, i] = count
                coverage[i, j] = coverage[j, i] = cover
                codes[i, j] = codes[j, i] = code
        group = {"channel": channel, "tag": tag, "trials": trials, "matrix": matrix,
                 "counts": counts, "coverage": coverage, "reason_codes": codes}
        reference = next((g.get("reference_trial_id") for g in old.get("groups", [])
                          if g["channel"] == channel and g["tag"] == tag and g.get("reference_mode") == "manual"), None)
        groups.append(summarize_group(group, config, reference))
    _check(cancel)
    if source_identity(source) != output["source_identity"]:
        raise ValueError("计算期间数据源已改变，未提交相似度结果。")
    if progress:
        progress(100, "判断窗两两相似度及参考一致率计算完成")
    result = {"parameters": config, "groups": groups, "stale": False, "anchor_revision": revision,
              "window_resize_revision": len(output.get("window_resize_history", [])),
              "source_identity": output["source_identity"], "pre_samples": pre, "post_samples": post,
              "meaning": "趋势判断窗 Pearson r；较早锚点对齐，不平移搜索、不拉伸、不取绝对值；仅描述形状重复性，不筛选原平均。"}
    return {**output, "judgment_similarity": result}


def ranked_matches(group, trial_id, fs):
    trials = group["trials"]
    index = next(i for i, trial in enumerate(trials) if trial["trial_id"] == trial_id)
    order = sorted((j for j in range(len(trials)) if j != index), key=lambda j: (
        not np.isfinite(group["matrix"][index, j]),
        -group["matrix"][index, j] if np.isfinite(group["matrix"][index, j]) else 0, trials[j]["trial_id"]))
    for rank, j in enumerate(order, 1):
        n = int(group["counts"][index, j])
        yield {"channel": group["channel"], "tag": group["tag"], "trial_id": trial_id,
               "other_trial_id": trials[j]["trial_id"], "other_tag_trial_id": trials[j]["tag_trial_id"],
               "rank": rank, "r": float(group["matrix"][index, j]), "compared_samples": n,
               "effective_ms": n/fs*1000, "minimum_window_coverage": float(group["coverage"][index, j]),
               "reason": (trials[index].get("reason") or trials[j].get("reason")
                          or REASONS[int(group["reason_codes"][index, j])])}


def export_similarity_report(output, directory, cancel=None):
    """Stream CSV/HTML rankings and per-group numeric trends without H5 rereads."""
    result = output.get("judgment_similarity")
    if not result or result.get("stale"):
        return
    _check(cancel)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from animal_peak_plotting import draw_similarity_heatmap
    folder = Path(directory)/"response_similarity"
    folder.mkdir(exist_ok=False)
    parameters = result["parameters"]
    (folder/"parameters.json").write_text(json.dumps(_json_safe({
        "parameters": parameters, "meaning": result["meaning"], "source": output["source"],
        "fs": output["fs"], "trend_settings": output["settings"],
        "anchor_revision": result["anchor_revision"],
        "window_resize_revision": result.get("window_resize_revision", 0),
        "reference_denominator": "同通道/Tag下与参考可比较的其他Trial，不含参考本身；另报有效比较率，防止少量可比较数据抬高比例。",
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["channel", "tag", "trial_id", "other_trial_id", "other_tag_trial_id", "rank", "r",
              "compared_samples", "effective_ms", "minimum_window_coverage", "reason"]
    summary_fields = ["channel", "tag", "trial_id", "tag_trial_id", "similar_count", "comparable_count", "similar_fraction", "median_r", "best_trial_id", "best_r", "reason"]
    group_fields = ["channel", "tag", "reference_mode", "reference_trial_id", "total_trials", "valid_windows", "similar_count", "comparable_count", "reference_fraction", "reference_coverage", "reference_threshold_met", "matched_group_fraction", "internal_pair_count", "internal_median_r", "internal_similar_pair_fraction", "internal_comparison_coverage", "matched_trial_ids"]
    labels = {"channel": "实际通道", "tag": "Tag", "trial_id": "当前原 Trial", "tag_trial_id": "Tag 内 Trial",
              "other_trial_id": "对方原 Trial", "other_tag_trial_id": "对方 Tag 内 Trial", "rank": "排名",
              "r": "趋势相似度 r", "compared_samples": "实际比较采样点数", "effective_ms": "有效比较长度 ms",
              "minimum_window_coverage": "两窗最低覆盖率", "reason": "不可比较原因",
              "similar_count": "相似伙伴数", "comparable_count": "可比较的其他 Trial 数",
              "similar_fraction": "相似伙伴比例", "median_r": "与其他 Trial 的中位数 r",
              "best_trial_id": "最佳匹配原 Trial", "best_r": "最高相似度 r", "reference_mode": "参考选择方式",
              "reference_trial_id": "参考原 Trial", "total_trials": "本组总 Trial 数", "valid_windows": "有效判断窗数",
              "reference_fraction": "参考一致率（不含参考自身）", "reference_coverage": "参考有效比较率",
              "reference_threshold_met": "是否满足当前参考一致率规则", "matched_group_fraction": "匹配组占全部 Trial 比例（含参考）",
              "internal_pair_count": "匹配组内部有效配对数", "internal_median_r": "匹配组内部中位数 r",
              "internal_similar_pair_fraction": "匹配组内部达标配对比例", "internal_comparison_coverage": "匹配组内部有效配对覆盖率",
              "matched_trial_ids": "匹配组原 Trial 编号（含参考）"}
    def display(value):
        return "不可比较" if isinstance(value, (float, np.floating)) and not np.isfinite(value) else escape(str(value))
    def cell(key, value):
        if key.endswith("fraction") or key in ("reference_coverage", "internal_comparison_coverage", "minimum_window_coverage"):
            return f"{value*100:.1f}%" if value is not None and np.isfinite(value) else "不可比较"
        if key == "reference_threshold_met":
            return "达到当前规则（不等于全部 Trial 彼此一致）" if value else "未达到或有效比较不足"
        if key == "reference_mode":
            return "自动推荐（探索性）" if value == "auto" else "人工指定"
        return display(value)
    with (folder/"all_rankings.csv").open("w", newline="", encoding="utf-8-sig") as handle, \
         (folder/"trial_summary.csv").open("w", newline="", encoding="utf-8-sig") as summaries, \
         (folder/"group_summary.csv").open("w", newline="", encoding="utf-8-sig") as group_file, \
         (folder/"judgment_windows.csv").open("w", newline="", encoding="utf-8-sig") as window_file, \
         (folder/"report.html").open("w", encoding="utf-8") as report:
        writer = csv.DictWriter(handle, fields); writer.writeheader()
        summary_writer = csv.DictWriter(summaries, summary_fields); summary_writer.writeheader()
        group_writer = csv.DictWriter(group_file, group_fields); group_writer.writeheader()
        window_fields = ["channel", "tag", "trial_id", "tag_trial_id", "anchor_local_sample", "first", "last", "window_start_ms", "window_end_ms", "peak_ms", "trough_ms", "peak_origin", "trough_origin", "reason"]
        window_writer = csv.DictWriter(window_file, window_fields); window_writer.writeheader()
        report.write('<!doctype html><meta charset="utf-8"><title>响应判断窗相似度报告</title><style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:5px}details{margin:12px 0}</style>')
        report.write("<h1>响应判断窗相似度</h1><p>"+escape(result["meaning"])+"</p><p>参数："+escape(str(parameters))+"</p><p>参考一致率不等于全部Trial彼此一致；不代表刺激锁时、幅值稳定或已验证的VEP。自动参考从本组选择，属于探索性结果。</p>")
        for group in result["groups"]:
            _check(cancel)
            channel, tag = group["channel"], group["tag"]
            record = {key: group.get(key) for key in group_fields}
            record.update(total_trials=len(group["trials"]), valid_windows=sum(not t.get("reason") for t in group["trials"]))
            group_writer.writerow(record)
            for row in group["trial_summaries"]:
                summary_writer.writerow({"channel": channel, "tag": tag, **row})
            report.write(f"<h2>ch{channel} / Tag {tag}</h2><table>" + "".join(f"<tr><th>{labels[k]}</th><td>{cell(k,v)}</td></tr>" for k, v in record.items())+"</table>")
            summary_columns = summary_fields[2:]
            report.write("<h3>逐 Trial 汇总</h3><table><tr>"+"".join(f"<th>{labels[k]}</th>" for k in summary_columns)+"</tr>")
            for row in group["trial_summaries"]:
                report.write("<tr>"+"".join(f"<td>{cell(k,row[k])}</td>" for k in summary_columns)+"</tr>")
            report.write("</table>")
            report.write(f'<p><img src="ch{channel}_tag{tag}_heatmap.png" style="max-width:100%"></p>')
            trials = group["trials"]
            for trial in trials:
                _check(cancel)
                window_writer.writerow({key: channel if key == "channel" else tag if key == "tag" else trial.get(key, "") for key in window_fields})
                report.write(f'<details><summary>原 Trial {trial["trial_id"]} / Tag 内 {trial["tag_trial_id"]}：完整匹配排名</summary><table><tr>'+"".join(f"<th>{labels[k]}</th>" for k in fields)+"</tr>")
                for row in ranked_matches(group, trial["trial_id"], output["fs"]):
                    _check(cancel); writer.writerow(row)
                    report.write("<tr>"+"".join(f"<td>{cell(k,row[k])}</td>" for k in fields)+"</tr>")
                report.write("</table></details>")
            arrays = {key: group[key] for key in ("matrix", "counts", "coverage", "reason_codes")}
            arrays["trial_ids"] = np.asarray([t["trial_id"] for t in trials])
            for trial in trials:
                if "trend" in trial:
                    arrays[f'trial_{trial["trial_id"]}_trend'] = trial["trend"]
                    arrays[f'trial_{trial["trial_id"]}_source_bounds'] = np.asarray([trial["first"], trial["last"], trial["anchor_local_sample"]])
            np.savez_compressed(folder/f"ch{channel}_tag{tag}_data.npz", **arrays)
            matrix_labels = [str(t["trial_id"]) for t in trials]
            with (folder/f"ch{channel}_tag{tag}_matrix.csv").open("w", newline="", encoding="utf-8-sig") as matrix_file:
                matrix_writer = csv.writer(matrix_file); matrix_writer.writerow(["原Trial", *matrix_labels])
                for trial, values in zip(trials, group["matrix"]):
                    matrix_writer.writerow([trial["trial_id"], *values])
            figure = Figure(figsize=(8, 7), layout="constrained"); FigureCanvasAgg(figure)
            draw_similarity_heatmap(figure, group)
            figure.savefig(folder/f"ch{channel}_tag{tag}_heatmap.png", dpi=160); figure.clear()
    return folder
