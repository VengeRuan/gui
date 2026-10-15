"""Read-only, conservative matching of alignment companions for a loaded H5."""
from datetime import datetime
from pathlib import Path
import re


DEFAULT_EXPERIMENT_ROOT = Path(r"D:\data\PilotStudy2\正式实验")
CSV_MIN_BYTES = 8 * 1024
CSV_MAX_BYTES = 10 * 1024
FILE_KINDS = ("log", "events", "detection")


def _identity(h5_path):
    """Use the first experiment date/animal, never the later export timestamp."""
    name = Path(h5_path).stem
    match = re.search(r"(?<!\d)(20\d{6})[_-](\d+)(?=[_-]|$)", name)
    if not match:
        return None
    date, animal = match.groups()
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return None
    block = re.search(r"(?:^|[_-])block[_-]?(\d+)(?=[_-]|$)", name[match.end():], re.I)
    return {"date": date, "animal": animal, "block": int(block[1]) if block else None}


def _directory_files(directory):
    try:
        return sorted((path for path in directory.iterdir() if path.is_file()), key=lambda path: path.name.casefold()), ""
    except FileNotFoundError:
        return [], f"目录不存在：{directory}"
    except OSError as exc:
        return [], f"目录无法访问：{directory}（{exc}）"


def _selection(candidates, missing_message, directory_error=""):
    candidates = sorted(candidates, key=lambda path: path.name.casefold())
    if directory_error:
        return {"selected": None, "candidates": [], "status": "unavailable", "message": directory_error}
    if len(candidates) == 1:
        return {"selected": candidates[0], "candidates": candidates, "status": "found", "message": "唯一匹配，已自动填入"}
    if candidates:
        return {"selected": None, "candidates": candidates, "status": "ambiguous",
                "message": f"符合条件的文件有 {len(candidates)} 个，请用浏览手动选择"}
    return {"selected": None, "candidates": [], "status": "missing", "message": missing_message}


def discover_alignment_files(h5_path, experiment_root=None):
    """Search only the two requested directories, never read a recording matrix.

    Log uses date/animal/block. Event CSV deliberately ignores block, requires
    the Letter filename and 8192–10240 bytes inclusive. Detection uses the full
    experiment date and exact animal token. Ambiguity never picks a newest file.
    """
    h5_path = Path(h5_path)
    identity = _identity(h5_path) if h5_path.suffix.lower() in (".h5", ".hdf5") else None
    if identity is None:
        return {"identity": None, "directories": {}, "files": {
            kind: {"selected": None, "candidates": [], "status": "unrecognized",
                   "message": "H5 文件名未识别出有效实验日期与动物编号，请手动浏览选择"}
            for kind in FILE_KINDS
        }}
    root = Path(experiment_root) if experiment_root is not None else DEFAULT_EXPERIMENT_ROOT
    date, animal, block = identity["date"], identity["animal"], identity["block"]
    log_directory = root / "logfiles" / date / animal
    detection_directory = root / "processed_data" / "行为学数据处理" / "chars_detected" / date
    log_files, log_error = _directory_files(log_directory)
    detection_files, detection_error = _directory_files(detection_directory)
    prefix = re.escape(date) + r"[_-]" + re.escape(animal)
    log_pattern = re.compile(r"^" + prefix + r"[_-]block[_-]?(\d+)\.txt$", re.I)
    logs = [path for path in log_files if (match := log_pattern.fullmatch(path.name)) and int(match[1]) == block]
    csv_pattern = re.compile(r"^" + re.escape(animal) + r"_Letter_experiment_data_" + re.escape(date) + r"(?:[_-].*)?\.csv$", re.I)
    csvs = []
    rejected_sizes = 0
    unreadable = 0
    for path in log_files:
        if not csv_pattern.fullmatch(path.name):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            unreadable += 1
            continue
        if CSV_MIN_BYTES <= size <= CSV_MAX_BYTES:
            csvs.append(path)
        else:
            rejected_sizes += 1
    iso_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    detection_pattern = re.compile(
        r"^(?:" + re.escape(iso_date) + "|" + re.escape(date) + r")[ _].*[_-]"
        + re.escape(animal) + r"_detection_results\.txt$", re.I,
    )
    detections = [path for path in detection_files if detection_pattern.fullmatch(path.name)]
    files = {
        "log": _selection(logs, "没有找到日期、动物编号与 block 对应的 log.txt", log_error),
        "events": _selection(csvs, "没有找到日期/动物匹配且大小为 8～10 KB 的 Letter Event CSV", log_error),
        "detection": _selection(detections, "没有找到日期与动物编号对应的视频检测 TXT", detection_error),
    }
    if block is None:
        files["log"] = _selection([], "H5 文件名没有 block 编号，log.txt 请手动选择")
    if rejected_sizes or unreadable:
        files["events"]["message"] += f"；大小不符 {rejected_sizes} 个，无法读取大小 {unreadable} 个"
    return {"identity": identity, "directories": {"log": log_directory, "events": log_directory,
                                                 "detection": detection_directory}, "files": files}
