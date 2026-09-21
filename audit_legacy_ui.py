"""Export a runtime inventory of the legacy Tk UI for Qt parity work.

Run from the project folder:
    python audit_legacy_ui.py

The report contains every constructed widget, its layout manager/location,
bound Tk variable/default value, combobox choices, enabled state, and all
Tk variables reachable from the main controller and embedded pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import tkinter as tk

import gui


OUTPUT = Path(__file__).with_name("legacy_ui_inventory.json")


def _json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return str(value)
    except Exception:
        return repr(value)


def _layout(widget):
    manager = widget.winfo_manager()
    if manager == "grid":
        return {"manager": manager, **{key: _json_value(value) for key, value in widget.grid_info().items()}}
    if manager == "pack":
        return {"manager": manager, **{key: _json_value(value) for key, value in widget.pack_info().items()}}
    if manager == "place":
        return {"manager": manager, **{key: _json_value(value) for key, value in widget.place_info().items()}}
    return {"manager": manager}


def _config(widget, root):
    result = {}
    for option in ("text", "textvariable", "variable", "state", "values", "command", "orient", "show"):
        try:
            value = widget.cget(option)
        except tk.TclError:
            continue
        result[option] = _json_value(value)
        if option in {"textvariable", "variable"} and value:
            try:
                result[f"{option}_value"] = _json_value(root.getvar(value))
            except tk.TclError:
                pass
    return result


def _walk(widget, root, rows):
    rows.append(
        {
            "path": str(widget),
            "class": widget.winfo_class(),
            "name": widget.winfo_name(),
            "layout": _layout(widget),
            "config": _config(widget, root),
            "requested_size": [widget.winfo_reqwidth(), widget.winfo_reqheight()],
            "children": [str(child) for child in widget.winfo_children()],
        }
    )
    for child in widget.winfo_children():
        _walk(child, root, rows)


def _variables(owner):
    result = {}
    for name, value in vars(owner).items():
        if not isinstance(value, tk.Variable):
            continue
        try:
            result[name] = {"tcl_name": str(value), "value": _json_value(value.get()), "type": type(value).__name__}
        except tk.TclError:
            continue
    return result


def _descendants(widget):
    result = []
    for child in widget.winfo_children():
        result.append(child)
        result.extend(_descendants(child))
    return result


def _page_summary(app):
    variable_names = {details["tcl_name"]: name for name, details in _variables(app).items()}
    pages = {}
    for page_name, frame in getattr(app, "page_frames", {}).items():
        widgets = [frame, *_descendants(frame)]
        controls = []
        for widget in widgets:
            widget_class = widget.winfo_class()
            if widget_class not in {"TButton", "Button", "TEntry", "TCombobox", "TCheckbutton", "TScale", "TProgressbar"}:
                continue
            config = _config(widget, app)
            bound_tcl = config.get("textvariable") or config.get("variable")
            controls.append(
                {
                    "class": widget_class,
                    "text": config.get("text", ""),
                    "bound_tcl_variable": bound_tcl,
                    "bound_attribute": variable_names.get(bound_tcl),
                    "default_value": config.get("textvariable_value", config.get("variable_value")),
                    "values": config.get("values"),
                    "state": config.get("state"),
                    "layout": _layout(widget),
                }
            )
        pages[page_name] = {
            "title": getattr(app, "page_titles", {}).get(page_name),
            "controls": controls,
            "widget_count": len(widgets),
        }
    return pages


def main() -> int:
    app = gui.AnalysisGUI()
    app.withdraw()
    app.update_idletasks()
    embedded = [child for child in app.winfo_children() if isinstance(child, tk.Toplevel)]
    report = {
        "main_window": {"geometry": app.winfo_geometry(), "variables": _variables(app), "widgets": []},
        "embedded_windows": [],
        "main_page_inventory": _page_summary(app),
    }
    _walk(app, app, report["main_window"]["widgets"])
    for window in embedded:
        item = {"path": str(window), "geometry": window.winfo_geometry(), "widgets": []}
        _walk(window, app, item["widgets"])
        report["embedded_windows"].append(item)
    pipeline = getattr(app, "_pipeline", None)
    if pipeline is None:
        pipeline = next((value for value in vars(app).values() if value.__class__.__name__ == "_EmbeddedPipelineGUI"), None)
    report["embedded_pipeline_variables"] = _variables(pipeline) if pipeline is not None else {}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Main variables: {len(report['main_window']['variables'])}")
    print(f"Embedded variables: {len(report['embedded_pipeline_variables'])}")
    print(f"Widgets: {len(report['main_window']['widgets']) + sum(len(item['widgets']) for item in report['embedded_windows'])}")
    print("Main page controls:", sum(len(page["controls"]) for page in report["main_page_inventory"].values()))
    app.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
