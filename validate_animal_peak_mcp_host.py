"""Developer-only synthetic GUI host for SSH validation; never opens EEG files.

Use a private temporary registry and a separate staged code directory. The main
program and the user's existing windows are not touched. Create <ready>.stop to
end this host; otherwise it automatically exits after 15 minutes.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir",required=True)
    parser.add_argument("--ready-file",required=True)
    parser.add_argument("--fallback-project",help="Original project for unchanged dependencies such as gui_core.pyc")
    args = parser.parse_args()
    if args.fallback_project: sys.path.insert(1,str(Path(args.fallback_project).resolve()))
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if os.name == "nt":
        os.environ.setdefault("QT_QPA_FONTDIR",str(Path(os.environ.get("SystemRoot","C:/Windows"))/"Fonts"))
    os.environ["ANIMAL_PEAK_MCP_SESSION_DIR"] = str(Path(args.registry_dir).resolve())
    from test_animal_peak_event_vep import prepared
    from animal_peak_dialog import QtCore, QtWidgets, AnimalPeakDialog
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source,markers,matches,output = prepared()
    parent = QtWidgets.QWidget()
    d = AnimalPeakDialog(parent,source,markers,matches,output["settings"])
    d.analysis_finished(output)
    d.tabs.setCurrentWidget(d.letter_vep_page)
    d.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen,True)
    d.parameter_toggle.setChecked(False); d.resize(1360,960); d.show()
    d.mcp_export_root.setText(str(Path(args.ready_file).resolve().parent/"exports"))
    d.mcp_enabled_check.setChecked(True)
    if d._mcp_bridge is None: raise RuntimeError(d.mcp_status.text())
    ready = Path(args.ready_file).resolve()
    ready.write_text(json.dumps(dict(session_id=d._mcp_bridge.session_id,pid=os.getpid(),
        source="synthetic validation only")),encoding="utf-8")
    print("Synthetic MCP validation host ready: " + d._mcp_bridge.session_id, flush=True)
    app.setQuitOnLastWindowClosed(False)
    stop = QtCore.QTimer(d)
    stop.timeout.connect(lambda: app.quit() if Path(str(ready)+".stop").exists() else None)
    stop.start(500)
    QtCore.QTimer.singleShot(15*60*1000,app.quit)
    try:
        app.exec()
    finally:
        if d.worker is not None:
            d.worker.cancel(); d.worker.wait(15000)
        d.mcp_enabled_check.setChecked(False)
        d.deleteLater(); parent.deleteLater(); app.processEvents()


if __name__ == "__main__": main()
