"""Regression audit: compare the migrated Qt shell with the Tk UI inventory.

Run on Windows (or with ``QT_QPA_PLATFORM=offscreen``) after UI edits.  It
does not execute long analyses; it verifies that every old controller
variable is retained and that each old logical action still has a Qt callback.
Matplotlib navigation is intentionally represented by its standard icon
toolbar instead of duplicated text buttons.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel, QProgressBar

from qt_gui import QtAnalysisGUI


def main() -> int:
    root = Path(__file__).resolve().parent
    inventory = json.loads((root / "legacy_ui_inventory.json").read_text(encoding="utf-8"))
    app = QApplication.instance() or QApplication([])
    window = QtAnalysisGUI()
    missing_variables = [name for name in inventory["main_window"]["variables"] if not hasattr(window, name)]
    # The legacy preview commands remain available through the compact
    # preview controls and Matplotlib's icon toolbar.  Verify callbacks,
    # rather than forcing a duplicate text-button panel into the page.
    required_actions = (
        "run_preprocess", "run_preprocess_all_channels", "open_remapped_overview",
        "run_bad_channel_check", "run_selected_ica", "run_ica_snr", "clear_ica",
        "open_overview",
    )
    missing_actions = [name for name in required_actions if not callable(getattr(window, name, None))]
    live_bindings = {
        "import_progress_var": QProgressBar,
        "preprocess_filter_progress_var": QProgressBar,
        "lfp_progress_var": QProgressBar,
        "preprocess_status_var": QLabel,
        "lfp_status_var": QLabel,
        "spike_status_var": QLabel,
    }
    incorrect_bindings = [name for name, expected in live_bindings.items() if not isinstance(getattr(window, name), expected)]
    print(f"variables: {len(inventory['main_window']['variables'])}; missing: {len(missing_variables)}")
    print("missing variable names:", ", ".join(missing_variables) or "none")
    print("missing legacy action callbacks:", ", ".join(missing_actions) or "none")
    print("incorrect live variable bindings:", ", ".join(incorrect_bindings) or "none")
    window.close()
    app.quit()
    return 1 if missing_variables or missing_actions or incorrect_bindings else 0


if __name__ == "__main__":
    raise SystemExit(main())
