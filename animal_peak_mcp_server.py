"""STDIO MCP adapter. Run in the workstation GUI environment, optionally via SSH.

No GUI or analysis is imported here. A live, explicitly enabled test-window
session is required. stdout is reserved exclusively for the MCP protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent, ToolAnnotations

from animal_peak_rpc import call_bridge, list_sessions as discover_sessions

server = FastMCP("AnimalStart workstation control", instructions=(
    "Control ONLY an explicitly enabled AnimalStart test-window session. First list_sessions, then get_state and describe_controls. "
    "Every mutation requires the chosen session_id and latest expected_revision from a read. Never silently retry writes after a timeout. "
    "set_parameters stages inputs; execute the corresponding apply/refresh action. Long tasks return job_id: poll get_jobs, do not restart. "
    "Use actual channel, Tag and ORIGINAL Trial IDs, not positional indices. Preserve manual points and chosen trials. "
    "filter/export and discarding unsaved results need explicit user authorization (confirm=true). "
    "Use get_plot_image to obtain PNG including title/axes; clipboard writes affect the workstation, not necessarily the local desktop. "
    "Data/output remain on workstation; exports are confined to the GUI-authorized export root. "
    "get_state identifies pending parameters and last applied results. A successful API call is not proof of physiological validity."))

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
CONFIRMED_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)


@server.tool(annotations=READ)
def list_sessions() -> dict:
    """List explicitly enabled GUI sessions of the SAME Windows account. Never returns authentication tokens.

    Registry entries may be stale after a crash; verify the chosen session using get_state.
    Empty list: open the AnimalStart test window on the workstation and enable its MCP control checkbox.
    """
    return dict(sessions=discover_sessions())


@server.tool(annotations=READ)
def get_state(session_id: str) -> dict:
    """Read current source, channel scope, staged/applied parameters, selected Trial lists, revision and active job.

    Read BEFORE any mutation; pass its revision as expected_revision. Human changes invalidate old revisions.
    """
    return call_bridge(session_id, "get_state")


@server.tool(annotations=READ)
def describe_controls(session_id: str) -> dict:
    """Discover ALL supported controls, bounds, enum values, tabs, plot targets and execute actions.

    GUI disabled/read-only flags and underlying analysis eligibility rules remain authoritative.
    """
    return call_bridge(session_id, "describe")


@server.tool(annotations=WRITE)
def set_parameters(session_id: str, expected_revision: int,
                   group: Literal["extraction","event","similarity","filter","view","manual"], values: dict[str,Any]) -> dict:
    """Stage a validated parameter patch using field names from describe_controls; not an automatic calculation.

    Event alignment=letter/dot/animal; duration_ms cannot be combined with explicit start_ms/end_ms.
    Dot/Letter look after onset, Animal looks before onset. Baseline defaults relative to Dot for Dot,
    LetterStart for others. Extraction aligned channels are read-only. Numeric values are never silently clipped.
    """
    return call_bridge(session_id,"set_parameters",dict(group=group,values=values),expected_revision=expected_revision)


@server.tool(annotations=WRITE)
def navigate(session_id: str, expected_revision: int, selection: dict[str,Any]) -> dict:
    """Select tab/channel/tag/ORIGINAL trial_id/branch/detail_page/reference_trial_id/rank_trial_id/similarity_tab.

    tab values: average,histogram,details,records,animal_reference,similarity,event.
    branch=peak/trough; detail_page is zero-based. trial_id automatically selects its detail page.
    reference_trial_id=null chooses automatic reference. Does not add channels or recompute peaks.
    """
    return call_bridge(session_id,"navigate",selection,expected_revision=expected_revision)


@server.tool(annotations=WRITE)
def set_trial_selection(session_id: str, expected_revision: int, target: Literal["peaks","event"],
                        trial_ids: list[int] | None) -> dict:
    """Use ORIGINAL Trial IDs: null=all eligible, []=none. Acts on current channel/Tag.

    peaks list affects peak/trough and legacy Animal reference averages, independent of event list.
    event list is SHARED across Letter/Dot/Animal alignments; current event config must be refreshed first.
    Event changes start a refresh job. Invalid trials are not silently accepted.
    """
    return call_bridge(session_id,"set_trial_selection",dict(target=target,trial_ids=trial_ids),expected_revision=expected_revision)


@server.tool(annotations=CONFIRMED_WRITE)
def execute(session_id: str, expected_revision: int, action: str, options: dict[str,Any] | None = None) -> dict:
    """Execute a whitelisted GUI action, not arbitrary Python or shell commands. See describe_controls.actions.

    extract (auto re-detect; discards edited results), refresh_average (preserve anchors and resize excerpts),
    refresh_event, compute_similarity, apply_manual/restore_manual, filter/restore_filter (FULL source duration),
    apply_global_y/auto_global_y, apply_event_y/auto_event_y, apply_width/auto_width, apply_letter_range,
    show_average, clear_manual_preview, reset_view, copy_plot, export_plot, export, cancel, close.
    options: confirm=true (required for filter/export/export_plot, or discarding unsaved edits),
    job_id (cancel), plot/index (image/reset), x_range/y_range (reset_view), filename (new PNG export_plot),
    directory (export subdirectory within GUI-authorized root). Exports never overwrite source H5.
    Poll get_jobs for asynchronous operations; don't rerun a timed-out mutation automatically.
    """
    if options is not None and "action" in options:
        raise ValueError("options不能覆盖action。")
    return call_bridge(session_id,"execute",dict(action=action,**(options or {})),expected_revision=expected_revision)


@server.tool(annotations=READ)
def get_jobs(session_id: str, job_id: str | None = None) -> dict:
    """Poll task state/progress/error/export_directory, including GUI-started jobs after MCP was enabled.

    Jobs belong to the live GUI process, not the SSH adapter. Reconnect/poll existing job instead of re-executing.
    Disabling/re-enabling MCP in the same window retains the controller's job history (latest 100).
    """
    return call_bridge(session_id,"get_jobs",{} if job_id is None else dict(job_id=job_id))


@server.tool(annotations=READ, structured_output=False)
def get_plot_image(session_id: str, plot: str = "event", index: int = 0) -> list:
    """Return the CURRENT full plot as PNG, INCLUDING title, channel/Tag/n, axes and curves.

    plot: event,average(0 peak/1 trough),histogram(0/1),details(current entire page),animal_reference,
    similarity(heatmap),manual_average(0/1). Does not copy toolbar/table or overwrite clipboard.
    Requires valid, non-busy current results; respects applied frame size/Y limits/current zoom.
    """
    response = call_bridge(session_id,"get_image",dict(plot=plot,index=index))
    image = response["result"]
    metadata = {key:value for key,value in image.items() if key != "data"}
    return [TextContent(type="text", text=json.dumps(dict(session_id=session_id,revision=response["revision"],**metadata))),
            ImageContent(type="image",mimeType=image["mime_type"],data=image["data"])]


@server.tool(annotations=READ)
def read_table(session_id: str, table: Literal["records","event","similarity","rankings"],
               offset: int = 0, limit: int = 100) -> dict:
    """Read displayed table headers and rows, including exclusion reasons. Page size 1..500.

    event/similarity/rankings reflect current group and computed state; records is full extraction table.
    """
    return call_bridge(session_id,"read_table",dict(table=table,offset=offset,limit=limit))


@server.tool(annotations=READ)
def read_records(session_id: str, filters: dict[str,int] | None = None, offset: int = 0, limit: int = 100) -> dict:
    """Read numerical extraction records, optional actual channel/tag/ORIGINAL trial_id filters. Page size 1..500."""
    if filters is not None and set(filters)-{"channel","tag","trial_id"}:
        raise ValueError("filters只允许channel/tag/trial_id。")
    return call_bridge(session_id,"read_records",dict(offset=offset,limit=limit,**(filters or {})))


@server.tool(annotations=READ)
def read_waveform(session_id: str, plot: str = "event", index: int = 0, curve: int = 0,
                  offset: int = 0, limit: int = 1000) -> dict:
    """Read plotted numerical x/y samples, paginated (1..4096). Returns curve_count and total sample count.

    Event curves: selected trial waves followed by mean. Details includes plotted lines/markers.
    Values follow current display convention; not a physical unit recalibration or raw-H5 file read.
    """
    return call_bridge(session_id,"read_waveform",dict(plot=plot,index=index,curve=curve,offset=offset,limit=limit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", help="Override private registry directory; must match GUI setting/environment")
    parser.add_argument("--diagnose", action="store_true", help="List registry sessions as JSON and exit; does not launch MCP")
    args = parser.parse_args()
    if args.session_dir:
        os.environ["ANIMAL_PEAK_MCP_SESSION_DIR"] = args.session_dir
    if args.diagnose:
        print(json.dumps(dict(sessions=discover_sessions()),ensure_ascii=False))
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
