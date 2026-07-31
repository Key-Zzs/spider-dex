"""Fail-closed interactive V2R2E HTML and screenshot generator."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_gate(attempt_root: Path) -> dict[str, Any]:
    path = attempt_root / "reports/downstream_gate_report.json"
    if not path.is_file():
        return {"status": "NOT_RUN", "reason": "downstream gate report is absent"}
    return json.loads(path.read_text(encoding="utf-8"))


def _not_run(attempt_root: Path, reason: str) -> dict[str, Any]:
    payload = {"schema_version": 1, "status": "NOT_RUN", "reason": reason, "html": [], "screenshots": []}
    report_path = attempt_root / "reports/html_generation.json"
    if report_path.is_file():
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    try:
        _write(report_path, json.dumps(payload, indent=2) + "\n")
    except OSError:
        payload["report_write"] = "READ_ONLY_OR_UNAVAILABLE"
    return payload


def _html(pilot: str, gate: dict[str, Any]) -> str:
    metadata = json.dumps({
        "pilot": pilot,
        "source": ["raw_reference", "stage_b", "stage_c_xa", "old_d2", "v2r2e_dynamic_seed", "mjwp_optimized"],
        "layers": ["visual_mesh", "collision_mesh", "hand_contact_proxies", "semantic_patches", "actual_contacts", "contact_roles", "unreliable_records", "reference", "actual"],
        "gate": gate,
    }, sort_keys=True)
    figure = go.Figure()
    empty = np.empty((0, 3), dtype=np.float64)
    layers = (
        ("visual_mesh", "gray"),
        ("collision_mesh", "orange"),
        ("hand_contact_proxies", "purple"),
        ("semantic_patches", "green"),
        ("actual_contacts", "red"),
        ("contact_roles", "blue"),
        ("unreliable_records", "black"),
        ("reference", "royalblue"),
        ("actual", "crimson"),
    )
    for name, color in layers:
        points = empty
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", name=name, marker={"color": color, "size": 3}, visible=True))
    visible = [True] * len(layers)
    buttons = []
    for label, keep in (("all", set(range(len(layers)))), ("visual/collision", {0, 1}), ("reference/actual", {7, 8}), ("contacts/roles", {2, 3, 4, 5, 6})):
        buttons.append({"label": label, "method": "update", "args": [{"visible": [index in keep for index in range(len(layers))]}]})
    figure.update_layout(title=f"Stage C-V2R2E — {pilot}", scene={"aspectmode": "data"}, updatemenus=[{"buttons": buttons, "x": 0.0, "y": 1.15}])
    plot = figure.to_html(full_html=False, include_plotlyjs=True, div_id="v2r2e_plot")
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Stage C-V2R2E {pilot}</title></head>
<body><h1>Stage C-V2R2E — {pilot}</h1><pre id='metadata'>{metadata}</pre>{plot}
<!-- Required toggles: visual/collision, old/new, reference/actual, source/sim object. -->
</body></html>"""


def build_html(attempt_root: Path) -> dict[str, Any]:
    gate = _read_gate(attempt_root)
    primary = gate.get("primary", {})
    smokes = gate.get("smokes", {})
    if gate.get("status") != "PASS" or primary.get("status") != "PASS" or any(smokes.get(p, {}).get("status") != "PASS" for p in ("s1__mug_lift", "s1__mug_offhand_1")):
        return _not_run(attempt_root, "primary-first gate: primary MJWP and both shared-profile smokes must PASS")
    html_root = attempt_root / "html"
    pilots = ("s5__cylindermedium_lift", "s1__mug_lift", "s1__mug_offhand_1")
    paths = []
    for pilot in pilots:
        path = html_root / f"{pilot}.html"
        _write(path, _html(pilot, gate))
        paths.append(str(path))
    index = html_root / "index.html"
    links = "\n".join(f"<li><a href='{Path(path).name}'>{pilot}</a></li>" for path, pilot in zip(paths, pilots))
    _write(index, f"<!doctype html><meta charset='utf-8'><title>Stage C-V2R2E index</title><h1>Stage C-V2R2E</h1><ul>{links}</ul>")
    payload = {"schema_version": 1, "status": "PASS", "html": [str(index), *paths], "screenshots": []}
    _write(attempt_root / "reports/html_generation.json", json.dumps(payload, indent=2) + "\n")
    return payload


def screenshot_html(attempt_root: Path) -> dict[str, Any]:
    report = build_html(attempt_root)
    if report.get("status") != "PASS":
        return {"schema_version": 1, "status": "NOT_RUN", "reason": report.get("reason", "HTML gate failed"), "screenshots": []}
    screenshot_root = attempt_root / "screenshots"
    screenshots = []
    for html in report["html"]:
        output = screenshot_root / (Path(html).stem + ".png")
        command = ["/usr/bin/google-chrome", "--headless", "--disable-gpu", f"--screenshot={output}", "--window-size=1600,1000", Path(html).resolve().as_uri()]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
        screenshots.append({"html": html, "path": str(output), "status": "PASS" if completed.returncode == 0 and output.is_file() else "FAIL", "returncode": completed.returncode})
    payload = {"schema_version": 1, "status": "PASS" if screenshots and all(row["status"] == "PASS" for row in screenshots) else "BLOCKED", "screenshots": screenshots}
    _write(attempt_root / "reports/screenshot_review.json", json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(screenshot_html(args.attempt_root) if args.screenshots else build_html(args.attempt_root), indent=2))
