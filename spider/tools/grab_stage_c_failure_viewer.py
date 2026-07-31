"""Failure-only Stage C viewer and deterministic Chrome screenshot capture.

This module deliberately has no acceptance gate and cannot emit an acceptance
artifact.  It consumes a compact JSON payload produced by the diagnostic tool
and writes only into the caller-selected diagnostic directory.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


DISCLAIMER = "FAILURE DIAGNOSTIC — NOT AN ACCEPTANCE ARTIFACT"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _trace(row: dict[str, Any], name: str, color: str, *, size: int = 5) -> go.Scatter3d:
    points = row.get(name, [])
    connected = name in {
        "source_human_right",
        "source_human_left",
        "actual_finger_trajectory",
        "semantic_patch_trajectory",
        "actual_contact_normals",
        "contact_force_vectors",
    }
    marker_symbol = "x" if name == "lost_contact_marker" else "circle"
    return go.Scatter3d(
        x=[None if point is None else point[0] for point in points],
        y=[None if point is None else point[1] for point in points],
        z=[None if point is None else point[2] for point in points],
        mode="lines+markers" if connected else "markers",
        name=name,
        marker={"size": size, "color": color, "symbol": marker_symbol},
        line={"width": 5, "color": color},
        text=row.get("actual_mujoco_contact_labels") if name == "actual_mujoco_contacts" else None,
        hovertemplate="%{text}<extra></extra>" if name == "actual_mujoco_contacts" else None,
    )


def _mesh(row: dict[str, Any], name: str, color: str, opacity: float) -> go.Mesh3d:
    mesh = row.get(name, {})
    vertices = mesh.get("vertices", [])
    faces = mesh.get("faces", [])
    return go.Mesh3d(
        x=[point[0] for point in vertices],
        y=[point[1] for point in vertices],
        z=[point[2] for point in vertices],
        i=[face[0] for face in faces],
        j=[face[1] for face in faces],
        k=[face[2] for face in faces],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=False,
    )


def build_failure_html(payload: dict[str, Any], output: Path) -> Path:
    """Write a self-contained full-hand failure diagnostic HTML."""
    rows = payload["frames"]
    first = rows[0]
    layer_specs = (
        ("object_source_visual_mesh", "#ffe066", 0.20),
        ("object_simulated_visual_mesh", "#d6b400", 0.48),
        ("object_collision_mesh", "#8d99ae", 0.22),
        ("stage_b_right_visual_mesh", "#adb5bd", 0.20),
        ("stage_b_left_visual_mesh", "#ced4da", 0.20),
        ("cxa_right_visual_mesh", "#ff9f1c", 0.30),
        ("cxa_left_visual_mesh", "#80ed99", 0.30),
        ("failed_reference_right_visual_mesh", "#9b5de5", 0.34),
        ("failed_reference_left_visual_mesh", "#00bbf9", 0.34),
        ("failed_actual_right_visual_mesh", "#f15bb5", 0.82),
        ("failed_actual_left_visual_mesh", "#2ec4b6", 0.82),
        ("failed_actual_right_collision_mesh", "#ff70a6", 0.16),
        ("failed_actual_left_collision_mesh", "#70d6ff", 0.16),
        ("semantic_patch_surface_mesh", "#39ff14", 0.95),
    )
    point_specs = (
        ("source_human_right", "#ef233c"),
        ("source_human_left", "#277da1"),
        ("stage_b_kinematic_wuji", "#adb5bd"),
        ("cxa_corrected_static_wuji", "#80ed99"),
        ("failed_dynamic_reference", "#9b5de5"),
        ("failed_dynamic_actual", "#f15bb5"),
        ("corrected_active_contact_anchors", "#70e000"),
        ("unreliable_source_records", "#6a040f"),
        ("actual_mujoco_contacts", "#ffffff"),
        ("valid_region_contacts", "#39ff14"),
        ("wrong_region_contacts", "#fb8500"),
        ("lost_contact_marker", "#ff1744"),
        ("penetration_points", "#d00000"),
        ("joint_limit_active_fingers", "#ff006e"),
        ("actual_finger_trajectory", "#43aa8b"),
        ("semantic_patch_trajectory", "#90be6d"),
        ("actual_contact_normals", "#bde0fe"),
        ("contact_force_vectors", "#00b4d8"),
    )
    trace_names = [spec[0] for spec in layer_specs] + [spec[0] for spec in point_specs]
    figure = go.Figure()
    for name, color, opacity in layer_specs:
        figure.add_trace(_mesh(first, name, color, opacity))
    for name, color in point_specs:
        figure.add_trace(_trace(first, name, color, size=7 if "contact" in name else 4))
    plot_frames = []
    for row in rows:
        data = [_mesh(row, *spec) for spec in layer_specs]
        data.extend(_trace(row, name, color, size=7 if "contact" in name else 4) for name, color in point_specs)
        plot_frames.append(go.Frame(name=str(row["source_frame"]), data=data))
    figure.frames = plot_frames
    steps = [
        {
            "method": "animate",
            "label": str(row["source_frame"]),
            "args": [[str(row["source_frame"])], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
        }
        for row in rows
    ]
    def indices(*names: str) -> set[int]:
        return {trace_names.index(name) for name in names}

    groups = {
        "all": set(range(len(trace_names))),
        "failure_core": indices(
            "object_simulated_visual_mesh", "failed_reference_right_visual_mesh", "failed_reference_left_visual_mesh",
            "failed_actual_right_visual_mesh", "failed_actual_left_visual_mesh", "semantic_patch_surface_mesh",
            "corrected_active_contact_anchors", "actual_mujoco_contacts", "valid_region_contacts",
            "wrong_region_contacts", "lost_contact_marker", "actual_finger_trajectory", "semantic_patch_trajectory",
        ),
        "patch_contacts": indices(
            "object_simulated_visual_mesh", "semantic_patch_surface_mesh", "corrected_active_contact_anchors",
            "actual_mujoco_contacts", "valid_region_contacts", "wrong_region_contacts", "lost_contact_marker",
            "actual_contact_normals", "contact_force_vectors",
        ),
        "visual_collision": indices(
            "object_simulated_visual_mesh", "object_collision_mesh", "failed_actual_right_visual_mesh",
            "failed_actual_left_visual_mesh", "failed_actual_right_collision_mesh", "failed_actual_left_collision_mesh",
        ),
        "reference_actual": indices(
            "object_simulated_visual_mesh", "semantic_patch_surface_mesh", "failed_reference_right_visual_mesh",
            "failed_reference_left_visual_mesh", "failed_actual_right_visual_mesh", "failed_actual_left_visual_mesh",
            "lost_contact_marker", "actual_finger_trajectory", "semantic_patch_trajectory",
        ),
        "source_stageb_cxa": indices(
            "object_source_visual_mesh", "source_human_right", "source_human_left", "stage_b_right_visual_mesh",
            "stage_b_left_visual_mesh", "cxa_right_visual_mesh", "cxa_left_visual_mesh",
        ),
        "force_penetration": indices(
            "object_collision_mesh", "failed_actual_right_collision_mesh", "failed_actual_left_collision_mesh",
            "actual_mujoco_contacts", "penetration_points", "actual_contact_normals", "contact_force_vectors",
        ),
    }
    buttons = [
        {"label": label, "method": "update", "args": [{"visible": [index in visible for index in range(len(layer_specs) + len(point_specs))]}]}
        for label, visible in groups.items()
    ]
    figure.update_layout(
        title=f"{DISCLAIMER}<br>{payload['best_attempt']} — full Wuji surfaces — first failure {payload['first_failure']['first_failure_source_frame']}",
        paper_bgcolor="#10131a",
        plot_bgcolor="#10131a",
        font={"color": "#f8f9fa"},
        height=760,
        scene={"aspectmode": "cube", "bgcolor": "#10131a"},
        sliders=[{"steps": steps, "active": 0, "currentvalue": {"prefix": "source frame "}}],
        updatemenus=[{"buttons": buttons, "direction": "right", "x": 0.0, "y": 1.09}],
        margin={"l": 0, "r": 0, "t": 110, "b": 30},
    )
    curves = payload["curves"]
    curve_figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        specs=[[{"secondary_y": True}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=("contact quality", "tracking and joint limits", "contact dynamics", "object tracking"),
    )
    colors = ["#00f5d4", "#f15bb5", "#fee440", "#00bbf9", "#fb5607", "#8338ec", "#80ed99", "#ff006e", "#90be6d", "#43aa8b", "#ff9f1c", "#bde0fe", "#ffffff"]
    color_index = 0
    for row_index, (group_name, series) in enumerate(curves.items(), start=1):
        for name, values in series.items():
            secondary = (row_index == 1 and "distance" in name) or (row_index == 3 and "depth" in name) or (row_index == 4 and "rotation" in name)
            curve_figure.add_trace(go.Scatter(x=payload["source_frames"], y=values, name=name, line={"color": colors[color_index % len(colors)]}), row=row_index, col=1, secondary_y=secondary if row_index in {1, 3, 4} else None)
            color_index += 1
    for event in payload["events"]:
        curve_figure.add_vline(x=event["source_frame"], line_dash="dash", line_color="#ffffff", opacity=0.34)
    curve_figure.update_layout(height=860, paper_bgcolor="#10131a", plot_bgcolor="#10131a", font={"color": "#f8f9fa"}, hovermode="x unified", margin={"l": 58, "r": 58, "t": 55, "b": 45})
    figure_html = figure.to_html(full_html=False, include_plotlyjs=False, div_id="failure-3d")
    curve_html = curve_figure.to_html(full_html=False, include_plotlyjs=False, div_id="failure-curves")
    metadata = json.dumps(payload["metadata"], indent=2, sort_keys=True)
    frame_map = json.dumps({str(row["source_frame"]): index for index, row in enumerate(rows)})
    layer_map = json.dumps({name: sorted(indices) for name, indices in groups.items()})
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{DISCLAIMER}</title>
<style>body{{margin:0;background:#10131a;color:#f8f9fa;font-family:system-ui}} .banner{{padding:14px 22px;background:#780000;font-weight:800;letter-spacing:.08em}} pre{{white-space:pre-wrap;padding:12px 22px;background:#171b26}} .note{{padding:10px 22px;color:#ffd166}} .facts{{display:flex;gap:24px;flex-wrap:wrap;padding:10px 22px;background:#171b26}} .facts b{{color:#39ff14}}</style>
<script>{get_plotlyjs()}</script></head><body><div class='banner'>{DISCLAIMER}</div>
<div class='note'>Full Wuji visual surfaces, collision proxies, connected semantic patch surface, source skeletons, reference and actual states. Legend entries can be toggled independently.</div>
<div class='facts'><span>first loss <b>{payload['first_failure']['first_failure_source_frame']}</b></span><span>finger <b>{payload['first_failure']['side']} {payload['first_failure']['finger']}</b></span><span>role <b>{payload['first_failure']['role_type']}</b></span><span>actual pair <b>{payload['first_failure']['actual_geom_pair']}</b></span></div>
{figure_html}{curve_html}<details><summary>Artifact metadata</summary><pre>{metadata}</pre></details>
<script>
const params=new URLSearchParams(location.search); const fmap={frame_map}; const lmap={layer_map};
const requested=params.get('frame'); if(requested && fmap[requested]!==undefined){{Plotly.animate('failure-3d',[requested],{{mode:'immediate',frame:{{duration:0,redraw:true}}}});}}
const layer=params.get('layers'); if(layer && lmap[layer]){{const n={len(layer_specs)+len(point_specs)}; Plotly.restyle('failure-3d',{{visible:Array.from({{length:n}},(_,i)=>lmap[layer].includes(i))}});}}
const globalBounds={json.dumps(payload['metadata']['global_bounds'])}; const defaultCloseCenter={json.dumps(payload['metadata']['close_center'])}; const focusCenters={json.dumps(payload['metadata']['frame_focus_centers'])};
const closeCenter=(requested && focusCenters[requested])?focusCenters[requested]:defaultCloseCenter;
const span=Math.max(...globalBounds[1].map((v,i)=>v-globalBounds[0][i]))*.58; const center=globalBounds[0].map((v,i)=>(v+globalBounds[1][i])/2);
const ranges={{global:center.map(v=>[v-span,v+span]),close:closeCenter.map(v=>[v-.075,v+.075]),top:closeCenter.map(v=>[v-.09,v+.09]),opposite:closeCenter.map(v=>[v-.09,v+.09])}};
const cameras={{global:{{eye:{{x:1.55,y:-1.55,z:1.2}}}},close:{{eye:{{x:1.15,y:-1.15,z:.75}}}},top:{{eye:{{x:0,y:0,z:2.25}}}},opposite:{{eye:{{x:-1.4,y:1.4,z:1.0}}}}}};
const view=params.get('view')||'global'; const range=ranges[view]||ranges.global; Plotly.relayout('failure-3d',{{'scene.camera':cameras[view]||cameras.global,'scene.xaxis.range':range[0],'scene.yaxis.range':range[1],'scene.zaxis.range':range[2]}});
</script></body></html>"""
    _write(output, html)
    return output


def render_chrome_screenshots(
    html: Path,
    screenshot_dir: Path,
    requests: list[dict[str, Any]],
    *,
    chrome: str = "/usr/bin/google-chrome",
) -> list[dict[str, Any]]:
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for request in requests:
        filename = request["filename"]
        target = screenshot_dir / filename
        query = urlencode({key: value for key, value in request.items() if key in {"frame", "view", "layers"}})
        url = html.resolve().as_uri() + "?" + query
        command = [
            chrome,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--virtual-time-budget=4000",
            "--window-size=1800,1200",
            f"--screenshot={target}",
            url,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
        rows.append(
            {
                **request,
                "path": str(target),
                "url": url,
                "status": "PASS" if completed.returncode == 0 and target.is_file() and target.stat().st_size else "FAIL",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-500:],
            }
        )
    return rows
