"""Self-contained real-geometry WebGL viewer for C-M1R evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


def _mesh(row: dict[str, Any], name: str, color: str, opacity: float) -> go.Mesh3d:
    mesh = row.get(name, {})
    vertices, faces = mesh.get("vertices", []), mesh.get("faces", [])
    return go.Mesh3d(
        x=[point[0] for point in vertices], y=[point[1] for point in vertices], z=[point[2] for point in vertices],
        i=[face[0] for face in faces], j=[face[1] for face in faces], k=[face[2] for face in faces],
        name=name, color=color, opacity=opacity, flatshading=False,
    )


def _points(row: dict[str, Any], name: str, color: str, *, size: int = 5) -> go.Scatter3d:
    values = row.get(name, [])
    connected = name in {"source_human_right", "source_human_left", "actual_contact_normals", "contact_force_vectors", "contact_target_world_trajectory", "actual_fingertip_trajectory"}
    return go.Scatter3d(
        x=[None if point is None else point[0] for point in values], y=[None if point is None else point[1] for point in values], z=[None if point is None else point[2] for point in values],
        mode="lines+markers" if connected else "markers", name=name, marker={"size": size, "color": color, "symbol": "x" if name == "lost_contact_marker" else "circle"},
        line={"width": 5, "color": color}, text=row.get("actual_mujoco_contact_labels") if name == "actual_mujoco_contacts" else None,
        hovertemplate="%{text}<extra></extra>" if name == "actual_mujoco_contacts" else None,
    )


def build_html(payload: dict[str, Any], html_root: Path) -> tuple[Path, Path]:
    """Write a genuine 3D mesh viewer and an explicit diagnostic index."""
    html_root.mkdir(parents=True, exist_ok=True)
    frames = payload["frames"]
    first = frames[0]
    mesh_specs = (
        ("object_source_visual_mesh", "#ffe066", 0.18), ("object_simulated_visual_mesh", "#d6b400", 0.48), ("object_collision_mesh", "#8d99ae", 0.22),
        ("stage_b_right_visual_mesh", "#adb5bd", 0.20), ("stage_b_left_visual_mesh", "#ced4da", 0.20),
        ("cxa_right_visual_mesh", "#ff9f1c", 0.28), ("cxa_left_visual_mesh", "#80ed99", 0.28),
        ("new_reference_right_visual_mesh", "#9b5de5", 0.34), ("new_reference_left_visual_mesh", "#00bbf9", 0.34),
        ("new_actual_right_visual_mesh", "#f15bb5", 0.82), ("new_actual_left_visual_mesh", "#2ec4b6", 0.82),
        ("new_actual_right_collision_mesh", "#ff70a6", 0.16), ("new_actual_left_collision_mesh", "#70d6ff", 0.16),
        ("semantic_patch_surface_mesh", "#39ff14", 0.95),
    )
    point_specs = (
        ("source_human_right", "#ef233c"), ("source_human_left", "#277da1"), ("actual_mujoco_contacts", "#ffffff"),
        ("actual_contact_normals", "#bde0fe"), ("contact_force_vectors", "#00b4d8"), ("penetration_points", "#d00000"),
        ("lost_contact_marker", "#ff1744"), ("contact_target_object_marker", "#b5179e"), ("contact_target_world_trajectory", "#90be6d"),
        ("actual_fingertip_trajectory", "#43aa8b"),
    )
    trace_names = [spec[0] for spec in mesh_specs] + [spec[0] for spec in point_specs]
    figure = go.Figure()
    for spec in mesh_specs:
        figure.add_trace(_mesh(first, *spec))
    for name, color in point_specs:
        figure.add_trace(_points(first, name, color, size=7 if "contact" in name else 4))
    figure.frames = [go.Frame(name=row["event_id"], data=[_mesh(row, *spec) for spec in mesh_specs] + [_points(row, name, color, size=7 if "contact" in name else 4) for name, color in point_specs]) for row in frames]
    sliders = [{"steps": [{"method": "animate", "label": f"{row['experiment']}:{row['source_frame']}", "args": [[row["event_id"]], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for row in frames], "active": 0, "currentvalue": {"prefix": "real evidence frame "}}]
    def indices(*names: str) -> set[int]:
        return {trace_names.index(name) for name in names}
    groups = {
        "all": set(range(len(trace_names))),
        "new_actual": indices("object_simulated_visual_mesh", "new_actual_right_visual_mesh", "new_actual_left_visual_mesh", "semantic_patch_surface_mesh", "actual_mujoco_contacts", "lost_contact_marker", "contact_target_world_trajectory", "actual_fingertip_trajectory"),
        "contact": indices("object_simulated_visual_mesh", "new_actual_right_visual_mesh", "new_actual_left_visual_mesh", "new_actual_right_collision_mesh", "new_actual_left_collision_mesh", "semantic_patch_surface_mesh", "actual_mujoco_contacts", "actual_contact_normals", "contact_force_vectors", "lost_contact_marker", "contact_target_world_trajectory", "actual_fingertip_trajectory"),
        "reference_actual": indices("object_simulated_visual_mesh", "semantic_patch_surface_mesh", "new_reference_right_visual_mesh", "new_reference_left_visual_mesh", "new_actual_right_visual_mesh", "new_actual_left_visual_mesh"),
        "source_stageb_cxa": indices("object_source_visual_mesh", "source_human_right", "source_human_left", "stage_b_right_visual_mesh", "stage_b_left_visual_mesh", "cxa_right_visual_mesh", "cxa_left_visual_mesh"),
        "collision": indices("object_collision_mesh", "new_actual_right_collision_mesh", "new_actual_left_collision_mesh", "penetration_points", "actual_mujoco_contacts"),
    }
    figure.update_layout(
        title=f"{payload['disclaimer']}<br>Real MuJoCo states, full Wuji meshes, connected semantic patch", paper_bgcolor="#10131a", plot_bgcolor="#10131a", font={"color": "#f8f9fa"}, height=800,
        scene={"aspectmode": "cube", "bgcolor": "#10131a"}, sliders=sliders,
        updatemenus=[{"buttons": [{"label": label, "method": "update", "args": [{"visible": [index in visible for index in range(len(trace_names))]}]} for label, visible in groups.items()], "direction": "right", "x": 0.0, "y": 1.10}], margin={"l": 0, "r": 0, "t": 120, "b": 30},
    )
    metrics = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("patch distance and contact", "force / penetration", "mode"))
    x = [row["source_frame"] for row in frames]
    metrics.add_trace(go.Scatter(x=x, y=[row["metrics"]["patch_distance_m"] for row in frames], name="patch distance m"), row=1, col=1)
    metrics.add_trace(go.Scatter(x=x, y=[int(row["metrics"]["physical_contact"]) for row in frames], name="physical contact"), row=1, col=1)
    metrics.add_trace(go.Scatter(x=x, y=[row["metrics"]["force_n"] for row in frames], name="force N"), row=2, col=1)
    metrics.add_trace(go.Scatter(x=x, y=[row["metrics"]["penetration_m"] for row in frames], name="penetration m"), row=2, col=1)
    metrics.add_trace(go.Scatter(x=x, y=[row["mode"] for row in frames], name="mode"), row=3, col=1)
    metrics.update_layout(height=700, paper_bgcolor="#10131a", plot_bgcolor="#10131a", font={"color": "#f8f9fa"}, hovermode="x unified")
    event_map = {row["event_id"]: index for index, row in enumerate(frames)}
    layer_map = {name: sorted(values) for name, values in groups.items()}
    focus_map = {
        row["event_id"]: [
            sum(point[axis] for point in row.get("semantic_patch_surface_mesh", {}).get("vertices", [])) / max(1, len(row.get("semantic_patch_surface_mesh", {}).get("vertices", [])))
            for axis in range(3)
        ]
        for row in frames
    }
    viewer = figure.to_html(full_html=False, include_plotlyjs=False, div_id="cm1r-3d")
    curve = metrics.to_html(full_html=False, include_plotlyjs=False, div_id="cm1r-curves")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Stage C C-M1R contact mode</title><script>{get_plotlyjs()}</script>
<style>body{{margin:0;background:#10131a;color:#f8f9fa;font-family:system-ui}}.banner{{padding:14px 22px;background:#780000;font-weight:800;letter-spacing:.06em}}.note{{padding:10px 22px;color:#ffd166}}pre{{white-space:pre-wrap;padding:12px 22px;background:#171b26}}</style></head><body>
<div class='banner'>{payload['disclaimer']}</div><div class='note'>This is a real 3D WebGL reconstruction: complete Wuji visual/collision meshes, visual/collision object meshes, connected green semantic patch surface, actual MuJoCo contacts, normals, forces, and object-frame target markers.</div>
{viewer}{curve}<details><summary>Viewer metadata</summary><pre>{json.dumps(payload['metadata'], indent=2, sort_keys=True)}</pre></details>
<script>const e={json.dumps(event_map)};const layers={json.dumps(layer_map)};const focus={json.dumps(focus_map)};const p=new URLSearchParams(location.search);const event=p.get('event');if(event&&e[event]!==undefined){{Plotly.animate('cm1r-3d',[event],{{mode:'immediate',frame:{{duration:0,redraw:true}}}});}}const group=p.get('layers');if(group&&layers[group]){{Plotly.restyle('cm1r-3d',{{visible:Array.from({{length:{len(trace_names)}}},(_,i)=>layers[group].includes(i))}});}}const view=p.get('view')||'global';const cameras={{global:{{eye:{{x:1.55,y:-1.55,z:1.2}}}},close:{{eye:{{x:1.1,y:-1.1,z:.7}}}},top:{{eye:{{x:0,y:0,z:2.2}}}}}};const center=focus[event]||[0,0,0];const span=view==='close'?0.085:view==='top'?0.10:0.35;const ranges={{'scene.xaxis.range':[center[0]-span,center[0]+span],'scene.yaxis.range':[center[1]-span,center[1]+span],'scene.zaxis.range':[center[2]-span,center[2]+span]}};if(view==='global'){{delete ranges['scene.xaxis.range'];delete ranges['scene.yaxis.range'];delete ranges['scene.zaxis.range'];}}Plotly.relayout('cm1r-3d',Object.assign({{'scene.camera':cameras[view]||cameras.global}},ranges));</script></body></html>"""
    page = html_root / "stage_c_cm1r_contact_mode.html"
    page.write_text(html, encoding="utf-8")
    index = html_root / "stage_c_cm1r_visual_index.html"
    index.write_text("<!doctype html><meta charset='utf-8'><title>Stage C C-M1R visual index</title><h1>" + payload["disclaimer"] + "</h1><p><a href='stage_c_cm1r_contact_mode.html'>open real 3D contact-mode viewer</a></p><ul>" + "".join(f"<li><a href='stage_c_cm1r_contact_mode.html?event={row['event_id']}&view=close&layers=contact'>{row['experiment']} source {row['source_frame']}</a></li>" for row in frames) + "</ul>", encoding="utf-8")
    return page, index


def render_chrome_screenshots(html: Path, screenshot_root: Path, requests: list[tuple[str, str, str]], *, chrome: str = "/usr/bin/google-chrome") -> list[dict[str, Any]]:
    screenshot_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for event, view, name in requests:
        target = screenshot_root / f"{name}_{view}.png"
        query = urlencode({"event": event, "view": view, "layers": "contact" if "m1" in name else "new_actual"})
        completed = subprocess.run([chrome, "--headless", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=5000", "--window-size=1800,1200", f"--screenshot={target}", html.resolve().as_uri() + "?" + query], capture_output=True, text=True, timeout=60, check=False)
        rows.append({"event": event, "view": view, "name": name, "path": str(target), "status": "PASS" if completed.returncode == 0 and target.is_file() and target.stat().st_size else "FAIL", "returncode": completed.returncode, "stderr_tail": completed.stderr[-500:]})
    return rows
