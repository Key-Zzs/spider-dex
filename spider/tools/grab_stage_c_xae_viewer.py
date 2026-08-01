"""Real-mesh HTML and Chrome screenshot evidence for Stage C-XAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import struct
from typing import Any
from urllib.parse import urlencode

import mujoco
import numpy as np
from plotly.offline import get_plotlyjs
from scipy.spatial.transform import Rotation
import trimesh

from spider.geometry.collision_audit import mesh_from_model, region_for_geom
from spider.tools.grab_stage_c_xae import Context, _closest_contact, _json_default, evaluate_samples, load_context
from spider.tools.grab_stage_c_xar_viewer import _baked_robot_surface


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _compact_mesh(mesh: trimesh.Trimesh, max_faces: int) -> dict[str, np.ndarray]:
    result = mesh
    if len(result.faces) > max_faces:
        selected = np.linspace(0, len(result.faces) - 1, max_faces, dtype=np.int64)
        result = trimesh.Trimesh(vertices=np.asarray(result.vertices), faces=np.asarray(result.faces)[selected], process=False)
        result.remove_unreferenced_vertices()
    return {"vertices": np.asarray(result.vertices, dtype=np.float32), "faces": np.asarray(result.faces, dtype=np.int32)}


def _object_mesh(ctx: Context, state: np.ndarray, max_faces: int = 900) -> dict[str, np.ndarray]:
    ctx.data.qpos[:] = state
    ctx.data.qvel[:] = 0.0
    mujoco.mj_forward(ctx.model, ctx.data)
    meshes: list[trimesh.Trimesh] = []
    for geom in range(ctx.model.ngeom):
        name = mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if name.startswith("right_object_") and int(ctx.model.geom_group[geom]) in {1, 3}:
            try:
                meshes.append(mesh_from_model(ctx.model, ctx.data, geom))
            except ValueError:
                continue
    if not meshes:
        raise RuntimeError("XAE viewer found no real object mesh geom")
    return _compact_mesh(trimesh.util.concatenate(meshes), max_faces)


def _contact_region_mesh(ctx: Context, state: np.ndarray, side: str, finger: str, max_faces: int = 350) -> dict[str, np.ndarray]:
    ctx.data.qpos[:] = state
    ctx.data.qvel[:] = 0.0
    mujoco.mj_forward(ctx.model, ctx.data)
    meshes: list[trimesh.Trimesh] = []
    for geom in range(ctx.model.ngeom):
        if int(ctx.model.geom_group[geom]) != 1:
            continue
        geom_side, geom_finger = region_for_geom(ctx.model, geom)
        if geom_side == side and geom_finger == finger:
            try:
                meshes.append(mesh_from_model(ctx.model, ctx.data, geom))
            except ValueError:
                continue
    if not meshes:
        raise RuntimeError(f"XAE viewer found no visual surface for {side}_{finger}")
    return _compact_mesh(trimesh.util.concatenate(meshes), max_faces)


def _patch_mesh_world(ctx: Context, frame: int, role_id: str) -> dict[str, np.ndarray]:
    entry = next(item for item in ctx.entries if item.role["role_id"] == role_id)
    pose = ctx.source_qpos[frame, -14:-7]
    rotation = Rotation.from_quat(pose[3:7][[1, 2, 3, 0]])
    vertices = rotation.apply(np.asarray(entry.patch_mesh.vertices)) + pose[:3]
    return _compact_mesh(trimesh.Trimesh(vertices=vertices, faces=entry.patch_mesh.faces, process=False), 700)


def _select_events(ctx: Context, final_states: np.ndarray) -> list[dict[str, Any]]:
    before = evaluate_samples(ctx, ctx.qpos)
    after = evaluate_samples(ctx, final_states)
    after_map = {(row["frame"], row["role"]): row for row in after}
    values = np.asarray([row["distance_m"] for row in before])
    ordered = sorted(before, key=lambda row: float(row["distance_m"]))
    p95_value = float(np.percentile(values, 95))
    candidates: list[tuple[str, dict[str, Any]]] = []
    used_frames: set[int] = set()

    def add_distinct(label: str, ranked: list[dict[str, Any]]) -> None:
        row = next(item for item in ranked if int(item["frame"]) not in used_frames)
        used_frames.add(int(row["frame"]))
        candidates.append((label, row))

    add_distinct("p95_contributor", sorted(before, key=lambda row: abs(float(row["distance_m"]) - p95_value)))
    add_distinct("max_distance", sorted(before, key=lambda row: float(row["distance_m"]), reverse=True))
    add_distinct(
        "typical_pass",
        sorted(ordered, key=lambda row: abs(float(row["distance_m"]) - float(np.median(values)))),
    )
    add_distinct(
        "e4_static_best",
        sorted(
            before,
            key=lambda row: float(row["distance_m"]) - float(after_map[(row["frame"], row["role"])]["distance_m"]),
            reverse=True,
        ),
    )
    boundary_tail = [
        row
        for row in before
        if float(row["distance_m"]) > 0.020
        and min(int(row["frame"]) - int(row["role_start"]), int(row["role_end"]) - int(row["frame"])) <= 2
    ]
    add_distinct("role_boundary_tail", sorted(boundary_tail, key=lambda row: float(row["distance_m"]), reverse=True))
    events: list[dict[str, Any]] = []
    for label, row in candidates:
        final = after_map[(row["frame"], row["role"])]
        events.append({
            "label": label,
            "frame": int(row["frame"]),
            "source_frame": int(row["source_frame"]),
            "role": row["role"],
            "side": row["side"],
            "finger": row["finger"],
            "region": row["robot_contact_region"],
            "before_distance_m": row["distance_m"],
            "after_distance_m": final["distance_m"],
        })
    return events


def build_payload(ctx: Context) -> dict[str, Any]:
    final_path = ctx.run_root / "repair/repaired_cxa_v2_final/trajectory_depenetrated_init.npz"
    with np.load(final_path, allow_pickle=False) as archive:
        final = np.asarray(archive["qpos"], dtype=np.float64)
    events = _select_events(ctx, final)
    indices = np.asarray([row["frame"] for row in events], dtype=np.int64)
    stages = {"stage_b": ctx.baseline[indices], "repaired_cxa": ctx.qpos[indices], "xae_final": final[indices]}
    hands = {
        stage: _baked_robot_surface(Path(ctx.physics["scene_act"]), states, geom_group=1, max_faces=500)
        for stage, states in stages.items()
    }
    payload_events: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        frame = event["frame"]
        entry = next(item for item in ctx.entries if item.role["role_id"] == event["role"])
        before_tip, before_closest, _normal, before_distance, _triangle = _closest_contact(ctx, ctx.qpos[frame], frame, entry)
        final_tip, final_closest, normal, final_distance, triangle = _closest_contact(ctx, final[frame], frame, entry)
        payload_events.append({
            **event,
            "object": _object_mesh(ctx, ctx.baseline[frame]),
            "patch": _patch_mesh_world(ctx, frame, event["role"]),
            "contact_region": _contact_region_mesh(ctx, final[frame], event["side"], event["finger"]),
            "hands": {
                stage: {side: hands[stage][side][event_index] for side in ("right", "left")}
                for stage in stages
            },
            "vectors": {
                "repaired_cxa": {"tip": before_tip, "closest": before_closest, "distance_m": before_distance},
                "xae_final": {"tip": final_tip, "closest": final_closest, "distance_m": final_distance},
                "normal": normal,
                "nearest_triangle_id": triangle,
            },
        })
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "scene": ctx.physics["scene_act"],
        "object_visual_mesh": str(Path(ctx.physics["collision_cache"]) / "visual/visual.obj"),
        "events": payload_events,
        "disclaimer": "All triangles come from the real scene_act hand/object meshes and frozen semantic patch face IDs; no qpos alignment, static offset, or mesh substitution is applied.",
    }


def _document(payload: dict[str, Any]) -> str:
    packed = json.dumps(_plain(payload), ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Stage C-XAE contact audit</title>
<style>body{{margin:0;background:#0d141b;color:#e9f1f7;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}#bar{{padding:12px 18px;background:#172431}}#scene{{height:83vh}}select{{padding:5px;background:#293b49;color:white}}#info{{white-space:pre-wrap;padding:10px 18px;background:#172431;font:13px ui-monospace,monospace}}</style><body><div id='bar'><b>Stage C-XAE：真实 mesh / semantic patch / distance vector</b>　事件 <select id='event'></select>　视角 <select id='view'><option value='world'>world</option><option value='object'>object</option><option value='contact'>contact close-up</option></select></div><div id='scene'></div><pre id='info'></pre><script>{get_plotlyjs()}</script><script>
const D={packed},$=x=>document.getElementById(x),q=new URLSearchParams(location.search);D.events.forEach((e,i)=>$('event').add(new Option(`${{e.label}} | src ${{e.source_frame}} | ${{e.region}}`,i)));if(q.has('event'))$('event').value=q.get('event');if(q.has('view'))$('view').value=q.get('view');
const C={{stage_b:'#ffd166',repaired_cxa:'#ef476f',xae_final:'#06d6a0',object:'#457b9d',patch:'#c77dff',region:'#00b4d8'}};function mesh(name,m,c,o){{return{{type:'mesh3d',name,x:m.vertices.map(v=>v[0]),y:m.vertices.map(v=>v[1]),z:m.vertices.map(v=>v[2]),i:m.faces.map(v=>v[0]),j:m.faces.map(v=>v[1]),k:m.faces.map(v=>v[2]),color:c,opacity:o,flatshading:true,hoverinfo:'name'}}}}function line(name,a,b,c,w=7){{return{{type:'scatter3d',mode:'lines+markers',name,x:[a[0],b[0]],y:[a[1],b[1]],z:[a[2],b[2]],line:{{color:c,width:w}},marker:{{size:3,color:c}}}}}}function draw(){{let e=D.events[+$('event').value],v=$('view').value,tr=[];for(let s of ['stage_b','repaired_cxa','xae_final'])for(let side of ['right','left'])tr.push(mesh(`${{s}} ${{side}}`,e.hands[s][side],C[s],s==='xae_final'?.45:.22));tr.push(mesh('object visual',e.object,C.object,.34),mesh('semantic patch surface',e.patch,C.patch,.70),mesh('assigned fingertip visual surface',e.contact_region,C.region,.40));tr.push(line('repaired distance',e.vectors.repaired_cxa.tip,e.vectors.repaired_cxa.closest,C.repaired_cxa),line('XAE final distance',e.vectors.xae_final.tip,e.vectors.xae_final.closest,C.xae_final));let center=e.vectors.xae_final.closest,eye=v==='world'?{{x:1.45,y:1.25,z:.85}}:v==='object'?{{x:.95,y:.25,z:.35}}:{{x:.42,y:.18,z:.16}};let range=v==='contact'?.045:v==='object'?.10:null,scene={{aspectmode:range?'cube':'data',camera:{{eye}}}};if(range){{scene.xaxis={{range:[center[0]-range,center[0]+range]}};scene.yaxis={{range:[center[1]-range,center[1]+range]}};scene.zaxis={{range:[center[2]-range,center[2]+range]}}}}Plotly.react('scene',tr,{{paper_bgcolor:'#0d141b',plot_bgcolor:'#0d141b',font:{{color:'#e9f1f7'}},scene,margin:{{l:0,r:0,t:20,b:0}},legend:{{orientation:'h'}}}},{{responsive:true}});$('info').textContent=`${{e.label}} | source frame ${{e.source_frame}} | role ${{e.role}} | ${{e.region}}\nrepaired C-XA distance=${{(e.before_distance_m*1000).toFixed(3)}} mm | XAE final=${{(e.after_distance_m*1000).toFixed(3)}} mm\nnearest patch triangle=${{e.vectors.nearest_triangle_id}}\n${{D.disclaimer}}`;}}$('event').onchange=draw;$('view').onchange=draw;draw();</script></body></html>"""


def render_chrome(html: Path, screenshot_root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    screenshot_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(payload["events"]):
        for view in ("world", "object", "contact"):
            target = screenshot_root / f"xae_{event['label']}_{event['source_frame']}_{view}.png"
            query = urlencode({"event": index, "view": view})
            completed = subprocess.run(
                [
                    "/usr/bin/google-chrome",
                    "--headless",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--virtual-time-budget=5000",
                    "--window-size=1800,1200",
                    f"--screenshot={target}",
                    html.resolve().as_uri() + "?" + query,
                ],
                capture_output=True, text=True, timeout=60, check=False,
            )
            rows.append({"event": event["label"], "source_frame": event["source_frame"], "view": view, "path": str(target), "status": "PASS" if completed.returncode == 0 and target.is_file() and target.stat().st_size else "FAIL", "stderr_tail": completed.stderr[-300:]})
    return rows


def render(run_root: str, paths_config: str, repaired_root: str) -> dict[str, Any]:
    ctx = load_context(paths_config, run_root, repaired_root)
    payload = build_payload(ctx)
    html_root = ctx.run_root / "html"
    html_root.mkdir(parents=True, exist_ok=True)
    html = html_root / "stage_c_xae_contact_audit.html"
    html.write_text(_document(payload), encoding="utf-8")
    index = html_root / "stage_c_xae_visual_index.html"
    links = "".join(f"<li><a href='stage_c_xae_contact_audit.html?event={i}&view=contact'>{row['label']} / source {row['source_frame']}</a></li>" for i, row in enumerate(payload["events"]))
    index.write_text(f"<!doctype html><meta charset='utf-8'><title>XAE visual index</title><h1>Stage C-XAE visual index</h1><ul>{links}</ul><p>{payload['disclaimer']}</p>", encoding="utf-8")
    screenshots = render_chrome(html, ctx.run_root / "screenshots", payload)
    manifest = {"schema_version": 1, "status": "PASS" if all(row["status"] == "PASS" for row in screenshots) else "FAIL", "renderer": "Chrome headless self-contained Plotly with real MuJoCo meshes", "html": str(html), "screenshots": screenshots}
    (ctx.run_root / "screenshots/XAE_SCREENSHOT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ctx.run_root / "screenshots/XAE_SCREENSHOT_REVIEW.md").write_text("# XAE screenshot review\n\nStatus: `PENDING_CODEX_IMAGE_REVIEW`.\n", encoding="utf-8")
    return {"html": str(html), "index": str(index), "manifest": str(ctx.run_root / "screenshots/XAE_SCREENSHOT_MANIFEST.json"), "count": len(screenshots), "status": manifest["status"]}


def verify_existing(run_root: str) -> dict[str, Any]:
    """Verify PNGs produced by an explicitly invoked host Chrome process.

    The managed test sandbox can deny Chrome crashpad even when the same
    command succeeds directly on the host.  This verifier never fabricates an
    image or accepts file existence alone: it checks the PNG signature, IHDR,
    exact screenshot dimensions, and non-trivial payload size for every row
    already declared by the generated manifest.
    """
    manifest_path = Path(run_root).resolve() / "screenshots/XAE_SCREENSHOT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["screenshots"]:
        target = Path(row["path"])
        header = target.read_bytes()[:24] if target.is_file() else b""
        valid_header = len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR"
        width, height = struct.unpack(">II", header[16:24]) if valid_header else (0, 0)
        size = target.stat().st_size if target.is_file() else 0
        valid = valid_header and (width, height) == (1800, 1200) and size >= 10_000
        row.update({
            "status": "PASS" if valid else "FAIL",
            "verification": {
                "png_signature": valid_header,
                "width": width,
                "height": height,
                "size_bytes": size,
                "minimum_size_bytes": 10_000,
            },
        })
    manifest["status"] = "PASS" if all(row["status"] == "PASS" for row in manifest["screenshots"]) else "FAIL"
    manifest["host_chrome_outputs_verified"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "count": len(manifest["screenshots"]), "status": manifest["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--paths-config", default="configs/local/paths.yaml")
    parser.add_argument("--repaired-root", required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.verify_existing:
        print(json.dumps(verify_existing(args.run_root), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(render(args.run_root, args.paths_config, args.repaired_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
