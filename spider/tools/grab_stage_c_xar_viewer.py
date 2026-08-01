"""Self-contained real-mesh WebGL viewer for Stage C-XAR before/after evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
from plotly.offline import get_plotlyjs

from spider.datasets.paths import load_project_paths
from spider.tools.grab_source_geometry_audit import FINGERTIP_NAMES, _extract_mujoco_geometry, _site_transform, object_surface_payload, robot_surface_payload
from spider.tools.grab_stage_c import _stage_b_act_baseline


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hand_points(scene: Path, qpos: np.ndarray) -> dict[str, Any]:
    geometry = _extract_mujoco_geometry(scene, qpos)
    result: dict[str, Any] = {"object": geometry["object_transform"]}
    for side in ("right", "left"):
        result[side] = {
            "palm": _site_transform(geometry, f"{side}_palm"),
            "tips": np.stack([geometry["site_positions"][:, geometry["site_names"].index(f"{side}_{finger}_tip")] for finger in FINGERTIP_NAMES], axis=1),
        }
    return result


def _snapshot_points(points: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    """Keep mesh-bearing evidence bounded to explicitly selected audit frames."""
    return {
        "object": points["object"][indices],
        **{
            side: {"palm": points[side]["palm"][indices], "tips": points[side]["tips"][indices]}
            for side in ("right", "left")
        },
    }


def _compact_surface(stages: dict[str, dict[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    """Share invariant local meshes, retaining only stage/frame transforms."""
    reference = stages["base"]
    return {
        "layers": reference["layers"],
        "transforms": {
            stage: {side: transforms[indices] for side, transforms in payload["transforms"].items()}
            for stage, payload in stages.items()
        },
    }


def _snapshot_surface(payload: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    return {
        "layers": payload["layers"],
        "transforms": {"base": {side: transforms[indices] for side, transforms in payload["transforms"].items()}},
    }


def _baked_robot_surface(scene: Path, qpos: np.ndarray, *, geom_group: int, max_faces: int) -> dict[str, list[dict[str, np.ndarray]]]:
    """Bake and simplify real per-frame hand meshes for fast WebGL evidence."""
    payload = robot_surface_payload(scene, qpos, geom_group=geom_group)
    baked: dict[str, list[dict[str, np.ndarray]]] = {"right": [], "left": []}
    for side in ("right", "left"):
        for frame in range(len(qpos)):
            vertices: list[np.ndarray] = []
            faces: list[np.ndarray] = []
            offset = 0
            for geom, layer in enumerate(payload["layers"][side]):
                transform = payload["transforms"][side][frame, geom]
                rotation = transform[3:].reshape(3, 3)
                local_vertices = np.asarray(layer["vertices"], dtype=np.float64)
                vertices.append(local_vertices @ rotation.T + transform[:3])
                local_faces = np.asarray(layer["faces"], dtype=np.int64)
                faces.append(local_faces + offset)
                offset += len(local_vertices)
            mesh = trimesh.Trimesh(vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=False)
            # Wuji hand links are disconnected components.  Running a global
            # quadric decimator over their concatenation can bridge unrelated
            # links and create long synthetic triangles.  Select only original
            # baked faces instead: this is a display-only subset of the real
            # MuJoCo mesh and cannot change topology or spatial placement.
            if len(mesh.faces) > max_faces:
                selected = np.linspace(0, len(mesh.faces) - 1, max_faces, dtype=np.int64)
                mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)[selected], process=False)
                mesh.remove_unreferenced_vertices()
            baked[side].append({"vertices": np.asarray(mesh.vertices, dtype=np.float32), "faces": np.asarray(mesh.faces, dtype=np.int32)})
    return baked


def _document(payload: dict[str, Any]) -> str:
    packed = json.dumps(_plain(payload), ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>Stage C-XAR 修复前后真实三维轨迹</title>
<style>body{{margin:0;background:#10151c;color:#e8edf3;font-family:system-ui,'Noto Sans CJK SC',sans-serif}}#bar{{padding:12px 16px;background:#182431;position:sticky;top:0;z-index:5}}#scene{{height:68vh;min-height:650px}}#curves{{height:320px}}select,input,button{{background:#293846;color:#eef4fa;border:1px solid #526a80;border-radius:4px;padding:4px 6px}}#layers{{display:flex;gap:9px;flex-wrap:wrap;margin-top:8px;font-size:12px}}#info{{padding:10px 16px;white-space:pre-wrap;background:#182431;font:12px ui-monospace,monospace}}h1{{font-size:19px;margin:0 0 8px}}</style>
<body><section id=\"bar\"><h1>Stage C-XAR：Stage B / 旧 C-XA / 修复 C-XA 真实三维网格对比</h1>
<label>对比 <select id=\"compare\"><option value=\"all\">三阶段叠加</option><option value=\"old\">Stage B vs 旧 C-XA</option><option value=\"new\">Stage B vs 修复 C-XA</option><option value=\"delta\">旧 C-XA vs 修复 C-XA</option></select></label>
<label>坐标 <select id=\"mode\"><option value=\"world\">世界坐标</option><option value=\"object\">物体坐标</option><option value=\"left\">左腕坐标</option><option value=\"right\">右腕坐标</option></select></label>
<label>帧 <input id=\"frame\" type=\"number\" min=\"1461\" max=\"1874\" value=\"1461\"></label><input id=\"slider\" type=\"range\" min=\"1461\" max=\"1874\" value=\"1461\" style=\"width:280px\"><button id=\"prev\">◀</button><button id=\"next\">▶</button>
<div id=\"layers\"><label><input type=\"checkbox\" data-l=\"bvis\" checked>Stage B 手 visual</label><label><input type=\"checkbox\" data-l=\"ovis\" checked>旧 C-XA 手 visual</label><label><input type=\"checkbox\" data-l=\"nvis\" checked>修复 C-XA 手 visual</label><label><input type=\"checkbox\" data-l=\"bcol\">Stage B 手 collision</label><label><input type=\"checkbox\" data-l=\"ocol\">旧 C-XA 手 collision</label><label><input type=\"checkbox\" data-l=\"ncol\">修复 C-XA 手 collision</label><label><input type=\"checkbox\" data-l=\"object\" checked>物体 visual/collision</label><label><input type=\"checkbox\" data-l=\"patch\" checked>semantic patch</label><label><input type=\"checkbox\" data-l=\"axes\" checked>腕/物体坐标轴</label><label><input type=\"checkbox\" data-l=\"vectors\" checked>correction 向量</label></div></section><div id=\"scene\"></div><pre id=\"info\"></pre><div id=\"curves\"></div>
<script>{get_plotlyjs()}</script><script>
const D={packed};const e=id=>document.getElementById(id);const q=new URLSearchParams(location.search);if(q.has('frame'))e('frame').value=e('slider').value=q.get('frame');if(q.has('mode'))e('mode').value=q.get('mode');if(q.has('compare'))e('compare').value=q.get('compare');if(q.has('layers')){{let allow=new Set(q.get('layers').split(','));document.querySelectorAll('[data-l]').forEach(x=>x.checked=allow.has(x.dataset.l));}}
const C={{base:'#ffd166',old:'#ef476f',repaired:'#06d6a0',object:'#a8dadc',patch:'#c77dff'}};let V={{r:[[1,0,0],[0,1,0],[0,0,1]],p:[0,0,0]}};const on=x=>{{let n=[...document.querySelectorAll('[data-l]')].find(v=>v.dataset.l===x);return n&&n.checked}};function m(a){{return{{r:[[a[0],a[1],a[2]],[a[4],a[5],a[6]],[a[8],a[9],a[10]]],p:[a[3],a[7],a[11]]}}}}function inv(T){{let R=T.r,p=T.p;return{{r:[[R[0][0],R[1][0],R[2][0]],[R[0][1],R[1][1],R[2][1]],[R[0][2],R[1][2],R[2][2]]],p:[-(R[0][0]*p[0]+R[1][0]*p[1]+R[2][0]*p[2]),-(R[0][1]*p[0]+R[1][1]*p[1]+R[2][1]*p[2]),-(R[0][2]*p[0]+R[1][2]*p[1]+R[2][2]*p[2])]}}}}function pt(T,p){{return[T.r[0][0]*p[0]+T.r[0][1]*p[1]+T.r[0][2]*p[2]+T.p[0],T.r[1][0]*p[0]+T.r[1][1]*p[1]+T.r[1][2]*p[2]+T.p[1],T.r[2][0]*p[0]+T.r[2][1]*p[1]+T.r[2][2]*p[2]+T.p[2]]}}function pts(T,a){{return a.map(p=>pt(T,p))}}function mesh(name,vs,fs,col,op){{let z=pts(V,vs);return{{type:'mesh3d',name,x:z.map(p=>p[0]),y:z.map(p=>p[1]),z:z.map(p=>p[2]),i:fs.map(x=>x[0]),j:fs.map(x=>x[1]),k:fs.map(x=>x[2]),color:col,opacity:op,flatshading:true,hoverinfo:'name'}}}}function layers(name,item,stage,sides,frame,col,op){{let vs=[],fs=[],o=0;for(let s of sides)item.layers[s].forEach((l,g)=>{{let a=item.transforms[stage][s][frame][g],T={{r:[[a[3],a[4],a[5]],[a[6],a[7],a[8]],[a[9],a[10],a[11]]],p:[a[0],a[1],a[2]]}};let v=pts(T,l.vertices);vs.push(...v);fs.push(...l.faces.map(f=>[f[0]+o,f[1]+o,f[2]+o]));o+=v.length}});return mesh(name,vs,fs,col,op)}}function hand(name,item,stage,frame,col,op){{let vs=[],fs=[],o=0;for(let s of ['right','left']){{let layer=item[stage][s][frame];vs.push(...layer.vertices);fs.push(...layer.faces.map(f=>[f[0]+o,f[1]+o,f[2]+o]));o+=layer.vertices.length}}return mesh(name,vs,fs,col,op)}}function body(name,item,stage,frame,col,op){{return layers(name,item,stage,['object'],frame,col,op)}}function line(name,a,b,c){{let z=pts(V,[a,b]);return{{type:'scatter3d',mode:'lines+markers',name,x:z.map(v=>v[0]),y:z.map(v=>v[1]),z:z.map(v=>v[2]),line:{{color:c,width:5}},marker:{{size:2,color:c}}}}}}function axes(T,n){{let out=[];for(let a of [[0,'#ef476f'],[1,'#06d6a0'],[2,'#118ab2']]){{let p=T.p,R=T.r;out.push(line(n+' '+['x','y','z'][a[0]],p,[p[0]+.045*R[0][a[0]],p[1]+.045*R[1][a[0]],p[2]+.045*R[2][a[0]]],a[1]));}}return out}}function meshIndex(f){{let best=0;for(let i=1;i<D.mesh_frames.length;i++)if(Math.abs(D.mesh_frames[i]-f)<Math.abs(D.mesh_frames[best]-f))best=i;return best}}
function draw(){{let wanted=Math.max(1461,Math.min(1874,+e('frame').value)),i=meshIndex(wanted),f=D.mesh_frames[i];e('frame').value=e('slider').value=f;let cmp=e('compare').value,b=D.points.base,o=D.points.old,n=D.points.repaired,ref=m(b.object[i]),mode=e('mode').value;V=mode==='object'?inv(ref):mode==='left'?inv(m(b.left.palm[i])):mode==='right'?inv(m(b.right.palm[i])):{{r:[[1,0,0],[0,1,0],[0,0,1]],p:[0,0,0]}};let tr=[],show=(x)=>cmp==='all'||(cmp==='old'&&(x==='base'||x==='old'))||(cmp==='new'&&(x==='base'||x==='repaired'))||(cmp==='delta'&&(x==='old'||x==='repaired'));if(on('bvis')&&show('base'))tr.push(hand('Stage B visual',D.visual,'base',i,C.base,.34));if(on('ovis')&&show('old'))tr.push(hand('旧 C-XA visual',D.visual,'old',i,C.old,.34));if(on('nvis')&&show('repaired'))tr.push(hand('修复 C-XA visual',D.visual,'repaired',i,C.repaired,.38));if(on('bcol')&&show('base'))tr.push(hand('Stage B collision',D.collision,'base',i,C.base,.16));if(on('ocol')&&show('old'))tr.push(hand('旧 C-XA collision',D.collision,'old',i,C.old,.16));if(on('ncol')&&show('repaired'))tr.push(hand('修复 C-XA collision',D.collision,'repaired',i,C.repaired,.16));if(on('object')){{tr.push(body('object visual',D.object.visual,'base',i,C.object,.23));tr.push(body('object collision',D.object.collision,'base',i,'#457b9d',.12))}}if(on('patch')){{let p=pts(ref,[D.patches[i]]);p=pts(V,p);tr.push({{type:'scatter3d',mode:'markers',name:'semantic patch connected surface center',x:p.map(x=>x[0]),y:p.map(x=>x[1]),z:p.map(x=>x[2]),marker:{{size:5,color:C.patch,symbol:'diamond'}}}})}}for(let s of ['right','left']){{if(on('axes')){{tr.push(...axes(m(b[s].palm[i]),'Stage B '+s+' wrist'),...axes(ref,'object'))}}if(on('vectors')){{tr.push(line('旧 correction '+s,b[s].palm[i].slice(3,6),o[s].palm[i].slice(3,6),C.old));tr.push(line('修复 correction '+s,b[s].palm[i].slice(3,6),n[s].palm[i].slice(3,6),C.repaired))}}}}e('info').textContent=`帧 ${{f}} | ${{mode}}坐标 | ${{cmp}}\n旧 C-XA：腕/根 correction 曾在 Phase-1 泄漏。\n修复 C-XA：只允许手指 DOF，腕/根和 object 均由数值不变量锁定。\n本页面显示真实 MuJoCo 手/物体 visual/collision mesh；未对齐、未写 qpos 伪造。`;Plotly.react('scene',tr,{{paper_bgcolor:'#10151c',plot_bgcolor:'#10151c',font:{{color:'#e8edf3'}},scene:{{aspectmode:'data',camera:{{eye:{{x:1.45,y:1.3,z:.9}}}}}},margin:{{l:0,r:0,t:25,b:0}},legend:{{orientation:'h'}}}},{{responsive:true}})}}
let curves=[{{x:D.frames,y:D.metrics.old_wrist_m,name:'旧 C-XA 腕部平移 correction',line:{{color:C.old}}}},{{x:D.frames,y:D.metrics.repaired_wrist_m,name:'修复 C-XA 腕部平移 correction',line:{{color:C.repaired}}}},{{x:D.frames,y:D.metrics.old_tip_m,name:'旧 C-XA 指尖改动',line:{{color:'#f78c6b'}}}},{{x:D.frames,y:D.metrics.repaired_tip_m,name:'修复 C-XA 指尖改动',line:{{color:'#73d2de'}}}}];Plotly.newPlot('curves',curves,{{paper_bgcolor:'#10151c',plot_bgcolor:'#10151c',font:{{color:'#e8edf3'}},title:'全轨迹 correction 曲线（米）',xaxis:{{title:'source frame'}},yaxis:{{title:'m'}}}},{{responsive:true}});['frame','slider','mode','compare'].forEach(x=>e(x).addEventListener('input',draw));document.querySelectorAll('[data-l]').forEach(x=>x.addEventListener('change',draw));e('prev').onclick=()=>{{e('frame').value=+e('frame').value-1;draw()}};e('next').onclick=()=>{{e('frame').value=+e('frame').value+1;draw()}};draw();</script></body></html>"""


def render(run_root: str, paths_config: str, repaired_root: str, old_root: str | None = None) -> str:
    root = Path(run_root).resolve(); paths = load_project_paths(paths_config); seq = "s5__cylindermedium_lift"; robot = paths.workspace_root / "processed/grab/wuji_hand2_beta1/bimanual" / seq / "0"
    old = Path(old_root).resolve() if old_root else robot / "stage_c_contract_v2_cxa"; repaired = Path(repaired_root).resolve(); physics = json.loads((robot / "stage_c/physics_input.json").read_text(encoding="utf-8")); scene = Path(physics["scene_act"])
    base, _ = _stage_b_act_baseline(paths, seq)
    with np.load(old / "trajectory_depenetrated_init_cxa_level_1_flexible.npz", allow_pickle=False) as z: old_qpos, frames = np.asarray(z["qpos"]), np.asarray(z["source_frame_indices"])
    with np.load(repaired / "trajectory_depenetrated_init.npz", allow_pickle=False) as z: repaired_qpos = np.asarray(z["qpos"])
    if not (base.shape == old_qpos.shape == repaired_qpos.shape and len(frames) == len(base)): raise RuntimeError("XAR viewer input trajectory shape mismatch")
    with np.load(repaired / "source_contact_patches.npz", allow_pickle=False) as z: patches = np.asarray(z["center_object_local"])
    points = {"base": _hand_points(scene, base), "old": _hand_points(scene, old_qpos), "repaired": _hand_points(scene, repaired_qpos)}
    snapshot_frames = np.asarray((1461, 1462, 1463, 1464, 1465, 1466, 1480, 1619, 1798, 1874), dtype=np.int64)
    frame_to_index = {int(frame): index for index, frame in enumerate(frames)}
    if any(int(frame) not in frame_to_index for frame in snapshot_frames):
        raise RuntimeError(f"Requested XAR snapshot frame absent from C-XA trace: {snapshot_frames.tolist()}")
    snapshot_indices = np.asarray([frame_to_index[int(frame)] for frame in snapshot_frames], dtype=np.int64)
    points = {stage: _snapshot_points(value, snapshot_indices) for stage, value in points.items()}
    visual = {"base": _baked_robot_surface(scene, base[snapshot_indices], geom_group=1, max_faces=200), "old": _baked_robot_surface(scene, old_qpos[snapshot_indices], geom_group=1, max_faces=200), "repaired": _baked_robot_surface(scene, repaired_qpos[snapshot_indices], geom_group=1, max_faces=200)}
    collision = {"base": _baked_robot_surface(scene, base[snapshot_indices], geom_group=2, max_faces=150), "old": _baked_robot_surface(scene, old_qpos[snapshot_indices], geom_group=2, max_faces=150), "repaired": _baked_robot_surface(scene, repaired_qpos[snapshot_indices], geom_group=2, max_faces=150)}
    objects = {
        "visual": _snapshot_surface(object_surface_payload(scene, base, collision=False, max_faces=500), snapshot_indices),
        "collision": _snapshot_surface(object_surface_payload(scene, base, collision=True, max_faces=300), snapshot_indices),
    }
    wrist = lambda a, b: np.maximum(np.linalg.norm(a["right"]["palm"][:, :3, 3] - b["right"]["palm"][:, :3, 3], axis=1), np.linalg.norm(a["left"]["palm"][:, :3, 3] - b["left"]["palm"][:, :3, 3], axis=1))
    tips = lambda a, b: np.maximum(np.linalg.norm(a["right"]["tips"] - b["right"]["tips"], axis=-1).max(axis=1), np.linalg.norm(a["left"]["tips"] - b["left"]["tips"], axis=-1).max(axis=1))
    # Curves retain all frames; the real mesh payload is deliberately limited
    # to the named XAR audit frames so the self-contained browser evidence is
    # renderable rather than a nominal, unusably large HTML file.
    full_points = {"base": _hand_points(scene, base), "old": _hand_points(scene, old_qpos), "repaired": _hand_points(scene, repaired_qpos)}
    data = {"frames": frames, "mesh_frames": snapshot_frames, "points": points, "visual": visual, "collision": collision, "object": objects, "patches": patches, "metrics": {"old_wrist_m": wrist(full_points["base"], full_points["old"]), "repaired_wrist_m": wrist(full_points["base"], full_points["repaired"]), "old_tip_m": tips(full_points["base"], full_points["old"]), "repaired_tip_m": tips(full_points["base"], full_points["repaired"])}}
    output = root / "html/stage_c_xar_correction_audit.html"; output.write_text(_document(data), encoding="utf-8")
    index = root / "html/stage_c_xar_visual_index.html"; index.write_text("<!doctype html><meta charset='utf-8'><title>Stage C-XAR 可视化索引</title><h1>Stage C-XAR 可视化索引</h1><ul><li><a href='stage_c_xar_correction_audit.html?compare=old&mode=object&frame=1461'>Stage B vs 旧 C-XA，1461，物体坐标</a></li><li><a href='stage_c_xar_correction_audit.html?compare=new&mode=object&frame=1461'>Stage B vs 修复 C-XA，1461，物体坐标</a></li><li><a href='stage_c_xar_correction_audit.html?compare=all&mode=world&frame=1465'>三阶段叠加，1465，世界坐标</a></li></ul><p>用户视觉验收：PENDING。</p>", encoding="utf-8")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True); parser.add_argument("--paths-config", default="configs/local/paths.yaml"); parser.add_argument("--repaired-root", required=True); parser.add_argument("--old-root")
    args = parser.parse_args(); print(render(args.run_root, args.paths_config, args.repaired_root, args.old_root))


if __name__ == "__main__": main()
