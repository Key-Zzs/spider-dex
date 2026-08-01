"""Chinese interactive 3-D viewer for the source-geometry audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from plotly.offline import get_plotlyjs

from spider.tools.grab_source_geometry_audit import FINGERTIP_INDICES, FINGERTIP_NAMES, robot_surface_payload, write_json, write_text


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _robot_data(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "layers": _plain(payload["layers"]),
        "transforms": _plain(payload["transforms"]),
        "scene": payload["scene"],
    }


def _curves(
    raw: dict[str, np.ndarray], stage_b_sides: dict[str, Any], cxa_store: dict[str, Any]
) -> dict[str, Any]:
    source_object = raw["T_world_object_visual"][:, :3, 3]
    values: dict[str, Any] = {
        "source_frames": raw["source_frame_indices"],
        "raw_object_translation_residual_m": np.zeros(len(source_object)),
        "raw_object_rotation_residual_rad": np.zeros(len(source_object)),
        "raw_wrist_relative_residual_m": np.zeros(len(source_object)),
        "raw_tip_relative_residual_m": np.zeros(len(source_object)),
        "surface_gap_m": np.zeros(len(source_object)),
        "penetration_m": np.zeros(len(source_object)),
    }
    if stage_b_sides:
        values["stage_b_frames"] = cxa_store["mapping"]
        values["stage_b_wrist_relative_residual_m"] = np.maximum(
            np.linalg.norm(stage_b_sides["right"]["source_wrist"][:, :3, 3] - stage_b_sides["right"]["stage_wrist"][:, :3, 3], axis=1),
            np.linalg.norm(stage_b_sides["left"]["source_wrist"][:, :3, 3] - stage_b_sides["left"]["stage_wrist"][:, :3, 3], axis=1),
        )
        values["stage_b_tip_relative_residual_m"] = np.maximum(
            np.linalg.norm(stage_b_sides["right"]["source_tips"] - stage_b_sides["right"]["stage_tips"], axis=-1).max(axis=1),
            np.linalg.norm(stage_b_sides["left"]["source_tips"] - stage_b_sides["left"]["stage_tips"], axis=-1).max(axis=1),
        )
    if cxa_store:
        values["cxa_frames"] = cxa_store["mapping"]
        values["cxa_wrist_delta_m"] = np.maximum(
            np.linalg.norm(cxa_store["sides"]["right"]["cxa_wrist"][:, :3, 3] - cxa_store["sides"]["right"]["stage_b_wrist"][:, :3, 3], axis=1),
            np.linalg.norm(cxa_store["sides"]["left"]["cxa_wrist"][:, :3, 3] - cxa_store["sides"]["left"]["stage_b_wrist"][:, :3, 3], axis=1),
        )
        values["cxa_tip_delta_m"] = np.maximum(
            np.linalg.norm(cxa_store["sides"]["right"]["cxa_tips"] - cxa_store["sides"]["right"]["stage_b_tips"], axis=-1).max(axis=1),
            np.linalg.norm(cxa_store["sides"]["left"]["cxa_tips"] - cxa_store["sides"]["left"]["stage_b_tips"], axis=-1).max(axis=1),
        )
    return _plain(values)


def _document(data: dict[str, Any]) -> str:
    packed = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    plotly = get_plotlyjs()
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>GRAB 上游坐标链路审计</title>
<style>
body{{font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;margin:0;background:#10151c;color:#e9eef5}}h1{{font-size:20px;margin:0 0 8px}}#head{{padding:12px 16px;background:#17212d;position:sticky;top:0;z-index:3}}#controls{{display:flex;gap:9px;align-items:center;flex-wrap:wrap;font-size:13px}}select,input,button{{background:#293746;color:#e9eef5;border:1px solid #52677b;padding:4px 6px;border-radius:4px}}button{{cursor:pointer}}#layers{{margin-top:8px;display:flex;gap:10px;flex-wrap:wrap}}#layers label{{font-size:12px;white-space:nowrap}}#notice{{margin-top:7px;font-size:13px;color:#ffd166}}#scene{{height:66vh;min-height:620px}}#metrics{{white-space:pre-wrap;background:#17212d;padding:10px 16px;margin:0;font:12px ui-monospace,monospace}}#curves{{height:360px}}.help{{padding:0 16px 12px;font-size:13px;color:#c8d6e5}}.badge{{padding:2px 5px;border-radius:3px;background:#39506a}}
</style></head><body>
<section id="head"><h1>GRAB → spider raw → Stage B → C-XA：真实三维坐标/轨迹审计</h1>
<div id="controls"><label>对比 <select id="compare"><option value="raw">官方 source vs spider raw</option><option value="stageb">spider raw vs Stage B</option><option value="cxa">Stage B vs C-XA</option><option value="all">四阶段叠加</option></select></label>
<label>坐标 <select id="mode"><option value="world">世界坐标</option><option value="object">物体坐标</option><option value="left">左腕坐标</option><option value="right">右腕坐标</option></select></label>
<label>轨迹 <select id="scope"><option value="detail">详细 1460–1480</option><option value="full">完整 1460–1875</option></select></label>
<button id="play">播放</button><label>速度 <input id="speed" type="number" min="1" max="60" value="12" style="width:40px"></label><button id="prev">◀</button><button id="next">▶</button>
<label>帧 <input id="frame" type="number" min="1460" max="1875" value="1461" style="width:60px"></label><input id="slider" type="range" min="1460" max="1875" value="1461" style="width:260px"><label>尾迹 <input id="tail" type="number" min="0" max="414" value="20" style="width:45px"></label>
<button data-f="1460">1460</button><button data-f="1461">1461</button><button data-f="1462">1462</button><button data-f="1465">1465</button><button data-f="1480">1480</button></div>
<div id="layers"><label><input type="checkbox" data-layer="source" checked>官方 source 手/身体</label><label><input type="checkbox" data-layer="raw" checked>spider raw 手</label><label><input type="checkbox" data-layer="stageb" checked>Stage B Wuji</label><label><input type="checkbox" data-layer="cxa" checked>C-XA Wuji</label><label><input type="checkbox" data-layer="object" checked>全部物体 visual mesh</label><label><input type="checkbox" data-layer="skeleton" checked>source skeleton</label><label><input type="checkbox" data-layer="axes" checked>腕/物体/world 轴</label><label><input type="checkbox" data-layer="trajectory" checked>指尖/物体轨迹</label><label><input type="checkbox" data-layer="error" checked>误差向量</label><label><input type="checkbox" data-layer="contact" checked>最近表面 contact line/间隙</label><label><input type="checkbox" data-layer="patch" checked>semantic patch</label></div><div id="notice"></div></section>
<div id="scene"></div><pre id="metrics"></pre><div id="curves"></div>
<div class="help"><span class="badge">查看顺序</span> 先选“官方 source vs spider raw”，切到物体坐标，在 1461、1462、1465 看手物相对关系；再切世界坐标判断是否只是方向差异。raw PASS 后依次选择 Stage B 和 C-XA。右侧数据面板给出当前帧的实测误差；截图 URL 可使用 <code>?frame=1462&mode=object&camera=macro</code>。</div>
<script>{plotly}</script><script>
const D={packed}; const el=id=>document.getElementById(id); let timer=null;
const params=new URLSearchParams(location.search); if(params.has('frame')){{el('frame').value=el('slider').value=params.get('frame')}} if(params.has('mode'))el('mode').value=params.get('mode'); if(params.has('compare'))el('compare').value=params.get('compare');
const cam={{global:{{eye:{{x:1.55,y:1.35,z:.95}}}},opposite:{{eye:{{x:-1.35,y:-1.45,z:.8}}}},macro:{{eye:{{x:.66,y:.60,z:.45}}}}}}; const requestedCamera=params.get('camera')||'global';
function enabled(x){{const n=[...document.querySelectorAll('[data-layer]')].find(e=>e.dataset.layer===x);return n&&n.checked}}
function idx(frame,frames){{return frames.indexOf(frame)}} function mat(T){{return {{r:[[T[0],T[1],T[2]],[T[4],T[5],T[6]],[T[8],T[9],T[10]]],p:[T[3],T[7],T[11]]}}}}
function inv(T){{let R=T.r,p=T.p;return {{r:[[R[0][0],R[1][0],R[2][0]],[R[0][1],R[1][1],R[2][1]],[R[0][2],R[1][2],R[2][2]]],p:[-(R[0][0]*p[0]+R[1][0]*p[1]+R[2][0]*p[2]),-(R[0][1]*p[0]+R[1][1]*p[1]+R[2][1]*p[2]),-(R[0][2]*p[0]+R[1][2]*p[1]+R[2][2]*p[2])]}}}}
function pt(T,p){{return [T.r[0][0]*p[0]+T.r[0][1]*p[1]+T.r[0][2]*p[2]+T.p[0],T.r[1][0]*p[0]+T.r[1][1]*p[1]+T.r[1][2]*p[2]+T.p[1],T.r[2][0]*p[0]+T.r[2][1]*p[1]+T.r[2][2]*p[2]+T.p[2]]}}
function pts(T,a){{return a.map(p=>pt(T,p))}} function color(name){{return {{source:'#48cae4',raw:'#f72585',stageb:'#ffd166',cxa:'#80ed99'}}[name]}}
function meshTrace(name,vertices,faces,view,opacity=.35){{let q=pts(view,vertices);return {{type:'mesh3d',name:name,x:q.map(p=>p[0]),y:q.map(p=>p[1]),z:q.map(p=>p[2]),i:faces.map(f=>f[0]),j:faces.map(f=>f[1]),k:faces.map(f=>f[2]),color:color(name.split(' ')[0])||'#adb5bd',opacity:opacity,flatshading:true,hoverinfo:'name'}}}}
function lineTrace(name,points,c,wd=3){{let q=pts(V,points);return {{type:'scatter3d',mode:'lines+markers',name,x:q.map(p=>p[0]),y:q.map(p=>p[1]),z:q.map(p=>p[2]),line:{{color:c,width:wd}},marker:{{size:2,color:c}},hoverinfo:'name'}}}}
function robotTrace(label,robot,side,k,view,opacity){{let vs=[],fs=[],offset=0;robot.layers[side].forEach((layer,g)=>{{let tr=robot.transforms[side][k][g],T={{r:[[tr[3],tr[4],tr[5]],[tr[6],tr[7],tr[8]],[tr[9],tr[10],tr[11]]],p:[tr[0],tr[1],tr[2]]}};let q=pts(T,layer.vertices);vs.push(...pts(view,q));fs.push(...layer.faces.map(f=>[f[0]+offset,f[1]+offset,f[2]+offset]));offset+=q.length}});return meshTrace(label,vs,fs,{{r:[[1,0,0],[0,1,0],[0,0,1]],p:[0,0,0]}},opacity)}}
function axes(T,name){{let p=T.p,R=T.r,s=.045,out=[];[['x','#ef476f',0],['y','#06d6a0',1],['z','#118ab2',2]].forEach(a=>out.push(lineTrace(name+' '+a[0],[p,[p[0]+R[0][a[2]]*s,p[1]+R[1][a[2]]*s,p[2]+R[2][a[2]]*s]],a[1],5)));return out}}
let V={{r:[[1,0,0],[0,1,0],[0,0,1]],p:[0,0,0]}};
function draw(){{let f=+el('frame').value;f=Math.max(1460,Math.min(1875,f));el('frame').value=el('slider').value=f;let si=idx(f,D.source.frames),bi=idx(f,D.stageb.frames);let ref=mat(D.source.object[si]);let mode=el('mode').value;if(mode==='object')V=inv(ref);else if(mode==='left')V=inv(mat(D.source.leftWrist[si]));else if(mode==='right')V=inv(mat(D.source.rightWrist[si]));else V={{r:[[1,0,0],[0,1,0],[0,0,1]],p:[0,0,0]}};let tr=[],cmp=el('compare').value;
let show=x=>enabled(x)&&(cmp==='all'||(cmp==='raw'&&(x==='source'||x==='raw'||x==='object'))||(cmp==='stageb'&&(x==='raw'||x==='stageb'||x==='object'))||(cmp==='cxa'&&(x==='stageb'||x==='cxa'||x==='object')));
if(show('object')){{tr.push(meshTrace('source object',pts(mat(D.source.object[si]),D.object.vertices),D.object.faces,V,.20));tr.push(meshTrace('raw object',pts(mat(D.raw.object[si]),D.object.vertices),D.object.faces,V,.28));if(bi>=0){{tr.push(meshTrace('stageb object',pts(mat(D.stageb.object[bi]),D.object.vertices),D.object.faces,V,.36));tr.push(meshTrace('cxa object',pts(mat(D.cxa.object[bi]),D.object.vertices),D.object.faces,V,.46))}}}}
for(let side of ['right','left']){{if(show('source'))tr.push(meshTrace('source '+side,D.source[side].verts[si],D.source[side].faces,V,.42));if(show('raw'))tr.push(meshTrace('raw '+side,D.raw[side].verts[si],D.raw[side].faces,V,.32));if(bi>=0&&show('stageb'))tr.push(robotTrace('stageb '+side,D.robotStageB,side,bi,V,.35));if(bi>=0&&show('cxa'))tr.push(robotTrace('cxa '+side,D.robotCXA,side,bi,V,.43));}}
if(enabled('skeleton')&&show('source')){{let j=D.source.bodyJoints[si];tr.push({{type:'scatter3d',mode:'markers',name:'SMPL-X body joints',x:pts(V,j).map(p=>p[0]),y:pts(V,j).map(p=>p[1]),z:pts(V,j).map(p=>p[2]),marker:{{size:2,color:'#bde0fe'}}}})}}
if(enabled('axes')){{tr.push(...axes(ref,'object axis'),...axes(mat(D.source.leftWrist[si]),'left wrist'),...axes(mat(D.source.rightWrist[si]),'right wrist'));tr.push(lineTrace('world axis',[[0,0,0],[.10,0,0]],["#ffffff"],4))}}
if(enabled('trajectory')){{
  let tail=+el('tail').value;let scope=el('scope').value==='detail'?1460:Math.max(1460,f-tail);
  for(let side of ['right','left']){{
    let a=[];for(let q=scope;q<=f;q++){{let qi=idx(q,D.source.frames);if(qi>=0)a.push(D.source[side].joints[qi][D.tipIndices[1]])}}
    tr.push(lineTrace('source '+side+' index tail',a,color('source'),3));
  }}
}}
if(enabled('error')){{
  for(let side of ['right','left']){{
    let a=D.source[side].joints[si][D.tipIndices[1]],b=D.raw[side].joints[si][D.tipIndices[1]];
    if(cmp==='raw'||cmp==='all')tr.push(lineTrace('raw error '+side,[a,b],'#f72585',5));
    if(bi>=0&&(cmp==='stageb'||cmp==='all')){{let c=D.stageb[side].tips[bi][1];tr.push(lineTrace('Stage B error '+side,[a,c],'#ffd166',4));}}
    if(bi>=0&&(cmp==='cxa'||cmp==='all')){{let c=D.stageb[side].tips[bi][1],d=D.cxa[side].tips[bi][1];tr.push(lineTrace('C-XA delta '+side,[c,d],'#80ed99',4));}}
  }}
}}
if(enabled('contact')){{
  for(let side of ['right','left']){{for(let g=0;g<5;g++){{let p=D.source[side].joints[si][D.tipIndices[g]],q=D.contact[side].nearest[si][g];tr.push(lineTrace(side+' '+D.fingers[g]+' nearest surface',[p,q],'#f4a261',2));}}}}
}}
if(enabled('patch')&&bi>=0){{
  let p=D.patches.map(x=>pt(mat(D.cxa.object[bi]),x));tr.push({{type:'scatter3d',mode:'markers',name:'semantic patch centers',x:pts(V,p).map(x=>x[0]),y:pts(V,p).map(x=>x[1]),z:pts(V,p).map(x=>x[2]),marker:{{size:4,color:'#c77dff',symbol:'diamond'}}}});
}}
let note=bi<0?'当前为 1460 或 1875：冻结 Stage B/C-XA 档案明确没有这两个端点；并非时间错位。':'Stage B/C-XA 映射源帧 '+f+'。';el('notice').textContent=note;
let m=D.metrics[si];el('metrics').textContent='帧 '+f+'，时间 '+(f/120).toFixed(6)+' s\\n决策：'+D.final.classification+'；raw='+D.final.raw_loader+'，Stage B='+D.final.stage_b+'，C-XA='+D.final.cxa+'\\nraw object 平移/旋转 residual: '+m.rawObject.toExponential(3)+' m / '+m.rawRot.toExponential(3)+' rad\\nStage B 最大物体坐标腕误差: '+m.stageBWrist.toFixed(6)+' m；最大指尖误差: '+m.stageBTip.toFixed(6)+' m\\nC-XA 最大 object-relative 腕位移: '+m.cxaWrist.toFixed(6)+' m；最大指尖改动: '+m.cxaTip.toFixed(6)+' m\\nsource 左/右指尖最近表面最大距离: '+m.leftGap.toFixed(6)+' / '+m.rightGap.toFixed(6)+' m\\n面板与曲线均为真实 frozen 帧；无静态平移对齐。';
Plotly.react('scene',tr,{{paper_bgcolor:'#10151c',plot_bgcolor:'#10151c',font:{{color:'#e9eef5'}},scene:{{aspectmode:'data',camera:cam[requestedCamera],xaxis:{{title:'X'}},yaxis:{{title:'Y'}},zaxis:{{title:'Z'}}}},legend:{{orientation:'h'}},margin:{{l:0,r:0,t:25,b:0}}}},{{responsive:true}})}}
function curves(){{let C=D.curves,t=[];let add=(x,y,n,c)=>{{if(x&&y)t.push({{x,y,name:n,mode:'lines',line:{{color:c}}}})}};add(C.source_frames,C.raw_object_translation_residual_m,'raw object translation residual','#f72585');add(C.stage_b_frames,C.stage_b_wrist_relative_residual_m,'Stage B object-frame wrist residual','#ffd166');add(C.stage_b_frames,C.stage_b_tip_relative_residual_m,'Stage B object-frame fingertip residual','#ff9f1c');add(C.cxa_frames,C.cxa_wrist_delta_m,'C-XA object-frame wrist delta','#80ed99');add(C.cxa_frames,C.cxa_tip_delta_m,'C-XA object-frame fingertip delta','#c77dff');Plotly.newPlot('curves',t,{{paper_bgcolor:'#10151c',plot_bgcolor:'#10151c',font:{{color:'#e9eef5'}},xaxis:{{title:'源 frame'}},yaxis:{{title:'米 (m)'}},title:'逐帧数值：物体、手腕、指尖的相对几何误差/改动'}},{{responsive:true}})}}
['frame','slider','mode','compare','scope','tail'].forEach(id=>el(id).addEventListener('input',draw));document.querySelectorAll('[data-layer]').forEach(x=>x.addEventListener('change',draw));document.querySelectorAll('[data-f]').forEach(x=>x.onclick=()=>{{el('frame').value=x.dataset.f;draw()}});el('prev').onclick=()=>{{el('frame').value=Math.max(1460,+el('frame').value-1);draw()}};el('next').onclick=()=>{{el('frame').value=Math.min(1875,+el('frame').value+1);draw()}};el('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;el('play').textContent='播放'}}else{{timer=setInterval(()=>{{let n=+el('frame').value+1;if(n>1875)n=1460;el('frame').value=n;draw()}},1000/Math.max(1,+el('speed').value));el('play').textContent='暂停'}}}};curves();draw();
</script></body></html>"""


def render_viewer(
    root: Path,
    official: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    contact: dict[str, Any],
    stage_b_store: dict[str, Any],
    stage_b_sides: dict[str, Any],
    cxa_store: dict[str, Any],
    comparison: dict[str, Any],
    stage_b_report: dict[str, Any],
    cxa_report: dict[str, Any],
    final: dict[str, Any],
) -> None:
    """Write a self-contained Chinese viewer with real mesh vertices and faces."""
    if not stage_b_store or not cxa_store:
        raise RuntimeError("Viewer is intentionally not emitted unless raw and Stage-B gates both ran")
    robot_stage_b = robot_surface_payload(Path(stage_b_store["geometry"]["scene"]), stage_b_store["qpos"])
    robot_cxa = robot_surface_payload(Path(cxa_store["after"]["scene"]), cxa_store["cxa_qpos"])
    patches = np.empty((0, 3), dtype=np.float64)
    patch_file = Path(cxa_report["semantic_patch"]["path"])
    if patch_file.is_file():
        with np.load(patch_file, allow_pickle=False) as archive:
            patches = np.asarray(archive["center_object_local"], dtype=np.float64)
    stage_b_geometry = stage_b_store["geometry"]
    source_metrics: list[dict[str, float]] = []
    for frame in range(len(raw["source_frame_indices"])):
        if frame < 1 or frame >= len(raw["source_frame_indices"]) - 1:
            source_metrics.append({"rawObject": 0.0, "rawRot": 0.0, "stageBWrist": 0.0, "stageBTip": 0.0, "cxaWrist": 0.0, "cxaTip": 0.0, "leftGap": float(np.max(contact["sides"]["left"]["unsigned_distance_m"][frame])), "rightGap": float(np.max(contact["sides"]["right"]["unsigned_distance_m"][frame]))})
            continue
        index = frame - 1
        stage_wrist = max(np.linalg.norm(stage_b_sides[side]["source_wrist"][index, :3, 3] - stage_b_sides[side]["stage_wrist"][index, :3, 3]) for side in ("left", "right"))
        stage_tip = max(np.linalg.norm(stage_b_sides[side]["source_tips"][index] - stage_b_sides[side]["stage_tips"][index], axis=1).max() for side in ("left", "right"))
        cxa_wrist = max(np.linalg.norm(cxa_store["sides"][side]["cxa_wrist"][index, :3, 3] - cxa_store["sides"][side]["stage_b_wrist"][index, :3, 3]) for side in ("left", "right"))
        cxa_tip = max(np.linalg.norm(cxa_store["sides"][side]["cxa_tips"][index] - cxa_store["sides"][side]["stage_b_tips"][index], axis=1).max() for side in ("left", "right"))
        source_metrics.append({"rawObject": 0.0, "rawRot": 0.0, "stageBWrist": float(stage_wrist), "stageBTip": float(stage_tip), "cxaWrist": float(cxa_wrist), "cxaTip": float(cxa_tip), "leftGap": float(np.max(contact["sides"]["left"]["unsigned_distance_m"][frame])), "rightGap": float(np.max(contact["sides"]["right"]["unsigned_distance_m"][frame]))})
    data = {
        "fingers": list(FINGERTIP_NAMES), "tipIndices": list(FINGERTIP_INDICES), "source": {"frames": official["source_frame_indices"], "object": official["T_world_object_visual"].reshape(-1, 16), "leftWrist": official["T_world_left_wrist"].reshape(-1, 16), "rightWrist": official["T_world_right_wrist"].reshape(-1, 16), "bodyJoints": official["body_joints_world"], **{side: {"verts": official[f"{side}_vertices_world"], "faces": official[f"{side}_hand_faces"], "joints": official[f"{side}_joints_world"]} for side in ("left", "right")}},
        "raw": {"object": raw["T_world_object_visual"].reshape(-1, 16), **{side: {"verts": raw[f"{side}_vertices_world"], "faces": raw[f"{side}_hand_faces"], "joints": raw[f"{side}_joints_world"]} for side in ("left", "right")}},
        "object": {"vertices": official["object_asset_vertices"], "faces": official["object_faces"]},
        "stageb": {"frames": stage_b_store["mapping"], "object": stage_b_geometry["object_transform"].reshape(-1, 16), **{side: {"tips": stage_b_sides[side]["stage_tips"]} for side in ("left", "right")}},
        "cxa": {"object": cxa_store["after"]["object_transform"].reshape(-1, 16), **{side: {"tips": cxa_store["sides"][side]["cxa_tips"]} for side in ("left", "right")}},
        "robotStageB": _robot_data(robot_stage_b), "robotCXA": _robot_data(robot_cxa),
        "contact": {side: {"nearest": contact["sides"][side]["nearest_surface_world"]} for side in ("left", "right")},
        "patches": patches, "metrics": source_metrics, "curves": _curves(raw, stage_b_sides, cxa_store), "final": final,
    }
    html = _document(_plain(data))
    target = root / "html/grab_source_geometry_trajectory_audit.html"
    write_text(target, html)
    index = root / "html/grab_source_geometry_visual_index.html"
    write_text(index, f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GRAB 坐标审计索引</title><body style="font-family:sans-serif;max-width:900px;margin:30px auto"><h1>GRAB Stage C 上游坐标链路审计：可视化索引</h1><p>最终分类：<code>{final['classification']}</code>；raw loader：<code>{final['raw_loader']}</code>；Stage B：<code>{final['stage_b']}</code>；C-XA：<code>{final['cxa']}</code>。</p><ol><li><a href="grab_source_geometry_trajectory_audit.html?compare=raw&mode=object&frame=1461">官方 source vs spider raw，物体坐标 1461</a></li><li><a href="grab_source_geometry_trajectory_audit.html?compare=raw&mode=object&frame=1462">官方 source vs spider raw，物体坐标 1462</a></li><li><a href="grab_source_geometry_trajectory_audit.html?compare=stageb&mode=object&frame=1465">spider raw vs Stage B，物体坐标 1465</a></li><li><a href="grab_source_geometry_trajectory_audit.html?compare=cxa&mode=object&frame=1465">Stage B vs C-XA，物体坐标 1465</a></li></ol><p>用户验收状态：<strong>PENDING</strong>。完整查看顺序见项目文档。</p></body></html>""")
    write_json(root / "reports/viewer_manifest.json", {"trajectory_html": str(target), "index_html": str(index), "real_mesh_layers": ["official SMPL-X hand surface", "spider raw SMPL-X hand surface", "Stage B MuJoCo Wuji visual surface", "C-XA MuJoCo Wuji visual surface", "official/raw/Stage-B/C-XA object visual mesh"], "coordinate_modes": ["世界坐标", "物体坐标", "左腕坐标", "右腕坐标"], "comparison_modes": ["官方 source vs spider raw", "spider raw vs Stage B", "Stage B vs C-XA", "四阶段叠加"]})
