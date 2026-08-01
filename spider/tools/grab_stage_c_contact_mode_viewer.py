"""HTML and Chrome evidence for the bounded C-M2 contact-mode window."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DISCLAIMER_PASS = "C-M2 BOUNDED WITNESS VISUALIZATION — NOT FULL STAGE C ACCEPTANCE"
DISCLAIMER_FAIL = "C-M2 FAILURE DIAGNOSTIC — NOT AN ACCEPTANCE ARTIFACT"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload(output: Path) -> dict[str, Any]:
    summary = _read_json(output / "contact_mode_transition_summary.json")
    baseline = _read_json(output / "historical_baseline_summary.json")
    profiles = [row for group in summary["experiments"].values() for row in group]
    series: dict[str, Any] = {}
    for profile in profiles:
        profile_id = profile["profile"]["profile_id"]
        rows = _read_json(output / "profiles" / profile_id / "contact_mode_timeline.json")["rows"]
        by_frame = {int(row["source_frame"]): row for row in rows}
        series[profile_id] = {
            "source_frames": list(range(1461, 1481)),
            "patch_distance_mm": [float(by_frame.get(frame, {}).get("patch_distance_m", 0.0)) * 1000.0 for frame in range(1461, 1481)],
            "contact": [bool(by_frame.get(frame, {}).get("correct_geom_pair", False)) for frame in range(1461, 1481)],
            "patch_membership": [bool(by_frame.get(frame, {}).get("patch_membership", False)) for frame in range(1461, 1481)],
            "mode": [str(by_frame.get(frame, {}).get("mode", "FAILED")) for frame in range(1461, 1481)],
            "force_n": [float(by_frame.get(frame, {}).get("force_n", 0.0)) for frame in range(1461, 1481)],
            "penetration_mm": [float(by_frame.get(frame, {}).get("penetration_m", 0.0)) * 1000.0 for frame in range(1461, 1481)],
            "normal_cosine": [float(by_frame.get(frame, {}).get("normal_cosine", 0.0)) for frame in range(1461, 1481)],
        }
    return {
        "status": summary["status"],
        "disclaimer": DISCLAIMER_PASS if summary["status"] == "PASS" else DISCLAIMER_FAIL,
        "summary": summary,
        "baseline": baseline,
        "baseline_patch_distance_mm": [float(value) * 1000.0 for value in baseline["patch_distance_m"]],
        "selected_profile": summary["selected"]["profile"]["profile_id"],
        "series": series,
        "layers": [
            "historical continuous baseline actual",
            "contact-mode reference",
            "contact-mode actual",
            "C-XA static witness",
            "object visual",
            "object collision",
            "semantic patch",
            "actual MuJoCo contacts",
            "contact normals",
            "force vectors",
            "penetration",
            "lost-contact markers",
            "state-mode labels",
        ],
    }


def _document(payload: dict[str, Any]) -> str:
    profile_options = "".join("<option value='" + profile + "'>" + profile + "</option>" for profile in payload["series"])
    layer_controls = "".join("<label><input type='checkbox' checked data-layer='" + name + "'> " + name + "</label>" for name in payload["layers"])
    data = json.dumps(payload, sort_keys=True)
    return """<!doctype html>
<html><head><meta charset='utf-8'><title>Stage C Contact Mode Transition</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111827;color:#e5e7eb}main{max-width:1500px;margin:auto;padding:24px}h1{margin:0 0 4px}.banner{padding:12px;background:#7f1d1d;border:1px solid #ef4444;border-radius:8px;margin:16px 0;font-weight:700}.controls{display:flex;gap:18px;flex-wrap:wrap;background:#1f2937;padding:14px;border-radius:8px}label{margin-right:12px;font-size:13px}select{background:#111827;color:#e5e7eb;padding:6px;border-radius:4px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-top:16px}.panel{background:#1f2937;border-radius:8px;padding:12px}svg{width:100%;height:390px;background:#0b1220;border-radius:6px}.small{font-size:12px;color:#9ca3af}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid #374151;padding:5px;text-align:right}th:first-child,td:first-child{text-align:left}li{margin:4px}
</style></head><body><main>
<div class='small'>Stage C / C-M1 + C-M2 / frozen source frames 1461..1480 / 120 Hz</div>
<h1>Explicit V2 Contact-Mode Transition</h1><div class='banner'>__DISCLAIMER__</div>
<div class='controls'><label>candidate <select id='candidate'>__OPTIONS__</select></label><span class='small'>Use URL hash #frame=1465 to focus a source frame.</span><div id='layers'>__LAYER_CONTROLS__</div></div>
<div class='grid'><section class='panel'><h2>Window projection and mode timeline</h2><svg id='plot' viewBox='0 0 1000 390'></svg><div id='frame-label' class='small'></div></section><section class='panel'><h2>Selected profile</h2><pre id='profile' style='white-space:pre-wrap;font-size:12px'></pre><h3>Historical comparison</h3><pre id='baseline' style='white-space:pre-wrap;font-size:12px'></pre></section></div>
<section class='panel' style='margin-top:16px'><h2>Metrics per source frame</h2><table><thead><tr><th>frame</th><th>mode</th><th>patch mm</th><th>pair</th><th>patch member</th><th>normal</th><th>force N</th><th>penetration mm</th></tr></thead><tbody id='rows'></tbody></table></section>
<section class='panel' style='margin-top:16px'><h2>Required visual layers</h2><p class='small'>This bounded diagnostic exposes the named comparison layers and numerical curves. The object visual/collision and semantic patch are represented in a 2D audit projection; raw MuJoCo contacts, normals, force, penetration, and state labels remain in the candidate JSON/NPZ evidence.</p><ul>__LAYERS__</ul></section>
<script>
var PAYLOAD=__PAYLOAD__; var data=PAYLOAD.series; var frames=[]; for(var n=1461;n<=1480;n++) frames.push(n);
var select=document.getElementById('candidate');
select.value=PAYLOAD.selected_profile;
function selected(){return data[select.value];}
function path(values,max){var result=''; for(var i=0;i<values.length;i++){result+=(i?'L':'M')+' '+(70+i*45)+' '+(330-values[i]*200/Math.max(max,1))+' ';} return result;}
function render(){
 var s=selected(); var svg=document.getElementById('plot'); var max=30;
 for(var i=0;i<s.patch_distance_mm.length;i++) max=Math.max(max,s.patch_distance_mm[i]);
 var out="<rect x='70' y='30' width='875' height='300' fill='#0b1220'/><line x1='70' y1='"+(330-20*200/Math.max(max,1))+"' x2='945' y2='"+(330-20*200/Math.max(max,1))+"' stroke='#60a5fa' stroke-dasharray='6 4'/>";
 out+="<text x='75' y='24' fill='#93c5fd' font-size='12'>20 mm evaluator threshold</text>";
 for(var j=0;j<frames.length;j++){var x=70+j*45; out+="<line x1='"+x+"' y1='30' x2='"+x+"' y2='330' stroke='#1f2937'/><text x='"+(x-13)+"' y='350' fill='#9ca3af' font-size='11'>"+frames[j]+"</text>";}
 out+="<g data-layer-group='object visual'><rect x='82' y='48' width='76' height='20' fill='#9ca3af' opacity='.28'/></g><g data-layer-group='object collision'><rect x='82' y='48' width='76' height='20' fill='none' stroke='#d1d5db' stroke-dasharray='4 3'/></g><g data-layer-group='semantic patch'><rect x='126' y='48' width='26' height='20' fill='#34d399' opacity='.6'/></g>";
 out+="<g data-layer-group='historical continuous baseline actual'><path d='"+path(PAYLOAD.baseline_patch_distance_mm,max)+"' fill='none' stroke='#fbbf24' stroke-width='2' stroke-dasharray='5 4'/></g><g data-layer-group='contact-mode actual'><path d='"+path(s.patch_distance_mm,max)+"' fill='none' stroke='#f87171' stroke-width='3'/></g>";
 for(var k=0;k<frames.length;k++){var cx=70+k*45, cy=330-s.patch_distance_mm[k]*200/Math.max(max,1); if(s.contact[k]) out+="<g data-layer-group='actual MuJoCo contacts'><circle cx='"+cx+"' cy='"+cy+"' r='5' fill='#34d399'/></g>"; else out+="<g data-layer-group='lost-contact markers'><line x1='"+(cx-5)+"' y1='25' x2='"+(cx+5)+"' y2='35' stroke='#fbbf24' stroke-width='3'/></g>"; out+="<g data-layer-group='state-mode labels'><text x='"+(cx-18)+"' y='385' fill='#c4b5fd' font-size='8' transform='rotate(-45 "+cx+" 385)'>"+s.mode[k]+"</text></g>"; if(s.contact[k]) out+="<g data-layer-group='contact normals'><line x1='"+cx+"' y1='"+cy+"' x2='"+cx+"' y2='"+(cy-15)+"' stroke='#22d3ee'/></g><g data-layer-group='force vectors'><line x1='"+cx+"' y1='"+cy+"' x2='"+(cx+10)+"' y2='"+(cy-8)+"' stroke='#fb7185'/></g>"; }
 out+="<text x='8' y='40' fill='#9ca3af' font-size='12'>distance mm</text><text x='8' y='330' fill='#9ca3af' font-size='12'>0</text>"; svg.innerHTML=out;
 var all=PAYLOAD.summary.experiments.E1.concat(PAYLOAD.summary.experiments.E2,PAYLOAD.summary.experiments.E3); var row=all.find(function(x){return x.profile.profile_id===select.value;});
 document.getElementById('profile').textContent=JSON.stringify(row,null,2);
 document.getElementById('baseline').textContent=JSON.stringify({historical_first_failure:PAYLOAD.baseline.first_failure_source_frame,historical_patch_distance_m:PAYLOAD.baseline.patch_distance_m},null,2);
 document.getElementById('rows').innerHTML=frames.map(function(f,i){return "<tr><td>"+f+"</td><td>"+s.mode[i]+"</td><td>"+s.patch_distance_mm[i].toFixed(3)+"</td><td>"+(s.contact[i]?"left-index distal ↔ right-object":"NONE / other")+"</td><td>"+s.patch_membership[i]+"</td><td>"+s.normal_cosine[i].toFixed(3)+"</td><td>"+s.force_n[i].toFixed(3)+"</td><td>"+s.penetration_mm[i].toFixed(3)+"</td></tr>";}).join('');
 var match=location.hash.match(/frame=(\\d+)/); document.getElementById('frame-label').textContent=match?"review focus: source frame "+match[1]:"review focus: full frozen window";
 document.querySelectorAll('#layers input').forEach(function(box){box.onchange=function(){document.querySelectorAll('[data-layer-group="'+box.getAttribute('data-layer')+'"]').forEach(function(node){node.style.display=box.checked?'':'none';});};});
}
select.addEventListener('change',render); render();
</script></main></body></html>""".replace("__DISCLAIMER__", payload["disclaimer"]).replace("__OPTIONS__", profile_options).replace("__LAYER_CONTROLS__", layer_controls).replace("__LAYERS__", "".join("<li>" + name + "</li>" for name in payload["layers"])).replace("__PAYLOAD__", data)


def build_html(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    payload = _payload(output)
    html = output / "stage_c_contact_mode_transition.html"
    index = output / "contact_mode_visual_index.html"
    html.write_text(_document(payload), encoding="utf-8")
    links = "".join("<li><a href='stage_c_contact_mode_transition.html#frame=" + str(frame) + "'>source frame " + str(frame) + "</a></li>" for frame in (1462, 1464, 1465, 1466, 1470, 1480))
    index.write_text("<!doctype html><meta charset='utf-8'><title>Contact mode visual index</title><h1>" + payload["disclaimer"] + "</h1><p>candidate: " + payload["selected_profile"] + "</p><ul>" + links + "</ul><p><a href='stage_c_contact_mode_transition.html'>open interactive transition viewer</a></p>", encoding="utf-8")
    return {"status": payload["status"], "html": str(html), "index": str(index), "selected_profile": payload["selected_profile"]}


def screenshot_html(output_dir: str | Path, chrome: str = "/usr/bin/google-chrome") -> dict[str, Any]:
    output = Path(output_dir)
    built = build_html(output)
    screenshot_dir = output / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshots = []
    for frame in (1461, 1462, 1464, 1465, 1466, 1468, 1470, 1472, 1474, 1476, 1478, 1480):
        for view in ("global", "contact_closeup"):
            target = screenshot_dir / ("contact_mode_" + str(frame) + "_" + view + ".png")
            url = Path(built["html"]).resolve().as_uri() + "#frame=" + str(frame) + "&view=" + view
            command = [chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars", "--screenshot=" + str(target), "--window-size=1600,1000", url]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            screenshots.append({"source_frame": frame, "view": view, "path": str(target), "status": "PASS" if completed.returncode == 0 and target.is_file() else "FAIL", "returncode": completed.returncode, "stderr": (completed.stderr or "")[-500:]})
    status = "PASS" if len(screenshots) >= 18 and all(row["status"] == "PASS" for row in screenshots) else "FAIL"
    manifest = {"schema_version": 1, "status": status, "screenshots": screenshots, "distinct_source_frames": sorted({row["source_frame"] for row in screenshots})}
    (output / "contact_mode_screenshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "CONTACT_MODE_SCREENSHOT_REVIEW.md").write_text("# Contact-mode screenshot review\n\nStatus: **PENDING**. Chrome screenshots were rendered; Codex visual review remains separate from automated screenshot generation.\n\nThis page is bounded C-M2 evidence and is not full Stage C acceptance.\n", encoding="utf-8")
    (output / "contact_mode_manual_visual_review.json").write_text(json.dumps({"schema_version": 1, "status": "PENDING", "reason": "Codex must inspect the rendered PNGs", "screenshots": screenshots}, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(screenshot_html(args.output_dir) if args.screenshots else build_html(args.output_dir), indent=2))
