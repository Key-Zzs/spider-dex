# GRAB to Wuji kinematic IK

`spider.tools.grab_pipeline` writes canonical data and SPIDER-compatible
`trajectory_keypoints.npz` under the external workspace, stages the source
object mesh as an external OBJ, invokes the existing `generate_xml.py`, then
invokes existing `ik_fast.py`. It does not invoke MJWP, physics optimization,
or contact tuning.

For a single shared GRAB object, the generated SPIDER scene carries it as the
right object and retains a masked compatibility left-object channel required by
the legacy bimanual interface. The canonical schema still preserves the true
source hand streams.

```bash
conda run -n spider-dex python -m spider.tools.grab_pipeline render-source \
  --canonical-dir <workspace>/processed/grab/canonical/s1__mug_lift
conda run -n spider-dex python -m spider.tools.grab_pipeline run-wuji-ik \
  --paths-config configs/local/paths.yaml --sequence-id s1__mug_lift
```

The post-run validator writes source-frame mapping, config, named actuators,
joint-limit/continuity metrics, and tracking statistics. The tracked smoke
threshold config deliberately reports an executable trajectory with excessive
tracking error as `AUTO_PIPELINE_PASS_MANUAL_REVIEW_REQUIRED` rather than
silently widening thresholds.

For manual acceptance, generate `grab_interactive_viewer source` and
`grab_interactive_viewer wuji`. These are self-contained Plotly HTML files with
frame sliders, orbit/zoom/pan controls, source/world axes, and source-vs-Wuji
overlay; use them instead of a fixed camera MP4 for coordinate-frame review.
