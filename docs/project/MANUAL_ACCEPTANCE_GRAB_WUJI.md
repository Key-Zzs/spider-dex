# Stage B manual acceptance

Current frozen primary pilot: `s5/cylindermedium_lift`, frames `[1460, 1876)`,
120 Hz, object `cylindermedium`. It was selected source-only because both
hands have sustained fingertip proximity to the real object surface; the
initial part includes the right-hand approach. `s1/mug_lift` is retained only
as a right-source smoke, not a bimanual primary.

Prefer the self-contained interactive HTML reports: they support orbit, pan,
zoom, trace visibility toggles, and frame sliders. The source report includes
world XYZ axes; the Wuji report overlays source hands on the real robot visual
mesh and the source object, so frame-local spatial disagreement is directly
visible. In the Wuji report, source right/left are red/blue and Wuji
right/left are purple/teal, respectively.

```bash
# Source replay
xdg-open <workspace>/processed/grab/canonical/s1__mug_lift/visualization/source_replay.mp4

# Wuji replay
xdg-open <workspace>/processed/grab/wuji_hand2_beta1/bimanual/s1__mug_lift/0/visualization_ik.mp4

# Generate interactive HTML if it has not already been generated.
conda run -n spider-dex python -m spider.tools.grab_interactive_viewer source \
  --canonical-dir <workspace>/processed/grab/canonical/s1__mug_lift
conda run -n spider-dex python -m spider.tools.grab_interactive_viewer wuji \
  --robot-dir <workspace>/processed/grab/wuji_hand2_beta1/bimanual/s1__mug_lift/0 \
  --canonical-dir <workspace>/processed/grab/canonical/s1__mug_lift

# Interactive Wuji scene after generation
conda run -n spider-dex python spider/viewers/mjcpu_viewer.py \
  --dataset-dir <workspace> --dataset-name grab --robot-type wuji_hand2_beta1 \
  --embodiment-type bimanual --task s1__mug_lift --data-id 0 --data-type kinematic
```

Canonical source replay:

- [ ] At three separated slider positions, orbit around the scene and confirm
  that the red/blue hand joints maintain a physically plausible grasp/approach
  relation to the object from every viewpoint. A persistent separation is a
  frame/scale failure, not a camera effect.
- [ ] The object mesh scale/pose is correct and shares the hand world frame.
- [ ] Neither wrist has a fixed roughly 10 cm centre offset.
- [ ] Left/right, palm orientation, thumb side, fingertip order, axes, and time
  synchronization are correct.
- [ ] The window contains pre-contact approach and interaction; no 90/180
  degree, axis-swap, or metre/millimetre error is visible.

Wuji replay:

- [ ] Toggle source hands and compare red source-right against purple
  Wuji-right, then blue source-left against teal Wuji-left; a side swap or
  wrong thumb side is a FAIL.
- [ ] Toggle source hands/object together and orbit the cup wall. The robot
  palm/fingers must approach the same side of the yellow source object; a
  visually wrong wall direction is a FAIL pending an IK/frame fix.
- [ ] Each Wuji wrist follows its corresponding source wrist.
- [ ] Thumb/index/middle/ring/pinky identity and side are correct. Persistent
  visibly reverse-bent four-finger poses are not normal for acceptance and
  must be recorded as FAIL rather than waived as a rendering artifact.
- [ ] There is no sustained limit lock, high-frequency jitter, frame jump, or
  swapped second hand; object motion matches source.
- [ ] The known fingertip tracking shortfall is acceptable, or is rejected
  with a concrete frame/time observation.

```text
Stage B manual acceptance: PASS / FAIL

Canonical source replay:
- sequence:
- result:
- issues:

Wuji IK replay:
- sequence:
- result:
- issues:

Additional observations:
```
