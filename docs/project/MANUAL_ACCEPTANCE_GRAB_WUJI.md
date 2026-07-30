# Stage B manual acceptance

Current primary pilot: `s1/mug_lift`, frames `[120, 240)`, 120 Hz, bimanual
source streams, object `mug`.

```bash
# Source replay
xdg-open <workspace>/processed/grab/canonical/s1__mug_lift/visualization/source_replay.mp4

# Wuji replay
xdg-open <workspace>/processed/grab/wuji_hand2_beta1/bimanual/s1__mug_lift/0/visualization_ik.mp4

# Interactive Wuji scene after generation
conda run -n spider-dex python spider/viewers/mjcpu_viewer.py \
  --dataset-dir <workspace> --dataset-name grab --robot-type wuji_hand2_beta1 \
  --embodiment-type bimanual --task s1__mug_lift --data-id 0 --data-type kinematic
```

Canonical source replay:

- [ ] Object mesh scale/pose is correct and shares the hand world frame.
- [ ] Neither wrist has a fixed roughly 10 cm centre offset.
- [ ] Left/right, palm orientation, thumb side, fingertip order, axes, and time
  synchronization are correct.
- [ ] The window contains pre-contact approach and interaction; no 90/180
  degree, axis-swap, or metre/millimetre error is visible.

Wuji replay:

- [ ] Each Wuji wrist follows its corresponding source wrist.
- [ ] Thumb/index/middle/ring/pinky identity and side are correct.
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
