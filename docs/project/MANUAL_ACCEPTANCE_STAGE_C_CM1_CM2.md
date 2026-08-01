# Manual acceptance: Stage C C-M1/C-M2

User visual review is `PENDING`. The generated page is a bounded failure
diagnostic, not full Stage C acceptance.

## Open the generated evidence

```bash
google-chrome \
  "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_cm1_cm2/20260731T200000Z-contact-mode/stage_c_contact_mode_transition.html"
```

```bash
google-chrome \
  "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_cm1_cm2/20260731T200000Z-contact-mode/contact_mode_visual_index.html"
```

If local-file restrictions prevent the page from loading:

```bash
cd "/home/deepcybo/workspace/dex/retarget/spider-dex/.local_artifacts/stage_c_cm1_cm2/20260731T200000Z-contact-mode"
python -m http.server 8000
```

Open [http://127.0.0.1:8000/contact_mode_visual_index.html](http://127.0.0.1:8000/contact_mode_visual_index.html).

## Review order

1. Open the visual index.
2. Review the historical baseline and source frames 1464, 1465, and 1466.
3. Select `e3_high_kp_anchor`, the best failed bounded candidate.
4. Check the 20 mm line, contact markers, lost-contact markers, and mode labels.
5. Inspect the ACQUIRE → REGRASP → FAILED transition and terminal frame 1480.
6. Toggle the object visual/collision, semantic patch, normals, force,
   penetration, and state-label layers.
7. Compare the page against `selected_contact_mode_profile.json` and the raw
   `contact_mode_trace.npz`.

## Checklist

- [ ] Historical frame-1465 loss is visible.
- [ ] The selected candidate is visibly labeled as failure diagnostic.
- [ ] Contact identity remains left-index distal ↔ right-object.
- [ ] No finger switch or patch enlargement is shown.
- [ ] ACQUIRE/REGRASP/FAILED labels agree with the JSON timeline.
- [ ] No object qpos overwrite, teleport, deep penetration, or force claim is
  implied by the page.
- [ ] The page is not mistaken for full Stage C acceptance.
