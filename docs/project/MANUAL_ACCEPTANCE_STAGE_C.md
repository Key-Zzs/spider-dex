# Manual Stage C acceptance

There is currently no primary Stage C HTML to accept. The required primary
contact-guided MJWP run failed before producing a finite optimized trajectory;
see `<workspace>/reports/stage_c_validation.json`.

Do not treat the Stage B HTML as Stage C evidence. Once the start-state physics
failure is repaired, open the generated primary `stage_c_acceptance.html` with:

```bash
google-chrome /absolute/path/to/stage_c_acceptance.html
```

Then check kinematic versus optimized penetration, anchors versus real mesh,
side/thumb/palm orientation, preserved true contacts, offhand false contacts,
object tracking, jitter, and collision/visual mesh alignment. User review
status is PENDING; Stage D must not start before an actual repaired Stage C
HTML is reviewed.
