# Manual acceptance — Stage C C-M1R

Final user review remains **PENDING**. Open the 3D HTML and verify:

- [ ] The scene uses complete Wuji and object meshes, not rectangles or point-only proxies.
- [ ] The semantic patch is a green connected surface.
- [ ] Frame 1461 begins in correct left-index contact and the numerical trace enters `RETAIN_PENDING`, then `RETAIN`.
- [ ] The M0 initial peak is not the old approximately 110 N impulse.
- [ ] The object-frame target follows the object.
- [ ] M1 visibly loses the same left-index contact at frame 1462; it does not swap fingers or patches.
- [ ] There is no object teleport, qpos write, deep penetration, or persistent controller oscillation.
- [ ] The M1 visual failure agrees with the reported `RETENTION_FAILURE` and M2/M3 are marked `NOT_RUN`.
- [ ] The page explicitly says it is not a full Stage C acceptance artifact.

User feedback template:

```text
Stage C C-M1R visual review: PASS / FAIL

M0 initial hold:
- result:
- observations:

M1 moving retention:
- result:
- observations:

3D visualization:
- result:
- issues:

Additional observations:
```
