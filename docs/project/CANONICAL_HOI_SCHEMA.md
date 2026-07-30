# Canonical HOI schema v1

`CanonicalHOISequence` is the versioned boundary between source adapters and
SPIDER conversion. It stores timestamps, FPS, complete source IDs, a coordinate
contract, real right/left hand presence, object trajectories, source metadata,
and provenance. Lengths are always metres.

Each present `HandSequence` carries a validity mask, world translation and
orientation, MANO pose/shape, world joints, joint names, model type/gender, and
optionally vertices. Missing hands are `None`; they are never zero-filled in
the canonical schema. Each `ObjectSequence` has a safe ID, source-relative mesh
path, validity mask, pose, scale, and source metadata.

Coordinates declare world frame, axis convention, handedness, rotation
representation, and source-to-canonical transform. Rotations are either valid
orthogonal matrices with determinant +1 or normalized `wxyz` quaternions.
Every time-varying array shares the timestamp dimension; NaN/Inf and object
dtypes are rejected.

Serialization is a numerical `canonical_sequence.npz` plus JSON-only
`canonical_metadata.json`. Loading always uses `allow_pickle=False`. Save is
atomic and includes enough metadata to keep resolved local source paths out of
portable/public manifests.
