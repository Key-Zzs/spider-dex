# Object collision assets

Stage C caches one collision asset per SHA-256 visual mesh under the configured
external workspace: `cache/objects/<hash-prefix>/`. Each cache contains an
unchanged visual OBJ, CoACD convex pieces, a manifest, and bbox/finite/area
validation. Cache construction is atomic and idempotent. The repository stores
only code and configuration; no source mesh or collision output is tracked.
