"""Pure, fail-closed helpers for Stage C-X task-equivalent contact assignment.

V1 identifies a source contact by the original hand/finger.  V2 never changes
that source evidence; it selects a bounded, same-side robot region to satisfy
the same source functional role and surface patch.  The functions here are
data-only so tests can verify invariants without MuJoCo or a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ROLES = ("THUMB_OPPOSITION", "PRIMARY_GRASP", "SUPPORT", "STABILIZATION", "TRANSIENT", "NON_INTERACTING")
NON_THUMB_NEIGHBORS = {
    "index": ("index", "middle"), "middle": ("index", "middle", "ring"),
    "ring": ("middle", "ring", "pinky"), "pinky": ("ring", "pinky"),
}


@dataclass(frozen=True)
class RobotRegion:
    region_id: str
    side: str
    finger: str | None
    kind: str
    allowed_roles: tuple[str, ...]
    capacity: int = 1


def default_robot_regions() -> list[RobotRegion]:
    """Return the finite Wuji region catalogue; no proximal-link fallback."""
    regions: list[RobotRegion] = []
    for side in ("right", "left"):
        for finger in FINGERS:
            roles = ("THUMB_OPPOSITION",) if finger == "thumb" else ("PRIMARY_GRASP", "SUPPORT", "STABILIZATION", "TRANSIENT")
            for kind in ("fingertip", "distal"):
                regions.append(RobotRegion(f"{side}_{finger}_{kind}", side, finger, kind, roles))
        regions.append(RobotRegion(f"{side}_palm_support", side, None, "palm", ("SUPPORT", "STABILIZATION")))
    return regions


def classify_role(source_finger: str, duration_frames: int, overlapping_fingers: Iterable[str]) -> tuple[str, float, list[str]]:
    """Classify only source-side facts, never robot IK or a chosen assignment."""
    overlap = set(overlapping_fingers)
    if source_finger not in FINGERS:
        raise ValueError(f"Unknown source finger: {source_finger}")
    if duration_frames <= 0:
        return "NON_INTERACTING", 1.0, ["inactive source contact"]
    if source_finger == "thumb" and any(finger != "thumb" for finger in overlap):
        return "THUMB_OPPOSITION", 0.9, ["thumb contact overlaps non-thumb source contact"]
    if source_finger != "thumb" and "thumb" in overlap and duration_frames >= 12:
        return "PRIMARY_GRASP", 0.8, ["sustained non-thumb contact overlaps source thumb"]
    if duration_frames >= 30:
        return "SUPPORT", 0.7, ["sustained source contact without opposition evidence"]
    if duration_frames >= 6:
        return "STABILIZATION", 0.6, ["short persistent source contact"]
    return "TRANSIENT", 0.55, ["brief source contact interval"]


def allowed_region(region: RobotRegion, source_side: str, source_finger: str, role: str, level: int) -> bool:
    """Enforce side, thumb, palm, and neighbor constraints for one level."""
    if level not in range(5):
        raise ValueError("relaxation level must be 0..4")
    if region.side != source_side or role not in region.allowed_roles:
        return False
    if role == "THUMB_OPPOSITION":
        return region.finger == "thumb"
    if region.kind == "palm":
        return level >= 3 and role in {"SUPPORT", "STABILIZATION"}
    if level <= 1:
        return region.finger == source_finger
    if level == 2:
        return source_finger != "thumb" and region.finger in NON_THUMB_NEIGHBORS[source_finger]
    return region.finger == "thumb" if source_finger == "thumb" else region.finger in FINGERS[1:]


def thin_wall_compatible(
    source_component: int,
    candidate_component: int,
    source_normal: np.ndarray,
    candidate_normal: np.ndarray,
    normal_cosine_min: float = 0.0,
) -> bool:
    """Reject Euclidean-near but disconnected/opposite thin-wall patches."""
    a, b = np.asarray(source_normal, dtype=float), np.asarray(candidate_normal, dtype=float)
    if source_component != candidate_component or np.linalg.norm(a) <= 1e-9 or np.linalg.norm(b) <= 1e-9:
        return False
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) >= normal_cosine_min


def assignment_cost(cost_terms: dict[str, float], weights: dict[str, float]) -> float:
    required = {"surface_patch", "normal", "functional_role", "identity_change", "reachability", "collision_risk", "tracking_deviation", "temporal_switch"}
    if set(cost_terms) != required or not required.issubset(weights):
        raise ValueError("assignment costs require every named V2 term")
    if any(not np.isfinite(float(cost_terms[key])) or float(cost_terms[key]) < 0 for key in required):
        raise ValueError("assignment costs must be finite and non-negative")
    return float(sum(float(weights[key]) * float(cost_terms[key]) for key in required))


def viterbi_assignment(frame_costs: np.ndarray, switch_cost: float, max_switches: int) -> tuple[np.ndarray, float, int]:
    """Bounded temporal assignment, never a per-frame nearest-point greedy rule."""
    costs = np.asarray(frame_costs, dtype=float)
    if costs.ndim != 2 or not len(costs) or not costs.shape[1] or max_switches < 0 or switch_cost < 0:
        raise ValueError("invalid bounded Viterbi assignment inputs")
    if not np.isfinite(costs).all():
        raise ValueError("assignment costs must be finite")
    frames, states = costs.shape
    dp = np.full((frames, states, max_switches + 1), np.inf)
    back = np.full((frames, states, max_switches + 1, 2), -1, dtype=np.int64)
    dp[0, :, 0] = costs[0]
    for frame in range(1, frames):
        for state in range(states):
            for switches in range(max_switches + 1):
                stay = dp[frame - 1, state, switches]
                if stay < dp[frame, state, switches]:
                    dp[frame, state, switches] = stay + costs[frame, state]
                    back[frame, state, switches] = (state, switches)
                if switches:
                    previous = dp[frame - 1, :, switches - 1].copy(); previous[state] = np.inf
                    prior = int(np.argmin(previous))
                    value = previous[prior] + switch_cost + costs[frame, state]
                    if value < dp[frame, state, switches]:
                        dp[frame, state, switches] = value
                        back[frame, state, switches] = (prior, switches - 1)
    state, switches = np.unravel_index(np.argmin(dp[-1]), dp[-1].shape)
    path = np.empty(frames, dtype=np.int64)
    for frame in range(frames - 1, -1, -1):
        path[frame] = state
        if frame:
            state, switches = back[frame, state, switches]
            if state < 0:
                raise RuntimeError("incomplete Viterbi backtrace")
    return path, float(np.min(dp[-1])), int(np.count_nonzero(np.diff(path)))


def select_minimum_successful_level(level_status: dict[int, str]) -> int | None:
    """Return the first strict PASS; reject skipped/missing earlier levels."""
    for level in range(1, 5):
        status = level_status.get(level)
        if status == "PASS":
            if any(level_status.get(previous) not in {"FAIL", "BLOCKED"} for previous in range(1, level)):
                raise ValueError("a later relaxation level cannot bypass an unevaluated earlier level")
            return level
    return None
