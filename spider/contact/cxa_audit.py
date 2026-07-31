"""Pure C-XA audit rules for the Stage-C V2 contact contract.

The helpers deliberately keep three questions separate: whether a source
sample is geometrically reliable, whether a source interval is a functional
role, and whether the evidence supports a Case-A or Case-B decision.  They do
not inspect robot reachability, so an infeasible robot assignment can never be
used to downgrade source evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

FUNCTIONAL_ROLE_TYPES = frozenset({
    "THUMB_OPPOSITION",
    "PRIMARY_GRASP",
    "SUPPORT",
    "STABILIZATION",
})
NONFUNCTIONAL_ROLE_TYPES = frozenset({"TRANSIENT", "NON_INTERACTING"})


def is_functional_role(role_type: str) -> bool:
    """Return whether a source role belongs in functional-role recall."""
    if role_type in FUNCTIONAL_ROLE_TYPES:
        return True
    if role_type in NONFUNCTIONAL_ROLE_TYPES:
        return False
    raise ValueError(f"unknown source role type: {role_type}")


def source_contact_reliability(
    signed_distance_m: float | None,
    unsigned_distance_m: float,
    contact_distance_m: float,
) -> tuple[str, list[str]]:
    """Classify one source sample without changing the source record.

    A watertight sign calculation establishes that a sample is inside/outside;
    it does *not* establish that an arbitrarily deep penetration is a reliable
    physical contact.  The immutable source contact tolerance is the only
    geometric allowance used here.
    """
    if contact_distance_m <= 0:
        raise ValueError("contact_distance_m must be positive")
    if unsigned_distance_m < 0:
        raise ValueError("unsigned_distance_m must be non-negative")
    if unsigned_distance_m <= contact_distance_m:
        return "RELIABLE_SOURCE", ["closest surface distance is within immutable contact tolerance"]
    if signed_distance_m is not None and signed_distance_m < -contact_distance_m:
        return "UNRELIABLE_SOURCE", [
            "source point is deeper than immutable contact tolerance",
            "watertight sign confidence alone does not make deep penetration a contact",
        ]
    return "NON_CONTACT", ["closest surface distance exceeds immutable contact tolerance"]


def summarize_functional_role_recall(role_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Audit V2 functional-role recall as an unweighted unique-role metric."""
    rows = list(role_rows)
    functional = [row for row in rows if is_functional_role(str(row["role_type"]))]
    failed = [
        {
            "role_id": row["role_id"],
            "role_type": row["role_type"],
            "coverage": float(row["coverage"]),
            "reason": "coverage below required per-role 0.80",
        }
        for row in functional
        if not bool(row["passed"])
    ]
    numerator = len(functional) - len(failed)
    denominator = len(functional)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": float(numerator / denominator) if denominator else 0.0,
        "weighting_rule": "unweighted unique functional source-role intervals; TRANSIENT and NON_INTERACTING are reported but excluded",
        "failed_samples": failed,
        "excluded_nonfunctional_roles": [
            {"role_id": row["role_id"], "role_type": row["role_type"]}
            for row in rows
            if not is_functional_role(str(row["role_type"]))
        ],
    }


def decide_cxa_case(
    *,
    metric_implementation_errors: Iterable[str],
    source_contact_labeling_errors: Iterable[str],
    patch_definition_errors: Iterable[str],
    assignment_levels_covered: bool,
    remaining_failure_reasons: Iterable[str],
) -> dict[str, Any]:
    """Apply the stated C-XA decision tree without threshold changes."""
    metric_errors = sorted(set(metric_implementation_errors))
    labeling_errors = sorted(set(source_contact_labeling_errors))
    patch_errors = sorted(set(patch_definition_errors))
    remaining = sorted(set(remaining_failure_reasons))
    case_a_reasons = metric_errors + labeling_errors + patch_errors
    if case_a_reasons:
        return {
            "decision": "CASE_A_IMPLEMENTATION_OR_EVALUATION_BUG",
            "case_a_reasons": case_a_reasons,
            "case_b_preconditions_met": False,
        }
    embodiment_reasons = {
        "ROBOT_REACHABILITY_LIMIT",
        "JOINT_LIMIT_CONFLICT",
        "CONTACT_ROLE_CONFLICT",
        "TEMPORAL_ASSIGNMENT_FAILURE",
    }
    case_b = assignment_levels_covered and bool(remaining) and set(remaining).issubset(embodiment_reasons)
    return {
        "decision": "CASE_B_TRUE_EMBODIMENT_INFEASIBILITY" if case_b else "DECISION_INSUFFICIENT_EVIDENCE",
        "case_a_reasons": [],
        "case_b_preconditions_met": case_b,
        "remaining_failure_reasons": remaining,
    }
