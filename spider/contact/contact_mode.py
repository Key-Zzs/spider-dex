"""Explicit V2 contact-mode state machine.

The state machine is deliberately independent of MuJoCo.  A runner supplies
one immutable-role observation per integration substep and receives a bounded
mode decision.  This keeps contact semantics, timing, and safety accounting
out of ad-hoc controller branches while making every transition serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ContactMode(str, Enum):
    PRE_CONTACT = "PRE_CONTACT"
    ACQUIRE = "ACQUIRE"
    RETAIN_PENDING = "RETAIN_PENDING"
    RETAIN = "RETAIN"
    RELEASE = "RELEASE"
    REGRASP = "REGRASP"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class FailureCode(str, Enum):
    ACQUIRE_TIMEOUT = "ACQUIRE_TIMEOUT"
    REGRASP_TIMEOUT = "REGRASP_TIMEOUT"
    WRONG_REGION_CONTACT = "WRONG_REGION_CONTACT"
    PATCH_DISTANCE_VIOLATION = "PATCH_DISTANCE_VIOLATION"
    ROLE_CONTINUITY_VIOLATION = "ROLE_CONTINUITY_VIOLATION"
    PENETRATION_VIOLATION = "PENETRATION_VIOLATION"
    FORCE_VIOLATION = "FORCE_VIOLATION"
    JOINT_LIMIT_VIOLATION = "JOINT_LIMIT_VIOLATION"
    TRACKING_VIOLATION = "TRACKING_VIOLATION"
    OBJECT_TRACKING_VIOLATION = "OBJECT_TRACKING_VIOLATION"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    INVALID_MAPPING = "INVALID_MAPPING"


@dataclass(frozen=True)
class ContactModeConfig:
    """Frozen state-machine bounds; evaluator thresholds are not relaxed."""

    patch_distance_m: float = 0.020
    normal_cosine_min: float = 0.50
    penetration_max_m: float = 0.003
    force_max_n: float = 150.0
    acquire_entry_distance_m: float = 0.025
    retain_hysteresis_distance_m: float = 0.025
    confirmation_substeps: int = 4
    acquire_timeout_ms: int = 40
    regrasp_timeout_ms: int = 40
    max_regrasp_attempts: int = 1
    sim_dt_s: float = 0.0005
    allow_regrasp: bool = True

    def __post_init__(self) -> None:
        if self.patch_distance_m != 0.020:
            raise ValueError("V2 patch-distance evaluator threshold is immutable at 0.020 m")
        if not 0.0 < self.acquire_entry_distance_m <= 0.030:
            raise ValueError("acquire entry distance must be in (0, 0.030] m")
        if not self.patch_distance_m < self.retain_hysteresis_distance_m <= 0.030:
            raise ValueError("retain hysteresis must be strictly above 0.020 and at most 0.030 m")
        if self.confirmation_substeps not in {2, 4, 8}:
            raise ValueError("confirmation_substeps must be 2, 4, or 8")
        if self.acquire_timeout_ms not in {20, 40, 80}:
            raise ValueError("acquire timeout must be 20, 40, or 80 ms")
        if self.regrasp_timeout_ms not in {20, 40, 80}:
            raise ValueError("regrasp timeout must be 20, 40, or 80 ms")
        if not 0 <= self.max_regrasp_attempts <= 2:
            raise ValueError("regrasp attempts must be bounded to 0, 1, or 2")
        if self.sim_dt_s <= 0.0:
            raise ValueError("sim_dt_s must be positive")
        if self.force_max_n <= 0.0:
            raise ValueError("force_max_n must be positive")

    @property
    def acquire_timeout_steps(self) -> int:
        return max(1, round(self.acquire_timeout_ms / 1000.0 / self.sim_dt_s))

    @property
    def regrasp_timeout_steps(self) -> int:
        return max(1, round(self.regrasp_timeout_ms / 1000.0 / self.sim_dt_s))


@dataclass(frozen=True)
class ContactObservation:
    """One real MuJoCo substep for the immutable assigned role."""

    source_frame: int
    source_timestamp_s: float
    sim_step: int
    substep: int
    role_active: bool
    assigned_patch: str
    assigned_robot_region: str
    physical_contact_present: bool
    correct_geom_pair: bool
    geom_pair: str
    patch_distance_m: float
    patch_membership: bool
    normal_cosine: float
    tangential_slip_m: float
    normal_gap_m: float
    penetration_m: float
    force_n: float
    force_impulse_ns: float
    joint_margin_fraction: float
    wrist_tracking_error_m: float
    fingertip_tracking_error_m: float
    object_tracking_position_m: float
    object_tracking_rotation_rad: float
    finite: bool
    joint_limit_valid: bool
    warning_count: int
    reference_qpos: tuple[float, ...] = ()
    actual_qpos: tuple[float, ...] = ()
    ctrl: tuple[float, ...] = ()
    regrasp_attempt: int = 0

    @property
    def correct_contact(self) -> bool:
        return self.physical_contact_present and self.correct_geom_pair

    @property
    def acquire_ok(self) -> bool:
        return (
            self.role_active
            and self.correct_contact
            and self.patch_membership
            and self.patch_distance_m <= 0.020
            and self.normal_cosine >= 0.50
            and self.penetration_m <= 0.003
            and self.finite
            and self.joint_limit_valid
            and self.warning_count == 0
        )

    @property
    def hard_safety_failure(self) -> FailureCode | None:
        if not self.finite or self.warning_count:
            return FailureCode.NUMERICAL_FAILURE
        if not self.joint_limit_valid:
            return FailureCode.JOINT_LIMIT_VIOLATION
        if self.penetration_m > 0.003:
            return FailureCode.PENETRATION_VIOLATION
        if not all(value == value and abs(value) != float("inf") for value in (self.force_n, self.patch_distance_m, self.normal_cosine)):
            return FailureCode.NUMERICAL_FAILURE
        return None


@dataclass(frozen=True)
class ContactTransition:
    source_frame: int
    sim_step: int
    substep: int
    previous_mode: ContactMode
    mode: ContactMode
    reason: str
    failure_code: FailureCode | None
    regrasp_attempt: int


class ContactModeMachine:
    """Fail-closed contact-mode state machine with explicit retain transfer.

    ``RETAIN_PENDING`` is intentionally separate from ``ACQUIRE``.  A real
    assigned contact at the first observation must never be treated as an
    acquisition problem: it receives a short confirmation interval while the
    physical controller transfers in bumplessly.  The runner owns controller
    blending; this class owns the observable transition contract.
    """

    def __init__(self, config: ContactModeConfig, *, role_active: bool = True) -> None:
        self.config = config
        self.mode = ContactMode.PRE_CONTACT
        self.role_active = role_active
        self.regrasp_attempts = 0
        self._mode_steps = 0
        self._confirmation_steps = 0
        self.contact_seen_in_acquire = False
        self.provisional_contact_counter = 0
        self.last_correct_contact_step: int | None = None
        self.failure_code: FailureCode | None = None
        self.transitions: list[ContactTransition] = []

    def _transition(self, observation: ContactObservation, mode: ContactMode, reason: str, failure: FailureCode | None = None) -> None:
        previous = self.mode
        self.mode = mode
        self._mode_steps = 0
        self._confirmation_steps = 0
        if mode not in {ContactMode.ACQUIRE, ContactMode.REGRASP}:
            self.contact_seen_in_acquire = False
            self.provisional_contact_counter = 0
        self.failure_code = failure
        self.transitions.append(
            ContactTransition(
                source_frame=observation.source_frame,
                sim_step=observation.sim_step,
                substep=observation.substep,
                previous_mode=previous,
                mode=mode,
                reason=reason,
                failure_code=failure,
                regrasp_attempt=self.regrasp_attempts,
            )
        )

    def _fail(self, observation: ContactObservation, code: FailureCode, reason: str) -> None:
        self._transition(observation, ContactMode.FAILED, reason, code)

    def observe(self, observation: ContactObservation) -> ContactMode:
        """Consume one substep and return the current mode."""
        if self.mode in {ContactMode.COMPLETE, ContactMode.FAILED}:
            return self.mode
        if observation.assigned_patch == "" or observation.assigned_robot_region == "":
            self._fail(observation, FailureCode.INVALID_MAPPING, "assigned contact mapping is empty")
            return self.mode
        safety = observation.hard_safety_failure
        if safety is None and observation.force_n > self.config.force_max_n:
            safety = FailureCode.FORCE_VIOLATION
        if safety is not None:
            self._fail(observation, safety, "hard safety gate failed")
            return self.mode

        self._mode_steps += 1
        if self.mode == ContactMode.PRE_CONTACT:
            if not observation.role_active:
                self._transition(observation, ContactMode.RELEASE, "role interval already ended")
            elif observation.acquire_ok:
                self._transition(observation, ContactMode.RETAIN_PENDING, "initial assigned physical contact is valid")
            else:
                # Distance alone is not a contact-mode success.  A mandatory
                # role without the required physical pair enters acquisition.
                self._transition(observation, ContactMode.ACQUIRE, "initial assigned physical contact is absent")
            return self.mode

        if self.mode == ContactMode.RETAIN_PENDING:
            if not observation.role_active:
                self._transition(observation, ContactMode.RELEASE, "role interval ended during retain confirmation")
            elif observation.acquire_ok:
                self._confirmation_steps += 1
                self.last_correct_contact_step = observation.sim_step
                if self._confirmation_steps >= self.config.confirmation_substeps:
                    self._transition(observation, ContactMode.RETAIN, "initial contact confirmation satisfied")
            elif self.config.allow_regrasp and self.regrasp_attempts < self.config.max_regrasp_attempts:
                self.regrasp_attempts += 1
                self._transition(observation, ContactMode.REGRASP, "initial contact lost during retain confirmation")
            else:
                self._fail(observation, FailureCode.ROLE_CONTINUITY_VIOLATION, "initial contact lost with no regrasp budget")
            return self.mode

        if self.mode == ContactMode.ACQUIRE:
            if not observation.role_active:
                self._transition(observation, ContactMode.RELEASE, "role interval ended during acquisition")
                return self.mode
            if observation.acquire_ok:
                self.contact_seen_in_acquire = True
                self.provisional_contact_counter += 1
                self.last_correct_contact_step = observation.sim_step
                self._confirmation_steps += 1
                if self._confirmation_steps >= self.config.confirmation_substeps:
                    self._transition(observation, ContactMode.RETAIN, "contact confirmation satisfied")
            else:
                self._confirmation_steps = 0
                if self.contact_seen_in_acquire and self.config.allow_regrasp and self.regrasp_attempts < self.config.max_regrasp_attempts:
                    # Do not burn the entire acquire timeout after a real,
                    # provisional contact disappears.  This transition occurs
                    # on the first failed observation (one control interval).
                    self.regrasp_attempts += 1
                    self._transition(observation, ContactMode.REGRASP, "provisional assigned contact lost before confirmation")
                elif self._mode_steps >= self.config.acquire_timeout_steps:
                    if self.config.allow_regrasp and self.regrasp_attempts < self.config.max_regrasp_attempts:
                        self.regrasp_attempts += 1
                        self._transition(observation, ContactMode.REGRASP, "acquire timeout while role remains active")
                    else:
                        self._fail(observation, FailureCode.ACQUIRE_TIMEOUT, "bounded acquire timeout")
            return self.mode

        if self.mode == ContactMode.REGRASP:
            if not observation.role_active:
                self._transition(observation, ContactMode.RELEASE, "role interval ended during regrasp")
                return self.mode
            if observation.acquire_ok:
                self._confirmation_steps += 1
                self.last_correct_contact_step = observation.sim_step
                if self._confirmation_steps >= self.config.confirmation_substeps:
                    self._transition(observation, ContactMode.RETAIN, "regrasp confirmation satisfied")
            else:
                self._confirmation_steps = 0
                if self._mode_steps >= self.config.regrasp_timeout_steps:
                    if self.regrasp_attempts < self.config.max_regrasp_attempts:
                        self.regrasp_attempts += 1
                        self._transition(observation, ContactMode.REGRASP, "bounded regrasp reset for next attempt")
                    else:
                        self._fail(observation, FailureCode.REGRASP_TIMEOUT, "bounded regrasp timeout")
            return self.mode

        if self.mode == ContactMode.RETAIN:
            if not observation.role_active:
                self._transition(observation, ContactMode.RELEASE, "frozen role interval ended")
            elif observation.acquire_ok:
                pass
            elif self.config.allow_regrasp and self.regrasp_attempts < self.config.max_regrasp_attempts:
                self.regrasp_attempts += 1
                reason = "retention loss within hysteresis" if observation.patch_distance_m <= self.config.retain_hysteresis_distance_m else "hard retention loss"
                self._transition(observation, ContactMode.REGRASP, reason)
            else:
                code = FailureCode.PATCH_DISTANCE_VIOLATION if observation.patch_distance_m > 0.020 else FailureCode.ROLE_CONTINUITY_VIOLATION
                self._fail(observation, code, "retention loss with no regrasp budget")
            return self.mode

        if self.mode == ContactMode.RELEASE:
            if observation.role_active:
                self._transition(observation, ContactMode.ACQUIRE, "role reactivated after a real release")
            return self.mode
        return self.mode

    def finish(self, observation: ContactObservation) -> ContactMode:
        """Close a frozen window without treating RELEASE as role success."""
        if self.mode not in {ContactMode.FAILED, ContactMode.COMPLETE}:
            if self.mode == ContactMode.RETAIN and observation.role_active and observation.acquire_ok:
                self._transition(observation, ContactMode.COMPLETE, "window terminal contact satisfies active role")
            elif self.mode == ContactMode.RELEASE and not observation.role_active:
                self._transition(observation, ContactMode.COMPLETE, "window terminal release complete")
            else:
                self._fail(observation, FailureCode.ROLE_CONTINUITY_VIOLATION, "window ended without terminal role state")
        return self.mode

    def transition_payload(self) -> list[dict[str, Any]]:
        return [
            {
                **asdict(row),
                "previous_mode": row.previous_mode.value,
                "mode": row.mode.value,
                "failure_code": row.failure_code.value if row.failure_code else None,
            }
            for row in self.transitions
        ]


def observation_payload(observation: ContactObservation) -> dict[str, Any]:
    """Serialize a complete observation for a JSONL-compatible timeline."""
    payload = asdict(observation)
    payload.update(
        {
            "correct_contact": observation.correct_contact,
            "acquire_ok": observation.acquire_ok,
            "mode_observation_hard_failure": observation.hard_safety_failure.value if observation.hard_safety_failure else None,
        }
    )
    return payload
