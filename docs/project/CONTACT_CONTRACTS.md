# Stage C contact contracts

Stage C has two separately versioned contact contracts.

| Contract | Meaning | Status authority |
| --- | --- | --- |
| V1 `EXACT_SOURCE_FINGER_CONTACT` | Same hand, same finger, same source-surface contact | Immutable C-R4 infeasibility report |
| V2 `TASK_EQUIVALENT_CONTACT` | Same hand, source patch and functional role; bounded robot-region reassignment | C-X validation report |

V1 is permanently `BLOCKED_BY_INFEASIBLE_EMBODIMENT_CONTACT`.  V2 does not
reproduce exact human contact and must never overwrite V1 metrics, threshold,
source contact denominator, pilot list, raw GRAB, or Stage B trajectory.

V2 reports both `exact_contact_recall_v1` and
`task_equivalent_contact_recall_v2`; they are not interchangeable.  Its
assignment is constrained to the source hand, a mesh-adjacent normal-consistent
object patch, functional role, finite robot region capacity, and bounded
temporal switching.  Thumb opposition remains thumb-only; palms are support or
stabilization-only.
