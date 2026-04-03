# tasks.py
#
# WHY THIS FILE EXISTS:
# The hackathon requires minimum 3 tasks: easy → medium → hard
# Each task must have a deterministic grader that scores 0.0–1.0
#
# JUDGING CHECK: Phase 2 runs all 3 tasks with Nemotron-3 Super and checks
# that easy_score > hard_score (difficulty scaling must be real)

from dataclasses import dataclass, field
from typing import List


@dataclass
class TaskConfig:
    task_id: str
    description: str
    difficulty: str          # easy | medium | hard
    num_riders: int
    num_orders: int
    max_steps: int
    zones: int               # number of city zones
    has_busy_riders: bool    # some riders already on delivery at start
    sla_pressure: float      # multiplier: lower = tighter deadlines (0.5–2.0)
    surge_at_step: int       # step when extra orders appear (0 = no surge)
    seed: int                # FIXED seed for reproducibility (Phase 2 check)


# ─────────────────────────────────────────────────────────────
# TASK 1 — easy
# Single zone, 3 idle riders, 4 orders, generous deadlines
# Agent just needs to learn: assign closest rider to each order
# Expected score for a good agent: 0.80–1.0
# Expected score for random agent: 0.40–0.60
# ─────────────────────────────────────────────────────────────
TASK_SINGLE_ZONE = TaskConfig(
    task_id="single_zone",
    description=(
        "Dispatch 4 orders across 3 idle riders in a single city zone. "
        "All riders are available. Deadlines are generous. "
        "Score = on-time delivery rate."
    ),
    difficulty="easy",
    num_riders=3,
    num_orders=4,
    max_steps=15,
    zones=1,
    has_busy_riders=False,
    sla_pressure=2.0,       # very generous deadlines
    surge_at_step=0,        # no surge
    seed=42,
)

# ─────────────────────────────────────────────────────────────
# TASK 2 — medium
# 5 zones, 4 riders (1 busy), 10 orders, normal deadlines
# Agent must learn: zone proximity matters + avoid busy riders
# Expected score for a good agent: 0.60–0.80
# Expected score for random agent: 0.25–0.45
# ─────────────────────────────────────────────────────────────
TASK_MULTI_ZONE = TaskConfig(
    task_id="multi_zone",
    description=(
        "Dispatch 10 orders across 5 city zones with 4 riders (1 already busy). "
        "Deadlines are moderate. Zone proximity and rider availability both matter. "
        "Score = weighted delivery rate minus cancellation penalty."
    ),
    difficulty="medium",
    num_riders=4,
    num_orders=10,
    max_steps=25,
    zones=5,
    has_busy_riders=True,   # 1 rider starts mid-delivery
    sla_pressure=1.0,       # normal deadlines
    surge_at_step=0,
    seed=42,
)

# ─────────────────────────────────────────────────────────────
# TASK 3 — hard
# 5 zones, 5 riders (2 busy), 20 orders, tight deadlines
# At step 10 a surge adds 5 more urgent orders
# Agent must learn: urgency triage + preemptive positioning
# Expected score for a good agent: 0.45–0.65
# Expected score for random agent: 0.10–0.25
# ─────────────────────────────────────────────────────────────
TASK_PEAK_HOUR = TaskConfig(
    task_id="peak_hour",
    description=(
        "Manage 20 orders across 5 zones with 5 riders (2 already busy) "
        "under tight SLA deadlines. At step 10 a demand surge adds 5 urgent orders. "
        "Score = multi-objective: delivery rate + on-time ratio + cancellation penalty."
    ),
    difficulty="hard",
    num_riders=5,
    num_orders=20,
    max_steps=35,
    zones=5,
    has_busy_riders=True,   # 2 riders start mid-delivery
    sla_pressure=0.6,       # tight deadlines
    surge_at_step=10,       # surge event
    seed=42,
)

# ─────────────────────────────────────────────────────────────
# Registry — used by /tasks endpoint and environment
# ─────────────────────────────────────────────────────────────
ALL_TASKS: List[TaskConfig] = [
    TASK_SINGLE_ZONE,
    TASK_MULTI_ZONE,
    TASK_PEAK_HOUR,
]

TASK_MAP = {t.task_id: t for t in ALL_TASKS}
