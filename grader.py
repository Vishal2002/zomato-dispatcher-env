# grader.py
#
# WHY THIS FILE EXISTS:
# Phase 1 check: bot enumerates tasks, runs each grader, verifies scores in 0.0–1.0
# Phase 2 check: easy_score must be visibly higher than hard_score
#
# CRITICAL RULES (from judging criteria):
#   1. NEVER return the same score always — instant disqualification
#   2. Scores MUST be in [0.0, 1.0] — no exceptions
#   3. Must be deterministic — same episode state = same score every time
#   4. Different logic per task — proves difficulty scaling is real

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.environment import ZomatoDispatcherEnv


class ZomatoDispatcherGrader:
    """
    Grades a completed (or in-progress) episode.
    Call grade(env) after reset+steps to get a float in [0.0, 1.0].
    """

    def grade(self, env: "ZomatoDispatcherEnv") -> float:
        task_id = env._task_config.task_id
        if task_id == "single_zone":
            return self._grade_single_zone(env)
        elif task_id == "multi_zone":
            return self._grade_multi_zone(env)
        elif task_id == "peak_hour":
            return self._grade_peak_hour(env)
        raise ValueError(f"Unknown task_id: {task_id}")

    # ──────────────────────────────────────────────────────
    # TASK 1 — easy: pure delivery rate
    # Score = delivered_on_time / total_orders
    # Simple, binary per order — easy to understand
    # ──────────────────────────────────────────────────────
    def _grade_single_zone(self, env: "ZomatoDispatcherEnv") -> float:
        orders = env._orders
        if not orders:
            return 0.0
        total = len(orders)
        on_time = sum(
            1 for o in orders
            if o.delivered and not o.cancelled
        )
        score = on_time / total
        return round(max(0.0, min(1.0, score)), 4)

    # ──────────────────────────────────────────────────────
    # TASK 2 — medium: delivery rate minus cancellation penalty
    # Score = (delivered/total) - 0.15 * (cancelled/total)
    # Partial credit for late deliveries too
    # ──────────────────────────────────────────────────────
    def _grade_multi_zone(self, env: "ZomatoDispatcherEnv") -> float:
        orders = env._orders
        if not orders:
            return 0.0
        total     = len(orders)
        delivered = sum(1 for o in orders if o.delivered)
        cancelled = sum(1 for o in orders if o.cancelled)

        delivery_rate      = delivered / total
        cancellation_ratio = cancelled / total

        score = delivery_rate - (0.15 * cancellation_ratio)
        return round(max(0.0, min(1.0, score)), 4)

    # ──────────────────────────────────────────────────────
    # TASK 3 — hard: multi-objective
    # 50% delivery rate
    # 30% on-time ratio (on_time / delivered)
    # 20% cancellation avoidance (1 - cancelled/total)
    # ──────────────────────────────────────────────────────
    def _grade_peak_hour(self, env: "ZomatoDispatcherEnv") -> float:
        orders = env._orders
        if not orders:
            return 0.0
        total     = len(orders)
        delivered = sum(1 for o in orders if o.delivered)
        on_time   = sum(1 for o in orders if o.delivered and not o.cancelled)
        cancelled = sum(1 for o in orders if o.cancelled)

        delivery_rate   = delivered / total
        on_time_ratio   = on_time / delivered if delivered > 0 else 0.0
        cancel_avoidance = 1.0 - (cancelled / total)

        score = (
            0.50 * delivery_rate
            + 0.30 * on_time_ratio
            + 0.20 * cancel_avoidance
        )
        return round(max(0.0, min(1.0, score)), 4)
