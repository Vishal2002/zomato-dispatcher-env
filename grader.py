# grader.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.environment import ZomatoDispatcherEnv


def _clamp(score: float) -> float:
    """Scores must be strictly between 0 and 1 (exclusive)."""
    return round(max(0.001, min(0.999, score)), 4)


class ZomatoDispatcherGrader:

    def grade(self, env: "ZomatoDispatcherEnv") -> float:
        task_id = env._task_config.task_id
        if task_id == "single_zone":
            return self._grade_single_zone(env)
        elif task_id == "multi_zone":
            return self._grade_multi_zone(env)
        elif task_id == "peak_hour":
            return self._grade_peak_hour(env)
        raise ValueError(f"Unknown task_id: {task_id}")

    def _grade_single_zone(self, env: "ZomatoDispatcherEnv") -> float:
        orders = env._orders
        if not orders:
            return 0.001
        total   = len(orders)
        on_time = sum(1 for o in orders if o.delivered and not o.cancelled)
        return _clamp(on_time / total)

    def _grade_multi_zone(self, env: "ZomatoDispatcherEnv") -> float:
        orders = env._orders
        if not orders:
            return 0.001
        total             = len(orders)
        delivered         = sum(1 for o in orders if o.delivered)
        cancelled         = sum(1 for o in orders if o.cancelled)
        delivery_rate     = delivered / total
        cancellation_ratio= cancelled / total
        score             = delivery_rate - (0.15 * cancellation_ratio)
        return _clamp(score)

    def _grade_peak_hour(self, env: "ZomatoDispatcherEnv") -> float:
        orders = env._orders
        if not orders:
            return 0.001
        total            = len(orders)
        delivered        = sum(1 for o in orders if o.delivered)
        on_time          = sum(1 for o in orders if o.delivered and not o.cancelled)
        cancelled        = sum(1 for o in orders if o.cancelled)
        delivery_rate    = delivered / total
        on_time_ratio    = on_time / delivered if delivered > 0 else 0.0
        cancel_avoidance = 1.0 - (cancelled / total)
        score = (
            0.50 * delivery_rate
            + 0.30 * on_time_ratio
            + 0.20 * cancel_avoidance
        )
        return _clamp(score)