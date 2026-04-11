"""
server/environment.py — Core ZomatoDispatcherEnv logic.

Implements the OpenEnv Environment interface (reset / step / state).
All game logic lives here: order/rider spawning, assignment,
world advancement, SLA tracking, surge events, and reward shaping.
"""
import random
import uuid
from typing import Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

from models import DispatchAction, DispatchObservation, OrderInfo, RiderInfo
from tasks import TASK_MAP, TaskConfig

# Steps to travel one zone, plus deliver
TRAVEL_TIME_PER_ZONE = 1
DELIVERY_TIME = 1

SKIP_IDS = {"SKIP", "skip", ""}


class ZomatoDispatcherEnv(Environment):
    """
    Food delivery dispatch RL environment.

    The agent acts as a fleet dispatcher: at each step it assigns
    one pending order to one available rider. The episode ends when
    all orders are resolved (delivered or cancelled) or max_steps
    is reached.

    Reward shaping:
      +0.3  — zone-matched assignment (rider already in restaurant zone)
      +1.0  — successful on-time delivery
      +0.4  — delivery after SLA already breached (partial credit)
      +2.0  — bonus for delivering every single order in the episode
      -2.0  — order cancelled (SLA deadline reached 0)
      -0.3  — invalid action (bad IDs, already-done order, busy rider)
    """

    def __init__(self, task_id: str = "multi_zone", seed: int = 42):
        self._task_id = task_id
        self._seed = seed
        self._rng = random.Random(seed)
        self._task_config: Optional[TaskConfig] = None
        self._step_count: int = 0
        self._episode_id: str = ""
        self._orders: List[OrderInfo] = []
        self._riders: List[RiderInfo] = []
        self._episode_reward: float = 0.0
        self._done: bool = False
        # Maps order_id → steps remaining until delivery completes
        self._in_transit: Dict[str, int] = {}

    # ── OpenEnv interface ──────────────────────────────────────────

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="zomato-dispatcher-env",
            description=(
                "Food delivery order dispatch RL environment inspired by "
                "Zomato/Swiggy. Assign orders to riders across city zones "
                "to maximise on-time delivery while avoiding SLA breaches."
            ),
            version="1.0.0",
        )

    def close(self) -> None:
        """Clean up resources (nothing persistent to release)."""
        pass

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> DispatchObservation:
        """Start a new episode. Spawns orders and riders deterministically."""
        if task_id:
            self._task_id = task_id
        if self._task_id not in TASK_MAP:
            self._task_id = "multi_zone"

        cfg = TASK_MAP[self._task_id]
        self._task_config = cfg
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._episode_reward = 0.0
        self._done = False
        self._in_transit = {}
        self._rng = random.Random(seed if seed is not None else cfg.seed)

        self._riders = self._spawn_riders(cfg)
        self._orders = self._spawn_orders(cfg)

        return self._build_observation()

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> DispatchObservation:
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)

    def step(
        self,
        action: DispatchAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> DispatchObservation:
        """
        Execute one dispatch decision.

        The agent assigns one order to one rider (or SKIPs).
        After assignment the world advances one time-step:
        in-transit orders get closer to delivery, SLA deadlines tick
        down, and the surge event fires if configured.
        """
        if self._done:
            return self._build_observation()

        reward = 0.0
        info: dict = {}

        # ── Validate and execute assignment ───────────────────────
        is_skip = (
            action.order_id in SKIP_IDS or action.rider_id in SKIP_IDS
        )

        if is_skip:
            info["skipped"] = True
        else:
            order = self._find_order(action.order_id)
            rider = self._find_rider(action.rider_id)

            if order is None:
                reward -= 0.3
                info["error"] = f"unknown order_id: {action.order_id}"
            elif rider is None:
                reward -= 0.3
                info["error"] = f"unknown rider_id: {action.rider_id}"
            elif order.delivered or order.cancelled:
                reward -= 0.3
                info["error"] = "order already resolved"
            elif order.assigned_rider is not None:
                reward -= 0.3
                info["error"] = "order already assigned"
            elif not rider.is_available:
                reward -= 0.3
                info["error"] = "rider not available"
            else:
                reward += self._assign_order(order, rider)
                info["assigned"] = f"{action.order_id} → {action.rider_id}"

        # ── Advance the world one tick ─────────────────────────────
        reward += self._advance_world()

        self._step_count += 1
        self._episode_reward += reward

        # Check done FIRST
        all_resolved = all(o.delivered or o.cancelled for o in self._orders)
        self._done = all_resolved or (self._step_count >= self._task_config.max_steps)

        if self._done and all(o.delivered for o in self._orders):
            self._episode_reward += 2.0

        # Build observation AFTER setting self._done
        return self._build_observation()

    async def step_async(
        self,
        action: DispatchAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> DispatchObservation:
        return self.step(action, timeout_s=timeout_s, **kwargs)

    # ── Spawning helpers ───────────────────────────────────────────

    def _spawn_riders(self, cfg: TaskConfig) -> List[RiderInfo]:
        riders = []
        for i in range(cfg.num_riders):
            zone = self._rng.randint(0, cfg.zones - 1)
            busy = cfg.has_busy_riders and i < 2
            riders.append(
                RiderInfo(
                    rider_id=f"RIDER-{i + 1}",
                    current_zone=zone,
                    is_available=not busy,
                    current_orders=[],
                    eta_to_free=self._rng.randint(2, 5) if busy else 0,
                    rating=round(self._rng.uniform(4.0, 5.0), 1),
                    deliveries_today=0,
                )
            )
        return riders

    def _spawn_orders(self, cfg: TaskConfig) -> List[OrderInfo]:
        orders = []
        for i in range(cfg.num_orders):
            deadline = max(4, int(self._rng.randint(8, 15) * cfg.sla_pressure))
            orders.append(
                OrderInfo(
                    order_id=f"ORD-{i + 1:03d}",
                    restaurant_zone=self._rng.randint(0, cfg.zones - 1),
                    customer_zone=self._rng.randint(0, cfg.zones - 1),
                    prep_time_remaining=self._rng.randint(1, 4),
                    sla_deadline=deadline,
                )
            )
        return orders

    def _spawn_surge(self, n: int, cfg: TaskConfig) -> List[OrderInfo]:
        """Create urgent surge orders (tight deadlines, minimal prep)."""
        surge_start = len(self._orders)
        return [
            OrderInfo(
                order_id=f"SURGE-{i + 1:03d}",
                restaurant_zone=self._rng.randint(0, cfg.zones - 1),
                customer_zone=self._rng.randint(0, cfg.zones - 1),
                prep_time_remaining=1,
                sla_deadline=8,
            )
            for i in range(n)
        ]

    # ── World simulation ───────────────────────────────────────────

    def _assign_order(self, order: OrderInfo, rider: RiderInfo) -> float:
        """Mark assignment and compute travel ETA. Returns step reward."""
        order.assigned_rider = rider.rider_id
        rider.is_available = False
        rider.current_orders.append(order.order_id)

        dist = abs(rider.current_zone - order.restaurant_zone)
        eta = dist * TRAVEL_TIME_PER_ZONE + DELIVERY_TIME + order.prep_time_remaining
        self._in_transit[order.order_id] = eta

        # Small bonus for zone match (no travel needed to restaurant)
        return 0.3 if rider.current_zone == order.restaurant_zone else 0.0

    def _advance_world(self) -> float:
        """
        Tick the world forward one step. Returns reward delta.

        Order of operations:
          1. Identify completed deliveries (ETA <= 0 after decrement)
          2. Mark orders delivered, free riders
          3. Decrement remaining ETAs
          4. Tick SLA deadlines; cancel breached orders
          5. Free busy-at-start riders (eta_to_free countdown)
          6. Inject surge orders at the configured step
        """
        reward = 0.0

        # Decrement all ETAs first, collect completed ones
        completed = []
        updated_transit: Dict[str, int] = {}
        for oid, remaining in self._in_transit.items():
            new_remaining = remaining - 1
            if new_remaining <= 0:
                completed.append(oid)
            else:
                updated_transit[oid] = new_remaining
        self._in_transit = updated_transit

        # Resolve completed deliveries
        for oid in completed:
            order = self._find_order(oid)
            if order and not order.cancelled:
                order.delivered = True
                # on-time = SLA deadline still positive
                reward += 1.0 if order.sla_deadline > 0 else 0.4
                rider = self._find_rider(order.assigned_rider)
                if rider:
                    rider.is_available = True
                    rider.current_orders = [
                        o for o in rider.current_orders if o != oid
                    ]
                    rider.deliveries_today += 1
                    rider.current_zone = order.customer_zone

        # Tick SLA deadlines
        for order in self._orders:
            if not order.delivered and not order.cancelled:
                order.sla_deadline -= 1
                order.prep_time_remaining = max(0, order.prep_time_remaining - 1)
                if order.sla_deadline <= 0:
                    order.cancelled = True
                    reward -= 2.0

        # Free riders who were busy at episode start
        for rider in self._riders:
            if not rider.is_available and rider.eta_to_free > 0:
                rider.eta_to_free -= 1
                if rider.eta_to_free == 0:
                    rider.is_available = True

        # Surge event
        cfg = self._task_config
        if cfg.surge_at_step > 0 and self._step_count == cfg.surge_at_step:
            self._orders.extend(self._spawn_surge(5, cfg))

        return reward

    # ── Lookup helpers ─────────────────────────────────────────────

    def _find_order(self, oid: str) -> Optional[OrderInfo]:
        return next((o for o in self._orders if o.order_id == oid), None)

    def _find_rider(self, rid: str) -> Optional[RiderInfo]:
        return next((r for r in self._riders if r.rider_id == rid), None)

    # ── Observation builder ────────────────────────────────────────

    def _build_observation(self, info: Optional[dict] = None) -> DispatchObservation:
        pending = sum(
            1 for o in self._orders
            if not o.assigned_rider and not o.cancelled and not o.delivered
        )
        active = sum(
            1 for o in self._orders
            if o.assigned_rider and not o.delivered and not o.cancelled
        )
        delivered = sum(1 for o in self._orders if o.delivered)
        cancelled = sum(1 for o in self._orders if o.cancelled)

        return DispatchObservation(
            step=self._step_count,
            task_id=self._task_id,
            max_steps=self._task_config.max_steps,
            done=self._done,
            reward=round(self._episode_reward, 4),
            orders=list(self._orders),
            riders=list(self._riders),
            pending_orders=pending,
            active_orders=active,
            delivered_count=delivered,
            cancelled_count=cancelled,
            episode_reward_so_far=round(self._episode_reward, 4),
        )