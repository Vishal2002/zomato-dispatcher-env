import random, uuid
from typing import List, Optional
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State
from models import DispatchAction, DispatchObservation, OrderInfo, RiderInfo
from tasks import TASK_MAP, TaskConfig

TRAVEL_TIME = 1
DELIVERY_TIME = 1

class ZomatoDispatcherEnv(Environment):
    def __init__(self, task_id="multi_zone", seed=42):
        self._task_id = task_id
        self._seed = seed
        self._rng = random.Random(seed)
        self._task_config = None
        self._step_count = 0
        self._episode_id = ""
        self._orders: List[OrderInfo] = []
        self._riders: List[RiderInfo] = []
        self._episode_reward = 0.0
        self._done = False
        self._in_transit = {}

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(name="zomato-dispatcher-env", description="Food delivery dispatch RL environment", version="1.0.0")

    def close(self): pass

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self._step_count)

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> DispatchObservation:
        if task_id: self._task_id = task_id
        if self._task_id not in TASK_MAP: self._task_id = "multi_zone"
        cfg = TASK_MAP[self._task_id]
        self._task_config = cfg
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._episode_reward = 0.0
        self._done = False
        self._in_transit = {}
        self._rng = random.Random(seed or cfg.seed)
        self._riders = self._spawn_riders(cfg)
        self._orders = self._spawn_orders(cfg)
        return self._build_observation()

    async def reset_async(self, seed=None, episode_id=None, **kwargs):
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)

    def step(self, action: DispatchAction, timeout_s=None, **kwargs) -> DispatchObservation:
        if self._done: return self._build_observation()
        reward = 0.0
        info = {}
        order = self._find_order(action.order_id)
        rider = self._find_rider(action.rider_id)
        if order is None: reward -= 0.3; info["error"] = f"unknown order_id: {action.order_id}"
        elif rider is None: reward -= 0.3; info["error"] = f"unknown rider_id: {action.rider_id}"
        elif order.delivered or order.cancelled: reward -= 0.3; info["error"] = "order already done"
        elif order.assigned_rider is not None: reward -= 0.3; info["error"] = "order already assigned"
        elif not rider.is_available: reward -= 0.3; info["error"] = "rider unavailable"
        else: reward += self._assign_order(order, rider); info["assigned"] = f"{action.order_id} → {action.rider_id}"
        reward += self._advance_world()
        if "assigned" not in info: reward -= 0.5
        self._step_count += 1
        self._episode_reward += reward
        all_resolved = all(o.delivered or o.cancelled for o in self._orders)
        self._done = all_resolved or self._step_count >= self._task_config.max_steps
        if self._done and all(o.delivered for o in self._orders): self._episode_reward += 2.0
        return self._build_observation(info=info)

    async def step_async(self, action, timeout_s=None, **kwargs):
        return self.step(action, timeout_s=timeout_s, **kwargs)

    def _spawn_riders(self, cfg):
        riders = []
        for i in range(cfg.num_riders):
            zone = self._rng.randint(0, cfg.zones - 1)
            busy = cfg.has_busy_riders and i < 2
            riders.append(RiderInfo(rider_id=f"RIDER-{i+1}", current_zone=zone, is_available=not busy, current_orders=[], eta_to_free=self._rng.randint(2,5) if busy else 0, rating=round(self._rng.uniform(4.0,5.0),1), deliveries_today=0, done=False, reward=0.0))
        return riders

    def _spawn_orders(self, cfg):
        orders = []
        for i in range(cfg.num_orders):
            deadline = max(4, int(self._rng.randint(8,15) * cfg.sla_pressure))
            orders.append(OrderInfo(order_id=f"ORD-{i+1:03d}", restaurant_zone=self._rng.randint(0,cfg.zones-1), customer_zone=self._rng.randint(0,cfg.zones-1), prep_time_remaining=self._rng.randint(1,4), sla_deadline=deadline, done=False, reward=0.0))
        return orders

    def _assign_order(self, order, rider):
        order.assigned_rider = rider.rider_id
        rider.is_available = False
        rider.current_orders.append(order.order_id)
        dist = abs(rider.current_zone - order.restaurant_zone)
        self._in_transit[order.order_id] = dist * TRAVEL_TIME + DELIVERY_TIME + order.prep_time_remaining
        return 0.3 if rider.current_zone == order.restaurant_zone else 0.0

    def _advance_world(self):
        reward = 0.0
        cfg = self._task_config
        completed = [oid for oid, t in self._in_transit.items() if t - 1 <= 0]
        for oid in completed:
            del self._in_transit[oid]
            order = self._find_order(oid)
            if order and not order.cancelled:
                order.delivered = True
                reward += 1.0 if order.sla_deadline > 0 else 0.4
                rider = self._find_rider(order.assigned_rider)
                if rider:
                    rider.is_available = True
                    rider.current_orders = [o for o in rider.current_orders if o != oid]
                    rider.deliveries_today += 1
                    rider.current_zone = order.customer_zone
        for oid in self._in_transit: self._in_transit[oid] -= 1
        for order in self._orders:
            if not order.delivered and not order.cancelled:
                order.sla_deadline -= 1
                order.prep_time_remaining = max(0, order.prep_time_remaining - 1)
                if order.sla_deadline <= 0: order.cancelled = True; reward -= 2.0
        for rider in self._riders:
            if not rider.is_available and rider.eta_to_free > 0:
                rider.eta_to_free -= 1
                if rider.eta_to_free == 0: rider.is_available = True
        if cfg.surge_at_step > 0 and self._step_count == cfg.surge_at_step:
            self._orders.extend(self._spawn_surge(5, cfg))
        return reward

    def _spawn_surge(self, n, cfg):
        return [OrderInfo(order_id=f"SURGE-{i+1:03d}", restaurant_zone=self._rng.randint(0,cfg.zones-1), customer_zone=self._rng.randint(0,cfg.zones-1), prep_time_remaining=1, sla_deadline=8, done=False, reward=0.0) for i in range(n)]

    def _find_order(self, oid): return next((o for o in self._orders if o.order_id == oid), None)
    def _find_rider(self, rid): return next((r for r in self._riders if r.rider_id == rid), None)

    def _build_observation(self, info=None):
        pending   = sum(1 for o in self._orders if not o.assigned_rider and not o.cancelled and not o.delivered)
        active    = sum(1 for o in self._orders if o.assigned_rider and not o.delivered and not o.cancelled)
        delivered = sum(1 for o in self._orders if o.delivered)
        cancelled = sum(1 for o in self._orders if o.cancelled)
        return DispatchObservation(step=self._step_count, task_id=self._task_id, max_steps=self._task_config.max_steps, orders=list(self._orders), riders=list(self._riders), pending_orders=pending, active_orders=active, delivered_count=delivered, cancelled_count=cancelled, episode_reward_so_far=round(self._episode_reward,4), done=self._done, reward=round(self._episode_reward,4), metadata=info or {})
    