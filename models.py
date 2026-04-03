from typing import List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class OrderInfo(Observation):
    order_id: str
    restaurant_zone: int
    customer_zone: int
    prep_time_remaining: int
    sla_deadline: int
    assigned_rider: Optional[str] = None
    delivered: bool = False
    cancelled: bool = False

class RiderInfo(Observation):
    rider_id: str
    current_zone: int
    is_available: bool
    current_orders: List[str] = Field(default_factory=list)
    eta_to_free: int = 0
    rating: float
    deliveries_today: int = 0

class DispatchAction(Action):
    order_id: str = Field(..., description="Order ID e.g. 'ORD-001'")
    rider_id: str = Field(..., description="Rider ID e.g. 'RIDER-1'")

class DispatchObservation(Observation):
    step: int
    task_id: str
    max_steps: int
    orders: List[OrderInfo]
    riders: List[RiderInfo]
    pending_orders: int
    active_orders: int
    delivered_count: int
    cancelled_count: int
    episode_reward_so_far: float
