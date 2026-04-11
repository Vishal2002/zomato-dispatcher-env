"""
models.py — Pydantic data contracts for ZomatoDispatcherEnv.

These models define the typed interface between the environment server
and RL agents/clients. All fields are serialisable to JSON.
"""
from typing import List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class OrderInfo(Observation):
    """Represents a single food delivery order."""
    order_id: str = Field(..., description="Unique order identifier e.g. 'ORD-001'")
    restaurant_zone: int = Field(..., description="Zone where the restaurant is located")
    customer_zone: int = Field(..., description="Zone where the customer is located")
    prep_time_remaining: int = Field(..., description="Steps until food is ready for pickup")
    sla_deadline: int = Field(..., description="Steps remaining before SLA breach (cancellation)")
    assigned_rider: Optional[str] = Field(None, description="Rider ID if assigned, else None")
    delivered: bool = Field(False, description="True if order has been delivered")
    cancelled: bool = Field(False, description="True if order was cancelled due to SLA breach")


class RiderInfo(Observation):
    """Represents a delivery rider."""
    rider_id: str = Field(..., description="Unique rider identifier e.g. 'RIDER-1'")
    current_zone: int = Field(..., description="Zone the rider is currently in")
    is_available: bool = Field(..., description="True if rider can accept a new assignment")
    current_orders: List[str] = Field(default_factory=list, description="Order IDs currently being handled")
    eta_to_free: int = Field(0, description="Steps until rider becomes available (0 if already free)")
    rating: float = Field(..., description="Rider rating (4.0–5.0)")
    deliveries_today: int = Field(0, description="Number of successful deliveries this episode")


class DispatchAction(Action):
    """
    Dispatcher's decision: assign one order to one rider.

    Use order_id='SKIP' / rider_id='SKIP' to explicitly pass a step
    (e.g. when no orders are pending or no riders are available).
    """
    order_id: str = Field(..., description="Order ID to assign e.g. 'ORD-001', or 'SKIP'")
    rider_id: str = Field(..., description="Rider ID to assign e.g. 'RIDER-1', or 'SKIP'")


class DispatchObservation(Observation):
    """
    Full environment state returned after reset() and step().

    Contains all orders and riders so the agent has complete
    visibility to make informed dispatch decisions.
    """
    step: int = Field(..., description="Current step number (0-indexed)")
    task_id: str = Field(..., description="Active task: single_zone | multi_zone | peak_hour")
    max_steps: int = Field(..., description="Maximum steps before episode ends")
    done: bool = Field(False, description="True if episode has ended")
    reward: float = Field(0.0, description="Reward from the last step")

    # Full state
    orders: List[OrderInfo] = Field(default_factory=list, description="All orders this episode")
    riders: List[RiderInfo] = Field(default_factory=list, description="All riders this episode")

    # Aggregated counters (convenient for agents / dashboards)
    pending_orders: int = Field(0, description="Orders not yet assigned")
    active_orders: int = Field(0, description="Orders currently en route")
    delivered_count: int = Field(0, description="Successfully delivered orders so far")
    cancelled_count: int = Field(0, description="Orders cancelled due to SLA breach")
    episode_reward_so_far: float = Field(0.0, description="Cumulative reward for this episode")