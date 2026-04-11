from __future__ import annotations
from typing import Optional

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import DispatchAction, DispatchObservation

class ZomatoDispatcherClient(EnvClient[DispatchAction, DispatchObservation, State]):

    def _step_payload(self, action: DispatchAction) -> dict:
        # Flat dict — WebSocket server deserializes this into DispatchAction directly
        return {
            "order_id": action.order_id,
            "rider_id": action.rider_id,
        }

    def _parse_result(self, payload: dict) -> StepResult[DispatchObservation]:
        # payload = response["data"] = {"observation": {...}, "reward": x, "done": bool}
        obs_data = payload.get("observation", payload)
        obs = DispatchObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(**{k: v for k, v in payload.items() 
                       if k in State.model_fields})

    def reset(
        self,
        task_id: str = "multi_zone",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> StepResult[DispatchObservation]:
        return super().reset(task_id=task_id, seed=seed, episode_id=episode_id, **kwargs)