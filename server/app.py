"""
server/app.py — FastAPI application entry point for ZomatoDispatcherEnv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from openenv.core.env_server import create_app

from server.environment import ZomatoDispatcherEnv
from models import DispatchAction, DispatchObservation
from tasks import ALL_TASKS

app: FastAPI = create_app(
    env=ZomatoDispatcherEnv,
    action_cls=DispatchAction,
    observation_cls=DispatchObservation,
    env_name="zomato-dispatcher-env",
)


@app.get("/tasks", tags=["environment"])
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "difficulty": t.difficulty,
                "description": t.description,
                "num_orders": t.num_orders,
                "num_riders": t.num_riders,
                "zones": t.zones,
                "max_steps": t.max_steps,
            }
            for t in ALL_TASKS
        ]
    }


@app.get("/baseline", tags=["evaluation"])
def baseline_info():
    return {
        "model": "gpt-4o-mini",
        "scores": {"single_zone": 0.999, "multi_zone": 0.999, "peak_hour": 0.405},
        "average": 0.80,
        "note": "Beating the baseline.",
    }


def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False,
    )


if __name__ == "__main__":
    main()