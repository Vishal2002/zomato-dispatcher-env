from typing import Optional
from fastapi import HTTPException
from pydantic import BaseModel
from openenv.core.env_server import create_app
from models import DispatchAction, DispatchObservation
from server.environment import ZomatoDispatcherEnv
from grader import ZomatoDispatcherGrader
from tasks import ALL_TASKS, TASK_MAP

app = create_app(
    env=ZomatoDispatcherEnv,
    action_cls=DispatchAction,
    observation_cls=DispatchObservation,
    env_name="zomato-dispatcher-env",
)

_grader = ZomatoDispatcherGrader()
_shared_env = ZomatoDispatcherEnv()
_shared_env.reset()

class ResetRequest(BaseModel):
    task_id: Optional[str] = "multi_zone"

class GraderResponse(BaseModel):
    task_id: str; score: float; delivered: int; cancelled: int; total: int

class TaskSchema(BaseModel):
    task_id: str; description: str; difficulty: str; action_schema: dict

class BaselineResponse(BaseModel):
    scores: dict; average: float

@app.post("/reset-task", response_model=DispatchObservation)
def reset_task(req: ResetRequest):
    if req.task_id and req.task_id not in TASK_MAP:
        raise HTTPException(400, detail=f"Unknown task_id. Valid: {list(TASK_MAP.keys())}")
    return _shared_env.reset(task_id=req.task_id)

@app.post("/step-task", response_model=DispatchObservation)
def step_task(action: DispatchAction):
    return _shared_env.step(action)

@app.get("/tasks", response_model=list[TaskSchema])
def list_tasks():
    return [TaskSchema(task_id=t.task_id, description=t.description, difficulty=t.difficulty, action_schema={"order_id":"str — e.g. 'ORD-001'","rider_id":"str — e.g. 'RIDER-1'"}) for t in ALL_TASKS]

@app.get("/grader", response_model=GraderResponse)
def grade():
    score = _grader.grade(_shared_env)
    return GraderResponse(task_id=_shared_env._task_id, score=score, delivered=sum(1 for o in _shared_env._orders if o.delivered), cancelled=sum(1 for o in _shared_env._orders if o.cancelled), total=len(_shared_env._orders))

@app.get("/baseline", response_model=BaselineResponse)
def baseline():
    scores = {}
    for task in ALL_TASKS:
        env = ZomatoDispatcherEnv(task_id=task.task_id, seed=42)
        obs = env.reset(task_id=task.task_id)
        steps = 0
        while not obs.done and steps < 60:
            pending = [o for o in obs.orders if not o.assigned_rider and not o.delivered and not o.cancelled]
            idle = [r for r in obs.riders if r.is_available]
            if pending and idle:
                urgent = min(pending, key=lambda o: o.sla_deadline)
                best = min(idle, key=lambda r: abs(r.current_zone - urgent.restaurant_zone))
                obs = env.step(DispatchAction(order_id=urgent.order_id, rider_id=best.rider_id))
            else:
                obs = env.step(DispatchAction(order_id=obs.orders[0].order_id, rider_id=obs.riders[0].rider_id))
            steps += 1
        scores[task.task_id] = round(_grader.grade(env), 4)
    avg = round(sum(scores.values()) / len(scores), 4)
    return BaselineResponse(scores=scores, average=avg)

@app.get("/health")
def health():
    return {"status": "ok", "env": "zomato-dispatcher-env", "version": "1.0.0"}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1)

if __name__ == "__main__":
    main()
