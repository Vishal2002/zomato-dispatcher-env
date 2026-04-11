# tests/test_environment.py
from server.environment import ZomatoDispatcherEnv
from models import DispatchAction

def test_reset():
    env = ZomatoDispatcherEnv()
    obs = env.reset(task_id="single_zone")
    assert obs.pending_orders == 4
    assert obs.step == 0
    assert not obs.done

def test_step():
    env = ZomatoDispatcherEnv()
    obs = env.reset(task_id="single_zone")
    action = DispatchAction(order_id="ORD-001", rider_id="RIDER-1")
    obs2 = env.step(action)
    assert obs2.step == 1

def test_full_episode():
    env = ZomatoDispatcherEnv()
    obs = env.reset(task_id="single_zone")
    steps = 0
    while not obs.done and steps < 20:
        pending = [o for o in obs.orders if not o.assigned_rider 
                   and not o.delivered and not o.cancelled]
        available = [r for r in obs.riders if r.is_available]
        if pending and available:
            action = DispatchAction(order_id=pending[0].order_id, 
                                    rider_id=available[0].rider_id)
        else:
            action = DispatchAction(order_id="SKIP", rider_id="SKIP")
        obs = env.step(action)
        steps += 1
    assert obs.delivered_count > 0

def test_all_tasks():
    for task_id in ["single_zone", "multi_zone", "peak_hour"]:
        env = ZomatoDispatcherEnv()
        obs = env.reset(task_id=task_id)
        assert obs.task_id == task_id