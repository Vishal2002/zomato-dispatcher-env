from client import ZomatoDispatcherClient
from models import DispatchAction

with ZomatoDispatcherClient(base_url="http://localhost:7860").sync() as env:
     
    for task in ["single_zone", "multi_zone", "peak_hour"]:
    
        result = env.reset(task_id=task)
        obs = result.observation
        print(f"Reset OK — {obs.pending_orders} pending orders")
    
        step = 0
        while not result.done and step < 50:   # ← use result.done + hard cap as safety net
            pending = [o for o in obs.orders 
                    if not o.assigned_rider and not o.delivered and not o.cancelled]
            available = [r for r in obs.riders if r.is_available]
            
            if pending and available:
                action = DispatchAction(
                    order_id=pending[0].order_id,
                    rider_id=available[0].rider_id
                )
            else:
                action = DispatchAction(order_id="SKIP", rider_id="SKIP")
            
            result = env.step(action)
            obs = result.observation
            step += 1
            print(f"Step {obs.step} | done={result.done} | delivered={obs.delivered_count} cancelled={obs.cancelled_count}")
        
        print(f"\nEpisode finished in {step} steps")
        print(f"Final reward: {obs.episode_reward_so_far}")
        print(f"Delivered: {obs.delivered_count} | Cancelled: {obs.cancelled_count}")