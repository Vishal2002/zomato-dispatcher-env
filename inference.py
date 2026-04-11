#!/usr/bin/env python3
import json
import os
import re
import textwrap
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from client import ZomatoDispatcherClient
from models import DispatchAction

load_dotenv()
load_dotenv()
# print(f"[DEBUG] API_KEY loaded: {'YES' if os.getenv('OPENAI_API_KEY') else 'NO'}")
# print(f"[DEBUG] HF_TOKEN loaded: {'YES' if os.getenv('HF_TOKEN') else 'NO'}")
# ── Config ─────────────────────────────────────────────────────
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS   = 40
TEMPERATURE = 0.0
MAX_TOKENS  = 150
TASK_IDS    = ["single_zone", "multi_zone", "peak_hour"]

_llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL) if API_KEY else None

# ── Prompts ─────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a food delivery dispatcher. Each step assign ONE pending order to ONE available rider.

CRITICAL RULES:
- PENDING = not assigned, not delivered, not cancelled
- AVAILABLE = is_available is true
- If ALL riders are busy (in transit) → SKIP
- If NO pending orders → SKIP  
- Otherwise ALWAYS assign — never SKIP when you can assign
- Priority: smallest sla_deadline first (most urgent)
- Tiebreak: rider whose zone matches order's restaurant_zone

Output ONLY valid JSON, no explanation:
{"order_id": "ORD-001", "rider_id": "RIDER-1"}
""").strip()

def build_prompt(obs) -> str:
    pending = sorted(
        [o for o in obs.orders 
         if not o.assigned_rider and not o.delivered and not o.cancelled],
        key=lambda o: o.sla_deadline
    )
    available = [r for r in obs.riders if r.is_available]
    
    # Show in-transit so LLM knows why riders are busy
    in_transit = [o for o in obs.orders 
                  if o.assigned_rider and not o.delivered and not o.cancelled]

    order_lines = "\n".join(
        f"  {o.order_id} deadline={o.sla_deadline} rest_zone={o.restaurant_zone} cust_zone={o.customer_zone}"
        for o in pending[:8]
    ) or "  NONE"

    rider_lines = "\n".join(
        f"  {r.rider_id} zone={r.current_zone} available={r.is_available}"
        for r in obs.riders  # show ALL riders, not just available
    ) or "  NONE"
    
    transit_lines = "\n".join(
        f"  {o.order_id} → {o.assigned_rider}"
        for o in in_transit
    ) or "  NONE"

    return f"""Step {obs.step} / {obs.max_steps}
Delivered: {obs.delivered_count} | Cancelled: {obs.cancelled_count} | In-transit: {len(in_transit)}

PENDING (unassigned, sorted by urgency):
{order_lines}

ALL RIDERS:
{rider_lines}

IN TRANSIT (already assigned, riders busy):
{transit_lines}

If no pending orders OR no available riders, reply: {{"order_id": "SKIP", "rider_id": "SKIP"}}
Otherwise assign the most urgent pending order to the best available rider."""

def parse_action(text: str, obs=None) -> DispatchAction:
    if not text:
        return greedy_fallback_action(obs)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    try:
        data = json.loads(text)
        if "order_id" in data and "rider_id" in data:
            action = DispatchAction(**data)
            # Validate the LLM's choice — fall back to greedy if invalid
            if obs and action.order_id != "SKIP":
                valid_orders = {o.order_id for o in obs.orders 
                               if not o.assigned_rider and not o.delivered and not o.cancelled}
                valid_riders = {r.rider_id for r in obs.riders if r.is_available}
                if action.order_id not in valid_orders or action.rider_id not in valid_riders:
                    print(f"  [LLM invalid] {action} — falling back to greedy")
                    return greedy_fallback_action(obs)
            return action
    except:
        pass
    return greedy_fallback_action(obs)

def greedy_fallback_action(obs) -> DispatchAction:
    """Smart greedy: most urgent order + zone-matched rider."""
    if obs is None:
        return DispatchAction(order_id="SKIP", rider_id="SKIP")
    
    pending = sorted(
        [o for o in obs.orders 
         if not o.assigned_rider and not o.delivered and not o.cancelled],
        key=lambda o: o.sla_deadline
    )
    available = [r for r in obs.riders if r.is_available]
    
    if not pending or not available:
        return DispatchAction(order_id="SKIP", rider_id="SKIP")
    
    urgent = pending[0]
    # prefer zone match, then nearest zone
    best = min(available, key=lambda r: (
        0 if r.current_zone == urgent.restaurant_zone else 1,
        abs(r.current_zone - urgent.restaurant_zone)
    ))
    return DispatchAction(order_id=urgent.order_id, rider_id=best.rider_id)


def run_episode(env, task_id: str) -> float:
    print(f"[START] task={task_id} env=zomato-dispatcher-env model={MODEL_NAME}")
    
    result = env.reset(task_id=task_id)
    obs = result.observation
    step = 0
    rewards = []

    while not result.done and step < MAX_STEPS:
        action = greedy_fallback_action(obs)
        error = None

        if _llm:
            try:
                completion = _llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(obs)},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                text = completion.choices[0].message.content or ""
                action = parse_action(text, obs)
            except Exception as e:
                error = str(e)

        result = env.step(action)
        obs = result.observation
        rewards.append(round(result.reward, 4))
        step += 1

        print(f"[STEP] step={step} action={action.model_dump_json()} reward={result.reward:.2f} done={result.done} error={error}")

    # Compute grader score
    total = len(obs.orders)
    delivered = obs.delivered_count
    cancelled = obs.cancelled_count

    if task_id == "single_zone":
        score = delivered / total if total else 0
    elif task_id == "multi_zone":
        delivery_rate = delivered / total if total else 0
        cancel_ratio = cancelled / total if total else 0
        score = delivery_rate - (0.15 * cancel_ratio)
    elif task_id == "peak_hour":
        on_time = sum(1 for o in obs.orders if o.delivered)
        on_time_ratio = on_time / delivered if delivered > 0 else 0
        cancel_avoidance = 1.0 - (cancelled / total if total else 0)
        score = (0.50 * (delivered / total)
                 + 0.30 * on_time_ratio
                 + 0.20 * cancel_avoidance)

    score = round(max(0.001, min(0.999, score)), 4)
    success = score >= 0.999
    rewards_str = ",".join(str(r) for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step} score={score:.3f} rewards={rewards_str}")
    return score
def main():
    scores = {}
    with ZomatoDispatcherClient(base_url=ENV_BASE_URL).sync() as env:
        for task_id in TASK_IDS:
            scores[task_id] = run_episode(env, task_id)

    avg = sum(scores.values()) / len(scores)
    print(f"\nFinal scores: {scores}")
    print(f"Average: {avg:.2f}")

    output = {"scores": scores, "average": round(avg, 2), "model": MODEL_NAME}
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved → baseline_scores.json")


if __name__ == "__main__":
    main()