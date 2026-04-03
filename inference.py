#!/usr/bin/env python3
# inference.py
#
# MANDATORY REQUIREMENTS (from hackathon):
#   - File MUST be named inference.py in root directory
#   - MUST use OpenAI Client for all LLM calls
#   - MUST read: API_BASE_URL, MODEL_NAME, HF_TOKEN from env vars
#   - MUST complete in < 20 minutes on 2 vCPU / 8 GB RAM
#   - MUST produce reproducible scores (seed=42 everywhere)
#
# Pattern follows the official inference.py example provided by organizers.

import json
import os
import re
import textwrap
import time
from typing import List, Optional
from dotenv import load_dotenv

import requests
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

# ── Credentials from environment variables ─────────────────────
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = os.getenv("OPENAI_API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")

# ── Inference settings ─────────────────────────────────────────
MAX_STEPS   = 40       # safety cap per episode
TEMPERATURE = 0.0      # deterministic output (reproducibility)
MAX_TOKENS  = 150
FALLBACK_ORDER = "ORD-001"
FALLBACK_RIDER = "RIDER-1"

# ── OpenAI client (as required by hackathon) ───────────────────
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

TASK_IDS = ["single_zone", "multi_zone", "peak_hour"]

# ── Prompts ────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a Zomato/Swiggy delivery dispatcher.
    Each step assign exactly ONE pending order to ONE available rider.

    CRITICAL RULES:
    - ONLY use order_ids listed under "PENDING orders" — never re-assign delivered/assigned ones
    - ONLY use rider_ids listed under "AVAILABLE riders" — never assign to busy riders
    - If PENDING shows "NONE" or AVAILABLE shows "NONE" — reply: {"order_id": "SKIP", "rider_id": "SKIP"}
    - Pick the order with SMALLEST deadline number (most urgent)
    - Pick the rider whose zone matches the order's zone

    Respond ONLY with valid JSON:
    {"order_id": "ORD-003", "rider_id": "RIDER-2"}
""").strip()


def build_user_prompt(obs: dict, history: List[str]) -> str:
    """Build a text prompt from the current observation — like the official example."""
    pending = [
        o for o in obs.get("orders", [])
        if not o.get("assigned_rider")
        and not o.get("delivered")
        and not o.get("cancelled")
    ]
    pending_sorted = sorted(pending, key=lambda o: o.get("sla_deadline", 99))

    idle_riders = [r for r in obs.get("riders", []) if r.get("is_available")]

    order_lines = "\n".join(
        f"  {o['order_id']}: rest_zone={o['restaurant_zone']} "
        f"cust_zone={o['customer_zone']} "
        f"deadline={o['sla_deadline']}steps"
        for o in pending_sorted[:8]   # top 8 most urgent
    ) or "  (none)"

    rider_lines = "\n".join(
        f"  {r['rider_id']}: zone={r['current_zone']} rating={r['rating']}"
        for r in idle_riders
    ) or "  (none)"

    history_str = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(f"""
        Step {obs.get('step', '?')} / {obs.get('max_steps', '?')}
        Pending: {obs.get('pending_orders', 0)} | Active: {obs.get('active_orders', 0)} | Delivered: {obs.get('delivered_count', 0)} | Cancelled: {obs.get('cancelled_count', 0)}

        Pending orders (most urgent first):
        {order_lines}

        Available riders:
        {rider_lines}

        Recent history:
        {history_str}

        Assign one order to one rider. Reply ONLY with JSON.
    """).strip()


def parse_action(response_text: str) -> dict:
    """
    Extract JSON action from model response.
    Mirrors parse_model_action() pattern from the official example.
    """
    if not response_text:
        return {"order_id": FALLBACK_ORDER, "rider_id": FALLBACK_RIDER}

    # Strip markdown code fences if present
    text = re.sub(r"```json|```", "", response_text).strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if "order_id" in data and "rider_id" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object with regex
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            data = json.loads(match.group(0))
            if "order_id" in data and "rider_id" in data:
                return data
        except json.JSONDecodeError:
            pass

    return {"order_id": FALLBACK_ORDER, "rider_id": FALLBACK_RIDER}


def greedy_fallback(obs: dict) -> dict:
    """Pure heuristic fallback when LLM fails — same logic as /baseline."""
    pending = [
        o for o in obs.get("orders", [])
        if not o.get("assigned_rider")
        and not o.get("delivered")
        and not o.get("cancelled")
    ]
    idle = [r for r in obs.get("riders", []) if r.get("is_available")]

    if not pending or not idle:
        return {"order_id": FALLBACK_ORDER, "rider_id": FALLBACK_RIDER}

    urgent = min(pending, key=lambda o: o.get("sla_deadline", 99))
    best   = min(idle, key=lambda r: abs(
        r.get("current_zone", 0) - urgent.get("restaurant_zone", 0)
    ))
    return {"order_id": urgent["order_id"], "rider_id": best["rider_id"]}


# ── Environment HTTP helpers ───────────────────────────────────

def env_reset(task_id: str) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/reset-task",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(order_id: str, rider_id: str) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step-task",
        json={"order_id": order_id, "rider_id": rider_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_grade() -> dict:
    r = requests.get(f"{ENV_BASE_URL}/grader", timeout=30)
    r.raise_for_status()
    return r.json()


# ── Single episode runner ──────────────────────────────────────
def run_episode(task_id: str) -> float:
    print(f"\n{'='*55}")
    print(f"  Task: {task_id}")
    print(f"{'='*55}")

    obs     = env_reset(task_id)
    history: List[str] = []
    step    = 0
    t0      = time.time()

    while not obs.get("done", False) and step < MAX_STEPS:
        user_msg = build_user_prompt(obs, history)
        try:
            completion = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLM error] {exc} — using greedy fallback")
            response_text = ""

        action = parse_action(response_text) if response_text else greedy_fallback(obs)

        # Skip step if LLM says nothing to do
        if action.get("order_id") == "SKIP":
            print(f"  step={step:02d} | [waiting — letting deliveries complete]")
            obs = env_step(FALLBACK_ORDER, FALLBACK_RIDER)
            history.append(f"step{step}:wait(r={obs.get('episode_reward_so_far',0):.1f})")
            step += 1
            continue

        print(f"  step={step:02d} | {action.get('order_id','?')} → {action.get('rider_id','?')}")

        # ✅ KEY FIX: /step-task returns observation directly, not wrapped
        obs = env_step(
            action.get("order_id", FALLBACK_ORDER),
            action.get("rider_id", FALLBACK_RIDER),
        )

        history.append(
            f"step{step}:{action.get('order_id')}→{action.get('rider_id')}"
            f"(r={obs.get('episode_reward_so_far', 0):.1f})"
        )
        step += 1

    elapsed = time.time() - t0
    grade   = env_grade()
    score   = grade.get("score", 0.0)

    print(f"\n  Result: delivered={grade.get('delivered')}/{grade.get('total')} "
          f"cancelled={grade.get('cancelled')} score={score:.4f} ({elapsed:.1f}s)")
    return score


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    print("\nZomato Dispatcher OpenEnv — Baseline Inference")
    print(f"Model : {MODEL_NAME}")
    print(f"Server: {ENV_BASE_URL}")

    # Wait for env server to be ready (important for Docker startup)
    print("\nWaiting for environment server...")
    for attempt in range(15):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print("  Server ready.")
                break
        except Exception:
            pass
        print(f"  Attempt {attempt+1}/15 ...")
        time.sleep(3)

    # Run all 3 tasks
    scores = {}
    for task_id in TASK_IDS:
        scores[task_id] = run_episode(task_id)

    # Print final results
    print(f"\n{'='*55}")
    print("  FINAL SCORES")
    print(f"{'='*55}")
    for tid, sc in scores.items():
        bar = "#" * int(sc * 20)
        print(f"  {tid:<20} {sc:.4f}  [{bar:<20}]")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average: {avg:.4f}")
    print(f"{'='*55}\n")

    # Save to JSON (for CI / reproducibility check)
    output = {"scores": scores, "average": avg, "model": MODEL_NAME}
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  Saved → baseline_scores.json")


if __name__ == "__main__":
    main()
