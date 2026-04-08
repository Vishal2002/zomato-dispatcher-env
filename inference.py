#!/usr/bin/env python3
import json
import os
import re
import textwrap
import time
from typing import List

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ── Credentials from environment variables ─────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

# HF router uses HF_TOKEN as API key
# fallback for local testing
API_KEY = HF_TOKEN
# print(API_KEY,"key")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# ── Inference settings ─────────────────────────────────────────
MAX_STEPS   = 40
TEMPERATURE = 0.0
MAX_TOKENS  = 150
FALLBACK_ORDER = "ORD-001"
FALLBACK_RIDER = "RIDER-1"

# ── OpenAI-compatible client (HF router compatible) ────────────
_client = None
if API_KEY:
    try:
        _client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as e:
        print(f"[Warning] Client init failed: {e}. Using fallback.")
else:
    print("[Warning] No API key found. Using greedy fallback.")

TASK_IDS = ["single_zone", "multi_zone", "peak_hour"]

# ── Prompts ────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a Zomato/Swiggy delivery dispatcher.

Assign exactly ONE pending order to ONE available rider.

Rules:
- Only pick from pending orders
- Only assign available riders
- If none available → {"order_id": "SKIP", "rider_id": "SKIP"}
- Choose smallest deadline first
- Prefer matching zones

Reply ONLY JSON:
{"order_id": "ORD-001", "rider_id": "RIDER-1"}
""").strip()


def build_user_prompt(obs: dict, history: List[str]) -> str:
    pending = [
        o for o in obs.get("orders", [])
        if not o.get("assigned_rider")
        and not o.get("delivered")
        and not o.get("cancelled")
    ]
    pending_sorted = sorted(pending, key=lambda o: o.get("sla_deadline", 99))

    idle = [r for r in obs.get("riders", []) if r.get("is_available")]

    order_lines = "\n".join(
        f"{o['order_id']} deadline={o['sla_deadline']}"
        for o in pending_sorted[:8]
    ) or "NONE"

    rider_lines = "\n".join(
        f"{r['rider_id']} zone={r['current_zone']}"
        for r in idle
    ) or "NONE"

    return f"""
Step {obs.get('step')}

PENDING:
{order_lines}

AVAILABLE:
{rider_lines}

History:
{history[-3:] if history else "None"}

Assign one order.
""".strip()


def parse_action(text: str) -> dict:
    if not text:
        return {"order_id": FALLBACK_ORDER, "rider_id": FALLBACK_RIDER}

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    try:
        data = json.loads(text)
        if "order_id" in data and "rider_id" in data:
            return data
    except:
        pass

    match = re.search(r'\{.*?\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return {"order_id": FALLBACK_ORDER, "rider_id": FALLBACK_RIDER}


def greedy_fallback(obs: dict) -> dict:
    pending = [
        o for o in obs.get("orders", [])
        if not o.get("assigned_rider")
        and not o.get("delivered")
    ]
    idle = [r for r in obs.get("riders", []) if r.get("is_available")]

    if not pending or not idle:
        return {"order_id": FALLBACK_ORDER, "rider_id": FALLBACK_RIDER}

    urgent = min(pending, key=lambda o: o.get("sla_deadline", 99))
    rider  = idle[0]

    return {"order_id": urgent["order_id"], "rider_id": rider["rider_id"]}


def env_reset(task_id):
    return requests.post(f"{ENV_BASE_URL}/reset-task", json={"task_id": task_id}).json()


def env_step(order_id, rider_id):
    return requests.post(
        f"{ENV_BASE_URL}/step-task",
        json={"order_id": order_id, "rider_id": rider_id},
    ).json()


def env_grade():
    return requests.get(f"{ENV_BASE_URL}/grader").json()


def run_episode(task_id):
    print(f"\n--- {task_id} ---", flush=True)
    print(f"[START] task={task_id}", flush=True)

    obs = env_reset(task_id)
    history = []
    step = 0

    while not obs.get("done") and step < MAX_STEPS:
        user_msg = build_user_prompt(obs, history)
        response_text = ""

        if _client:
            try:
                completion = _client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as e:
                print(f"[LLM error] {e}", flush=True)
        else:
            print(f"step={step} | no client → fallback", flush=True)

        action = parse_action(response_text) if response_text else greedy_fallback(obs)

        # Handle SKIP — still advance world
        if action.get("order_id") == "SKIP":
            obs = env_step(FALLBACK_ORDER, FALLBACK_RIDER)
        else:
            obs = env_step(
                action.get("order_id", FALLBACK_ORDER),
                action.get("rider_id", FALLBACK_RIDER),
            )

        reward = obs.get("episode_reward_so_far", 0)
        print(f"[STEP] step={step} reward={reward:.4f}", flush=True)

        history.append(str(action))
        step += 1

    score = env_grade().get("score", 0)
    print(f"[END] task={task_id} score={score:.4f} steps={step}", flush=True)
    return score

def main():
    print("Running inference")

    scores = {}
    for t in TASK_IDS:
        scores[t] = run_episode(t)

    avg = sum(scores.values()) / len(scores)

    print("\nFinal:", scores)
    print("Average:", avg)


    output = {
        "scores": scores,
        "average": round(avg,2),
        "model": MODEL_NAME
    }

    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved → baseline_scores.json")


if __name__ == "__main__":
    main()