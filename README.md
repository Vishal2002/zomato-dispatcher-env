# Zomato Dispatcher OpenEnv

---
title: Zomato Dispatcher OpenEnv
emoji: 🛵
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - logistics
  - food-delivery
  - real-world
short_description: Food delivery dispatch RL environment — OpenEnv compatible
---


> **Food delivery order dispatch** — a real-world RL environment built on the
> [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

An AI agent acts as a **fleet dispatcher** for a food delivery platform
(inspired by Zomato/Swiggy). It observes incoming orders and available
delivery riders across city zones, then assigns orders to riders to maximise
on-time delivery while avoiding SLA breaches.

---

## Motivation

Last-mile food delivery dispatch is one of the most latency-sensitive
optimisation problems in consumer tech. Companies like Zomato, Swiggy, and
DoorDash solve it thousands of times per minute. Traditional rule-based systems
(nearest-rider, FIFO) leave significant efficiency on the table. An RL agent
trained in this environment can learn policies that balance urgency, proximity,
rider load, and demand surges simultaneously.

---

## Environment description

| Property | Value |
|---|---|
| City model | 5 zones (Koramangala, Indiranagar, HSR Layout, Whitefield, Jayanagar) |
| Riders | 3–5 per episode (task-dependent) |
| Orders | 4–20 + surge (task-dependent) |
| Episode length | 15–35 steps |
| Action | Assign 1 order to 1 rider per step |
| Observation | All orders + all riders + deadlines + reward so far |

---

## Action space

```json
{ "order_id": "ORD-003", "rider_id": "RIDER-2" }
```

---

## Observation space (key fields)

```json
{
  "step": 5,
  "task_id": "multi_zone",
  "pending_orders": 3,
  "delivered_count": 4,
  "cancelled_count": 1,
  "orders": [
    {
      "order_id": "ORD-003",
      "restaurant_zone": 2,
      "customer_zone": 4,
      "sla_deadline": 7,
      "assigned_rider": null,
      "delivered": false,
      "cancelled": false
    }
  ],
  "riders": [
    {
      "rider_id": "RIDER-2",
      "current_zone": 2,
      "is_available": true,
      "rating": 4.8,
      "eta_to_free": 0
    }
  ],
  "done": false
}
```

---

## Tasks

### Task 1 — `single_zone` *(easy)*
3 idle riders, 4 orders, 1 zone, generous deadlines.
Score = on-time deliveries / total orders.

### Task 2 — `multi_zone` *(medium)*
4 riders (1 busy), 10 orders, 5 zones, normal deadlines.
Score = delivery rate − 0.15 × cancellation rate.

### Task 3 — `peak_hour` *(hard)*
5 riders (2 busy), 20 orders, tight deadlines. Surge at step 10 adds 5 urgent orders.
Score = 50% delivery rate + 30% on-time ratio + 20% cancellation avoidance.

---

## Reward function

| Event | Reward |
|---|---|
| On-time delivery | +1.0 |
| Late delivery | +0.4 |
| SLA breach (cancelled) | −2.0 |
| Rider in same zone as restaurant | +0.3 |
| Invalid action | −0.3 |
| Step with no valid assignment | −0.5 |
| All orders delivered | +2.0 bonus |

---

## Baseline scores (seed=42)

| Task | Random agent | Greedy agent |
|---|---|---|
| single_zone | ~0.35 | ~0.75 |
| multi_zone | ~0.25 | ~0.58 |
| peak_hour | ~0.12 | ~0.40 |

---

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-key"
python inference.py
```

### Docker

```bash
docker build -t zomato-dispatcher-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your-key" \
  zomato-dispatcher-env
```

### Quick API test

```bash
# Health check
curl http://localhost:7860/health

# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "single_zone"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"order_id": "ORD-001", "rider_id": "RIDER-1"}'

# Grade the episode
curl http://localhost:7860/grader
```

---

## Project structure

```
zomato-dispatcher-env/
├── inference.py          ← LLM agent (OpenAI client, required filename)
├── models.py             ← Pydantic typed Action + Observation models
├── tasks.py              ← 3 task configs (easy/medium/hard)
├── grader.py             ← Deterministic graders (0.0–1.0)
├── openenv.yaml          ← OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
├── README.md
└── server/
    ├── app.py            ← FastAPI server + all 7 endpoints
    └── environment.py    ← Core RL simulation logic
```

---

## License

MIT
