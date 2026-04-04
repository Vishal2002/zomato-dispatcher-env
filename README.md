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
  - dispatch
short_description: Food delivery dispatch RL environment — OpenEnv compatible
---

# Zomato Dispatcher OpenEnv

> **Food delivery order dispatch** — a real-world RL environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

An AI agent acts as a **fleet dispatcher** for a food delivery platform inspired by Zomato/Swiggy. It observes incoming orders and available riders across city zones, then assigns orders to riders to maximise on-time delivery while avoiding SLA breaches.

## Tasks

| Task | Difficulty | Description |
|---|---|---|
| `single_zone` | Easy | 4 orders, 3 riders, 1 zone, generous deadlines |
| `multi_zone` | Medium | 10 orders, 4 riders, 5 zones, normal deadlines |
| `peak_hour` | Hard | 20 orders, 5 riders, tight deadlines + surge at step 10 |

## Baseline scores (gpt-4o-mini agent)

| Task | Score |
|---|---|
| single_zone | 1.00 |
| multi_zone | 0.655 |
| peak_hour | 0.475 |
| **Average** | **0.71** |

## Quick test
```bash
curl https://Vishal200-zomato-dispatcher-env.hf.space/health
curl https://Vishal200-zomato-dispatcher-env.hf.space/tasks
curl https://Vishal200-zomato-dispatcher-env.hf.space/baseline
```

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Execute action |
| `/state` | GET | Current state |
| `/tasks` | GET | List all tasks |
| `/grader` | GET | Grade episode |
| `/baseline` | GET | Run baseline on all tasks |

## Local setup
```bash
pip install "openenv-core[core]>=0.2.1" fastapi uvicorn pydantic openai requests
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## License

MIT
