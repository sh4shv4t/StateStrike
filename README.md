---
title: RyFlow
emoji: "🌊"
colorFrom: purple
colorTo: red
sdk: docker
python_version: "3.11"
app_file: app.py
pinned: false
tags:
  - openenv
---

  _____ _        _       ____  _        _ _
 / ____| |      | |     / __ \| |      (_) |
| (___ | |_ __ _| |_ ___| |  | | |_   _ _| |_
 \___ \| __/ _` | __/ _ \ |  | | | | | | | __|
 ____) | || (_| | ||  __/ |__| | | |_| | | |_
|_____/ \__\__,_|\__\___|\___\_\_|\__,_|_|\__|

StateStrike Security Audit Environment
An OpenEnv-ready stateful API security environment for real-world vulnerability triage.

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-00C853)
![MIT License](https://img.shields.io/badge/License-MIT-green)
![HF Spaces](https://img.shields.io/badge/HF%20Spaces-Ready-yellow)

## Environment Description and Motivation
StateStrike models a practical security engineering workflow: systematic API auditing to discover, classify, and chain exploitable behaviors in a production-like service.

Unlike toy game environments, the agent performs genuine tasks security teams run in real engagements:
- Endpoint reachability mapping
- Vulnerability probing and classification
- Stateful exploit-chain execution

This design creates measurable operational value: better API hardening and earlier detection of latency-amplifying attack paths.

## Action Space
| Field | Type | Description | Values |
|---|---|---|---|
| endpoint | EndpointChoice | Target API operation | POST /users, GET /users/{id}, POST /orders, GET /orders, GET /health |
| payload_strategy | PayloadStrategy | Payload mutation strategy | valid, redos, oversized, malformed |
| target_user_id | Optional[int] | User context for stateful calls | null or integer user id |

## Observation Space
| Field | Type | Description |
|---|---|---|
| step | int | Current episode step |
| endpoint_called | str | Executed endpoint |
| http_status | int | HTTP response code |
| latency_ms | float | Request latency in milliseconds |
| response_body | dict[str, Any] | Parsed response payload |
| session_order_count | int | Number of orders created in session |
| endpoints_discovered | list[str] | Reachable endpoints found so far |
| vulnerabilities_found | list[str] | Confirmed vulnerability labels |
| task_progress | float | Normalized task completion in [0.0, 1.0] |

## Task Descriptions
| Task | Difficulty | Max Steps | Success Threshold | Description |
|---|---|---:|---:|---|
| endpoint_discovery | easy | 20 | 0.60 | Find all reachable API endpoints |
| vulnerability_probe | medium | 30 | 0.50 | Find and classify vulnerabilities (redos, db_degradation) |
| exploit_chain | hard | 60 | 0.75 | Execute full stateful exploit chain with evidence |

## Reward Function
Step reward is normalized to [0.0, 1.0] and shaped by true task progress:

R_step = clamp(Delta task_score + bonuses - penalties)

Components:
- Delta task score: max(0, score_t - score_t-1), capped to 0.30
- +0.05 for a newly discovered endpoint
- +0.10 for a newly confirmed vulnerability
- -0.02 for repeated identical no-op action
- +0.20 terminal completion bonus when task is solved

Anti-hacking properties:
- One-time vulnerability flags prevent bounty farming
- Chain cooldown and order-growth guards prevent POST/GET cycling exploits
- Baseline latency updated via EMA only on successful steps
- Connection failures produce neutral reward and never corrupt baseline

## Setup Instructions
### Docker (single command)
```bash
docker build -t statestrike .
docker run -p 7860:7860 statestrike
```

### Local Python
```bash
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn honeypot.app:app --host 0.0.0.0 --port 8000
HONEYPOT_URL=http://localhost:8000 uvicorn statestrike_env.environment:app --host 0.0.0.0 --port 7860
python inference.py
```

### HF Space URL
Set this to your deployed environment Space URL:
- https://sh4shv4t-statestrike-env.hf.space

## Baseline Scores
| Task | Baseline Score | Model |
|------|---------------:|-------|
| endpoint_discovery | 0.600 | Qwen/Qwen2.5-72B-Instruct |
| vulnerability_probe | 0.400 | Qwen/Qwen2.5-72B-Instruct |
| exploit_chain | 0.000 | Qwen/Qwen2.5-72B-Instruct |

## OpenEnv Compliance Checklist
- [x] Real-world task framing (security audit)
- [x] Typed Pydantic action/observation/state models
- [x] reset(), step(), state(), close() implemented
- [x] Three graded tasks (easy, medium, hard)
- [x] Graders produce normalized scores in [0.0, 1.0]
- [x] Partial-progress reward shaping
- [x] Root inference.py with [START]/[STEP]/[END] format
- [x] Root openenv.yaml manifest
- [x] Single-container Docker runtime with /health and /reset

## Architecture Diagram
```text
+-------------------------------+
| HF Space Container            |
|  +-------------------------+  |
|  | Honeypot API :8000      |  |
|  +-------------------------+  |
|  | OpenEnv Server :7860    |  |
|  | /reset /step /state     |  |
|  +-------------------------+  |
+---------------+---------------+
                |
                v
       inference.py (LLM agent)
```

## License
MIT
