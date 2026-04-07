#
#   ███████╗████████╗ █████╗ ████████╗███████╗███████╗████████╗██████╗ ██╗██╗  ██╗███████╗
#   ██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔══██╗██║██║ ██╔╝██╔════╝
#   ███████╗   ██║   ███████║   ██║   █████╗  ███████╗   ██║   ██████╔╝██║█████╔╝ █████╗
#   ╚════██║   ██║   ██╔══██║   ██║   ██╔══╝  ╚════██║   ██║   ██╔══██╗██║██╔═██╗ ██╔══╝
#   ███████║   ██║   ██║  ██║   ██║   ███████╗███████║   ██║   ██║  ██║██║██║  ██╗███████╗
#   ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝
#

**An RL agent that learns to break your API before attackers do.**

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-compliant-00C853.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![HF Spaces Ready](https://img.shields.io/badge/HF%20Spaces-ready-FFCC00.svg)

## Abstract
Traditional API fuzzers are overwhelmingly stateless and syntax-oriented. They excel at malformed payload mutation, but many high-impact production failures emerge only after valid, stateful interaction sequences. This mismatch creates blind spots for vulnerabilities that require temporal structure: repeated writes before reads, user-specific data growth, or semantically valid requests that trigger worst-case algorithmic behavior.

StateStrike frames API fuzzing as a Markov Decision Process (MDP). The agent observes API response state (status codes, latency drift, vulnerability markers), selects actions over endpoint and payload strategy, and receives shaped rewards when it induces measurable infrastructure degradation. This includes ReDoS-class compute spikes, deliberate unindexed aggregation slowdowns, and chain-completion bonuses that bias exploration toward meaningful CRUD trajectories rather than random endpoint spraying.

The design draws from stateful REST fuzzing and RL-based vulnerability discovery literature: RESTler (Atlidakis et al., ICSE 2019), TitanFuzz (Deng et al., 2023) for intelligent black-box fuzzing motivation, classical RL formulation from Sutton & Barto (2018), ReDoS impact characterization from Davis et al. (USENIX Security 2018), and OpenEnv architecture patterns (Meta x Hugging Face OpenEnv specification, 2025).

## Architecture (ASCII)
```text
                         reward / observation
                   +-------------------------------+
                   |                               |
                   v                               |
+----------------------+      WebSocket /ws    +-------------------------+
|  Baseline / RL Agent | --------------------> |  StateStrike OpenEnv    |
|  (agent/runner.py)   | <-------------------- |  Server (reset/step/state)
+----------------------+                        +-----------+-------------+
                                                            |
                                                            | HTTP actions
                                                            v
                                                +-------------------------+
                                                | Vulnerable Honeypot API |
                                                | (FastAPI + SQLite)      |
                                                +-----------+-------------+
                                                            |
                                                            | telemetry events
                                                            v
                                                +-------------------------+
                                                | telemetry.json + SSE     |
                                                | + Streamlit Dashboard    |
                                                +-------------------------+
```

## Reward Function
StateStrike optimizes the following shaped objective:

$$
R_t = \alpha \cdot \log\left(\frac{L_t}{L_{base}}\right) + \beta \cdot S_t + \gamma \cdot E_t - \delta \cdot P_t
$$

Where:
- $L_t$: response latency at step $t$ (ms)
- $L_{base}$: rolling 10-step baseline latency
- $S_t$: state-transition bonus ($+10$ on POST->GET chain completion)
- $E_t$: exploitation bounty ($+500$ for 5xx or timeout-like latency)
- $P_t$: fuzzing penalty ($-1$ for fast 400 spam)
- $\alpha=1.0, \beta=10, \gamma=500, \delta=1$

Why each term exists:
- Log latency term rewards proportional degradation, limiting reward inflation from absolute latency variance.
- Chain bonus favors deep logical sequencing, discouraging shallow endpoint roulette.
- Exploitation bounty creates sparse high-value supervision for meaningful vulnerability discovery.
- Fuzzing penalty reduces reward hacking from cheap malformed-request spam while preserving costly 400 paths (e.g., ReDoS).

## Quickstart
> Compatibility note: local installs require Python 3.11.x due to pinned package versions (notably `pydantic==2.7.1`). If your host uses Python 3.13+, prefer the Docker path below.

### Docker (one-command)
```bash
docker compose up --build
```

### Manual Python
```bash
python -m pip install -r requirements.txt
cp .env.example .env

# terminal 1
uvicorn honeypot.app:app --host 0.0.0.0 --port 8000

# terminal 2
python -m statestrike_env.server

# terminal 3
streamlit run dashboard/app.py --server.port 8501

# terminal 4
python -m agent.runner --steps 200
```

## OpenEnv Interface Contract
Environment endpoint: `ws://localhost:8001/ws`

WebSocket frame methods:
- `reset()` -> `StateStrikeObservation`
- `step(action)` -> `StateStrikeObservation`
- `state()` -> `StateStrikeState`

Core semantics:
- Per-connection isolated `StateStrikeSession`
- Episode terminates when `step_count >= 200` or `cumulative_reward < -50`
- Honeypot outages return synthetic observations (`status=0`, `latency_ms=0`) instead of crashing

## Vulnerability Model
### VULN-1: ReDoS in `POST /users`
Regex: `^([a-zA-Z0-9]+\s?)*[a-zA-Z0-9]+$` with `re.fullmatch(..., re.DOTALL)`.
Input pattern `"aaaa...a!"` induces catastrophic backtracking (Davis et al., 2018).

### VULN-2: Stateful DB Degradation in `GET /orders`
Only reached after sustained sequence pressure (`>20` orders for a user). Endpoint runs intentional $O(n^2)$ in-memory aggregation and adds I/O-like sleep, modeling unindexed analytical degradation. This is explicitly sequence-dependent, reflecting the statefulness emphasized in RESTler-style analysis.

## Evaluation Alignment
| Hackathon Criterion | Where Addressed |
|---|---|
| Task definition clarity | `statestrike_env/models.py`, reward section in this README |
| Grader correctness | `statestrike_env/grader.py`, `tests/test_grader.py` |
| Reward logic soundness | `statestrike_env/grader.py` docstring + formula above |
| OpenEnv compliance | `statestrike_env/server.py`, `statestrike_env/__init__.py` |
| LLM scoring potential | This README + typed observation/state models |
| Real-world applicability | `honeypot/app.py` (ReDoS + stateful degradation) |
| Code quality | Typed signatures, docstrings, logging, constants, tests |

## Project Layout
```text
statestrike/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile.honeypot
├── Dockerfile.env
├── honeypot/
├── statestrike_env/
├── agent/
├── dashboard/
├── tests/
└── scripts/
```

## License
MIT License.

## Team
StateStrike Core Team
- AI/ML + Systems Architecture: OpenEnv Hackathon submission build
- Security Research Focus: Stateful API fuzzing for infrastructure degradation discovery
