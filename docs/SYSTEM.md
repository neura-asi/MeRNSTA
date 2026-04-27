# MeRNSTA — System Overview & Runbook (SYSTEM.md)

**Memory‑Ranked Neuro‑Symbolic Transformer Architecture**
**Scope:** one‑file source of truth for *what the system is*, *how to run it*, *how it works end‑to‑end*, and *what to improve next toward AGI‑grade autonomy.*

---

## 0) Executive Summary

MeRNSTA is a **memory‑first cognitive stack**: a FastAPI web + API platform that wraps a **Phase‑2 autonomous cognitive core** with **persistent, ranked memory**, **hybrid vector backends** (symbolic HRR, analogical VSA, semantic embeddings), and a **multi‑agent layer** (planner/critic/debater/reflector…). It is **config‑driven** (no hardcoding), deploys from **SQLite → PostgreSQL/pgvector**, and exposes a **visualizer dashboard** and **Prometheus metrics**. The design goal is *long‑horizon coherent behavior* via structured memory, contradiction control, and self‑maintenance cycles.

---

## 1) What the System Is

* **Core identity:** A **cognitive OS** for LLMs with *addressable, auditable memory* (S‑P‑O triplets + embeddings) and *agentic control* over retrieval, reasoning, and self‑care (compression, reconciliation, drift handling).
* **Key differentiators vs. generic wrappers:**

  * **Hybrid memory synthesis**: routes queries to **HRRFormer** (symbolic), **VecSymR** (analogical), and **semantic** backends, then **fuses** results (confidence, recency, overlap, source attribution).
  * **Contradiction‑aware generation**: Bayesian/semantic detectors + **logit guard** to suppress contradictory tokens; **volatility & personality‑based decay** to stabilize memory.
  * **Multi‑agent reasoning**: 20+ specialized agents orchestrated by a registry; debate/critique/reflect cycles with memory context.
  * **Observable + configurable**: Prometheus metrics, structured logs, visualizers, and **all thresholds/ports/models in config.yaml/.env**.
  * **Enterprise path**: Celery/Redis background jobs, SQLite→Postgres/pgvector, JWT, rate‑limiting.

---

## 2) Quick Start (Unified Full Stack)

```bash
# 1) Install
git clone https://github.com/icedmoca/mernsta
cd mernsta
pip install -r requirements.txt

# 2) (Optional) Start Ollama / tokenizer
./scripts/start_ollama.sh start

# 3) Launch full system (web + API + agents + background)
python main.py run
```

**Access points**

* Web Chat & Visualizer: `http://localhost:8000/chat` (visualizer routes enabled via config)
* System Bridge API (OpenAPI): `http://localhost:8001/docs`
* Health checks: `http://localhost:8000/health`

**Other modes**

* Web only: `python main.py web --port 8000`
* API only: `python main.py api --port 8001`
* OS integration (daemon/headless/interactive): `python system/integration_runner.py --mode daemon`
* Enterprise suite: `python start_enterprise.py`

**Troubleshooting**

* Validate embeddings/tokenizer: `python utils/ollama_checker.py --validate`
* API/Web health: `curl http://localhost:8001/health` / `curl http://localhost:8000/health`
* Dynamic port retry is enabled and config‑driven.

---

## 3) End‑to‑End Architecture & Data/Control Flow

**Entry & orchestration**

* `main.py run` → `system/unified_runner.py` starts **Web UI (web/main.py)**, **System Bridge API (api/system\_bridge.py)**, initializes **Phase‑2 Cognitive Core**, loads **agents.registry**, and kicks off **background loops**.

**Request path**

1. **Ingress**: Web chat or `/ask` hits the **System Bridge API**.
2. **Extraction**: `storage/phase2_cognitive_system.py` extracts S‑P‑O triplets (spaCy + regex fallbacks), computes volatility/contradictions/emotion, persists to DB.
3. **Retrieval (hybrid)**: Query is **routed** to HRR/VecSymR/Semantic backends (`vector_memory/hybrid_memory.py`), **vectorized in parallel**, then **fused** by confidence, recency, and semantic overlap (with source attribution).
4. **Reasoning**: The **multi‑agent layer** (planner, critic, debater, reflector, …) operates over retrieved context; **LLM fallback** is gate‑kept by memory (contradiction/logit guard).
5. **Response & maintenance**: Reply returned; **Celery** jobs handle reconciliation, compression, health checks; **Prometheus/structlog** record metrics and audits. Visualizer presents contradictions, clusters, metrics, events.

**Persistence & scaling**

* **SQLite (WAL, pooled)** for edge/dev; **PostgreSQL + pgvector** for scale. All via `storage/db_utils.py` with retry and pooling.

---

## 4) Cognitive Mechanics (Why It Works)

* **Bayesian surprise ranking**: Highlights salient tokens/facts to prioritize retention and retrieval.
* **Hybrid contradiction detector**: Rule+semantic score with PPO‑style tuning of sensitivity (γ) → **logit penalties** suppress conflicting continuations.
* **Volatility & personality decay**: Unstable topics lose influence; profile multipliers bias retention (loyal/skeptical/emotional/analytical).
* **Predictive causal modeling**: Links facts over time, estimates drift, and triggers **meta‑goals** (compression, reconciliation, health audits).
* **Episodic memory & trust**: Episodes, contradiction history, and subject‑level trust scores modulate reinforcement.

---

## 5) Subsystems Map

* **API/Web:** `api/system_bridge.py`, `web/main.py` (+ routes for agents/visualizer)
* **Cognitive Core:** `storage/phase2_cognitive_system.py`, `storage/memory_log.py`, `storage/spacy_extractor.py`
* **Vector/Memory:** `vector_memory/hybrid_memory.py` (+ HRR/VecSymR adapters), `embedder.py`
* **Agents:** `agents/registry.py`, `agents/base.py`, specialized agents (planner/critic/debater/reflector/etc.)
* **Cortex helpers:** `cortex/*` (contradiction, reconciliation, PPO tuner, memory ops, personality)
* **Observability:** `monitoring/logger.py`, `monitoring/metrics.py` (Prometheus `/dashboard/metrics`)
* **Background:** `tasks/task_queue.py` (Celery/Redis)
* **Config:** `config/settings.py`, `config/environment.py`, `config/reloader.py`

---

## 6) Memory & Retrieval Details

* **Triplets**: (subject, predicate, object) with timestamps, frequency, contradiction, volatility, confidence, emotion, episode, session\_id, user\_profile\_id; media\_type (text/image/audio) + BLOBs for multimodal.
* **Hybrid routing**: math/logic → HRR; analogical/comparative → VecSymR; default → semantic; all can run in parallel.
* **Fusion & attribution**: Confidence/recency/overlap weighting; keep provenance for audit and explanation.
* **Active forgetting**: temporal and volatility decay; pruning; cluster compression.

---

## 7) Multi‑Agent Layer (Capabilities Snapshot)

* **Planning & Strategy**: Planner, Recursive Planner, Decision Planner, Task Selector, Strategy Evaluator/Optimizer, Planning Integration, Meta Router.
* **Analysis & Reasoning**: Critic, Debater, Reflector, Hypothesis Generator, Intent Modeler, Architect Analyzer, Drift Analysis.
* **Memory & Learning**: Memory Consolidator, Reflex Anticipator, Daemon Drift Watcher, Cognitive Repair, Self Healer, Repair Simulator.
* **System Management**: Upgrade Manager, Meta Monitor, Execution Monitor, Command Router, File Writer, Edit Loop, Code Refactorer, Registry.
* **Communication/Interface**: Personality Engine, Reflective Engine, Self Prompter, Drift Execution Engine.

---

## 8) APIs & UX

**System Bridge API**

* `/ask`, `/memory` (search/recent/facts/contradictions), `/goal`, `/reflect`, `/personality`, `/status`
* Agent‑facing APIs: `/agent/context`, `/agent/contradictions`, `/agent/trust_score/{subject}`, `/agent/reflect`, `/agent/memory_health`, `/agent/search_triplets`

**Visualizer**

* Dashboard pages for facts, contradictions, clusters, metrics (Tailwind + D3/JS modules)

**Security**

* JWT for protected memory endpoints; rate limiting, input validation; CORS controls.

---

## 9) Deployment & Ops

* **Profiles**: Dev (SQLite/WAL), Production (Postgres/pgvector). Redis optional for caching/queues.
* **Scaling**: Horizontal API workers + Celery workers; pgvector for ANN similarity; load balancers in front of web/API.
* **Observability**: Prometheus metrics (cognitive + infra), structured logs, health endpoints; contradiction/cluster visualizers for cognitive triage.
* **Resilience**: Connection pooling + WAL; retry wrappers; background auto‑reconciliation and compression.

---

## 10) Performance & Benchmarks (from internal paper)

* Hybrid fusion success \~**88.9%**; contradiction F1 \~**0.89**; long‑context BLEU +**52%** over baseline; token‑latency \~**2.1ms** at 1M+ facts (pgvector).
  *(Figures are system‑reported; reproduce via included tests and dashboards.)*

---

## 11) Practical Runbook

* **First boot**: run, open web/chat, hit `/dashboard/metrics`; confirm HRR/VecSymR availability in logs; if missing, system falls back to default embedder.
* **Data check**: Post a few preferences; verify triplets in `/dashboard/facts`; induce a contradiction and watch `/dashboard/contradictions`.
* **Throughput**: Add pgvector + Redis; enable Celery beat; watch compression/reconciliation rates.
* **Security**: Set `API_SECURITY_TOKEN`; enable JWT; tune rate limits; set CORS.

---

## 12) Roadmap: Upgrades to Push Toward AGI‑Grade Autonomy

> Concrete, staged improvements with code touchpoints and acceptance checks.

### A. **Closed‑Loop Autonomy & Tool Use**

* **Goal manager** (meta‑goals → executable plans); wire to Planner/Task Selector for *self‑initiated* actions.
* **Reliable toolformer** layer: standardized ToolSpec, success detectors, and rollback policies.
* **Acceptance**: Given an objective (e.g., “summarize my docs weekly”), system schedules, runs, verifies output quality, and self‑corrects on failure.

### B. **Grounded World Model & Memory Graph**

* **Typed knowledge graph** over triplets with **temporal/causal edges**; incremental schema (entities/relations/events, provenance, uncertainty).
* **Acceptance**: Counterfactual and temporal queries (`what changed since X?`, `if Y then likely Z?`) yield stable, explainable answers.

### C. **Long‑Horizon Planning & Execution Monitors**

* Add **progress monitors** with invariant checks; **hierarchical plans** (HATP‑style) stored in plan\_memory; checkpoint/rollback per step.
* **Acceptance**: Multi‑day tasks recover from restarts; invariant breaches trigger clarification or replanning automatically.

### D. **Self‑Reflection That Updates Policy**

* Convert reflections into **policy deltas** (prompt/program fragments) gated by trust & tests; archive deltas with lineage.
* **Acceptance**: A/B evaluate before/after deltas on regression suites; retain only improvements.

### E. **Learning from Feedback (RLHF‑lite & Bandits)**

* Integrate **preference logging** and **bandit** selection among prompts/agents; PPO‑style tuning already present for contradictions → extend to **response policies**.
* **Acceptance**: Online win‑rate improves on held‑out tasks without manual hyper‑tuning.

### F. **Robust Tool Sandbox & Safety**

* Per‑tool **resource caps** (CPU/mem/fs/network), syscall filters, allow‑lists; structured error taxonomies; safe rollbacks.
* **Acceptance**: Adverse prompts cannot cause filesystem/network abuse; all tool use is audited with reversible effects.

### G. **Interpretability & Causal Debriefs**

* Auto‑generate **decision traces** linking answer → retrieved facts → agents’ votes → logit guard effects; expose in visualizer.
* **Acceptance**: For any response, operators can trace *why* in <10s.

### H. **Self‑Play, Curriculum, and Bench Farm**

* Synthetic **contradiction curricula**; **multi‑agent self‑play** debates; task generators to harden retrieval and fusion.
* **Acceptance**: Year‑over‑year improvement on open‑world evals (e.g., long‑context QA, contradiction mines, tool‑using tasks).

### I. **Scalable Memory Hygiene**

* **Aging & compaction** policies with topic‑aware summarization; cold‑storage tiers; shard by subject.
* **Acceptance**: 10M+ facts with bounded recall latency and stable precision.

### J. **Environment Interfaces**

* First‑class connectors for files, email, calendars, terminals, and codebases; typed events into memory with provenance.
* **Acceptance**: Agent can *act in the world* under policy, log effects, and learn from outcomes.

---

## 13) Configuration Principles (No Hardcoding)

* All thresholds/ports/models/paths in `config.yaml` + `.env` → loaded by `config/settings.py` with hot‑reload.
* Example knobs: network ports, hybrid memory backends, similarity thresholds, debate mode, visualizer flags, pooling sizes.

---

## 14) Security, Privacy, Compliance

* **AuthZ/AuthN**: JWT for memory routes; role hooks available.
* **Rate limiting & validation** on API.
* **Audit logging** via structlog; provenance on memory facts and vector sources.
* **Data controls**: selective forgetting by subject/episode; encryption hooks provided in storage layer.

---

## 15) Appendix: Minimal Repo Map

* **Entry/Orchestration**: `main.py`, `system/unified_runner.py`, `system/integration_runner.py`, `start_enterprise.py`
* **API/Web**: `api/system_bridge.py`, `web/main.py`, `web/routes/*`
* **Cognitive/Memory**: `storage/phase2_cognitive_system.py`, `storage/memory_log.py`, `vector_memory/*`, `embedder.py`
* **Agents**: `agents/registry.py`, `agents/base.py`, `agents/*`
* **Cortex**: `cortex/*` (contradiction, reconciliation, ppo\_tuner, memory\_ops, personality\_ops, …)
* **Observability/Tasks**: `monitoring/*`, `tasks/task_queue.py`
* **Config**: `config/*`

---

## 16) Repro & Validation

* Unit/functional tests for startup, memory backends, multimodal, normalization, dashboard, enterprise.
* CLI: `scripts/cortex.py`, demo contracts, validators; dashboards for metrics and cognitive states.

---

### Final Note

Treat MeRNSTA as a **memory‑centric agi platform**. The AGI trajectory is about making the loop **closed, grounded, and accountable**: goals → plans → actions → observations → memory → policy deltas.
