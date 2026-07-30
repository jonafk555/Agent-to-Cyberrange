# Cyber Range QA

An implementation-ready foundation for an autonomous cyber-range QA platform using LangGraph. It is designed for authorized lab/range environments and separates **facts**, **reasoning**, **state**, and **control**:

```text
facts (tools) -> observations -> specialist proposal -> supervisor decision -> one action -> event
                                      ^                                      |
                                      +----------- shared memory -----------+
```

The supervisor is the only workflow controller. Specialists implement OODA (`observe`, `orient`, `decide`, `act`) and return proposals; they never route the graph. Tool wrappers are intentionally fact-only and can be replaced with real SSH/WinRM/PowerShell/Nmap/etc. adapters.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
export OPENAI_API_KEY=...
python examples/trace.py
pytest
```

The LLM is created in `src/cyberqa/llm.py` and injected into `Agents` from `main.py`:

```bash
export CYBERQA_LLM_PROVIDER=openai
export CYBERQA_LLM_MODEL=gpt-4.1-mini
export OPENAI_API_KEY=sk-...
python -m cyberqa.main
```

Without `OPENAI_API_KEY`, the graph uses a safe observe-only fallback. The API key is read from the environment and is never placed in graph state or tool evidence.

External services are optional for the dry-run example. Set `REDIS_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and `RABBITMQ_URL` to enable production adapters. `CYBERQA_LLM_MODEL` defaults to `gpt-4.1-mini`.

## Package map

- `models.py`: Pydantic domain contracts, evidence, attack paths, approvals, events.
- `state.py`: LangGraph state and shared-memory projections.
- `nodes.py`: supervisor and specialist nodes; dynamic next-action selection.
- `graph.py`: compiled StateGraph topology and conditional routing.
- `tools.py`: fact-only command/tool adapter contracts and safe dry-run adapters.
- `events.py`: RabbitMQ event bus with in-process fallback.
- `memory.py`: Redis checkpoint/working-memory and Neo4j-compatible knowledge graph repository.
- `approval.py`: policy engine and interrupt-compatible approval gate.
- `examples/trace.py`: dynamic validation, attack-path, debugging, and approval traces.

## Graph topology

`START -> supervisor -> {validation | testing | debugging | judge | reporting | approval | END}`. Every specialist returns to `supervisor`; no specialist can select a different specialist. The supervisor chooses the next action from current evidence, unresolved goals, failures, and repair history. An iteration budget and repeated-signature guard stop non-progress loops.

## Neo4j graph schema

Nodes: `Host`, `Service`, `User`, `Credential`, `Vulnerability`, `AttackPath`, `Flag`.
Relationships: `(:Host)-[:EXPOSES]->(:Service)`, `(:User)-[:CAN_AUTHENTICATE_WITH]->(:Credential)`, `(:Host)-[:TRUSTS|REACHES]->(:Host)`, `(:AttackPath)-[:STARTS_AT|USES|REQUIRES|ENDS_AT]->(any)`, `(:Host)-[:HAS_VULNERABILITY]->(:Vulnerability)`.

Upserts are idempotent and are emitted from observations/events, not from agent prose. See `KnowledgeGraphRepository.upsert_observation`.

## Approval policy

Service restart, firewall/DNS/route corrections, package installation, and time synchronization are autonomous. Domain/forest/ADCS rebuilds, GPO replacement, credential reset, user deletion, and other destructive mutations emit `APPROVAL_REQUIRED` and suspend at the approval node. The approval record includes the exact action, scope, reason, expected impact, rollback, and evidence.

## Security posture

This package is an orchestration skeleton for controlled cyber ranges. Keep credentials in a secret manager, restrict tool identities, require allow-listed targets, log immutable evidence, and run destructive actions only behind the approval gate.
