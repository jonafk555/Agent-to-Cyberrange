# Cyber Range QA

An implementation-ready foundation for an autonomous cyber-range QA platform using LangGraph. It is designed for authorized lab/range environments and separates **facts**, **reasoning**, **state**, and **control**:

```text
Supervisor -> Specialist ReAct reason -> ToolNode -> tool result -> reason -> done
     ^                                                          |
     +---------------- shared evidence/events ------------------+
                              |
                         interrupt() -> Human -> Command(resume=...)
```

The supervisor is the only workflow controller. Each specialist binds an allow-listed set of LangChain tools and runs a ReAct loop (`reason -> ToolNode -> reason`) until it has enough facts. Specialists never route to another specialist. Tool wrappers are intentionally fact-only and can be replaced with real SSH/WinRM/PowerShell/Nmap/etc. adapters. Tool failures, unusable results, destructive actions, and repeated no-progress paths pause with `interrupt()`; the CLI resumes the checkpoint using `Command(resume=...)`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
export OPENAI_API_KEY=...
python examples/trace.py
pytest
```

Copy `.env.example` to `.env` to configure the run without placing settings in the command line:

```bash
cp .env.example .env
# edit .env, then run:
python -m cyberqa.main
```

Command-line arguments override `.env`. Keep real API keys and lab credentials in `.env` only for local testing; `.env` is git-ignored. The optional AD variables are configuration references for credential-aware adapters and are not written into reconnaissance reports.

The LLM is created in `src/cyberqa/llm.py` and injected into `Agents` from `main.py`:

```bash
export CYBERQA_LLM_PROVIDER=openai
export CYBERQA_LLM_MODEL=gpt-4.1-mini
export OPENAI_API_KEY=sk-...
python -m cyberqa.main
```

To let the ReAct agents call Kali Linux tools, explicitly authorize the lab targets:

```bash
export CYBERQA_ALLOWED_TARGETS="127.0.0.1,localhost,10.10.10.10,dc01.lab.local"
python -m cyberqa.main --target 10.10.10.10 \
  --scenario-id ad-lab-01 \
  --objective "Validate LDAP and test the authorized attack path" \
  --max-iterations 12
```

The CLI uses LangGraph streaming while the LLM remains the decision-maker. It reports reasoning status, selected tools, command execution, results, and graph state updates without exposing private chain-of-thought. It stays in interactive mode after the task finishes: type a new objective at `你：` to continue the same conversation/checkpoint session; type `exit` or `quit` to leave. Add `--once` for a single non-interactive run.

Every new session begins with an `initial_recon` Agent. It collects OS/release, interfaces and routes, DNS, firewall, listening ports, ACLs, local/domain users, privileges, SUID/SGID files, range configuration, and initial `nxc`, Impacket, and BloodHound observations. The result is written to `reports/<scenario-id>-initial-recon.md`. The report contains observed facts and command output, not model guesses.

`build_kali_registry()` provides fixed adapters for `nmap`, `nxc` (SMB/LDAP recon), `impacket-rpcdump`, `bloodhound-python`, `dig`, `curl`, `ldapsearch`, `smbclient`, `ip`, `cat`, `nft`, and `timedatectl`. Commands run with `create_subprocess_exec` (no shell), a timeout, and the target policy. A CIDR entry such as `10.0.0.0/24` authorizes every address in that network. After an Nmap result, discovered IPv4 addresses are added to the runtime policy and reported as `[Target]`; this lets the ReAct agent continue with the discovered hosts. The observation store records `tool + target + action + parameters` signatures, returns cached results, and prevents repeated identical probes from consuming another subprocess call. Install the corresponding Kali packages first, such as `nmap`, `netexec`, `impacket-scripts`, `bloodhound.py`, `dnsutils`, `curl`, `ldap-utils`, and `smbclient`.

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

`START -> supervisor -> {validation | testing | debugging | judge | reporting | approval | human_help | END}`. Validation, testing, and debugging contain nested `reason -> tools -> reason` ReAct subgraphs. Every specialist returns to `supervisor`; no specialist can select a different specialist. The supervisor chooses the next specialist from current evidence, unresolved goals, failures, and repair history. An iteration budget and repeated-signature guard route non-progress loops to `human_help`. `MemorySaver` is the default checkpoint backend; production deployments can inject a durable checkpointer.

## Neo4j graph schema

Nodes: `Host`, `Service`, `User`, `Credential`, `Vulnerability`, `AttackPath`, `Flag`.
Relationships: `(:Host)-[:EXPOSES]->(:Service)`, `(:User)-[:CAN_AUTHENTICATE_WITH]->(:Credential)`, `(:Host)-[:TRUSTS|REACHES]->(:Host)`, `(:AttackPath)-[:STARTS_AT|USES|REQUIRES|ENDS_AT]->(any)`, `(:Host)-[:HAS_VULNERABILITY]->(:Vulnerability)`.

Upserts are idempotent and are emitted from observations/events, not from agent prose. See `KnowledgeGraphRepository.upsert_observation`.

## Approval policy

Service restart, firewall/DNS/route corrections, package installation, and time synchronization are autonomous. Domain/forest/ADCS rebuilds, GPO replacement, credential reset, user deletion, and other destructive mutations emit `APPROVAL_REQUIRED` and suspend at the approval node. The approval record includes the exact action, scope, reason, expected impact, rollback, and evidence.

## Security posture

This package is an orchestration skeleton for controlled cyber ranges. Keep credentials in a secret manager, restrict tool identities, require allow-listed targets, log immutable evidence, and run destructive actions only behind the approval gate.
