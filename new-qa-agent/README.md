# Cyber Range QA

An implementation-ready foundation for an autonomous cyber-range QA platform using LangGraph. It is designed for authorized lab/range environments and separates **facts**, **reasoning**, **state**, and **control**:

```text
Supervisor -> Specialist ReAct reason -> ToolNode -> tool result -> reason -> done
     ^                                                          |
     +---------------- shared evidence/events ------------------+
                              |
                         interrupt() -> Human -> Command(resume=...)
```

The supervisor is the only workflow controller. Each specialist binds an allow-listed set of LangChain tools and runs a ReAct loop (`reason -> ToolNode -> reason`) until it has enough facts. Specialists never route to another specialist. Tool wrappers are intentionally fact-only and can be replaced with real SSH/WinRM/PowerShell/Nmap/etc. adapters. Recoverable command failures first return complete evidence to the same Agent, which has a bounded opportunity to correct parameters, choose another reviewed adapter, or pivot to a justified AD path. Only a non-recoverable failure, exhausted repair budget, destructive action, or repeated no-progress path pauses with `interrupt()`; the CLI resumes the checkpoint using `Command(resume=...)`.

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
export CYBERQA_ALLOWED_TARGETS="10.10.10.0/24"
python -m cyberqa.main --target 10.10.10.0/24 \
  --scenario-id ad-lab-01 \
  --objective "Validate LDAP and test the authorized attack path" \
  --max-iterations 12
```

The CLI uses LangGraph streaming while the LLM remains the decision-maker. It reports reasoning status, selected tools, command execution, results, and graph state updates without exposing private chain-of-thought. It stays in interactive mode after the task finishes: type a new objective at `你：` to continue the same conversation/checkpoint session; type `exit` or `quit` to leave. Add `--once` for a single non-interactive run.

Human responses are first-class execution input. For example, `run nmap -F against 10.10.10.0/24`, `use NXC LDAP users`, or `approve to as-rep roasting; username file: /path/users.txt` is converted into a frozen reviewed decision when it names a known tool/capability. Other natural-language guidance is retained as `human_instruction` and is passed to the structured Supervisor as a mandatory constraint. `CYBERQA_AD_*` settings supplied in the response are applied to the current process; password values are redacted from the conversation and reports.

Every new session begins with a bounded `initial_recon` Agent. It collects the local runtime baseline, then starts network access with `nmap -sn` for a CIDR (or `nmap -F` for one host). If ICMP discovery yields no hosts, it adapts once to `-F`; discovered non-local hosts then receive the reviewed `nmap -sC -sV` service baseline. Loopback and the Kali runner's own interface addresses are never target-scanned. Account-dependent LDAP/NXC/Impacket probes are deferred until the evidence and credential context justify them. The result is written to `reports/<scenario-id>-initial-recon.md`. The report contains observed facts and command output, not model guesses.

`build_kali_registry()` provides reviewed adapters for `nmap`, `nxc` (SMB/LDAP recon), Impacket, `dig`, `curl`, `ldapsearch`, `smbclient`, `ip`, `cat`, `nft`, and `timedatectl`. Nmap profiles include `host_discovery` (`-sn`), `fast` (`-F`), and `default` (`-sC -sV`); the default profile is only used on an individual discovered host, never as the first CIDR probe. LDAP and SMB expose reviewed repair profiles such as LDAP `rootdse`/`subtree`/`starttls_rootdse`/`gssapi_rootdse` and SMB `anonymous`/`smb2`/`smb3`/`port445`; the Agent may also choose validated read-only argv fragments. The adapter validates those fragments and always injects the authorized target/module itself. A non-zero LDAP, SMB, NXC, or Nmap result is stored with full stdout/stderr, `error_kind`, and `recoverable` metadata. The inner ReAct loop then inspects every result in a multi-tool batch, avoids the same effective command, and gets at most three recovery failures before Human-in-the-loop. Commands run with `create_subprocess_exec` (no shell), a timeout, and the target policy. A CIDR entry such as `10.0.0.0/24` authorizes every address in that network, except loopback and the scanner's local interface addresses. After an Nmap result, discovered IPv4 addresses inside the authorized CIDR are added to the runtime policy and reported as `[Target]`; this lets the ReAct agent continue with the discovered hosts. Evidence retains the complete redacted stdout/stderr and adds line counts plus a useful-output summary; the CLI displays only that summary. The observation store is SQLite at `CYBERQA_OBSERVATION_DB` (default `.cyberqa/observations.sqlite3`) and records the effective argv, target, parameters, result, and failure across process restarts. Identical effective argv is cached and the initial baseline is bounded; deferred AD probes are selected only after evidence synthesis. `force_refresh=True` remains an explicit API/operator option and is not exposed to the model-visible tool schema. Install the corresponding Kali packages first, such as `nmap`, `netexec`, `impacket-scripts`, `dnsutils`, `curl`, `ldap-utils`, and `smbclient`.

After reconnaissance, non-secret values such as the discovered AD domain, base DN, DC, DNS servers, and networks are written to `CYBERQA_DISCOVERED_ENV` (default `.cyberqa/discovered.env`) and loaded on the next process start. Existing secrets are never inferred or written. Target profiles preserve domain/forest relationships: an LDAP authentication or forest-context error is not automatically labelled as a network outage, and a different domain/forest is retained as a cross-forest candidate for later trust analysis.

The AD strategy layer is evidence-driven rather than prompt-only. It treats a domain/DC, username source, credential validation, LDAP bind, SPNs, and relationship collection as separate facts. With no credential and a username source it dispatches one approved AS-REP assessment; with no username source it runs one bounded anonymous identity phase and then asks for a source if no users are found. With a validated credential it enumerates users/SPNs, only selects Kerberoast when SPNs exist, collects relationships once, and then evaluates the evidence. Failed methods are recorded in `method_history` and are not retried under a new action description.

When no domain credential is configured, the planner first performs a bounded anonymous identity probe across LDAP, SMB, and NXC LDAP. It then aggregates the results: if usernames are found, it prioritizes `asrep_roasting_assessment`; AS-REP does not require a domain credential. A username source can also be `CYBERQA_AD_USERS_FILE` (one username per line). Anonymous NXC is disabled by default and is enabled only by the bounded identity phase or `CYBERQA_ALLOW_ANONYMOUS_NXC=1`. AS-REP assessment remains behind the credential-material approval gate. If anonymous paths produce no usernames, the Agent asks for a username source instead of repeating recon.

Without `OPENAI_API_KEY`, the graph uses a safe observe-only fallback. The API key is read from the environment and is never placed in graph state or tool evidence.

External services are optional for the dry-run example. Set `REDIS_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and `RABBITMQ_URL` to enable production adapters. `CYBERQA_LLM_MODEL` defaults to `gpt-4.1-mini`.

## Package map

- `models.py`: Pydantic domain contracts, evidence, attack paths, approvals, events.
- `state.py`: LangGraph state and shared-memory projections.
- `nodes.py`: supervisor and specialist nodes; decision dispatch, approval consumption, and evidence projection.
- `ad_strategy.py`: deterministic AD prerequisite/transition guard between LLM planning and tool execution.
- `graph.py`: compiled StateGraph topology and conditional routing.
- `tools.py`: fact-only command/tool adapter contracts and safe dry-run adapters.
- `events.py`: RabbitMQ event bus with in-process fallback.
- `memory.py`: Redis checkpoint/working-memory and Neo4j-compatible knowledge graph repository.
- `approval.py`: policy engine, exact decision fingerprints, and interrupt-compatible approval gate.
- `examples/trace.py`: dynamic validation, attack-path, debugging, and approval traces.

## Graph topology

`START -> supervisor -> {validation | testing | debugging | judge | reporting | approval | human_help | END}`. Validation, testing, and debugging contain nested `reason -> tools -> reason` ReAct subgraphs. Every specialist returns to `supervisor`; no specialist can select a different specialist. The supervisor chooses the next specialist from current evidence, unresolved goals, failures, and repair history. An iteration budget and repeated-signature guard route non-progress loops to `human_help`. `MemorySaver` is the default checkpoint backend; production deployments can inject a durable checkpointer.

## Neo4j graph schema

Nodes: `Host`, `Service`, `User`, `Credential`, `Vulnerability`, `AttackPath`, `Flag`.
Relationships: `(:Host)-[:EXPOSES]->(:Service)`, `(:User)-[:CAN_AUTHENTICATE_WITH]->(:Credential)`, `(:Host)-[:TRUSTS|REACHES]->(:Host)`, `(:AttackPath)-[:STARTS_AT|USES|REQUIRES|ENDS_AT]->(any)`, `(:Host)-[:HAS_VULNERABILITY]->(:Vulnerability)`.

Upserts are idempotent and are emitted from observations/events, not from agent prose. See `KnowledgeGraphRepository.upsert_observation`.

## Approval policy

Service restart, firewall/DNS/route corrections, package installation, and time synchronization are autonomous. Domain/forest/ADCS rebuilds, GPO replacement, credential reset, user deletion, and other destructive mutations emit an approval request and suspend at the approval node. An approval resumes the frozen decision once; sensitive tools additionally require the matching capability and exact tool parameters. The approval record includes the exact action, scope, reason, expected impact, rollback, and evidence.

## Security posture

This package is an orchestration skeleton for controlled cyber ranges. Keep credentials in a secret manager, restrict tool identities, require allow-listed targets, log immutable evidence, and run destructive actions only behind the approval gate.
