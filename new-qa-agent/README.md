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

`--max-iterations` is retained as a progress/telemetry hint for compatibility; it is not the autonomous stop switch. Repeated effective commands are blocked by the durable observation ledger and sent back to Supervisor for replanning.

The observation cache stores the effective reviewed command identity (adapter, target, action/argv, profile, and non-secret parameters) together with its evidence/result and failure classification; it does not store the LLM's private reasoning. It is saved in SQLite at `CYBERQA_OBSERVATION_DB` (default `.cyberqa/observations.sqlite3`) and survives process restarts. Use `--clear-observation-cache` to clear it before a run, `CYBERQA_OBSERVATION_DB=:memory:` for a temporary empty cache, or `CYBERQA_OBSERVATION_TTL_SECONDS` to expire entries lazily. `force_refresh=True` is available to an explicit API/operator call, but is not model-visible.

After each fresh tool result, the Agent creates a safe post-tool evidence analysis containing usable content, unresolved questions, candidate reviewed tools, and a suggested next action. It is printed immediately and stored in `evidence_analyses` for the Supervisor. This is advisory planning memory, not a fixed pipeline; cache hits reuse prior evidence without another analysis/model call.

QA planning is assertion-driven as well as evidence-driven. Each task bootstraps one or more `QAAssertion` records from the objective and records the visibility mode (`white_box`, `gray_box`, or `black_box`). `EvidenceSufficiency` evaluates the strongest observed level from C0 Unknown, C1 Inferred, C2 Enumerated, C3 Functionally Verified, C4 Exploitability Verified, and C5 End-to-End Verified. The Supervisor chooses the least-invasive reviewed method needed by the unresolved assertion; it does not escalate an assertion to hash cracking or exploitation after its required threshold is already met. For example, AS-REP configuration evidence can complete at C2, while an explicitly requested end-to-end attack path requires C5.

Human responses are first-class semantic execution input. Compound guidance is passed to the structured Supervisor as a high-priority intent containing all requested goals, constraints, exclusions, ordering, and continuation—not reduced to the first named tool. Explicit capability approvals still become frozen reviewed decisions; `CYBERQA_AD_*` settings and username-file paths supplied in natural language are applied to the current process, while password values are redacted from the conversation and reports. A short negative response rejects the previous proposal and sends the Agent to an alternate autonomous path; only an explicit stop/abort ends the task.

Every new session begins with a runner-identity bootstrap. It runs only `inspect_interfaces` locally to record Kali's IP addresses as `runner_ips`; it does not inspect Kali's OS, routes, DNS, ports, users, or privileges. Those addresses are exclusion metadata, not cyber-range targets. The Supervisor then starts remote network reconnaissance with `nmap -sn` for a CIDR (or `nmap -F` for one host). If ICMP discovery yields no hosts, it adapts once to `-F`; discovered non-runner hosts in the authorized network then receive the reviewed `nmap -sC -sV` service baseline. Kali, loopback, the runner's interface addresses, and out-of-scope addresses are excluded from reconnaissance and validation. Account-dependent LDAP/NXC/Impacket probes are deferred until the evidence and credential context justify them. After each remote result, the Supervisor continues with the next authorized host/service or the next evidence-driven AD path; Human is used only when autonomous alternatives are genuinely exhausted, an approval is required, or the operator explicitly stops. The result is written to `reports/<scenario-id>-initial-recon.md` with the runner IPs recorded separately from QA evidence.

`build_kali_registry()` provides reviewed adapters for `nmap`, `nxc` (SMB/LDAP recon), Impacket, `dig`, `curl`, `ldapsearch`, `smbclient`, `ip`, `cat`, `nft`, and `timedatectl`. Local `inspect_*` adapters are blocked at the tool boundary except `inspect_interfaces` during `runner_identity`; Kali's OS, routes, DNS, ports, users, and privileges are not QA evidence. Nmap profiles include `host_discovery` (`-sn`), `fast` (`-F`), and `default` (`-sC -sV`); the default profile is only used on an individual discovered host, never as the first CIDR probe. LDAP and SMB expose reviewed repair profiles such as LDAP `rootdse`/`subtree`/`starttls_rootdse`/`gssapi_rootdse` and SMB `anonymous`/`smb2`/`smb3`/`port445`; the Agent may also choose validated read-only argv fragments. The adapter validates those fragments and always injects the authorized target/module itself. A non-zero LDAP, SMB, NXC, or Nmap result is stored with full stdout/stderr, `error_kind`, and `recoverable` metadata. The inner ReAct loop then inspects every result in a multi-tool batch, avoids the same effective command, and gets at most three recovery failures before Human-in-the-loop. Commands run with `create_subprocess_exec` (no shell), a timeout, and the target policy. A CIDR entry such as `10.0.0.0/24` authorizes every address in that network, except loopback and the scanner's local interface addresses. After an Nmap result, discovered IPv4 addresses inside the authorized CIDR are added to the runtime policy and reported as `[Target]`; this lets the ReAct agent continue with the discovered hosts. Evidence retains the complete redacted stdout/stderr and adds line counts plus a useful-output summary; the CLI displays only that summary. The observation store is SQLite at `CYBERQA_OBSERVATION_DB` (default `.cyberqa/observations.sqlite3`) and records the effective argv, target, parameters, result, and failure across process restarts. Identical effective argv is cached and the initial baseline is bounded; deferred AD probes are selected only after evidence synthesis. `force_refresh=True` remains an explicit API/operator option and is not exposed to the model-visible tool schema. Install the corresponding Kali packages first, such as `nmap`, `netexec`, `impacket-scripts`, `dnsutils`, `curl`, `ldap-utils`, and `smbclient`.

After reconnaissance, non-secret values such as the discovered AD domain, base DN, DC, DNS servers, and networks are written to `CYBERQA_DISCOVERED_ENV` (default `.cyberqa/discovered.env`) and loaded on the next process start. Existing secrets are never inferred or written. Target profiles preserve domain/forest relationships: an LDAP authentication or forest-context error is not automatically labelled as a network outage, and a different domain/forest is retained as a cross-forest candidate for later trust analysis.

The AD strategy layer is evidence-driven rather than prompt-only. It treats a domain/DC, username source, credential validation, LDAP bind, SPNs, and relationship collection as separate facts. After each tool result, the post-tool evidence analysis exposes what became usable and which reviewed capabilities are candidates; the Supervisor may choose AS-REP, local hash cracking, credential validation, authenticated enumeration, relationship collection, or another justified path instead of following a fixed pipeline. The deterministic AD layer only supplies prerequisite/completion guards; a concrete safe non-terminal Supervisor decision remains in control. With no username source it runs one bounded anonymous identity phase and then asks for a source if no users are found. With a validated credential it can enumerate users/SPNs, select Kerberoast when SPNs exist, collect relationships, and evaluate the evidence according to the accumulated findings. Judge/END is admitted only by the completion gate after remote baseline and required bounded AD evidence are present. Failed methods are recorded in `method_history` and are not retried under a new action description.
If the model emits `end/human_help` without a concrete approval, missing-input, tool, or scope blocker, the Supervisor automatically asks itself to re-plan and continue. After three consecutive refusals to select a path, Human is surfaced as a genuine planning-boundary fallback.

When no domain credential is configured, the planner first performs a bounded anonymous identity probe across LDAP, SMB, and NXC LDAP. It then aggregates the results: if usernames are found, it prioritizes `asrep_roasting_assessment`; AS-REP does not require a domain credential. A username source can also be `CYBERQA_AD_USERS_FILE` (one username per line). Anonymous NXC is disabled by default and is enabled only by the bounded identity phase or `CYBERQA_ALLOW_ANONYMOUS_NXC=1`. AS-REP assessment remains behind the credential-material approval gate. If anonymous paths produce no usernames, the Agent asks for a username source instead of repeating recon.

When AS-REP produces ticket material, the adapter writes only a mode-600 local hash artifact under `CYBERQA_CREDENTIAL_MATERIAL_DIR` (default `.cyberqa/credential-material`) and records its reference/count, never the hash contents. If `CYBERQA_AD_WORDLIST` or the standard Kali wordlist is available, the Supervisor can select the approval-gated `hash_cracking_assessment`; a recovered password remains process-local, then must pass `credential_validation` before `enumerate_domain_users`, SPN/Kerberoast selection, relationship collection, or other authenticated AD QA. A failed crack does not become a credential and does not cause an automatic retry loop.

Without `OPENAI_API_KEY`, the graph uses a safe observe-only fallback. The API key is read from the environment and is never placed in graph state or tool evidence.

External services are optional for the dry-run example. Set `REDIS_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and `RABBITMQ_URL` to enable production adapters. `CYBERQA_LLM_MODEL` defaults to `gpt-4.1-mini`.

## Package map

- `models.py`: Pydantic domain contracts, evidence, attack paths, approvals, events.
- `state.py`: LangGraph state and shared-memory projections.
- `nodes.py`: supervisor and specialist nodes; decision dispatch, approval consumption, and evidence projection.
- `ad_strategy.py`: deterministic AD prerequisite/transition guard between LLM planning and tool execution.
- `qa_assessment.py`: visibility mode, QA assertions, evidence levels, and evidence-sufficiency evaluation.
- `graph.py`: compiled StateGraph topology and conditional routing.
- `tools.py`: fact-only command/tool adapter contracts and safe dry-run adapters.
- `events.py`: RabbitMQ event bus with in-process fallback.
- `memory.py`: Redis checkpoint/working-memory and Neo4j-compatible knowledge graph repository.
- `approval.py`: policy engine, exact decision fingerprints, and interrupt-compatible approval gate.
- `examples/trace.py`: dynamic validation, attack-path, debugging, and approval traces.

## Graph topology

`START -> runner_identity -> supervisor -> {validation | testing | debugging | judge | reporting | approval | human_help | END}`. `runner_identity` only records Kali IPs for target exclusion. Validation, testing, and debugging contain nested `reason -> tools -> reason` ReAct subgraphs. Every specialist returns to `supervisor`; no specialist can select a different specialist. The supervisor chooses the next specialist from current remote evidence, unresolved goals, failures, and repair history. The durable observation ledger records effective tool/target/argv identities; a repeated identity sends control back to Supervisor for replanning, while `--max-iterations` is telemetry only. Human input is reserved for an explicit stop, an approval boundary, or a genuinely exhausted autonomous path. `MemorySaver` is the default checkpoint backend; production deployments can inject a durable checkpointer.

## Neo4j graph schema

Nodes: `Host`, `Service`, `User`, `Credential`, `Vulnerability`, `AttackPath`, `Flag`.
Relationships: `(:Host)-[:EXPOSES]->(:Service)`, `(:User)-[:CAN_AUTHENTICATE_WITH]->(:Credential)`, `(:Host)-[:TRUSTS|REACHES]->(:Host)`, `(:AttackPath)-[:STARTS_AT|USES|REQUIRES|ENDS_AT]->(any)`, `(:Host)-[:HAS_VULNERABILITY]->(:Vulnerability)`.

Upserts are idempotent and are emitted from observations/events, not from agent prose. See `KnowledgeGraphRepository.upsert_observation`.

## Approval policy

Service restart, firewall/DNS/route corrections, package installation, and time synchronization are autonomous. Domain/forest/ADCS rebuilds, GPO replacement, credential reset, user deletion, and other destructive mutations emit an approval request and suspend at the approval node. An approval resumes the frozen decision once; sensitive tools additionally require the matching capability and exact tool parameters. The approval record includes the exact action, scope, reason, expected impact, rollback, and evidence.

## Security posture

This package is an orchestration skeleton for controlled cyber ranges. Keep credentials in a secret manager, restrict tool identities, require allow-listed targets, log immutable evidence, and run destructive actions only behind the approval gate.
