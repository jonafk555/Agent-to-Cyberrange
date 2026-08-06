# Implementation notes

## Dynamic control loop

The compiled graph is intentionally small because the behavior is not encoded in edge order:

```text
                 +------------------+
                 |    supervisor    |  (only controller)
                 +---------+--------+
                           |
        +------------------+------------------+
        |                  |                  |
   validation         testing/debugging   judge/reporting
        |                  |                  |
        +------------------+------------------+
                           |
                    approval (policy)
                           |
                       supervisor
```

The supervisor receives current evidence, unresolved goals, prior action signatures, hypotheses, repair history, and the scenario objective. It returns exactly one `Decision`. A specialist can collect facts and emit a proposal, but cannot choose the next node. The `route` function only interprets the supervisor's decision.

## Assertion and evidence sufficiency control

The Supervisor is not driven by a fixed attack chain. A task contains `QAAssertion` records with a visibility mode and required evidence threshold. `EvidenceSufficiency` compares the assertion against durable evidence and exposes the least-invasive remaining methods:

```text
QA objective/specification
        ↓
QAAssertion (what must be answered, target, required C-level)
        ↓
Evidence + EvidenceOpportunity memory
        ↓
EvidenceSufficiency (current level, missing facts, next methods)
        ↓
Supervisor selects one distinct authorized Decision
        ↓
Tool Gateway / approval / audit
```

The levels are C0 Unknown, C1 Inferred, C2 Enumerated, C3 Functionally Verified, C4 Exploitability Verified, and C5 End-to-End Verified. The completion gate uses these thresholds for new tasks: an assertion that only needs configuration evidence can finish at C2, while an explicitly requested end-to-end attack-path assertion remains open until C5. This prevents an available credential-material or exploit tool from becoming an automatic escalation.

## AD decision contract

AD method selection has a deterministic prerequisite/completion guard between the model and the broker. The model remains the Supervisor and may choose any concrete safe non-terminal path; the guard only prevents unsafe prerequisites, premature terminal transitions, and repeated no-op planning:

```text
domain/DC + username source + no credential
    -> Supervisor analyzes each result and ranks available capabilities
    -> AS-REP, local hash cracking, credential validation, or another justified path
       (only when its evidence and prerequisites support it)
    -> remaining bounded identity/recon evidence
    -> Supervisor chooses the next unresolved path
    -> judge/report only after the completion gate
domain/DC + no credential + no username source
    -> one bounded anonymous LDAP/SMB/NXC-LDAP identity phase
    -> username source found: AS-REP
    -> no username source: human help, no guessed accounts or empty-credential loop
validated credential
    -> user/SPN enumeration
    -> Kerberoast only when SPNs exist
    -> bounded relationship collection
    -> judge/report
```

Each capability is mapped to one primary reviewed adapter. An approval grant freezes the target, action, capability, parameters, and allowed adapter set for one specialist dispatch. The grant is consumed when that dispatch returns. A cached observation is evidence that the method was already attempted; it is not permission to silently retry it.

`method_history` records the effective tool, target, outcome, argv, and evidence id. The supervisor receives this ledger and the deterministic guard rejects an immediately repeated decision. Failure categories remain distinct: authentication/bind failure, missing username source, invalid arguments, connectivity failure, and tool/runtime failure each produce a different operator-facing next step.

Each fresh tool result also passes through a post-tool evidence analysis. The analysis is intentionally compact and safe: it records usable content, unresolved questions, candidate reviewed tools, and an evidence-backed suggested next action. It is shown to the operator and stored in `evidence_analyses` for the Supervisor, but remains advisory so the Supervisor can reason across capabilities instead of following a hard-coded pipeline. Cache hits reuse prior evidence without another analysis/model call.

In production, bind `Agents._reason` to a structured-output model (`with_structured_output(Decision, method="function_calling")` for the supervisor and role-specific proposal models for specialists). The fallback is intentionally conservative and only observes.

## Validation contract

Validation is a three-part assertion for every service:

```json
{"running": true, "reachable": true, "functional": true}
```

The service is valid only when all three facts are supported by evidence. Examples: LDAP must complete a bind/search; Kerberos must obtain/validate a ticket; SMB must negotiate and list an authorized share; WinRM must authenticate and execute a harmless command; databases must establish a protocol-level connection; HTTP must return an expected health or challenge marker.

## Attack-path contract

Testing emits an `AttackPath` with `expected_steps`, `observed_steps`, `result`, and `evidence_ids`. It may add `alternatives`; the judge compares path length, prerequisites, and expected flag reachability. A path that is technically exploitable but materially exceeds the intended complexity is `degraded` rather than silently passing.

## Debugging contract

The debugging agent keeps hypotheses as first-class state. A repair is not successful because a command returned zero: it must produce `REPAIR_COMPLETED`, then validation must re-establish the affected functional assertion. The recommended event progression is:

```text
LDAP_FAILED -> HYPOTHESIS_GENERATED -> EVIDENCE_COLLECTED
-> HYPOTHESIS_VERIFIED -> REPAIR_COMPLETED -> SERVICE_VALIDATED
```

## Event envelope and RabbitMQ

All events use `Event`: `id`, `type`, `run_id`, `emitted_by`, `timestamp`, `target`, `evidence_ids`, `payload`. Publish to topic exchange `cyberqa.events` with routing keys such as `LDAP_FAILED`, `DNS_FAILED`, `KERBEROS_FAILED`, `FLAG_RETRIEVED`, `ATTACK_PATH_VALIDATED`, `SCENARIO_FAILED`, `REPAIR_COMPLETED`, and `APPROVAL_REQUIRED`. Consumers update projections (Redis), the Neo4j graph, dashboards, and immutable audit storage.

## Example traces

### Attack-path validation

```text
supervisor: missing evidence for intended Kerberoast path -> testing/request_service_tickets
testing: expected=[enumerate_spn, request_tgs, crack_or_validate] observed=[enumerate_spn, request_tgs, validate]
testing: ATTACK_PATH_VALIDATED result=passed evidence=[nmap/ldap/impacket]
supervisor: compare path complexity -> judge
judge: solvable=true difficulty=appropriate score=86
```

### Debugging and repair

```text
supervisor: LDAP functional assertion failed -> debugging/generate_hypotheses
debugging: DNS forwarder, time skew, firewall hypotheses
debugging: evidence ranks DNS forwarder first -> debugging/correct_dns
debugging: REPAIR_COMPLETED (autonomous action)
supervisor: revalidate LDAP bind/search -> validation
validation: SERVICE_VALIDATED functional=true
```

### Approval workflow

```text
supervisor: AS-REP or another credential-material capability is selected; policy=approval_required
approval: APPROVAL_REQUIRED request_id=... status=pending
human: approve/reject outside the specialist loop
resume: dispatch the frozen approved decision once; tools verify target, capability, allowed adapter, and exact parameters
```

To suspend/resume with LangGraph's production checkpointer, compile with a checkpointer and use `interrupt()` in `approval`; the skeleton keeps approval as an explicit node so deployments can select their approval transport (UI, ticketing, or signed API) without changing specialist logic.
