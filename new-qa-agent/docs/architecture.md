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

In production, bind `Agents._reason` to a structured-output model (`with_structured_output(Decision)` for the supervisor and role-specific proposal models for specialists). The fallback is intentionally conservative and only observes.

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
supervisor: ADCS rebuild is highest-value repair; policy=approval_required
approval: APPROVAL_REQUIRED request_id=... status=pending
human: approve/reject outside the specialist loop
resume: supervisor re-evaluates with the approval decision and fresh evidence
```

To suspend/resume with LangGraph's production checkpointer, compile with a checkpointer and use `interrupt()` in `approval`; the skeleton keeps approval as an explicit node so deployments can select their approval transport (UI, ticketing, or signed API) without changing specialist logic.
