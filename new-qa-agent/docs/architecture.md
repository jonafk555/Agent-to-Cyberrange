# Cochise core plus Cyber Range QA extensions

The execution semantics are Cochise. Cyber Range QA is an optional
observational layer around that loop, not a competing planner or permission
system.

## Control loop

~~~text
Red-team scenario + optional range reference
                    ↓
Planner creates or compacts the attack plan
                    ↓
Planner calls perform_task or asks the human
                    ↓
Fresh Executor reasons about one task
                    ↓
Executor calls execute_command over persistent SSH
                    ↓
Real output becomes the next tool message
                    ↓
Knowledge merges credentials, entities, and findings
                    ↓
Planner selects the next task
~~~

The Planner owns long-lived history and knowledge. Every task receives a fresh
Executor, so the Executor can focus on the current hypothesis while still
seeing the accumulated findings. This is the mechanism that lets the agent
change direction after a failed command or a newly discovered credential.

## Cochise contracts

Planner tools:

- perform_task: assign one concrete red-team task to a fresh Executor;
- ask_human: pause when a required artifact is missing, an approach is
  exhausted, or the next decision needs operator input;
- add/update compromised account;
- add/update entity information.

Executor tools:

- execute_command: arbitrary command execution on the configured SSH target,
  annotated with MITRE technique/procedure;
- ask_human;
- the four knowledge update functions.

The LLM receives tool results as tool messages and continues the
observe-act loop. Prompt text is guidance; the structured tool mapping and
program control loop are the runtime contract.

## Knowledge

Knowledge is the original Cochise in-memory model. It keeps compromised
accounts with username, password or hash, context, and a dirty flag. It also
keeps entity information and merges dirty findings back into the persistent
Planner. The run JSON log and final report therefore contain sensitive
engagement material; protect the run directory and do not use this mode
against an unauthorized system.

## Red-team scenario

src/cyberqa/templates/scenario.md contains the AD-oriented operating rules
ported from Cochise: password spraying, AS-REP roasting, Kerberos and NTLM
paths, credential/hash handling, attacker VM tooling, artifact recovery, and
bounded retry guidance. The target network is not hardcoded. Target host,
account, and any range-specific assumptions come from the environment and
optional operator context.

## QA extension

src/cyberqa/qa_extensions.py can load arbitrary JSON/YAML range data. It
extracts common declarative assertion lists and presents the remaining
environment metadata to the scenario as reference context. It deliberately
does not:

- create a fixed tool registry;
- expand the authorized target;
- turn commands in a document into executable actions;
- stop the Planner when an assertion is incomplete;
- redact the Cochise knowledge model.

At the end of a run it writes qa-assessment.md. The assessment is a
post-run triage appendix with observed/unverified labels; it is not a
replacement for raw command evidence or the red-team report.

## Artifacts and replay

The CLI creates a unique run directory before connecting to the target:

~~~text
<runs-root>/<scenario>_<UTC timestamp>_<short id>/
├── logs/run-<timestamp>.json
├── run-meta.json
├── report.md
└── qa-assessment.md
~~~

The JSON file is the Cochise event stream and is compatible with the replay
and analysis commands. report.md includes the scenario, the final planner
output, and the full knowledge rendering. The optional QA appendix is kept
separate so it cannot overwrite red-team findings.

## Human-in-the-loop

The human interaction implementation is used by both Planner and Executor.
The Executor detects missing artifact messages from command output and asks
for a path or guidance. After bounded recovery attempts it asks the human
again instead of silently inventing a file or assuming that a failed method
worked. Reply stop, quit, exit, abort, or cancel to end the run.
