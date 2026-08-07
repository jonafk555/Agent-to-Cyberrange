# Cyber Range QA — Cochise core

This project is Cochise's autonomous red-team loop plus optional Cyber Range
QA helpers. It is intended for an explicitly authorized, isolated lab. The
planner is allowed to choose the next red-team task, the executor is allowed
to choose and run the next command through SSH, and every result is fed back
into the next planning decision.

~~~text
scenario + optional range reference
              ↓
Planner creates/evolves an attack plan
              ↓
Executor proposes a hypothesis and calls execute_command
              ↓
SSH attacker VM returns real output
              ↓
Knowledge stores findings and discovered credentials
              ↓
Planner selects the next task or asks the human
~~~

There is no LangChain/LangGraph layer and no fixed QA graph. LiteLLM is the
provider adapter, so the same agent can use OpenAI, Claude, Gemini, Ollama, or
another OpenAI-compatible local model.

## Run on Kali

~~~bash
cd /path/to/new-qa-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
~~~

Set the LLM and the SSH target in .env:

~~~dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=...

TARGET_HOST=192.168.56.100
TARGET_USERNAME=root
TARGET_PASSWORD=kali
~~~

The target is the authorized range host reached by the persistent SSH
connection. The agent does not assume that every range has the same host,
domain, file layout, or specification.

~~~bash
python -m cyberqa.main
~~~

Each invocation creates a new directory:

~~~text
runs/<scenario>_<UTC timestamp>_<id>/
├── logs/run-<timestamp>.json
├── run-meta.json
├── report.md
└── qa-assessment.md       # only when --spec is supplied
~~~

The JSON log contains the raw Cochise event stream: configuration, planner and
executor history, tool calls, tool results, knowledge updates, and completion
events. Knowledge intentionally retains the credentials that the agent records
through add_compromised_account, so protect the run directory.

## Providers

OpenAI:

~~~dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=...
~~~

Claude:

~~~dotenv
LLM_PROVIDER=claude
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=...
~~~

Gemini:

~~~dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=...
~~~

Ollama:

~~~dotenv
LLM_PROVIDER=local
LLM_MODEL=llama3.1
LOCAL_LLM_BACKEND=ollama
LLM_BASE_URL=http://127.0.0.1:11434
~~~

For LM Studio, vLLM, or another OpenAI-compatible endpoint, set
LOCAL_LLM_BACKEND=openai-compatible, LLM_MODEL, and LLM_BASE_URL.

## Optional Cyber Range QA context

--spec path/to/range.yaml accepts any JSON/YAML environment reference. The
loader recognizes common declarative assertion keys such as assertions,
checks, tests, and requirements, but the file is never converted into
commands and never becomes a tool allowlist. It is added as informational
context for the Cochise scenario. After the run, qa-assessment.md provides a
conservative post-run triage appendix; it does not gate exploitation or claim
that keyword overlap proves a pass.

WIN-2024-010-AKAIRYU_spec.yaml is only a sample fixture. No range-specific
schema is required.

If a requested scenario/spec file is unavailable, the agent pauses and asks the
operator for a path, permission to continue, or stop. The executor also asks
the human when command output shows that a required artifact is missing or when
it has exhausted recovery attempts.

## Architecture

- src/cyberqa/planner.py: persistent plan, task selection, re-planning, and
  cross-task knowledge merge.
- src/cyberqa/executor.py: short-lived task executor with unrestricted
  Cochise-style execute_command, knowledge tools, and human recovery.
- src/cyberqa/ssh_connection.py: persistent SSH connection to the target.
- src/cyberqa/knowledge.py: in-memory red-team knowledge including plaintext
  account/password/hash material recorded by the model.
- src/cyberqa/common.py: provider-neutral LiteLLM calls and tool schemas.
- src/cyberqa/qa_extensions.py: optional, non-gating range specification and
  QA appendix support.
- src/cyberqa/templates/scenario.md: red-team AD scenario and operating
  rules, including credential attacks, Kerberos/AD paths, and artifact
  recovery guidance.

The original QA-only allowlist, fixed tool registry, approval broker,
assertion gate, and graph orchestration modules were removed from the
execution path. All external actions now follow the Cochise tool-calling
loop.

## Replay and analysis

~~~bash
cyberqa-replay runs/<run>/logs/run-<timestamp>.json
cyberqa-analyze-logs index-rounds runs/<run>/logs/run-<timestamp>.json
cyberqa-analyze-graphs runs/<run>/logs/run-<timestamp>.json
~~~

Use the raw log and report only inside the authorized engagement boundary.
