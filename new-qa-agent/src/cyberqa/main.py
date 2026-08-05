from __future__ import annotations

import asyncio
import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from dotenv import dotenv_values, load_dotenv

from .graph import build_graph
from .llm import build_llm
from .nodes import Agents
from .tools import build_kali_registry, summarize_output


load_dotenv()
_discovered_env = os.getenv("CYBERQA_DISCOVERED_ENV", ".cyberqa/discovered.env")
load_dotenv(_discovered_env, override=False)
for _key, _value in dotenv_values(_discovered_env).items():
    # A blank value in .env should not hide a previously discovered safe value.
    if _value and not os.getenv(_key):
        os.environ[_key] = _value


def write_initial_recon_report(values: dict, scenario_id: str) -> str:
    report_dir = Path(os.getenv("CYBERQA_REPORT_DIR", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{scenario_id}-initial-recon.md"
    evidence = [item for item in values.get("evidence", []) if getattr(item, "action", "") == "initial_recon"]
    lines = [f"# Cyber Range Initial Reconnaissance — {scenario_id}", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}",
             f"Target: `{values.get('target', 'not specified')}`", "",
             "This report contains observed facts from the mandatory baseline reconnaissance phase.", ""]
    expected_keys = (
        "CYBERQA_EXPECTED_LOCAL_USERS", "CYBERQA_EXPECTED_DOMAIN_USERS",
        "CYBERQA_EXPECTED_PRIVILEGED_GROUPS", "CYBERQA_EXPECTED_OPEN_PORTS",
        "CYBERQA_EXPECTED_NETWORKS",
    )
    lines.extend(["## Expected QA baseline", "", "Configured expectations are compared by the QA Agent in later analysis:", ""])
    for key in expected_keys:
        lines.append(f"- `{key}`: `{os.getenv(key, '')}`")
    lines.append("")
    lines.extend(["## Runtime configuration discovered by the Agent", "",
                  "Only non-secret values are persisted; credentials are never inferred or written here.", ""])
    for key, value in sorted(values.get("runtime_config", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Target and domain profiles", "", "```json",
                  json.dumps(values.get("target_profiles", {}), indent=2, ensure_ascii=False, default=str),
                  "```", "", "## Evidence synthesis", "", "```json",
                  json.dumps(values.get("evidence_synthesis", {}), indent=2, ensure_ascii=False, default=str),
                  "```", ""])
    for item in evidence:
        lines.extend([
            f"## {item.source} — `{item.target}`",
            f"- Exit code: `{item.exit_code}`",
            f"- Observed at: `{item.observed_at.isoformat()}`",
            "",
            "### Facts",
            "```json",
            json.dumps(item.facts, indent=2, ensure_ascii=False, default=str),
            "```",
            "",
            "### stdout",
            "```text",
            item.stdout or "",
            "```",
            "",
            "### stderr",
            "```text",
            item.stderr or "",
            "```",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
from .models import ADKnowledge


def interrupt_payload(value) -> dict:
    """Convert LangGraph's version-dependent Interrupt wrapper to a dict."""
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    value = getattr(value, "value", value)
    return value if isinstance(value, dict) else {"question": str(value)}


def read_human_input(prompt: str = "你：") -> str:
    """Read from the controlling terminal even when stdin is redirected."""
    print(prompt, end="", flush=True)
    stream = sys.stdin
    tty = None
    try:
        if not getattr(stream, "isatty", lambda: False)():
            try:
                tty = open("/dev/tty", "r", encoding="utf-8", errors="replace")
                stream = tty
            except OSError:
                pass
        line = stream.readline()
        if line == "":
            raise EOFError
        return line.strip()
    finally:
        if tty is not None:
            tty.close()


def print_progress(event: str, data: dict) -> None:
    if event == "reasoning_start":
        print(f"[{data['agent']}] 正在讀取對話與 evidence，分析下一步...", flush=True)
    elif event == "supervisor_decision":
        print(f"\n[Supervisor] -> {data['agent']} | action={data['action']} | target={data['target']}", flush=True)
    elif event == "reasoned":
        tools = ", ".join(data.get("tool_calls", [])) or "完成目前推理"
        print(f"[{data['agent']}] 分析完成，下一步：{tools}", flush=True)
    elif event == "tool_start":
        print(f"[Tool] 執行：{' '.join(shlex.quote(x) for x in data['argv'])}", flush=True)
    elif event == "tool_result":
        raw = "\n".join(
            value for value in (data.get("stdout", ""), data.get("stderr", "")) if value
        )
        preview = summarize_output(raw, max_lines=16, max_chars=2400)
        line_count = len([line for line in raw.splitlines() if line.strip()])
        print(f"[Tool] 結果 exit={data['exit_code']} | lines={line_count}", flush=True)
        if preview:
            print(preview, flush=True)
    elif event == "tool_cached":
        print(f"[Tool] 快取命中：{data['tool']} target={data['target']} "
              f"signature={data['signature']}", flush=True)
    elif event == "target_discovered":
        print(f"[Target] 發現並加入授權清單：{data['target']}", flush=True)
    elif event == "agent_done":
        print(f"[{data['agent']}] 回報 {data['evidence_count']} 筆 evidence，返回 Supervisor", flush=True)
    elif event == "agent_error":
        print(f"[{data['agent']}] Agent error，已記錄 evidence：{data['error']}", flush=True)
    elif event == "event_error":
        print(f"[EventBus] {data['event_type']} 發布失敗，流程繼續：{data['error']}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cyber-range QA ReAct multi-agent")
    parser.add_argument("--target", default=os.getenv("CYBERQA_TARGET", "127.0.0.1"), help="Authorized lab target")
    parser.add_argument("--objective", default=os.getenv("CYBERQA_OBJECTIVE", "Validate the range and produce a QA scorecard"))
    parser.add_argument("--scenario-id", default=os.getenv("CYBERQA_SCENARIO_ID", "demo"))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("CYBERQA_MAX_ITERATIONS", "8")))
    parser.add_argument("--allowed-targets", default=None,
                        help="Comma-separated allowlist; defaults to CYBERQA_ALLOWED_TARGETS")
    parser.add_argument("--once", action="store_true",
                        help="Run one task and exit instead of opening an interactive session")
    return parser.parse_args()


async def run(args: argparse.Namespace | None = None) -> None:
    # This is the LLM API connection point. Without OPENAI_API_KEY the package
    # deliberately runs in offline, observe-only fallback mode.
    args = args or parse_args()
    if args.allowed_targets:
        configured_targets = {item.strip() for item in args.allowed_targets.split(",") if item.strip()}
    else:
        configured_targets = {item.strip() for item in os.getenv("CYBERQA_ALLOWED_TARGETS", "").split(",") if item.strip()}
    # An explicit --target is an operator authorization for this run. Preserve
    # configured ranges, but never make the selected target fail its own policy.
    configured_targets.add(args.target)
    os.environ["CYBERQA_ALLOWED_TARGETS"] = ",".join(sorted(configured_targets))
    app = build_graph(Agents(llm=build_llm(), tools=build_kali_registry(on_event=print_progress),
                             on_progress=print_progress))
    async def stream_graph(input_state, task_config):
        """Stream graph updates while the LLM remains the decision-maker."""
        interrupt_value = None
        async for update in app.astream(input_state, task_config, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            if "__interrupt__" in update:
                interrupt_value = update["__interrupt__"]
                continue
            for node, patch in update.items():
                if node == "initial_recon":
                    snapshot = await app.aget_state(task_config)
                    report_path = write_initial_recon_report(snapshot.values, snapshot.values.get("scenario_id", "scenario"))
                    print(f"[Recon] baseline report written: {report_path}", flush=True)
                if node in {"initial_recon", "supervisor", "validation", "testing", "debugging", "judge", "reporting"}:
                    print(f"[Graph] {node} node completed; state updated", flush=True)
        snapshot = await app.aget_state(task_config)
        # LangGraph versions differ: some expose interrupts in stream updates,
        # others expose them only on pending task metadata in the checkpoint.
        if interrupt_value is None:
            for task in getattr(snapshot, "tasks", ()):
                pending = getattr(task, "interrupts", ())
                if pending:
                    interrupt_value = pending
                    break
        if interrupt_value is None and "approval" in getattr(snapshot, "next", ()):
            interrupt_value = {
                "kind": "approval",
                "request": snapshot.values.get("pending_action", {}),
                "question": "Approve this action? Reply approve or reject.",
            }
        # Some LangGraph releases expose the state flag but omit interrupt
        # metadata from the snapshot. Keep the CLI usable in that case, but
        # mark it synthetic so resume uses a fresh graph input instead of an
        # invalid Command(resume=...) call.
        if interrupt_value is None and snapshot.values.get("needs_human"):
            interrupt_value = {
                "kind": "human_help",
                "question": "Agent 需要你的下一步指示。",
                "options": ["retry_with_correction", "validation", "testing", "debugging", "abort"],
                "synthetic": True,
            }
        return snapshot.values, interrupt_value

    async def execute_task(objective: str, target: str, scenario_id: str):
        task_config = {"configurable": {"thread_id": str(uuid4())}}
        initial = {"run_id": str(uuid4()), "scenario_id": scenario_id, "objective": objective,
                   "target": target, "iteration": 0, "max_iterations": args.max_iterations,
                   "hosts": {}, "evidence": [], "events": [], "approvals": [], "action_history": [],
                   "method_history": [],
                   "completed_goals": [], "errors": [], "memory": {}, "human_requests": [],
                   "react_steps": 0, "needs_human": False, "aborted": False,
                   "baseline_complete": False, "approved_grant": None,
                   "no_progress_count": 0,
                   "discovered_targets": [target], "recon_coverage": {},
                   "ad_knowledge": ADKnowledge().model_dump(), "capability_history": [],
                   "target_profiles": {}, "evidence_synthesis": {}, "runtime_config": {},
                   "messages": [HumanMessage(content=objective)]}
        try:
            result, interrupt_value = await stream_graph(initial, task_config)
        except Exception as exc:
            print(f"\n[Graph error] {type(exc).__name__}: {exc}", flush=True)
            print("此任務已停止，但互動 session 仍可繼續輸入下一個任務。", flush=True)
            return {"iteration": 0, "events": [], "evidence": [], "errors": [str(exc)]}
        while interrupt_value:
            request = interrupt_payload(interrupt_value)
            print("\n[Human input required]", flush=True)
            if request.get("problem"):
                print(f"問題摘要：{request['problem']}", flush=True)
            if request.get("evidence_summary"):
                print(f"相關證據：{request['evidence_summary']}", flush=True)
            print(f"請回覆你的處置方向（可用自然語言）：{request.get('question', '請提供下一步指示')}", flush=True)
            try:
                answer = read_human_input()
            except EOFError:
                print("\n[Input closed] 未收到人類指示，任務安全停止。", flush=True)
                return {"iteration": 0, "events": [], "evidence": [], "errors": ["human input closed"]}
            if answer.lower() in {"quit", "exit"}:
                return None
            try:
                if request.get("synthetic"):
                    result, interrupt_value = await stream_graph({
                        "needs_human": False,
                        "no_progress_count": 0,
                        "action_history": [],
                        "messages": [HumanMessage(content=f"Human guidance: {answer}")],
                    }, task_config)
                else:
                    result, interrupt_value = await stream_graph(Command(resume=answer), task_config)
            except Exception as exc:
                print(f"\n[Resume error] {type(exc).__name__}: {exc}", flush=True)
                return {"iteration": 0, "events": [], "evidence": [], "errors": [str(exc)]}
        print(f"\n[Task completed] iterations={result.get('iteration')} "
              f"events={len(result.get('events', []))} evidence={len(result.get('evidence', []))}", flush=True)
        return result

    result = await execute_task(args.objective, args.target, args.scenario_id)
    if args.once or result is None:
        return

    print("\n互動模式已啟動。輸入下一個任務；輸入 exit 或 quit 離開。", flush=True)
    while True:
        objective = read_human_input("\n你：")
        if objective.lower() in {"quit", "exit", "q"}:
            print("Agent session ended.")
            return
        if not objective:
            continue
        result = await execute_task(objective, args.target, args.scenario_id)
        if result is None:
            return


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
