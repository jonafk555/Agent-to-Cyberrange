from __future__ import annotations

import asyncio
import argparse
import os
import shlex
from uuid import uuid4
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from .graph import build_graph
from .llm import build_llm
from .nodes import Agents
from .tools import build_kali_registry


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
        output = (data.get("stdout") or data.get("stderr") or "").strip().replace("\n", " ")
        print(f"[Tool] 結果 exit={data['exit_code']} | {output[:240]}", flush=True)
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
    parser.add_argument("--target", default="127.0.0.1", help="Authorized lab target")
    parser.add_argument("--objective", default="Validate the range and produce a QA scorecard")
    parser.add_argument("--scenario-id", default="demo")
    parser.add_argument("--max-iterations", type=int, default=8)
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
    config = {"configurable": {"thread_id": str(uuid4())}}

    async def stream_graph(input_state):
        """Stream graph updates while the LLM remains the decision-maker."""
        interrupt_value = None
        async for update in app.astream(input_state, config, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            if "__interrupt__" in update:
                interrupt_value = update["__interrupt__"]
                continue
            for node, patch in update.items():
                if node in {"supervisor", "validation", "testing", "debugging", "judge", "reporting"}:
                    print(f"[Graph] {node} node completed; state updated", flush=True)
        snapshot = await app.aget_state(config)
        # LangGraph versions differ: some expose interrupts in stream updates,
        # others expose them only on pending task metadata in the checkpoint.
        if interrupt_value is None:
            for task in getattr(snapshot, "tasks", ()):
                pending = getattr(task, "interrupts", ())
                if pending:
                    interrupt_value = pending
                    break
        return snapshot.values, interrupt_value

    async def execute_task(objective: str, target: str, scenario_id: str):
        initial = {"run_id": str(uuid4()), "scenario_id": scenario_id, "objective": objective,
                   "target": target, "iteration": 0, "max_iterations": args.max_iterations,
                   "hosts": {}, "evidence": [], "events": [], "approvals": [], "action_history": [],
                   "completed_goals": [], "errors": [], "memory": {}, "human_requests": [],
                   "react_steps": 0, "needs_human": False, "aborted": False,
                   "no_progress_count": 0,
                   "messages": [HumanMessage(content=objective)]}
        try:
            result, interrupt_value = await stream_graph(initial)
        except Exception as exc:
            print(f"\n[Graph error] {type(exc).__name__}: {exc}", flush=True)
            print("此任務已停止，但互動 session 仍可繼續輸入下一個任務。", flush=True)
            return {"iteration": 0, "events": [], "evidence": [], "errors": [str(exc)]}
        while interrupt_value:
            print("\n[Human input required]", interrupt_value, flush=True)
            answer = input("你：").strip()
            if answer.lower() in {"quit", "exit"}:
                return None
            try:
                result, interrupt_value = await stream_graph(Command(resume=answer))
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
        objective = input("\n你：").strip()
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
