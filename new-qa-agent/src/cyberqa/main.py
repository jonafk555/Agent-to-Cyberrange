from __future__ import annotations

import asyncio
from uuid import uuid4

from .graph import build_graph
from .llm import build_llm
from .nodes import Agents


async def run() -> None:
    # This is the LLM API connection point. Without OPENAI_API_KEY the package
    # deliberately runs in offline, observe-only fallback mode.
    app = build_graph(Agents(llm=build_llm()))
    result = await app.ainvoke({"run_id": str(uuid4()), "scenario_id": "demo", "objective": "Validate the range and produce a QA scorecard", "iteration": 0, "max_iterations": 8, "hosts": {}, "evidence": [], "events": [], "approvals": [], "action_history": [], "completed_goals": [], "errors": [], "memory": {}})
    print({"iterations": result.get("iteration"), "events": [e.type for e in result.get("events", [])], "scorecard": result.get("scorecard")})


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
