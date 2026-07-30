"""Offline trace showing dynamic supervisor decisions and an approval boundary."""
import asyncio
from uuid import uuid4

from cyberqa.graph import build_graph
from cyberqa.llm import build_llm
from cyberqa.models import Role
from cyberqa.nodes import Agents
from cyberqa.tools import DryRunTool, ToolRegistry


async def main():
    tools = ToolRegistry({"validation": DryRunTool("validation", {"port": 389, "reachable": True, "functional": True}),
                          "testing": DryRunTool("testing", {"expected": ["request_tgt", "use_ticket"], "observed": ["request_tgt", "use_ticket"], "result": "passed"}),
                          "debugging": DryRunTool("debugging", {"hypothesis": "DNS forwarder unavailable", "verified": True})})
    app = build_graph(Agents(llm=build_llm(), tools=tools))
    state = {"run_id": str(uuid4()), "scenario_id": "ad-lab-01", "objective": "Validate LDAP, test an attack path, and score the scenario", "iteration": 0, "max_iterations": 5, "hosts": {}, "evidence": [], "events": [], "approvals": [], "action_history": [], "completed_goals": [], "errors": [], "memory": {}}
    result = await app.ainvoke(state)
    for event in result.get("events", []):
        print(event.type, event.target)
    print("approval requests:", len(result.get("approvals", [])))


if __name__ == "__main__":
    asyncio.run(main())
