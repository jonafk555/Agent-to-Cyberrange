from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .models import Role
from .nodes import Agents
from .routing import entry_route, route, route_after_agent, route_after_approval
from .state import QAState

__all__ = ["build_graph", "entry_route", "route", "route_after_agent", "route_after_approval"]


def _memory_saver() -> MemorySaver:
    # State intentionally contains domain models and enums. Register only
    # those application types instead of allowing arbitrary msgpack imports.
    allowed = [
        ("cyberqa.models", name) for name in (
            "ADKnowledge", "ADRisk", "ApprovalRequest", "AttackPath", "CapabilitySpec",
            "Decision", "Evidence", "Event", "Host", "Hypothesis", "Role", "Scorecard", "ToolParameters",
            "Service", "ServiceProtocol",
        )
    ]
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))


def build_graph(agents: Agents | None = None, checkpointer=None):
    agents = agents or Agents()
    graph = StateGraph(QAState)
    graph.add_node("supervisor", agents.supervisor)
    graph.add_node("initial_recon", agents.initial_recon)

    async def validation(s):
        return await agents.specialist(Role.VALIDATION, s)

    async def testing(s):
        return await agents.specialist(Role.TESTING, s)

    async def debugging(s):
        return await agents.specialist(Role.DEBUGGING, s)

    async def judge(s):
        return await agents.specialist(Role.JUDGE, s)

    async def reporting(s):
        return await agents.specialist(Role.REPORTING, s)

    graph.add_node("validation", validation)
    graph.add_node("testing", testing)
    graph.add_node("debugging", debugging)
    graph.add_node("judge", judge)
    graph.add_node("reporting", reporting)
    graph.add_node("approval", agents.approval)
    graph.add_node("human_help", agents.human_help)
    graph.add_conditional_edges(START, entry_route, {"initial_recon": "initial_recon", "supervisor": "supervisor"})
    graph.add_conditional_edges("initial_recon", route_after_agent, {"human_help": "human_help", "supervisor": "supervisor"})
    graph.add_conditional_edges("supervisor", route, {"validation":"validation", "testing":"testing", "debugging":"debugging", "judge":"judge", "reporting":"reporting", "approval":"approval", "human_help":"human_help", END:END})
    for node in ("validation", "testing", "debugging", "judge", "reporting"):
        graph.add_conditional_edges(node, route_after_agent, {"human_help": "human_help", "supervisor": "supervisor"})
    graph.add_conditional_edges(
        "approval", route_after_approval,
        {"validation": "validation", "testing": "testing", "debugging": "debugging",
         "judge": "judge", "reporting": "reporting", "human_help": "human_help", END: END},
    )
    graph.add_edge("human_help", "supervisor")
    return graph.compile(checkpointer=checkpointer or _memory_saver())
