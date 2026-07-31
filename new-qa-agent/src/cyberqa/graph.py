from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .models import Role
from .nodes import Agents
from .state import QAState


def route(state: QAState) -> str:
    if state.get("aborted"):
        return END
    if state.get("needs_human"):
        return "human_help"
    decision = state.get("last_decision")
    if not decision or decision.next_agent == "end":
        return END
    if decision.approval_required:
        return "approval"
    return decision.next_agent.value


def build_graph(agents: Agents | None = None, checkpointer=None):
    agents = agents or Agents()
    graph = StateGraph(QAState)
    graph.add_node("supervisor", agents.supervisor)

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
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route, {"validation":"validation", "testing":"testing", "debugging":"debugging", "judge":"judge", "reporting":"reporting", "approval":"approval", "human_help":"human_help", END:END})
    for node in ("validation", "testing", "debugging", "judge", "reporting", "approval"):
        graph.add_edge(node, "supervisor")
    graph.add_edge("human_help", "supervisor")
    return graph.compile(checkpointer=checkpointer or MemorySaver())
