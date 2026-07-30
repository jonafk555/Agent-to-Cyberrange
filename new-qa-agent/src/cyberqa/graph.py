from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .models import Role
from .nodes import Agents
from .state import QAState


def route(state: QAState) -> str:
    decision = state.get("last_decision")
    if not decision or decision.next_agent == "end":
        return END
    if decision.approval_required:
        return "approval"
    return decision.next_agent.value


def build_graph(agents: Agents | None = None):
    agents = agents or Agents()
    graph = StateGraph(QAState)
    graph.add_node("supervisor", agents.supervisor)
    graph.add_node("validation", lambda s: agents.specialist(Role.VALIDATION, s))
    graph.add_node("testing", lambda s: agents.specialist(Role.TESTING, s))
    graph.add_node("debugging", lambda s: agents.specialist(Role.DEBUGGING, s))
    graph.add_node("judge", lambda s: agents.specialist(Role.JUDGE, s))
    graph.add_node("reporting", lambda s: agents.specialist(Role.REPORTING, s))
    graph.add_node("approval", agents.approval)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route, {"validation":"validation", "testing":"testing", "debugging":"debugging", "judge":"judge", "reporting":"reporting", "approval":"approval", END:END})
    for node in ("validation", "testing", "debugging", "judge", "reporting", "approval"):
        graph.add_edge(node, "supervisor")
    return graph.compile()
