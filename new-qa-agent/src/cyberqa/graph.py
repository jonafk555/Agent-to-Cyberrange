from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .models import Role
from .nodes import Agents
from .state import QAState


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
    return decision.next_agent.value if isinstance(decision.next_agent, Role) else decision.next_agent


def entry_route(state: QAState) -> str:
    return "supervisor" if state.get("baseline_complete") else "initial_recon"


def route_after_agent(state: QAState) -> str:
    return "human_help" if state.get("needs_human") else "supervisor"


def route_after_approval(state: QAState) -> str:
    if state.get("aborted"):
        return END
    decision = state.get("last_decision")
    if not decision or not isinstance(decision.next_agent, Role):
        return "human_help"
    return decision.next_agent.value


def route_after_human(state: QAState) -> str:
    # A malformed/incomplete human instruction should ask the human again,
    # rather than going through the LLM with a stale no-progress state.
    return "human_help" if state.get("needs_human") else "supervisor"


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
    graph.add_conditional_edges("human_help", route_after_human,
                               {"human_help": "human_help", "supervisor": "supervisor"})
    return graph.compile(checkpointer=checkpointer or _memory_saver())
