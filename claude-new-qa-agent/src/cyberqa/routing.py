"""Centralized routing policy for the QA graph.

All node-transition decisions live here so the precedence order
(aborted > needs_human > terminal > approval > specialist) is defined
exactly once instead of being re-derived inside each edge callback.
"""
from __future__ import annotations

from langgraph.graph import END

from .models import Role
from .state import QAState

SPECIALISTS = ("validation", "testing", "debugging", "judge", "reporting")
HUMAN_HELP = "human_help"
SUPERVISOR = "supervisor"
APPROVAL = "approval"
INITIAL_RECON = "initial_recon"


def _next_agent_node(state: QAState) -> str | None:
    """Return the specialist node named by the current decision, if any."""
    decision = state.get("last_decision")
    if not decision or decision.next_agent == "end":
        return None
    if isinstance(decision.next_agent, Role):
        return decision.next_agent.value
    return decision.next_agent


def route(state: QAState) -> str:
    """Primary supervisor edge. Single source of transition precedence."""
    if state.get("aborted"):
        return END
    if state.get("needs_human"):
        return HUMAN_HELP
    target = _next_agent_node(state)
    if target is None:
        return END
    decision = state.get("last_decision")
    if decision and decision.approval_required:
        return APPROVAL
    return target


def entry_route(state: QAState) -> str:
    return SUPERVISOR if state.get("baseline_complete") else INITIAL_RECON


def route_after_agent(state: QAState) -> str:
    return HUMAN_HELP if state.get("needs_human") else SUPERVISOR


def route_after_approval(state: QAState) -> str:
    if state.get("aborted"):
        return END
    target = _next_agent_node(state)
    decision = state.get("last_decision")
    # After an approval we must land on a concrete specialist; anything else
    # (end / bare "approval") is an inconsistent state -> ask the human.
    if target is None or not isinstance(decision.next_agent, Role):
        return HUMAN_HELP
    return target
