from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from .models import ApprovalRequest, AttackPath, Decision, Evidence, Event, Host, Hypothesis, Scorecard


def merge_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


class QAState(TypedDict, total=False):
    run_id: str
    scenario_id: str
    objective: str
    target: str
    phase: str
    hosts: Annotated[dict[str, Host], merge_dict]
    evidence: Annotated[list[Evidence], lambda a, b: a + b]
    events: Annotated[list[Event], lambda a, b: a + b]
    attack_paths: Annotated[dict[str, AttackPath], merge_dict]
    hypotheses: list[Hypothesis]
    approvals: Annotated[list[ApprovalRequest], lambda a, b: a + b]
    scorecard: Scorecard | None
    memory: Annotated[dict[str, Any], merge_dict]
    messages: Annotated[list[Any], add_messages]
    last_decision: Decision | None
    pending_action: dict[str, Any] | None
    iteration: int
    max_iterations: int
    action_history: list[str]
    completed_goals: list[str]
    errors: list[str]
    human_requests: Annotated[list[dict[str, Any]], lambda a, b: a + b]
    react_steps: int
    needs_human: bool
    aborted: bool
    observation_index: Annotated[dict[str, Any], merge_dict]
    no_progress_count: int
