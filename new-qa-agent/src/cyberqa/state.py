from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from .models import ADKnowledge, ApprovalRequest, AttackPath, Decision, Evidence, Event, Host, Hypothesis, Scorecard


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
    # A scorecard is terminal state only when the Supervisor's completion gate
    # explicitly authorized the Judge to create it. This also makes old
    # checkpoints with an unguarded scorecard recoverable.
    scorecard_authorized: bool
    judge_authorized: bool
    memory: Annotated[dict[str, Any], merge_dict]
    messages: Annotated[list[Any], add_messages]
    last_decision: Decision | None
    pending_action: dict[str, Any] | None
    # One-shot grant for one frozen decision. Specialists consume it after the
    # approved action is dispatched.
    approved_grant: dict[str, Any] | None
    iteration: int
    max_iterations: int
    action_history: list[str]
    # Number of times the Supervisor had to re-plan because the proposed
    # effective command was already present in the execution ledger.
    replan_count: int
    # Automatic continuation turns after a model tried to stop without a
    # concrete blocker. This is not an iteration limit; it is a bounded guard
    # against an LLM repeatedly refusing to choose a next path.
    autonomous_replan_count: int
    autonomous_continuation_required: bool
    method_history: list[dict[str, Any]]
    completed_goals: list[str]
    errors: list[str]
    human_requests: Annotated[list[dict[str, Any]], lambda a, b: a + b]
    react_steps: int
    needs_human: bool
    # Set when a human has supplied an explicit next action.  The supervisor
    # must dispatch this frozen decision instead of asking the LLM to reinterpret
    # the instruction on the next tick.
    human_directive: bool
    # Full operator guidance is kept as a first-class planning input.  It is
    # not dependent on the LLM remembering a HumanMessage from an earlier
    # checkpoint.
    human_instruction: str
    human_directives: Annotated[list[dict[str, Any]], lambda a, b: a + b]
    # Allows a specialist to use the read-only recovery tool set after a
    # recoverable command failure. It is cleared when that specialist returns.
    recovery_mode: bool
    aborted: bool
    baseline_complete: bool
    observation_index: Annotated[dict[str, Any], merge_dict]
    no_progress_count: int
    # IPs belonging to the QA runner; these are exclusion metadata, not QA
    # targets and must never enter recon/validation coverage.
    runner_ips: list[str]
    discovered_targets: list[str]
    recon_coverage: Annotated[dict[str, Any], merge_dict]
    ad_knowledge: ADKnowledge
    capability_history: list[dict[str, Any]]
    target_profiles: Annotated[dict[str, Any], merge_dict]
    evidence_synthesis: dict[str, Any]
    runtime_config: Annotated[dict[str, str], merge_dict]
