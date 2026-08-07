"""Situational-awareness header for planner/specialist prompts.

The graph is a long-running autonomous loop, so on every turn the model
needs to re-orient: who am I, where in the loop am I, what has already
been done, and how much budget is left. Building this once here (rather
than letting each prompt improvise) keeps every agent grounded and makes
autonomous cyber-range QA behave coherently across iterations.
"""
from __future__ import annotations

from .state import QAState

# How many consecutive no-progress turns before we inject explicit
# self-rescue guidance. Kept low so the loop breaks out of ruts quickly.
STALL_THRESHOLD = 2

# QA validation stages. This is a *cyber-range QA* agent: the goal is to
# prove the range was built correctly and is solvable as designed. The
# red-team-style recon -> exploit motion is only one mechanism of proof,
# so these stages are framed as validation milestones, not attack goals.
# Each stage carries a human-readable "gate" describing what evidence
# lets the agent legitimately consider the next stage reached. The gates
# are advisory orientation, NOT hard control flow: the agent decides.
_QA_STAGES: list[tuple[str, str, str]] = [
    ("recon",
     "map the range: which hosts/services are reachable",
     "at least one host discovered and one service profiled"),
    ("enumerate",
     "profile the identity/AD surface the build is supposed to expose",
     "identity surface known (users, AS-REP candidates, SPNs) or a domain credential validated"),
    ("validate",
     "prove a designed attack path actually works end to end",
     "an attack path validated, a flag/objective reached, or QA coverage broadly complete"),
    ("report",
     "synthesize the QA verdict: did the build meet its intended design",
     "scorecard produced summarizing what was proven vs. blocked"),
]


def _budget_line(state: QAState) -> str:
    it = state.get("iteration", 0)
    max_it = state.get("max_iterations", 0)
    remaining = max(max_it - it, 0) if max_it else "?"
    stall = state.get("no_progress_count", 0)
    react = state.get("react_steps", 0)
    parts = [f"iteration {it}/{max_it or '?'} ({remaining} left)"]
    if react:
        parts.append(f"react_steps={react}")
    if stall:
        parts.append(f"no_progress_streak={stall}")
    return "budget: " + ", ".join(parts)


def _progress_line(state: QAState) -> str:
    targets = state.get("discovered_targets") or []
    evidence = state.get("evidence") or []
    goals = state.get("completed_goals") or []
    profiled = state.get("target_profiles") or {}
    return (
        f"progress: {len(targets)} target(s) discovered, "
        f"{len(profiled)} profiled, {len(evidence)} evidence item(s), "
        f"{len(goals)} goal(s) completed"
    )


def _last_action_line(state: QAState) -> str:
    history = state.get("method_history") or []
    if not history:
        return "last_action: (none yet)"
    last = history[-1]
    tool = last.get("tool") or last.get("action") or "?"
    target = last.get("target") or ""
    outcome = last.get("outcome") or last.get("status") or ""
    tail = f" -> {outcome}" if outcome else ""
    where = f" on {target}" if target else ""
    return f"last_action: {tool}{where}{tail}"


def _pending_line(state: QAState) -> str:
    flags = []
    if state.get("needs_human"):
        flags.append("AWAITING HUMAN")
    decision = state.get("last_decision")
    if decision is not None and getattr(decision, "approval_required", False):
        flags.append("APPROVAL PENDING")
    if state.get("aborted"):
        flags.append("ABORTED")
    reqs = state.get("human_requests") or []
    if reqs:
        flags.append(f"{len(reqs)} open human request(s)")
    return "status_flags: " + (", ".join(flags) if flags else "nominal")


def _infer_stage_index(state: QAState) -> int:
    """Infer which QA validation stage the run has reached from evidence.

    Advisory only. We read observable progress signals rather than trusting
    a stored phase string, so the header stays honest even if the agent
    jumped around. The agent is free to disagree and act otherwise.
    """
    digest = state.get("memory_digest") or {}
    creds = digest.get("credentials") or {}
    identity = digest.get("identity") or {}
    qa = digest.get("qa_progress") or {}

    scorecard_done = state.get("scorecard") is not None
    paths = state.get("attack_paths") or {}
    path_validated = any(
        getattr(p, "validated", None) or (isinstance(p, dict) and p.get("validated"))
        for p in (paths.values() if isinstance(paths, dict) else paths)
    )
    coverage_pct = qa.get("coverage_pct") or 0.0

    identity_known = bool(
        identity.get("known_user_count")
        or identity.get("asrep_candidates")
        or identity.get("known_spn_count")
        or creds.get("has_validated_domain_credential")
    )
    targets = state.get("discovered_targets") or []
    profiled = state.get("target_profiles") or {}

    if scorecard_done:
        return 3  # report
    if path_validated or coverage_pct >= 80.0:
        return 2  # validate
    if identity_known:
        return 1  # enumerate (surface known, past pure recon)
    if targets and profiled:
        return 1  # recon gate met -> working on enumerate
    return 0  # recon


def _stage_lines(state: QAState) -> list[str]:
    idx = _infer_stage_index(state)
    lines = ["qa_stage_machine (advisory — you decide when to advance):"]
    for i, (name, purpose, gate) in enumerate(_QA_STAGES):
        if i < idx:
            marker = "[x]"
        elif i == idx:
            marker = "[>]"
        else:
            marker = "[ ]"
        lines.append(f"  {marker} {name}: {purpose}")
    if idx < len(_QA_STAGES):
        _, _, gate = _QA_STAGES[idx]
        lines.append(f"  -> gate to advance past '{_QA_STAGES[idx][0]}': {gate}")
    return lines


def _todo_lines(state: QAState) -> list[str]:
    digest = state.get("memory_digest") or {}
    recommended = digest.get("recommended_next") or []
    qa = digest.get("qa_progress") or {}
    pending = qa.get("pending_checks") or []
    blocked = qa.get("blocked_checks") or []
    if not recommended and not pending:
        return ["todo: no pending QA checks — consider synthesizing the scorecard"]
    lines = [
        f"todo: {len(pending)} pending, {len(blocked)} blocked, "
        f"coverage={qa.get('coverage_pct', 0.0)}% — top next checks:"
    ]
    for item in recommended[:4]:
        if isinstance(item, dict):
            cid = item.get("id", "?")
            comp = item.get("component", "")
            obj = item.get("objective", "")
            comp_s = f" ({comp})" if comp else ""
            lines.append(f"  - {cid}{comp_s}: {obj}")
        else:
            lines.append(f"  - {item}")
    return lines


def _stall_lines(state: QAState) -> list[str]:
    stall = state.get("no_progress_count", 0)
    if stall < STALL_THRESHOLD:
        return []
    return [
        f"!! STALL ALERT: {stall} turns with no new progress. Do NOT repeat the "
        "same effective action or argv. Re-orient deliberately: pick a DIFFERENT "
        "hypothesis, a different target/service, or a different capability from the "
        "todo list. If a check is genuinely blocked (e.g. missing domain credential), "
        "pursue an allowed alternative that needs none (e.g. AS-REP roasting) or "
        "escalate to Human — do not spin on the same blocked path.",
    ]


def build_status_header(state: QAState, role: str | None = None) -> str:
    """Return a compact, human-readable orientation block for a prompt."""
    role_label = role or "supervisor"
    lines = [
        "=== SITUATIONAL AWARENESS ===",
        f"you_are: {role_label} agent in an autonomous cyber-range QA loop",
        f"run: {state.get('run_id', '?')} | scenario: {state.get('scenario_id', '?')}",
        f"objective: {state.get('objective', '(unspecified)')}",
        f"phase: {state.get('phase', 'unknown')} | "
        f"baseline_complete={bool(state.get('baseline_complete'))}",
        _budget_line(state),
        _progress_line(state),
        _last_action_line(state),
        _pending_line(state),
    ]
    lines += _stage_lines(state)
    lines += _todo_lines(state)
    lines += _stall_lines(state)
    lines.append(
        "note: this is cyber-range QA — recon/exploit only serve to prove the "
        "build works as designed. You are an autonomous agent: reason, judge and "
        "decide for yourself within the safety guardrails; the stages and todo "
        "above are orientation, not fixed orders."
    )
    lines.append("=== END SITUATIONAL AWARENESS ===")
    return "\n".join(lines)
