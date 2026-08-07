"""Diff B: a small shared knowledge component.

cochise accumulates facts in a cross-turn Knowledge Base so the strategic brain
reasons over what is already known instead of re-deriving it each turn. Our graph
state already carries discovered_targets / recon_coverage / evidence_analyses, but
they are scattered; this module gives the Supervisor one compact, deterministic
view (a digest) plus the non-prescriptive reasoning leads loaded from templates.

Pure/deterministic: no I/O beyond reading the static leads file once, so it stays
safe to call on every supervisor turn without polluting context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_leads() -> str:
    """Load the non-prescriptive AD reasoning leads (see leads.md).

    Falls back to an empty string when the file is missing so callers can treat
    the leads as an optional hint rather than a hard dependency.
    """
    path = Path(__file__).resolve().parents[2] / "templates" / "leads.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""


def build_knowledge_digest(state: dict[str, Any]) -> str:
    """Summarize accumulated cross-turn facts into a compact digest.

    Deterministic ordering (sorted) so the same knowledge yields the same text,
    keeping the digest stable across turns and avoiding spurious context churn.
    """
    discovered = state.get("discovered_targets") or []
    coverage = state.get("recon_coverage") or {}
    analyses = state.get("evidence_analyses") or []
    open_q = state.get("unresolved_questions") or []

    lines: list[str] = ["## Knowledge base (accumulated facts)"]

    hosts = sorted({str(t) for t in discovered})
    lines.append(f"Discovered hosts ({len(hosts)}): " + (", ".join(hosts) if hosts else "none yet"))

    if coverage:
        covered = sorted(f"{k}={v}" for k, v in coverage.items())
        lines.append("Recon coverage: " + ", ".join(covered))
    else:
        lines.append("Recon coverage: none recorded")

    lines.append(f"Evidence analyses recorded: {len(analyses)}")

    if open_q:
        pending = sorted({str(q) for q in open_q})
        lines.append("Unresolved questions:")
        lines.extend(f"  - {q}" for q in pending)
    else:
        lines.append("Unresolved questions: none")

    return "\n".join(lines)


def build_knowledge_gaps(state: dict[str, Any]) -> str:
    """Diff C: make the planner knowledge-gap driven.

    Rather than marching a fixed pipeline, surface the open questions the QA
    still needs answered, ranked so the highest-value gap is first: assertions
    that are not yet sufficiently evidenced, then explicitly unresolved
    questions. The planner is asked to pick the next action that closes the
    most valuable gap; this is guidance, not a mandatory order.
    """
    assertions = state.get("qa_assertions") or []
    sufficiency = state.get("evidence_sufficiency") or {}
    open_q = state.get("unresolved_questions") or []

    gaps: list[str] = []
    for assertion in assertions:
        key = assertion.get("id") if isinstance(assertion, dict) else str(assertion)
        label = assertion.get("statement", key) if isinstance(assertion, dict) else str(assertion)
        level = sufficiency.get(key)
        if level in (None, "insufficient", "partial", False):
            gaps.append(f"unmet assertion: {label}")

    gaps.extend(f"open question: {q}" for q in sorted({str(q) for q in open_q}))

    lines = ["## Knowledge gaps (drive the next action from these)"]
    if gaps:
        lines.extend(f"  {i+1}. {g}" for i, g in enumerate(gaps))
        lines.append(
            "Choose the least-invasive reviewed capability that closes the "
            "highest-value gap above. If every gap is already sufficiently "
            "evidenced, finish evaluation instead of escalating."
        )
    else:
        lines.append("  none open — objective may be complete; prefer 'end'.")
    return "\n".join(lines)


def build_planner_context(state: dict[str, Any]) -> str:
    """Combine the accumulated knowledge digest, open gaps, and reasoning leads.

    This is what the Supervisor sees so it orients from known facts, is driven
    by outstanding knowledge gaps, and uses hints rather than a fixed pipeline.
    Leads are appended only when available.
    """
    parts = [build_knowledge_digest(state), build_knowledge_gaps(state)]
    leads = load_leads()
    if leads:
        parts.append(leads)
    return "\n\n".join(parts)
