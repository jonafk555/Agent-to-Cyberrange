"""Memory consolidation.

The graph accumulates a lot of parallel state (evidence, method_history,
recon_coverage, ad_knowledge, target_profiles, evidence_synthesis). Feeding all
of that raw to the planner every turn is exactly why the agent's "memory" feels
scattered and why it re-runs work: the model has to re-derive conclusions from
low-level facts on every step.

``consolidate_memory`` collapses that state into a single, compact, stable
digest -- "what we know, what's proven/blocked/unknown, what QA is done, what to
do next, what NOT to repeat". The planner and specialists orient on this digest
first, so decisions stay coherent across turns.
"""
from __future__ import annotations

from typing import Any

from .ad_knowledge_base import qa_coverage


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value) if isinstance(value, dict) else {}


def _open_services(target_profiles: dict[str, Any], evidence_synthesis: dict[str, Any]) -> dict[str, list[str]]:
    services: dict[str, list[str]] = {}
    targets = (evidence_synthesis or {}).get("targets", {})
    for target, bucket in targets.items():
        names: list[str] = []
        for service in bucket.get("open_services", []) if isinstance(bucket, dict) else []:
            if isinstance(service, dict):
                names.append(f"{service.get('port')}/{service.get('service', '')}".strip("/"))
        if names:
            services[target] = sorted(set(names))
    return services


def _flatten_service_names(open_services: dict[str, list[str]]) -> set[str]:
    return {part.lower() for names in open_services.values() for part in names}


def consolidate_memory(
    *,
    knowledge: dict[str, Any] | None,
    target_profiles: dict[str, Any] | None,
    evidence_synthesis: dict[str, Any] | None,
    recon_coverage: dict[str, Any] | None,
    method_history: list[dict[str, Any]] | None,
    evidence_sources: Any = (),
    has_credential: bool = False,
) -> dict[str, Any]:
    """Build the compact planning digest from the projected memory."""
    knowledge = _as_dict(knowledge)
    target_profiles = target_profiles or {}
    evidence_synthesis = evidence_synthesis or {}
    recon_coverage = recon_coverage or {}
    method_history = method_history or []

    open_services = _open_services(target_profiles, evidence_synthesis)
    coverage = qa_coverage(
        knowledge=knowledge,
        evidence_sources=evidence_sources,
        method_history=method_history,
        has_credential=has_credential,
        open_services=_flatten_service_names(open_services),
    )

    domain_controllers = sorted(
        target for target, profile in target_profiles.items()
        if isinstance(profile, dict) and profile.get("domain")
    )

    # "do_not_repeat" is the single authoritative list of completed semantic
    # checks per target. The planner must treat these as closed.
    do_not_repeat: list[str] = []
    for target, profile in recon_coverage.items():
        if not isinstance(profile, dict):
            continue
        for check, meta in profile.get("checks", {}).items():
            if isinstance(meta, dict) and meta.get("status") == "completed":
                do_not_repeat.append(f"{target}:{check}")

    findings = _findings(knowledge, coverage)

    return {
        "environment": {
            "primary_domain": knowledge.get("domain"),
            "domains": knowledge.get("domains", []),
            "forests": knowledge.get("forests", []),
            "domain_controllers": domain_controllers,
            "cross_forest_targets": knowledge.get("cross_forest_targets", []),
            "unresolved_targets": evidence_synthesis.get("unresolved_targets", []),
        },
        "credentials": {
            "has_validated_domain_credential": bool(knowledge.get("credentials_validated")),
            "validated": knowledge.get("credentials_validated", []),
        },
        "identity": {
            "known_user_count": len(knowledge.get("users", [])),
            "sample_users": sorted(knowledge.get("users", []))[:15],
            "asrep_candidates": knowledge.get("asrep_candidates", []),
            "known_spn_count": len(knowledge.get("spns", [])),
        },
        "services": open_services,
        "findings": findings,
        "qa_progress": {
            "coverage_pct": coverage["coverage_pct"],
            "completed_checks": coverage["completed"],
            "pending_checks": coverage["pending"],
            "blocked_checks": coverage["blocked"],
        },
        "recommended_next": coverage["recommended_next"],
        "do_not_repeat": sorted(do_not_repeat),
    }


def _findings(knowledge: dict[str, Any], coverage: dict[str, Any]) -> dict[str, list[str]]:
    """Summarize proven / blocked / unknown at the QA-category level."""
    proven: list[str] = []
    for key, label in (
        ("asrep_candidates", "AS-REP roastable accounts present"),
        ("spns", "SPN-bearing service accounts present"),
        ("acl_edges", "Dangerous ACL edges collected"),
        ("delegation", "Delegation configuration collected"),
        ("adcs_findings", "AD CS template findings present"),
        ("trusts", "Domain/forest trusts enumerated"),
        ("credentials_validated", "Domain credential validated"),
    ):
        if knowledge.get(key):
            proven.append(label)
    blocked = [f"{item['id']}: {item['reason']}" for item in coverage["blocked"]]
    completed = set(coverage["completed"])
    unknown = [
        c["component"] for c in coverage["checks"]
        if c["status"] == "pending" and c["id"] not in completed
    ]
    return {"proven": proven, "blocked": blocked, "unknown": unknown[:10]}
