"""Evidence-driven AD method selection.

The language model may explain and rank findings, but it must not override
hard prerequisites or turn a failed identity path into an endless recon loop.
This module contains the small deterministic policy that sits between the
model's proposal and the execution broker.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Decision, Role, ToolParameters


@dataclass(frozen=True)
class ADContext:
    domain: str | None
    target: str
    has_credentials: bool
    credential_validation_attempted: bool
    credentials_validated: bool
    username_source: bool
    users: tuple[str, ...]
    ldap_bound: bool
    identity_attempted: bool
    identity_complete: bool
    asrep_attempted: bool
    asrep_succeeded: bool
    domain_users_attempted: bool
    spns: tuple[str, ...]
    kerberoast_attempted: bool
    bloodhound_attempted: bool


def _knowledge(state: dict[str, Any]) -> dict[str, Any]:
    value = state.get("ad_knowledge", {}) or {}
    return value.model_dump() if hasattr(value, "model_dump") else value


def _items(state: dict[str, Any], needle: str) -> list[Any]:
    return [item for item in state.get("evidence", [])
            if needle in str(getattr(item, "source", "")).lower()]


def _attempted(state: dict[str, Any], *needles: str) -> bool:
    lowered = tuple(item.lower() for item in needles)
    if any(any(needle in str(getattr(item, "source", "")).lower() for needle in lowered)
           for item in state.get("evidence", [])):
        return True
    for record in state.get("method_history", []):
        text = " ".join(str(record.get(key, "")) for key in ("tool", "action", "capability"))
        if any(needle in text.lower() for needle in lowered):
            return True
    return False


def _successful(items: list[Any]) -> bool:
    return any(getattr(item, "exit_code", None) in (None, 0) for item in items)


def _error_kind(item: Any) -> str | None:
    facts = getattr(item, "facts", {}) or {}
    if not isinstance(facts, dict):
        return None
    nested = facts.get("tool_result") if isinstance(facts.get("tool_result"), dict) else {}
    return facts.get("error_kind") or nested.get("error_kind")


def _target(state: dict[str, Any], domain: str | None) -> str:
    profiles = state.get("target_profiles", {}) or {}
    for target, profile in profiles.items():
        if domain and profile.get("domain") == domain and profile.get("connectivity") == "reachable":
            return target
    for target in state.get("discovered_targets", []):
        if "/" not in str(target):
            return str(target)
    return state.get("target", "environment")


def derive_context(state: dict[str, Any]) -> ADContext:
    knowledge = _knowledge(state)
    users = tuple(sorted({str(user) for user in knowledge.get("users", [])}))
    users_file = os.getenv("CYBERQA_AD_USERS_FILE", "")
    file_available = bool(users_file and Path(users_file).expanduser().is_file())
    decision = state.get("last_decision")
    if decision:
        params = decision.tool_parameters.model_dump(mode="json", exclude_none=True)
        users = tuple(sorted(set(users) | {str(user) for user in params.get("users", [])}))
        file_available = file_available or bool(params.get("users_file"))
    domain = knowledge.get("domain") or os.getenv("CYBERQA_AD_DOMAIN")
    credentials = bool(os.getenv("CYBERQA_AD_USERNAME") and os.getenv("CYBERQA_AD_PASSWORD"))
    credential_items = _items(state, "ad_credential_validation")
    credential_validation_attempted = bool(credential_items) or _attempted(state, "credential_validation")
    credential_validated = bool(knowledge.get("credentials_validated")) or _successful(credential_items)
    ldap_items = [item for item in state.get("evidence", [])
                  if "ldap" in str(getattr(item, "source", "")).lower()]
    ldap_bound = _successful(ldap_items)
    identity_attempted = _attempted(state, "ldap_bind", "smb_negotiate", "nxc_ldap")
    identity_complete = all(_attempted(state, needle) for needle in
                            ("ldap_bind", "smb_negotiate", "nxc_ldap"))
    # An approval-scope rejection is not an AS-REP execution. This distinction
    # lets an existing checkpoint recover after the grant bug is fixed instead
    # of permanently treating the skipped action as completed.
    asrep_items = [item for item in state.get("evidence", [])
                   if str(getattr(item, "source", "")).lower().startswith("ad-capability:ad_asrep")]
    asrep_approval_rejections = [item for item in state.get("evidence", [])
                                 if "asrep" in str(getattr(item, "source", "")).lower()
                                 and _error_kind(item)
                                 in {"approval_scope", "approval_required"}]
    actual_asrep_records = [record for record in state.get("method_history", [])
                            if "asrep" in str(record.get("tool", "")).lower()
                            and record.get("error_kind") not in {"approval_scope", "approval_required"}]
    asrep_attempted = bool(asrep_items or actual_asrep_records or _attempted(state, "ad-capability:ad_asrep"))
    if asrep_approval_rejections and not asrep_items:
        asrep_attempted = False
    return ADContext(
        domain=domain,
        target=_target(state, domain),
        has_credentials=credentials,
        credential_validation_attempted=credential_validation_attempted,
        credentials_validated=credential_validated,
        username_source=bool(users or file_available),
        users=users,
        ldap_bound=ldap_bound,
        identity_attempted=identity_attempted,
        identity_complete=identity_complete,
        asrep_attempted=asrep_attempted,
        asrep_succeeded=_successful(asrep_items),
        domain_users_attempted=_attempted(state, "ad_domain_users"),
        spns=tuple(sorted({str(value) for value in knowledge.get("spns", [])})),
        kerberoast_attempted=_attempted(state, "kerberoast"),
        bloodhound_attempted=_attempted(state, "bloodhound"),
    )


def recommend(state: dict[str, Any]) -> Decision | None:
    """Return a hard-rule AD decision, or ``None`` for LLM planning.

    Returning ``None`` is intentional: it means the deterministic guard has
    no stronger fact-based transition and the model may select a read-only
    discovery/reporting step. It does not mean arbitrary tools are exposed.
    """
    if state.get("scorecard"):
        return Decision(
            next_agent="end", objective="complete", action="end",
            target=state.get("target", "environment"),
            justification="Evidence has been evaluated and the QA scorecard is complete.",
        )
    context = derive_context(state)
    if not context.domain:
        return None

    if not context.has_credentials and not context.credentials_validated:
        if context.username_source and not context.asrep_attempted:
            params = {"users": list(context.users[:500])} if context.users else {}
            configured_file = os.getenv("CYBERQA_AD_USERS_FILE")
            if configured_file:
                params = {"users_file": configured_file}
            return Decision(
                next_agent=Role.TESTING,
                objective="assess unauthenticated AD exposure",
                action="asrep_roasting_assessment",
                target=context.target,
                justification=(
                    "Domain/DC and a candidate username source are known, while no domain credential is "
                    "available. Run the bounded AS-REP assessment before any authenticated or empty-credential probe."
                ),
                capability="asrep_roasting_assessment",
                expected_evidence=["asrep_candidates", "ticket_obtained_or_blocked"],
                tool_parameters=ToolParameters.model_validate(params),
            )
        if not context.username_source:
            if not context.identity_complete:
                return Decision(
                    next_agent=Role.VALIDATION,
                    objective="establish unauthenticated AD identity context",
                    action="anonymous_identity_probe",
                    target=context.target,
                    justification=(
                        "No credential or username source is available. Run each anonymous identity probe "
                        "once to derive a bounded username source; do not retry empty-credential variants."
                    ),
                    expected_evidence=["domain_name", "users", "anonymous_access_or_blocked"],
                    tool_parameters=ToolParameters(allow_anonymous_nxc=True),
                )
            return Decision(
                next_agent="end", objective="human_help", action="provide_asrep_username_source",
                target=context.target,
                justification=(
                    "Anonymous LDAP/SMB identity probes are exhausted and produced no candidate usernames. "
                    "Provide CYBERQA_AD_USERS_FILE or explicit range-issued usernames; the agent will not guess accounts."
                ),
            )
        # A completed AS-REP branch is now an evidence-review transition, not
        # a reason to fall back to nmap, SMB, and LDAP in a loop. A non-zero
        # adapter result is intercepted by the specialist and routed to the
        # operator with its concrete error before this transition is reached.
        if context.asrep_attempted:
            return Decision(
                next_agent=Role.JUDGE, objective="evaluate AD evidence", action="evaluate_ad_evidence",
                target=context.target,
                justification=(
                    "The unauthenticated AS-REP branch has already been executed. Evaluate its redacted result "
                    "and record whether ticket material was observed or the branch was blocked; do not rerun it."
                ),
            )

    if context.has_credentials and not context.credentials_validated:
        if context.credential_validation_attempted:
            return Decision(
                next_agent="end", objective="human_help", action="review_credential_validation",
                target=context.target,
                justification=(
                    "The supplied domain credential was already tested and did not validate. "
                    "Provide a corrected range credential or a different authorized target; do not retry the same login."
                ),
            )
        return Decision(
            next_agent=Role.TESTING,
            objective="validate supplied range credential",
            action="credential_validation",
            target=context.target,
            justification="A domain credential is configured but has not been validated on the scoped target.",
            capability="credential_validation",
            expected_evidence=["authentication_success_or_failure"],
        )

    if context.credentials_validated and not context.domain_users_attempted:
        return Decision(
            next_agent=Role.TESTING,
            objective="enumerate domain identity and SPN context",
            action="enumerate_domain_users",
            target=context.target,
            justification="The supplied credential is validated; enumerate users/SPNs once before choosing Kerberoast or relationship collection.",
            capability="enumerate_domain_users",
            expected_evidence=["users", "spns", "preauth_flags"],
        )

    if context.credentials_validated and context.spns and not context.kerberoast_attempted:
        return Decision(
            next_agent=Role.TESTING,
            objective="assess exposed service-account tickets",
            action="kerberoasting_assessment",
            target=context.target,
            justification="SPNs are present in evidence and the credential is validated; assess the bounded Kerberoast path once.",
            capability="kerberoasting_assessment",
            expected_evidence=["spn_accounts", "ticket_obtained_or_blocked"],
        )

    if context.credentials_validated and not context.bloodhound_attempted:
        return Decision(
            next_agent=Role.TESTING,
            objective="collect AD relationship evidence",
            action="bloodhound_collection",
            target=context.target,
            justification="The credential is validated; collect bounded AD relationship evidence for ACL/delegation/trust analysis.",
            capability="bloodhound_collection",
            expected_evidence=["groups", "sessions", "acl_edges", "delegation", "trusts"],
        )
    if context.credentials_validated and context.bloodhound_attempted:
        return Decision(
            next_agent=Role.JUDGE,
            objective="evaluate accumulated AD QA evidence",
            action="evaluate_ad_evidence",
            target=context.target,
            justification="The bounded credentialed AD methods are complete; evaluate proven, blocked, and unknown findings once.",
        )
    return None
