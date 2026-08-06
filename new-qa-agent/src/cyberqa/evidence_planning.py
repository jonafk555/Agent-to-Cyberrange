"""Evidence-to-opportunity projection for adaptive QA planning.

This module is intentionally a capability map, not an attack pipeline.  It
turns facts already observed by an allow-listed tool into reviewed candidates
for the Supervisor.  The model can still select a different justified tool;
these projections make sure useful non-AD and non-AS-REP facts are not lost
when the model is offline or returns an incomplete interpretation.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import ADRisk, Evidence, EvidenceOpportunity
from .tools import LOCAL_EXECUTION_TARGET, is_local_target

_SERVICE_TOOLS: tuple[tuple[frozenset[str], tuple[str, ...], str], ...] = (
    (
        frozenset({"445", "139", "smb", "microsoft-ds", "netbios-ssn", "samba"}),
        ("smb_negotiate", "nxc_smb_recon"),
        "SMB service was observed; establish negotiation/access facts before selecting deeper QA checks.",
    ),
    (
        frozenset({"389", "636", "3268", "3269", "ldap", "ldaps", "global catalog"}),
        ("ldap_bind", "nxc_ldap_recon"),
        "LDAP/Global Catalog service was observed; derive directory identity and naming-context facts.",
    ),
    (
        frozenset({"80", "443", "8080", "8443", "http", "https", "iis", "web"}),
        ("http_health_check",),
        "HTTP(S) service was observed; collect a bounded health/banner response before inferring application paths.",
    ),
    (
        frozenset({"53", "dns"}),
        ("check_dns_resolution",),
        "DNS service was observed; resolve the authorized domain/service names to establish routing context.",
    ),
    (
        frozenset({"135", "593", "rpc", "msrpc"}),
        ("impacket_rpc_recon",),
        "RPC service was observed; collect the reviewed endpoint/interface inventory for later correlation.",
    ),
)


def _present(value: Any) -> bool:
    return value not in (None, "", [], {}, False, 0)


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _target_allowed(state: Mapping[str, Any], target: Any) -> bool:
    value = str(target or "").strip()
    if not value or value == LOCAL_EXECUTION_TARGET or is_local_target(value):
        return False
    runner_ips = {str(item).strip() for item in _values(state.get("runner_ips"))}
    return value not in runner_ips


def _wordlist_available(state: Mapping[str, Any]) -> bool:
    runtime = state.get("runtime_config")
    configured = runtime.get("CYBERQA_AD_WORDLIST") if isinstance(runtime, Mapping) else None
    configured = configured or os.getenv("CYBERQA_AD_WORDLIST")
    return bool(configured and Path(str(configured)).expanduser().is_file())


def _service_tokens(service: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("port", "protocol", "service", "name", "product", "banner"):
        value = service.get(key)
        if _present(value):
            tokens.add(str(value).strip().lower())
    return tokens


def _service_rows(facts: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key in ("open_ports", "services", "service_inventory"):
        for value in _values(facts.get(key)):
            if isinstance(value, Mapping):
                rows.append(value)
    return rows


def _merge_opportunity(
    output: OrderedDict[tuple[str, str], EvidenceOpportunity],
    *,
    tool: str,
    target: str,
    reason: str,
    evidence_fields: list[str],
    prerequisites_met: list[str] | None = None,
    prerequisites_missing: list[str] | None = None,
    capability: str | None = None,
    risk: ADRisk = ADRisk.READ_ONLY,
    expected_evidence: list[str] | None = None,
) -> None:
    if not tool or not target:
        return
    key = (tool, target)
    existing = output.get(key)
    if existing is None:
        output[key] = EvidenceOpportunity(
            key=f"{tool}:{target}",
            tool=tool,
            action=tool,
            target=target,
            reason=reason,
            evidence_fields=list(dict.fromkeys(evidence_fields)),
            prerequisites_met=list(dict.fromkeys(prerequisites_met or [])),
            prerequisites_missing=list(dict.fromkeys(prerequisites_missing or [])),
            capability=capability,
            risk=risk,
            expected_evidence=list(dict.fromkeys(expected_evidence or [])),
        )
        return
    existing.reason = f"{existing.reason} {reason}".strip()
    existing.evidence_fields = list(dict.fromkeys(existing.evidence_fields + evidence_fields))
    existing.prerequisites_met = list(dict.fromkeys(existing.prerequisites_met + (prerequisites_met or [])))
    existing.prerequisites_missing = list(dict.fromkeys(
        existing.prerequisites_missing + (prerequisites_missing or [])
    ))
    existing.expected_evidence = list(dict.fromkeys(existing.expected_evidence + (expected_evidence or [])))


def derive_evidence_opportunities(
    state: Mapping[str, Any], evidence: Evidence, available_tools: list[str] | tuple[str, ...]
) -> list[EvidenceOpportunity]:
    """Derive bounded, evidence-backed candidates for one fresh result.

    The function never executes a tool and never expands the allow-list.  It
    only describes candidates whose tool is already registered and whose
    target came from the observed result or the durable remote-target set.
    """
    facts = evidence.facts if isinstance(evidence.facts, Mapping) else {}
    available = {str(name) for name in available_tools if not str(name).startswith("inspect_")}
    target = str(evidence.target or "").strip()
    output: OrderedDict[tuple[str, str], EvidenceOpportunity] = OrderedDict()

    def add(tool: str, reason: str, fields: list[str], **kwargs: Any) -> None:
        if tool in available and _target_allowed(state, target):
            _merge_opportunity(
                output, tool=tool, target=target, reason=reason,
                evidence_fields=fields, **kwargs,
            )

    service_rows = _service_rows(facts)
    for row in service_rows:
        tokens = _service_tokens(row)
        for markers, tools, reason in _SERVICE_TOOLS:
            if not tokens.isdisjoint(markers):
                fields = [key for key in ("open_ports", "services", "service_inventory") if facts.get(key)]
                for tool in tools:
                    add(tool, reason, fields or ["service_observation"], expected_evidence=["service_details"])

    if _present(facts.get("discovered_targets")):
        discovered = [
            str(item) for item in _values(facts.get("discovered_targets"))
            if _target_allowed(state, item)
        ]
        if "check_port" in available:
            next_target = discovered[0] if discovered else target
            if _target_allowed(state, next_target):
                _merge_opportunity(
                    output,
                    tool="check_port",
                    target=next_target,
                    reason="A new authorized remote host was discovered; establish its service baseline before deeper checks.",
                    evidence_fields=["discovered_targets"],
                    expected_evidence=["open_ports", "service_inventory"],
                )

    if _present(facts.get("domain_name")) or _present(facts.get("domain")):
        domain_field = "domain_name" if _present(facts.get("domain_name")) else "domain"
        add(
            "check_dns_resolution",
            "A domain identity was observed; resolve its names and controller references before relying on them.",
            [domain_field],
            expected_evidence=["resolved_addresses"],
        )
        add(
            "ldap_bind",
            "A domain identity was observed; a bounded LDAP root query can establish naming context and access mode.",
            [domain_field],
            expected_evidence=["domain_name", "ldap_access"],
        )

    if _present(facts.get("users")) or _present(facts.get("asrep_candidates")):
        fields = [key for key in ("users", "asrep_candidates") if _present(facts.get(key))]
        add(
            "ad_asrep_roasting",
            "Candidate domain principals were observed; assess the reviewed pre-authentication exposure when the range policy permits it.",
            fields,
            capability="asrep_roasting_assessment",
            risk=ADRisk.CREDENTIAL_MATERIAL,
            prerequisites_met=["candidate username source"],
            expected_evidence=["asrep_candidates", "ticket_obtained_or_blocked"],
        )

    if _present(facts.get("spns")):
        add(
            "ad_kerberoasting",
            "Service-principal names were observed; assess the reviewed ticket-exposure path rather than repeating enumeration.",
            ["spns"],
            capability="kerberoasting_assessment",
            risk=ADRisk.CREDENTIAL_MATERIAL,
            prerequisites_met=["SPN candidates"],
            expected_evidence=["ticket_obtained_or_blocked"],
        )

    if _present(facts.get("asrep_hash_file")) or _present(facts.get("asrep_hash_count")):
        missing = [] if _wordlist_available(state) else [
            "approved cracking wordlist"
        ]
        add(
            "ad_hash_cracking",
            "Protected AS-REP hash material was observed; assess it with the explicitly scoped local wordlist, never as plaintext.",
            [key for key in ("asrep_hash_file", "asrep_hash_count") if _present(facts.get(key))],
            capability="hash_cracking_assessment",
            risk=ADRisk.CREDENTIAL_MATERIAL,
            prerequisites_met=["AS-REP hash material"],
            prerequisites_missing=missing,
            expected_evidence=["crack_status", "cracked_account_or_not_found"],
        )

    if _present(facts.get("hash_cracked")) or _present(facts.get("cracked_users")):
        add(
            "ad_credential_validation",
            "A recovered credential result was observed; validate it against the authorized service before using it for authenticated discovery.",
            [key for key in ("hash_cracked", "cracked_users") if _present(facts.get(key))],
            capability="credential_validation",
            risk=ADRisk.AUTHENTICATION_TEST,
            prerequisites_met=["recovered credential candidate"],
            expected_evidence=["authentication_success_or_failure"],
        )

    if _present(facts.get("credentials_validated")):
        add(
            "ad_domain_users",
            "A validated domain identity was observed; enumerate users and relationship-relevant controls once, then adapt from the returned facts.",
            ["credentials_validated"],
            capability="enumerate_domain_users",
            prerequisites_met=["validated domain credential"],
            expected_evidence=["users", "groups", "spns", "delegation_flags"],
        )
        add(
            "ad_bloodhound_collection",
            "A validated identity was observed; collect relationship evidence to evaluate ACL, delegation, trust, and session paths.",
            ["credentials_validated"],
            capability="bloodhound_collection",
            prerequisites_met=["validated domain credential"],
            expected_evidence=["groups", "acl_edges", "delegation", "trusts"],
        )

    if any(_present(facts.get(key)) for key in ("groups", "acl_edges", "delegation", "adcs_findings", "trusts")):
        add(
            "ad_bloodhound_collection",
            "Relationship or privilege-path evidence was observed; correlate it with the remaining authorized graph coverage before deciding whether a path is proven.",
            [key for key in ("groups", "acl_edges", "delegation", "adcs_findings", "trusts") if _present(facts.get(key))],
            capability="privilege_path_assessment",
            expected_evidence=["prerequisite_edges", "blocked_or_proven"],
        )

    return list(output.values())[:24]
