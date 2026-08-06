"""Assertion and evidence-sufficiency primitives for adaptive QA.

The Supervisor owns the decision about what to do next.  This module only
answers a narrower question: how much evidence has been collected for a QA
assertion, and whether another verification depth is still required.
"""
from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import (
    AssertionStatus,
    Evidence,
    EvidenceLevel,
    EvidenceSufficiency,
    QAAssertion,
    VisibilityMode,
)


_LEVEL_RANK = {level: index for index, level in enumerate(EvidenceLevel)}


def evidence_level_rank(level: EvidenceLevel | str) -> int:
    try:
        return _LEVEL_RANK[EvidenceLevel(level)]
    except (TypeError, ValueError):
        return _LEVEL_RANK[EvidenceLevel.C0]


def parse_visibility_mode(value: Any, specification_available: bool = False) -> VisibilityMode:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "white": VisibilityMode.WHITE_BOX,
        "whitebox": VisibilityMode.WHITE_BOX,
        "gray": VisibilityMode.GRAY_BOX,
        "grey": VisibilityMode.GRAY_BOX,
        "graybox": VisibilityMode.GRAY_BOX,
        "greybox": VisibilityMode.GRAY_BOX,
        "black": VisibilityMode.BLACK_BOX,
        "blackbox": VisibilityMode.BLACK_BOX,
    }
    if normalized in aliases:
        return aliases[normalized]
    compact = normalized.replace("_", "")
    if compact in aliases:
        return aliases[compact]
    try:
        return VisibilityMode(normalized)
    except ValueError:
        return VisibilityMode.WHITE_BOX if specification_available else VisibilityMode.BLACK_BOX


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _service_hint(text: str) -> str | None:
    service_markers = (
        ("ldap", ("ldap", "389", "636", "3268", "3269", "global catalog")),
        ("smb", ("smb", "445", "139", "microsoft-ds", "netbios")),
        ("http", ("https", "http", "443", "80", "8080", "8443", "iis", "web")),
        ("dns", ("dns", "53")),
        ("rpc", ("rpc", "135", "593", "msrpc")),
    )
    for service, markers in service_markers:
        if _contains_any(text, markers):
            return service
    return None


def _is_local_assertion_target(value: str) -> bool:
    host = str(value or "").strip().lower()
    if host in {"environment", "local-kali", "local_kali", "kali", "runner"}:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def build_bootstrap_assertions(
    objective: str,
    target: str,
    visibility: VisibilityMode = VisibilityMode.BLACK_BOX,
) -> list[QAAssertion]:
    """Create conservative assertions from an operator objective.

    This is only a bootstrap.  A supplied specification or future assertion
    planner can replace/extend it.  The important property is that the
    required evidence depth follows the QA question instead of a fixed AD
    attack sequence.
    """
    text = re.sub(r"\s+", " ", str(objective or "").strip().lower())
    assertions = [
        QAAssertion(
            id="environment-baseline",
            statement="Authorized target and required environment baseline are observable.",
            target=target,
            assertion_type="environment",
            source="operator",
            visibility=visibility,
            required_evidence_level=EvidenceLevel.C2,
            allowed_methods=["check_port", "check_dns_resolution", "ldap_bind", "smb_negotiate"],
        )
    ]
    if _contains_any(text, ("functional", "function", "正常", "功能", "service", "服務", "protocol", "協定")):
        service_hint = _service_hint(text)
        assertions.append(QAAssertion(
            id="functional-validation",
            statement=(
                f"The requested {service_hint} service or protocol is functionally usable, not merely reachable."
                if service_hint else
                "The requested service or protocol is functionally usable, not merely reachable."
            ),
            target=target,
            assertion_type="service_function",
            source="operator",
            visibility=visibility,
            required_evidence_level=EvidenceLevel.C3,
            allowed_methods=["ldap_bind", "smb_negotiate", "http_health_check", "check_dns_resolution"],
        ))
    if _contains_any(text, ("configuration", "config", "組態", "設定")):
        assertions.append(QAAssertion(
            id="configuration-validation",
            statement="The requested configuration condition is directly observable in the authorized environment.",
            target=target,
            assertion_type="configuration",
            source="operator",
            visibility=visibility,
            required_evidence_level=EvidenceLevel.C2,
            allowed_methods=["ldap_bind", "nxc_ldap_recon", "check_port"],
        ))
    if _contains_any(text, ("weakness", "vulnerab", "exploit", "利用", "弱點", "可利用")):
        assertions.append(QAAssertion(
            id="exploitability-validation",
            statement="The requested weakness is proven exploitable with the minimum authorized test.",
            target=target,
            assertion_type="exploitability",
            source="operator",
            visibility=visibility,
            required_evidence_level=EvidenceLevel.C4,
            allowed_methods=["controlled_functional_test", "minimal_exploit_validation"],
        ))
    if _contains_any(text, ("end-to-end", "end to end", "attack path", "完整攻擊鏈", "攻擊鏈", "flag", "domain admin")):
        assertions.append(QAAssertion(
            id="end-to-end-validation",
            statement="The explicitly requested authorized attack path reaches its stated end condition.",
            target=target,
            assertion_type="attack_path",
            source="operator",
            visibility=visibility,
            required_evidence_level=EvidenceLevel.C5,
            allowed_methods=["minimal_exploit_validation", "end_to_end_attack_validation"],
        ))
    return assertions


def load_specification_assertions(
    reference: str | None,
    target: str,
    visibility: VisibilityMode,
) -> list[QAAssertion]:
    """Load explicit JSON assertions for white/gray-box tasks.

    The specification is data, not executable instructions.  Only the
    closed QAAssertion schema is accepted; malformed entries are ignored so
    a broken optional file cannot widen tool scope or crash task startup.
    """
    if not reference:
        return []
    try:
        payload = json.loads(Path(reference).expanduser().read_text(
            encoding="utf-8", errors="replace"
        )[:2_000_000])
    except (OSError, TypeError, ValueError):
        return []
    if isinstance(payload, Mapping):
        rows = payload.get("assertions", [])
    else:
        rows = payload
    if not isinstance(rows, list):
        return []
    output: list[QAAssertion] = []
    for row in rows[:100]:
        if not isinstance(row, Mapping):
            continue
        values = dict(row)
        values.setdefault("target", target)
        if _is_local_assertion_target(str(values.get("target", ""))):
            continue
        values["source"] = "specification"
        values["visibility"] = visibility
        try:
            output.append(QAAssertion.model_validate(values))
        except (TypeError, ValueError):
            continue
    return output


def _target_matches(assertion_target: str, evidence_target: str) -> bool:
    if assertion_target in {"", "*", evidence_target}:
        return True
    try:
        network = ipaddress.ip_network(assertion_target, strict=False)
        address = ipaddress.ip_address(evidence_target)
        return address in network
    except ValueError:
        return False


def infer_evidence_level(evidence: Evidence) -> EvidenceLevel:
    facts = evidence.facts if isinstance(evidence.facts, Mapping) else {}
    method_text = f"{evidence.source} {evidence.action}".lower()
    explicit = facts.get("evidence_level")
    trusted_explicit = (
        bool(facts.get("evidence_level_verified"))
        and str(evidence.source).lower().startswith("evidence-verification:")
    )
    if explicit and trusted_explicit:
        try:
            return EvidenceLevel(str(explicit).upper())
        except ValueError:
            pass
    if any(bool(facts.get(key)) for key in (
        "end_to_end_verified", "attack_path_complete", "final_goal_achieved", "flag_retrieved",
    )) and any(marker in method_text for marker in (
        "end_to_end", "attack_path", "final_goal", "flag", "goal_validation",
    )):
        return EvidenceLevel.C5
    if any(bool(facts.get(key)) for key in (
        "exploitability_verified", "vulnerability_triggered", "controlled_exploit_success",
        "effect_observed",
    )) and any(marker in method_text for marker in (
        "exploit", "controlled", "vulnerability", "effect_validation",
    )):
        return EvidenceLevel.C4
    if any(bool(facts.get(key)) for key in (
        "functional", "functional_verified", "protocol_verified", "authentication_success",
        "health_ok", "service_functional",
    )):
        return EvidenceLevel.C3
    if any(bool(facts.get(key)) for key in (
        "open_ports", "service_inventory", "discovered_targets", "domain_name", "domain",
        "users", "spns", "groups", "acl_edges", "delegation", "trusts", "asrep_candidates",
    )):
        return EvidenceLevel.C2
    if any(bool(facts.get(key)) for key in ("inferred", "hypothesis", "confidence")):
        return EvidenceLevel.C1
    return EvidenceLevel.C0


def _opportunity_value(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    return item if isinstance(item, dict) else {}


def _evidence_supports_assertion(assertion: QAAssertion, evidence: Evidence) -> bool:
    """Reject high-level evidence from an unrelated service.

    Evidence levels describe depth, not semantic ownership.  A functional
    HTTP probe on the same host must not satisfy an LDAP-function assertion.
    Assertions without a recognizable service remain intentionally generic.
    """
    if assertion.assertion_type != "service_function":
        return True
    service = _service_hint(assertion.statement.lower())
    if not service:
        return True
    markers = {
        "ldap": ("ldap", "389", "636", "3268", "3269", "global catalog"),
        "smb": ("smb", "445", "139", "microsoft-ds", "netbios"),
        "http": ("http", "80", "443", "8080", "8443", "iis", "web"),
        "dns": ("dns", "53"),
        "rpc": ("rpc", "135", "593", "msrpc"),
    }[service]
    facts = evidence.facts if isinstance(evidence.facts, Mapping) else {}
    tokens = [str(evidence.source), str(evidence.action)]
    for key in ("protocol", "service", "name", "product", "banner"):
        value = facts.get(key)
        if isinstance(value, (list, tuple, set)):
            tokens.extend(str(item) for item in value)
        elif value is not None:
            tokens.append(str(value))
    for key in ("open_ports", "services", "service_inventory"):
        for row in facts.get(key, []) if isinstance(facts.get(key), list) else []:
            if isinstance(row, Mapping):
                tokens.extend(str(row.get(field, "")) for field in (
                    "port", "protocol", "service", "name", "product", "banner"
                ))
    haystack = " ".join(tokens).lower()
    return any(marker in haystack for marker in markers)


def _evidence_is_usable(evidence: Evidence) -> bool:
    """Treat failed commands as diagnostics unless partial evidence is explicit."""
    if evidence.exit_code in (None, 0):
        return True
    facts = evidence.facts if isinstance(evidence.facts, Mapping) else {}
    return bool(facts.get("partial_evidence"))


def assess_sufficiency(
    assertion: QAAssertion,
    evidence: Iterable[Evidence],
    opportunities: Iterable[Mapping[str, Any] | Any] = (),
) -> tuple[QAAssertion, EvidenceSufficiency]:
    relevant = [
        item for item in evidence
        if _target_matches(assertion.target, str(item.target))
        and _evidence_supports_assertion(assertion, item)
        and not str(item.target).lower() in {"environment", "local-kali", "runner"}
    ]
    levels = [
        (infer_evidence_level(item), item)
        for item in relevant
        if _evidence_is_usable(item)
    ]
    current = max((level for level, _ in levels), key=evidence_level_rank, default=EvidenceLevel.C0)
    evidence_ids = [item.id for level, item in levels if level == current]
    contradictory = any(
        isinstance(item.facts, Mapping) and item.facts.get("contradictory")
        for item in relevant
    )
    methods: list[str] = []
    for raw in opportunities:
        item = _opportunity_value(raw)
        if str(item.get("target", "")) == assertion.target or assertion.target == "*":
            tool = str(item.get("tool", "")).strip()
            if tool and tool not in methods:
                methods.append(tool)
    for method in assertion.allowed_methods:
        if method not in methods:
            methods.append(method)
    required_rank = evidence_level_rank(assertion.required_evidence_level)
    current_rank = evidence_level_rank(current)
    sufficient = not contradictory and current_rank >= required_rank
    if contradictory:
        status = "contradictory"
    elif sufficient:
        status = "sufficient"
    elif relevant and all(item.exit_code not in (None, 0) for item in relevant):
        status = "blocked"
    else:
        status = "insufficient"
    result = assertion.model_copy(update={
        "status": (
            assertion.status if assertion.status in {
                AssertionStatus.PASS, AssertionStatus.FAIL, AssertionStatus.NOT_APPLICABLE,
            } else AssertionStatus.IN_PROGRESS if relevant else AssertionStatus.NOT_STARTED
        ),
        "evidence_ids": list(dict.fromkeys(assertion.evidence_ids + [item.id for item in relevant])),
    })
    missing = [] if sufficient else [f"evidence level {assertion.required_evidence_level.value}"]
    if contradictory:
        missing = ["reconcile contradictory evidence before concluding"]
    sufficiency = EvidenceSufficiency(
        assertion_id=assertion.id,
        current_level=current,
        required_level=assertion.required_evidence_level,
        sufficient=sufficient,
        status=status,
        evidence_ids=evidence_ids,
        missing_evidence=missing,
        next_methods=[] if sufficient else methods[:12],
        reason=(
            "Evidence meets the requested threshold; Judge may now evaluate the assertion."
            if sufficient else
            "Evidence is below the requested threshold; choose the least invasive justified method."
        ),
    )
    return result, sufficiency


def refresh_assessment(
    assertions: Iterable[QAAssertion | Mapping[str, Any]],
    evidence: Iterable[Evidence],
    opportunities: Iterable[Mapping[str, Any] | Any] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_items = list(evidence)
    opportunity_items = list(opportunities)
    updated: list[dict[str, Any]] = []
    sufficiencies: list[dict[str, Any]] = []
    for raw in assertions:
        assertion = raw if isinstance(raw, QAAssertion) else QAAssertion.model_validate(raw)
        refreshed, sufficiency = assess_sufficiency(assertion, evidence_items, opportunity_items)
        updated.append(refreshed.model_dump(mode="json"))
        sufficiencies.append(sufficiency.model_dump(mode="json"))
    return updated, sufficiencies
