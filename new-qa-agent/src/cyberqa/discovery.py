from __future__ import annotations

import ipaddress
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .models import Evidence


def _domain_from_dn(value: str) -> str | None:
    parts = re.findall(r"DC=([^,\s]+)", value, flags=re.IGNORECASE)
    return ".".join(parts).lower() if parts else None


def _identity_facts(evidence: Evidence) -> dict[str, str]:
    text = f"{evidence.stdout}\n{evidence.stderr}"
    facts = evidence.facts if isinstance(evidence.facts, dict) else {}
    found: dict[str, str] = {}
    explicit_domain = facts.get("domain_name") or facts.get("domain")
    if explicit_domain:
        found["domain"] = str(explicit_domain).lower()
    for label, key in (("defaultNamingContext", "domain"),
                       ("rootDomainNamingContext", "forest")):
        match = re.search(rf"{label}\s*[:=]\s*([^\r\n]+)", text, re.IGNORECASE)
        if match and (domain := _domain_from_dn(match.group(1))):
            found[key] = domain
            if key == "domain":
                found["base_dn"] = match.group(1).strip()
    if "domain" not in found:
        patterns = (
            r"\bdomain(?:_name)?\s*[:=]\s*([A-Za-z0-9_.-]+)",
            r"\bDNS domain\s*[:=]\s*([A-Za-z0-9_.-]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and "." in match.group(1):
                found["domain"] = match.group(1).strip(" .").lower()
                break
    if "forest" not in found and facts.get("forest"):
        found["forest"] = str(facts["forest"]).lower()
    return found


def build_target_profiles(evidence_items: Iterable[Evidence], previous: dict[str, Any] | None = None,
                          primary_domain: str | None = None) -> dict[str, Any]:
    profiles = {key: dict(value) for key, value in (previous or {}).items()}
    detected_domains: list[str] = []
    for evidence in evidence_items:
        target = evidence.target
        profile = profiles.setdefault(target, {"target": target, "status": "observed", "sources": []})
        profile["sources"] = sorted(set(profile.get("sources", [])) | {evidence.source})
        identity = _identity_facts(evidence)
        profile.update(identity)
        if identity.get("domain"):
            detected_domains.append(identity["domain"])
        text = f"{evidence.stdout}\n{evidence.stderr}".lower()
        if evidence.exit_code == 0 or any(marker in text for marker in (
            "operations error", "stronger authentication required", "access denied", "logon failure"
        )):
            profile["connectivity"] = "reachable"
        elif any(marker in text for marker in ("no route to host", "network is unreachable")):
            profile["connectivity"] = "unreachable_observed"
        elif evidence.exit_code not in (None, 0):
            # Authentication, DNS and forest-context failures must not be
            # collapsed into a false network-down conclusion.
            profile.setdefault("connectivity", "unknown")
            profile["deferred_reason"] = "tool_failed_or_identity_context_missing"
    selected_primary = (primary_domain or (detected_domains[0] if detected_domains else None))
    for profile in profiles.values():
        domain = profile.get("domain")
        forest = profile.get("forest") or domain
        if selected_primary and domain:
            profile["domain_relation"] = (
                "primary_domain" if domain == selected_primary else
                "foreign_forest_or_domain"
            )
            if domain != selected_primary:
                profile["deferred_for_cross_forest"] = True
        elif profile.get("connectivity") != "unreachable_observed":
            profile.setdefault("domain_relation", "unknown_domain")
        if forest:
            profile["forest"] = forest
    return profiles


def synthesize_evidence(evidence_items: Iterable[Evidence], target_profiles: dict[str, Any]) -> dict[str, Any]:
    items = list(evidence_items)
    targets: dict[str, Any] = {}
    for evidence in items:
        bucket = targets.setdefault(evidence.target, {
            "observations": 0, "successful": 0, "failed": 0,
            "sources": [], "open_services": [], "latest_errors": [],
        })
        bucket["observations"] += 1
        bucket["successful" if evidence.exit_code in (None, 0) else "failed"] += 1
        bucket["sources"] = sorted(set(bucket["sources"]) | {evidence.source})
        facts = evidence.facts if isinstance(evidence.facts, dict) else {}
        for service in facts.get("open_ports") or []:
            if service not in bucket["open_services"]:
                bucket["open_services"].append(service)
        if evidence.exit_code not in (None, 0):
            error = (evidence.stderr or evidence.stdout or "non-zero exit").strip()[-500:]
            if error and error not in bucket["latest_errors"]:
                bucket["latest_errors"] = (bucket["latest_errors"] + [error])[-5:]
    foreign = [target for target, profile in target_profiles.items()
               if profile.get("deferred_for_cross_forest")]
    unresolved = [target for target, profile in target_profiles.items()
                  if profile.get("domain_relation") == "unknown_domain"
                  or profile.get("connectivity") == "unknown"]
    all_users = sorted({str(user) for item in items
                        for user in ((item.facts.get("users") or []) if isinstance(item.facts, dict) else [])})
    has_domain = bool(target_profiles and any(profile.get("domain") for profile in target_profiles.values()))
    has_credential = bool(os.getenv("CYBERQA_AD_USERNAME") and os.getenv("CYBERQA_AD_PASSWORD")) or any(
        bool(item.facts.get("credentials_validated"))
        for item in items if isinstance(item.facts, dict)
    )
    anonymous_attempted = any(
        ("ldap" in item.source.lower() or "nxc" in item.source.lower())
        and item.exit_code not in (None, 0) for item in items
    )
    if has_domain and not has_credential and all_users:
        next_actions = ["asrep_roasting_assessment"]
    elif has_domain and not has_credential and not anonymous_attempted:
        next_actions = ["anonymous_ldap_or_nxc_user_enumeration"]
    elif has_domain and not has_credential and anonymous_attempted and not all_users:
        next_actions = ["request_username_source_or_enable_anonymous_enumeration"]
    else:
        next_actions = []
    return {
        "total_observations": len(items),
        "targets": targets,
        "target_profiles": target_profiles,
        "cross_forest_candidates": sorted(foreign),
        "unresolved_targets": sorted(set(unresolved)),
        "candidate_users": all_users,
        "next_actions": next_actions,
    }


def derive_runtime_config(evidence_items: Iterable[Evidence], target_profiles: dict[str, Any],
                          current: dict[str, str] | None = None) -> dict[str, str]:
    config = dict(current or {})
    domains = sorted({str(profile["domain"]) for profile in target_profiles.values()
                      if profile.get("domain")})
    if len(domains) == 1:
        config.setdefault("CYBERQA_AD_DOMAIN", domains[0])
        config.setdefault("CYBERQA_AD_BASE_DN", ",".join(f"DC={part}" for part in domains[0].split(".")))
    for target, profile in target_profiles.items():
        if profile.get("domain") == config.get("CYBERQA_AD_DOMAIN") and _is_ip(target):
            config.setdefault("CYBERQA_AD_DC", target)
            break
    nameservers: set[str] = set()
    networks: set[str] = set()
    for evidence in evidence_items:
        for match in re.findall(r"^nameserver\s+(\S+)", evidence.stdout, re.MULTILINE):
            nameservers.add(match)
        facts = evidence.facts if isinstance(evidence.facts, dict) else {}
        for target in facts.get("discovered_targets") or []:
            if _is_ip(target):
                networks.add(str(ipaddress.ip_network(f"{target}/24", strict=False)))
    if nameservers:
        config.setdefault("CYBERQA_DISCOVERED_DNS_SERVERS", ",".join(sorted(nameservers)))
    if networks:
        config.setdefault("CYBERQA_DISCOVERED_NETWORKS", ",".join(sorted(networks)))
    return config


def apply_and_persist_runtime_config(config: dict[str, str]) -> str:
    safe = {key: str(value) for key, value in config.items() if value and "PASSWORD" not in key}
    for key, value in safe.items():
        os.environ.setdefault(key, value)
    path = Path(os.getenv("CYBERQA_DISCOVERED_ENV", ".cyberqa/discovered.env")).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{key}={json.dumps(value)}" for key, value in sorted(safe.items())) + "\n",
                    encoding="utf-8")
    return str(path)


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False
