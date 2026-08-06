"""Deterministic projection of operator language into executable intent."""

from __future__ import annotations

import re
import shlex
from typing import Any

from .models import HumanIntent
from .tools import is_local_target


_TOOL_ALIASES = (
    ("ad_hash_cracking", ("hashcat", "hash crack", "crack the hash", "破解雜湊", "破解 hash")),
    ("ad_credential_validation", ("credential validation", "validate credential", "驗證憑證", "驗證帳密")),
    ("ad_asrep_roasting", ("as-rep", "asrep", "as_rep", "getnpusers", "as-rep 評估")),
    ("ad_bloodhound_collection", ("bloodhound", "sharphound")),
    ("nxc_ldap_recon", ("nxc ldap", "netexec ldap")),
    ("nxc_smb_recon", ("nxc smb", "netexec smb")),
    ("smb_negotiate", ("smb", "smbclient")),
    ("ldap_bind", ("ldap", "ldapsearch")),
    ("check_port", ("nmap", "port scan", "服務枚舉", "服務偵察", "service enumeration")),
)

_NEGATION_MARKERS = ("不要", "勿", "禁止", "不執行", "不要再", "拒絕", "do not", "don't", "never")
_ORDER_MARKERS = ("then", "after", "before", "also", "and", "然後", "之後", "完成後", "接著", "後續", "並", "同時", "再")


def _targets(text: str) -> list[str]:
    values = re.findall(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?(?![\w.])", text)
    return list(dict.fromkeys(value for value in values if not is_local_target(value)))


def _excluded_targets(text: str) -> list[str]:
    """Extract targets explicitly negated by the operator.

    Target negation is clause-local.  A negation before one host must not
    accidentally carry over a redirect such as ``改掃 10.0.0.2`` in the next
    clause.
    """
    matches = list(re.finditer(
        r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?(?![\w.])", text
    ))
    excluded: list[str] = []
    for match in matches:
        clause_start = max(
            (text.rfind(separator, 0, match.start()) for separator in (",", "，", ";", "；", "。", "\n")),
            default=-1,
        ) + 1
        clause = text[clause_start:match.end()].lower()
        if any(marker in clause for marker in _NEGATION_MARKERS):
            excluded.append(match.group(0))
    return list(dict.fromkeys(excluded))


def _has_order(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _ORDER_MARKERS)


def _tool_occurrences(text: str) -> list[tuple[int, str]]:
    lowered = text.lower()
    found: list[tuple[int, str]] = []
    for tool, aliases in _TOOL_ALIASES:
        positions = [lowered.find(alias.lower()) for alias in aliases]
        position = min((item for item in positions if item >= 0), default=-1)
        # The protocol word inside ``nxc ldap``/``nxc smb`` is not a second
        # standalone LDAP/SMB probe.
        if tool == "ldap_bind" and re.search(r"(?:nxc|netexec)\s+ldap", lowered):
            position = -1
        if tool == "smb_negotiate" and re.search(r"(?:nxc|netexec)\s+smb", lowered):
            position = -1
        if position >= 0:
            found.append((position, tool))
    return sorted(found)


def _forbidden_tools(text: str, occurrences: list[tuple[int, str]]) -> list[str]:
    lowered = text.lower()
    forbidden: list[str] = []
    for position, tool in occurrences:
        prefix = lowered[max(0, position - 28):position]
        last_negation = max((prefix.rfind(marker) for marker in _NEGATION_MARKERS), default=-1)
        last_redirect = max(prefix.rfind(marker) for marker in ("改用", "instead", "use"))
        if last_negation >= 0 and last_redirect <= last_negation:
            forbidden.append(tool)
    return list(dict.fromkeys(forbidden))


def _nmap_argv(text: str) -> list[str]:
    """Keep only reviewed nmap options explicitly supplied by the operator."""
    try:
        tokens = shlex.split(text.replace("，", " ").replace("、", " "))
    except ValueError:
        tokens = re.findall(r"--?[A-Za-z0-9-]+|[^\s，、]+", text)
    flags = {
        "-6", "-F", "-f", "-n", "-Pn", "-sn", "-sC", "-sS", "-sT", "-sU", "-sV",
        "-T0", "-T1", "-T2", "-T3", "-T4", "-T5", "--open", "--reason",
        "--traceroute", "--version-light",
    }
    value_flags = {"-T", "-p", "--host-timeout", "--max-retries", "--max-rate", "--min-rate", "--top-ports", "--version-intensity"}
    unsupported: list[str] = []
    result: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in flags:
            result.append(token)
        elif token in value_flags and index + 1 < len(tokens):
            result.extend((token, tokens[index + 1]))
            index += 1
        elif any(token.startswith(f"{flag}=") for flag in value_flags):
            result.append(token)
        elif token.startswith("-"):
            unsupported.append(token)
        index += 1
    if unsupported:
        raise ValueError(
            "Unsupported nmap option(s): " + ", ".join(dict.fromkeys(unsupported))
        )
    return result


def _step_parameters(text: str, tool: str) -> dict[str, Any]:
    lowered = text.lower()
    if tool == "check_port":
        argv = _nmap_argv(text)
        profile = "host_discovery" if "-sn" in argv or "主機發現" in text else "fast" if "-f" in lowered or "快速" in text else "default"
        return {"profile": profile, **({"argv": argv} if argv else {})}
    if tool == "nxc_ldap_recon":
        profile = "groups" if "group" in lowered or "群組" in text else "users"
        return {"profile": profile, "allow_anonymous_nxc": True}
    if tool == "nxc_smb_recon":
        profile = next((name for name in ("shares", "users", "groups", "sessions", "pass-pol") if name in lowered), "shares")
        return {"profile": profile, "allow_anonymous_nxc": True}
    if tool == "smb_negotiate":
        return {"profile": "smb3" if "smb3" in lowered else "anonymous"}
    if tool == "ad_asrep_roasting":
        return {}
    return {}


def parse_human_intent(text: str, state: dict[str, Any] | None = None) -> HumanIntent:
    """Parse operator constraints without pretending to solve the whole task.

    The LLM may rank alternatives, but it must not be the only place where a
    negation, ordering constraint, or exact reviewed argv exists.
    """
    state = state or {}
    raw = text.strip()
    occurrences = _tool_occurrences(raw)
    forbidden = _forbidden_tools(raw, occurrences)
    active = [(position, tool) for position, tool in occurrences if tool not in forbidden]
    ordered_tools = list(dict.fromkeys(tool for _, tool in active))
    if not ordered_tools and ("主機發現" in raw or "host discovery" in raw.lower()):
        ordered_tools = ["check_port"]
    requested_targets = _targets(raw)
    excluded_targets = [name for name in ("local-kali", "local_kali", "kali", "environment") if name in raw.lower()]
    excluded_targets.extend(value for value in re.findall(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", raw) if is_local_target(value))
    excluded_targets.extend(_excluded_targets(raw))
    parsing_errors: list[str] = []
    step_statuses: list[str] = []
    # A single adapter can represent multiple ordered semantic steps. Keep
    # those steps distinct so host discovery and service enumeration cannot
    # collapse into one default nmap action.
    if ("主機發現" in raw or "host discovery" in raw.lower()) and (
        "服務枚舉" in raw or "service enumeration" in raw.lower()
    ):
        ordered_steps = ["network_host_discovery", "service_enumeration"]
        step_parameters = {
            "network_host_discovery": {"profile": "host_discovery"},
            "service_enumeration": {"profile": "default"},
        }
        step_statuses = ["pending", "pending"]
    else:
        ordered_steps = ordered_tools
        steps: list[dict[str, Any]] = []
        for tool in ordered_steps:
            try:
                steps.append(_step_parameters(raw, tool))
                step_statuses.append("pending")
            except ValueError as exc:
                parsing_errors.append(str(exc))
                steps.append({"profile": "default"})
                step_statuses.append("blocked")
        step_parameters = {tool: params for tool, params in zip(ordered_steps, steps)}
    goals = [segment.strip() for segment in re.split(r"[,，。;；\n]", raw) if segment.strip()][:12]
    return HumanIntent(
        raw_instruction=raw,
        goals=goals,
        ordered_steps=ordered_steps,
        step_parameters=step_parameters,
        step_statuses=step_statuses,
        parsing_errors=parsing_errors,
        requested_targets=requested_targets,
        excluded_targets=list(dict.fromkeys(excluded_targets)),
        forbidden_tools=forbidden,
        required_tools=list(dict.fromkeys(
            "check_port" if step in {"network_host_discovery", "service_enumeration"} else step
            for step in ordered_steps
        )),
        has_ordering=_has_order(raw) or len(ordered_steps) > 1,
        continue_autonomously=not any(marker in raw.lower() for marker in ("abort", "stop", "停止", "中止", "結束")),
        rejected_previous=any(marker in raw.lower() for marker in ("不要再", "拒絕上一個", "改用", "不要執行")),
    )
