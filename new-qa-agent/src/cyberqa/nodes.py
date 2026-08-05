from __future__ import annotations

import json
import os
import re
from typing import Any, Annotated, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from .approval import ApprovalPolicy, approved_tools_for_decision, decision_fingerprint
from .ad_playbooks import (capability_catalog, get_capability,
                           normalize_capability_parameters)
from .ad_strategy import recommend as recommend_ad_method
from .discovery import (apply_and_persist_runtime_config, build_target_profiles,
                         derive_runtime_config, synthesize_evidence)
from .events import EventBus
from .execution_broker import CapabilityBroker
from .models import (ADRisk, Decision, Event, Evidence, Hypothesis, Role, Scorecard, Service,
                      ServiceProtocol, ToolParameters)
from .state import QAState
from .tools import LOCAL_EXECUTION_TARGET, ToolRegistry, is_local_target


AD_QA_PLAYBOOK = """For an AD range, do not stop at one domain controller. Inventory all valuable
authorized targets and services: domain controllers, DNS, LDAP/LDAPS, Kerberos, SMB/file servers,
member servers, workstations, web/IIS, WinRM, MSSQL, AD CS/CA, backup and management systems, trusts,
and high-value shares. Track host/service coverage and do not call the task complete after DC recon.

Use evidence-driven, read-only QA checks for applicable abuse classes: anonymous LDAP, SMB signing and
share access, Kerberoasting/SPNs, AS-REP roasting, NTLM relay prerequisites, delegation, privileged
group and ACL edges, LAPS/GPO exposure, AD CS enrollment weaknesses, DNS/trust misconfiguration,
stale principals, and excessive local-admin paths. Never invent credentials or claim exploitability;
enumerate prerequisites and state what is proven, blocked, or unknown. Prefer a different target/service
or diagnostic probe over repeating a cached command.
"""

SYSTEM = AD_QA_PLAYBOOK + """You are a cyber-range QA specialist operating only on authorized targets. Use OODA:
observe facts, orient against the objective and prior evidence, decide one justified action, and act
through the supplied fact-only tools. Inspect every tool result before selecting the next tool. Continue
until the objective is complete. Never invent facts, credentials, vulnerabilities, or successful attacks.
Build a compact evidence summary after each probe. If no domain credentials exist, do not repeat empty-
credential SMB/LDAP/NXC probes; pivot to domain discovery, supplied username-file AS-REP assessment,
anonymous access only when evidence supports it, or another justified path."""


def _is_abort_instruction(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {
        "abort", "stop", "exit", "quit", "cancel", "end", "停止", "中止", "離開", "結束",
    }


def _is_rejection_instruction(text: str) -> bool:
    """Recognize a short rejection without treating it as task termination."""
    normalized = re.sub(r"[\s，。,.!！?？]+", "", text.strip().lower())
    return normalized in {
        "no", "nope", "否", "不", "不是", "不要", "不可以", "拒絕", "不授權", "未授權",
    }


def _is_multi_step_instruction(text: str) -> bool:
    """Keep compound natural-language guidance in the Supervisor semantic path."""
    normalized = text.lower()
    return any(marker in normalized for marker in (
        " and ", " then ", " after ", " before ", " also ",
        "然後", "之後", "完成後", "再", "並", "同時", "優先", "除了", "不要只",
    ))


class ReactState(TypedDict, total=False):
    """Small private state contract used by each specialist ReAct subgraph."""
    messages: Annotated[list[Any], add_messages]
    failed_tool_signatures: list[str]
    tool_signatures: list[str]
    needs_human: bool
    human_request: dict[str, Any]
    recovery_failures: list[str]
    recovery_steps: int


class Agents:
    def __init__(self, llm: BaseChatModel | None = None, tools: ToolRegistry | None = None,
                 events: EventBus | None = None, policy: ApprovalPolicy | None = None,
                 on_progress: Callable[[str, dict[str, Any]], None] | None = None):
        self.llm, self.tools, self.events, self.policy = llm, tools or ToolRegistry(), events or EventBus(), policy or ApprovalPolicy()
        self.on_progress = on_progress
        self.broker = CapabilityBroker()

    def progress(self, event: str, **data: Any) -> None:
        if self.on_progress:
            self.on_progress(event, data)

    @staticmethod
    def _evidence_is_novel(state: QAState, evidence: Evidence, cached: bool) -> bool:
        """Count new facts, not merely a different recon command, as progress."""
        if cached:
            return False
        facts = evidence.facts if isinstance(evidence.facts, dict) else {}
        previous_targets = {item.target for item in state.get("evidence", [])}
        if evidence.target not in previous_targets:
            return True
        previous_facts = [item.facts for item in state.get("evidence", [])
                          if isinstance(item.facts, dict)]
        for key in ("discovered_targets", "open_ports", "users", "spns", "asrep_candidates",
                    "credentials_validated", "ticket_obtained_or_blocked", "groups", "acl_edges",
                    "delegation", "adcs_findings", "trusts"):
            current = {json.dumps(value, sort_keys=True, default=str)
                       for value in facts.get(key, [])}
            known = {json.dumps(value, sort_keys=True, default=str)
                     for old in previous_facts for value in old.get(key, [])}
            if current - known:
                return True
        if facts.get("domain_name") and not any(old.get("domain_name") for old in previous_facts):
            return True
        # A first result from a source/target pair is useful diagnostic
        # evidence. Subsequent command variants with no new facts are not.
        pair = (evidence.source, evidence.target)
        return not any((item.source, item.target) == pair for item in state.get("evidence", []))

    @staticmethod
    def _recon_check_key(evidence: Evidence) -> tuple[str, str]:
        """Map a result to a durable semantic recon check and its category."""
        source = evidence.source.lower()
        facts = evidence.facts if isinstance(evidence.facts, dict) else {}
        argv = " ".join(str(item) for item in facts.get("argv", []))
        if "nxc_smb" in source:
            profile = next((name for name in ("shares", "users", "groups", "sessions", "pass-pol")
                            if f"--{name}" in argv), "default")
            return f"nxc_smb:{profile}", "nxc_smb"
        if "nxc_ldap" in source:
            profile = next((name for name in ("users", "groups", "sessions", "pass-pol")
                            if f"--{name}" in argv), "default")
            return f"nxc_ldap:{profile}", "nxc_ldap"
        if "check_port" in source or "nmap" in source:
            if " -sn" in f" {argv} ":
                profile = "host_discovery"
            elif " -F" in f" {argv} ":
                profile = "fast"
            elif "--top-ports 1000" in argv:
                profile = "top1000"
            elif "--top-ports 100" in argv:
                profile = "top100"
            elif " -p " in f" {argv} ":
                profile = "ad_tcp"
            else:
                profile = "default"
            return f"nmap:{profile}", "nmap"
        if "ldap_bind" in source or "ldapsearch" in source:
            if "-ZZ" in argv:
                profile = "starttls_rootdse"
            elif "-Y GSSAPI" in argv:
                profile = "gssapi_rootdse"
            elif "-s sub" in argv:
                profile = "subtree"
            else:
                profile = "rootdse"
            return f"ldap:{profile}", "ldap"
        if "smb_negotiate" in source or "smbclient" in source:
            if "-m SMB2" in argv:
                profile = "smb2"
            elif "-m SMB3" in argv:
                profile = "smb3"
            elif "-p 445" in argv:
                profile = "port445"
            else:
                profile = "anonymous"
            return f"smb:{profile}", "smb"
        if "check_dns" in source or "dig" in source:
            return "dns:resolution", "dns"
        return f"{evidence.source}:{evidence.action}", "other"

    @classmethod
    def _build_recon_coverage(cls, state: QAState, evidence_items: list[Evidence]) -> dict[str, Any]:
        """Persist target -> semantic check status, not just source names."""
        previous = state.get("recon_coverage", {}) or {}
        coverage: dict[str, Any] = {}
        for target, raw in previous.items():
            if is_local_target(str(target)):
                continue
            if isinstance(raw, dict):
                coverage[target] = {
                    "checks": dict(raw.get("checks", {})),
                    "semantic": dict(raw.get("semantic", {})),
                    "remaining": list(raw.get("remaining", [])),
                }
            else:
                coverage[target] = {"checks": {str(item): {"status": "completed"}
                                                for item in raw},
                                     "semantic": {}, "remaining": []}
        for evidence in [*state.get("evidence", []), *evidence_items]:
            # Local runtime observations are useful context, but are never a
            # network recon check and must not create a host-coverage debt.
            if is_local_target(evidence.target):
                continue
            key, category = cls._recon_check_key(evidence)
            target = evidence.target
            entry = coverage.setdefault(target, {"checks": {}, "semantic": {}, "remaining": []})
            status = "completed" if evidence.exit_code in (None, 0) else "blocked"
            check = entry["checks"].setdefault(key, {
                "status": status, "attempts": 0, "evidence_ids": [],
            })
            check["status"] = "completed" if status == "completed" else check.get("status", status)
            check["attempts"] = int(check.get("attempts", 0)) + 1
            if evidence.id not in check["evidence_ids"]:
                check["evidence_ids"].append(evidence.id)
            check["last_argv"] = facts_argv = (
                evidence.facts.get("argv", []) if isinstance(evidence.facts, dict) else []
            )
            entry["semantic"][category] = {
                "checks": sorted(name for name in entry["checks"] if name.startswith(f"{category}:")),
                "attempts": sum(int(item.get("attempts", 0)) for name, item in entry["checks"].items()
                                if name.startswith(f"{category}:")),
                "has_success": any(item.get("status") == "completed"
                                    for name, item in entry["checks"].items()
                                    if name.startswith(f"{category}:")),
            }
        return coverage

    def _project_observations(self, state: QAState, new_evidence: list[Evidence]) -> dict[str, Any]:
        """Build one cumulative view used by every future planning decision."""
        all_evidence = [*state.get("evidence", []), *new_evidence]
        scope_target = str(state.get("target", ""))
        recon_evidence = [
            item for item in all_evidence
            if not is_local_target(item.target)
            and (item.target == scope_target or self.tools.target_policy.allows(item.target))
        ]
        old_profiles = {
            target: profile for target, profile in (state.get("target_profiles", {}) or {}).items()
            if not is_local_target(str(target))
            and (str(target) == scope_target or self.tools.target_policy.allows(str(target)))
        }
        old_knowledge = state.get("ad_knowledge", {}) or {}
        if hasattr(old_knowledge, "model_dump"):
            old_knowledge = old_knowledge.model_dump()
        profiles = build_target_profiles(recon_evidence, old_profiles, old_knowledge.get("domain"))
        synthesis = synthesize_evidence(recon_evidence, profiles)
        recon_coverage = self._build_recon_coverage(
            state, [item for item in recon_evidence if not is_local_target(item.target)]
        )
        recon_coverage = {
            target: profile for target, profile in recon_coverage.items()
            if str(target) == scope_target or self.tools.target_policy.allows(str(target))
        }
        synthesis["recon_coverage"] = recon_coverage
        runtime = derive_runtime_config(all_evidence, profiles, state.get("runtime_config", {}))
        if runtime:
            apply_and_persist_runtime_config(runtime)
        knowledge = dict(old_knowledge)
        for field in ("users", "spns", "asrep_candidates", "credentials_validated", "groups", "acl_edges",
                      "delegation", "adcs_findings", "trusts"):
            values = set(knowledge.get(field, []))
            for item in all_evidence:
                facts = item.facts if isinstance(item.facts, dict) else {}
                values.update(str(value) for value in facts.get(field, []))
            knowledge[field] = sorted(values)
        for item in all_evidence:
            facts = item.facts if isinstance(item.facts, dict) else {}
            if facts.get("domain_name") and not knowledge.get("domain"):
                knowledge["domain"] = str(facts["domain_name"])
        domains = set(knowledge.get("domains", []))
        forests = set(knowledge.get("forests", []))
        domains.update(str(profile["domain"]) for profile in profiles.values() if profile.get("domain"))
        forests.update(str(profile["forest"]) for profile in profiles.values() if profile.get("forest"))
        knowledge["domains"] = sorted(domains)
        knowledge["forests"] = sorted(forests)
        knowledge["target_domains"] = {
            target: profile["domain"] for target, profile in profiles.items() if profile.get("domain")
        }
        knowledge["cross_forest_targets"] = sorted(
            target for target, profile in profiles.items() if profile.get("deferred_for_cross_forest")
        )
        if not knowledge.get("domain") and knowledge.get("domains"):
            knowledge["domain"] = knowledge["domains"][0]
        return {"target_profiles": profiles, "evidence_synthesis": synthesis,
                "runtime_config": runtime, "ad_knowledge": knowledge,
                "recon_coverage": recon_coverage}

    @staticmethod
    def _known_prerequisites(state: QAState) -> set[str]:
        knowledge = state.get("ad_knowledge", {}) or {}
        if hasattr(knowledge, "model_dump"):
            knowledge = knowledge.model_dump()
        known: set[str] = set()
        if knowledge.get("domain") or os.getenv("CYBERQA_AD_DOMAIN") or state.get("target_profiles"):
            known.add("domain_inventory")
        if knowledge.get("users"):
            known.add("user enumeration")
            known.add("candidate username source")
        decision = state.get("last_decision")
        tool_parameters = decision.tool_parameters.model_dump() if decision else {}
        if decision and (tool_parameters.get("users") or tool_parameters.get("users_file")):
            known.add("user enumeration")
            known.add("candidate username source")
        users_file = os.getenv("CYBERQA_AD_USERS_FILE", "")
        # A path supplied by the operator is an intentional candidate source.
        # Let the reviewed adapter report a missing/unreadable file instead of
        # silently falling back to anonymous recon.
        if users_file:
            known.add("candidate username source")
        if knowledge.get("credentials_validated"):
            known.add("validated domain credential")
            known.add("authorized identity or approved anonymous path")
        evidence = state.get("evidence", [])
        for item in evidence:
            facts = item.facts if isinstance(item.facts, dict) else {}
            if item.exit_code in (None, 0) and "ldap" in item.source.lower():
                known.add("valid LDAP access or explicitly allowed anonymous LDAP")
            if facts.get("lockout_policy"):
                known.add("lockout_policy")
            if facts.get("dns_resolved") or "dns" in item.source.lower():
                known.add("DNS resolution")
        if knowledge.get("acl_edges") or knowledge.get("delegation") or knowledge.get("trusts"):
            known.add("bloodhound_collection or equivalent relationship evidence")
        if os.getenv("CYBERQA_APPROVED_TEST_PASSWORD"):
            known.add("approved_test_password")
        if os.getenv("CYBERQA_AD_USERNAME") and os.getenv("CYBERQA_AD_PASSWORD"):
            known.add("human_supplied_or_range_issued_credential")
        return known

    async def _human_problem(self, state: QAState, kind: str, raw: str = "") -> str:
        """Create an operator-facing issue summary without exposing hidden reasoning."""
        failures = [
            f"{e.source} target={e.target} exit={e.exit_code} stderr={e.stderr[-1000:]}"
            for e in state.get("evidence", [])[-8:]
            if e.exit_code not in (None, 0) or e.facts.get("ok") is False
        ]
        fallback = (failures[-1] if failures else raw[-1500:]) or "流程沒有取得新的可用觀測結果。"
        if not self.llm:
            return fallback
        try:
            import asyncio
            response = await asyncio.wait_for(self.llm.ainvoke([
                SystemMessage(content=(
                    "Summarize an authorized AD QA problem for a human operator in Traditional Chinese. "
                    "Give only what failed, likely category, exact useful evidence, and what decision "
                    "the operator should provide. Do not reveal hidden chain-of-thought."
                )),
                HumanMessage(content=json.dumps({"kind": kind, "failures": failures, "raw": raw[-3000:]}, ensure_ascii=False)),
            ]), timeout=15)
            text = response.content if isinstance(response.content, str) else str(response.content)
            return text.strip() or fallback
        except Exception:
            return fallback

    @staticmethod
    def _conversation_context(messages: list[Any]) -> list[Any]:
        """Keep only messages valid as outer conversational context.

        Tool messages belong to the ReAct subgraph that created them. Passing
        an orphan ToolMessage into a new OpenAI request causes a 400 error.
        Tool results remain available to the current inner loop and are also
        projected into the durable evidence list.
        """
        return [
            message for message in messages[-20:]
            if not isinstance(message, ToolMessage)
            and not getattr(message, "tool_calls", None)
        ]

    @staticmethod
    def _react_context(messages: list[Any]) -> list[Any]:
        """Keep only a valid AI tool-call -> ToolMessage sequence."""
        valid: list[Any] = []
        pending_calls: set[str] = set()
        for message in messages[-30:]:
            if isinstance(message, ToolMessage):
                if message.tool_call_id in pending_calls:
                    valid.append(message)
                continue
            valid.append(message)
            if isinstance(message, AIMessage):
                pending_calls = {call.get("id") for call in (message.tool_calls or []) if call.get("id")}
            elif not isinstance(message, SystemMessage):
                pending_calls = set()
        return valid

    async def _reason(self, role: Role, state: QAState, instruction: str) -> dict[str, Any]:
        if not self.llm:
            return {"action": "observe", "target": "environment", "justification": "Collect missing facts before changing state."}
        prompt = json.dumps({"objective": state.get("objective"), "phase": state.get("phase"),
                             "target": state.get("target", "environment"),
                             "evidence": [e.model_dump() for e in state.get("evidence", [])[-20:]],
                             "instruction": instruction})
        self.progress("reasoning_start", agent=role.value)
        conversation = self._conversation_context(state.get("messages", []))
        response = await self.llm.ainvoke([
            SystemMessage(content=SYSTEM + "\nYou are the workflow supervisor. Return only a valid decision JSON."),
            *conversation,
            HumanMessage(content=prompt),
        ])
        return json.loads(response.content)

    async def _structured_supervisor(self, state: QAState) -> Decision:
        """Ask the model for a typed routing decision, never free-form JSON."""
        if not self.llm:
            return Decision(next_agent=Role.VALIDATION, objective=state.get("objective", "QA"),
                            action="observe", target=state.get("target", "environment"),
                            justification="Collect missing facts before changing state.")
        model = self.llm.with_structured_output(Decision, method="function_calling")
        failures = [
            {
                "source": evidence.source,
                "target": evidence.target,
                "exit_code": evidence.exit_code,
                "stderr": evidence.stderr[-2000:],
                "argv": evidence.facts.get("argv") if isinstance(evidence.facts, dict) else None,
                "error_kind": evidence.facts.get("error_kind") if isinstance(evidence.facts, dict) else None,
                "recoverable": evidence.facts.get("recoverable", False) if isinstance(evidence.facts, dict) else False,
            }
            for evidence in state.get("evidence", [])[-20:]
            if evidence.exit_code not in (None, 0) or evidence.facts.get("ok") is False
        ]
        prompt = json.dumps({
            "objective": state.get("objective"),
            "target": state.get("target", "environment"),
            "phase": state.get("phase"),
            "evidence": [e.model_dump(mode="json") for e in state.get("evidence", [])[-20:]],
            "evidence_synthesis": state.get("evidence_synthesis", {}),
            "observed_signatures": list(state.get("observation_index", {}).keys())[-50:],
            "available_tools": list(self.tools.tools),
            "discovered_targets": state.get("discovered_targets", []),
            "recon_coverage": state.get("recon_coverage", {}),
            "no_progress_count": state.get("no_progress_count", 0),
            "method_history": state.get("method_history", [])[-30:],
            "tool_failures": failures,
            "ad_knowledge": state.get("ad_knowledge", {}),
            "target_profiles": state.get("target_profiles", {}),
            "runtime_config": state.get("runtime_config", {}),
            "operator_instruction": state.get("human_instruction", ""),
            "operator_instruction_history": state.get("human_directives", [])[-10:],
            "operator_rejections": [
                item for item in state.get("human_directives", [])[-10:]
                if item.get("intent") == "reject_previous"
            ],
            "approved_tool_parameters": state.get("last_decision").tool_parameters.model_dump(mode="json") if state.get("last_decision") else {},
            "capabilities": capability_catalog(),
            "instruction": "Read the complete evidence.stdout, evidence.stderr, facts, and output_summary fields; the CLI progress preview is not the evidence. Reason over the complete evidence synthesis, not only the last result. Advance one or more unresolved findings. Cover all discovered hosts/services and cross-forest candidates. Treat recon_coverage as authoritative: a completed semantic check is not a new task merely because the action wording changed. Never repeat an identical effective argv; a different reviewed argv/profile is allowed only when it has an explicit expected evidence gain. Treat LDAP authentication, DNS context, forest mismatch, permissions, and command syntax as different hypotheses. The latest operator instruction is semantic, high-priority execution guidance, not a single fixed command: extract all requested goals, constraints, exclusions, ordering, target changes, username sources, and requested continuation, then combine them with the autonomous plan. Do not discard later clauses after recognizing one tool name. If the operator rejected the previous proposal, never reproduce that action/target/parameters; choose the next justified autonomous alternative. If domain credentials are absent and domain/DC plus a candidate username source are known, prioritize asrep_roasting_assessment immediately; AS-REP does not require a domain credential. Do not loop over empty-credential SMB/LDAP/NXC probes. If no username source exists, use only explicitly allowed anonymous enumeration or ask Human for CYBERQA_AD_USERS_FILE. For a CIDR, host discovery must begin with nmap -sn or -F; after hosts are found, perform nmap -sC -sV (or an explicitly justified reviewed profile) against each authorized non-local host before declaring the network empty. If discovery returns no hosts, adapt with the other discovery method and inspect routing/DNS instead of stopping. For check_port choose a profile or safe argv deliberately; for NXC choose a profile or safe argv deliberately. Return one primary decision plus useful next_options and exact tool_parameters, including argv/users_file when supplied.",
        })
        self.progress("reasoning_start", agent=Role.SUPERVISOR.value)
        response = await model.ainvoke([
                SystemMessage(content=(
                    "You are the workflow supervisor for an authorized cyber-range QA agent. "
                    "Choose dynamically based on the conversation and evidence. Tool failures are "
                    "diagnostic evidence: send them to debugging, do not blindly repeat them. "
                    "Select an AD capability when applicable and fill prerequisites, expected_evidence, "
                    "risk, tool_parameters, and next_options. You may propose a multi-step chain; the execution broker "
                    "will enforce scope and approvals. Treat the latest operator instruction as an explicit constraint, "
                    "not optional context. "
                    "Do not execute tools. Return a Decision object."
            )),
            *self._conversation_context(state.get("messages", [])),
            HumanMessage(content=prompt),
        ])
        return response if isinstance(response, Decision) else Decision.model_validate(response)

    def _network_recon_transition(self, state: QAState) -> Decision | None:
        """Return the next staged Nmap action for an authorized network.

        This is a bounded guard, not a fixed AD workflow: the LLM remains free
        to choose identity, trust, or testing paths once each discovered host
        has a service baseline. It only prevents the common mistake of running
        -sC/-sV against a whole CIDR before host discovery, or stopping after
        an ICMP-filtered -sn result.
        """
        network = str(state.get("target", ""))
        if "/" not in network:
            return None
        coverage = state.get("recon_coverage", {}) or {}
        network_checks = coverage.get(network, {}).get("checks", {}) if isinstance(coverage.get(network), dict) else {}
        discovery_done = any(
            network_checks.get(f"nmap:{profile}", {}).get("status") == "completed"
            for profile in ("host_discovery", "fast")
        )
        if not discovery_done:
            return Decision(
                next_agent=Role.VALIDATION, objective="discover live authorized range hosts",
                action="network_host_discovery", target=network,
                justification="Start CIDR reconnaissance with host discovery before service detection.",
                tool_parameters=ToolParameters(profile="host_discovery"),
                expected_evidence=["discovered_targets"],
            )
        hosts = []
        for value in state.get("discovered_targets", []):
            host = str(value)
            if (
                host == network
                or is_local_target(host)
                or not self.tools.target_policy.allows(host)
            ):
                continue
            if host not in hosts:
                hosts.append(host)
        for host in hosts:
            profile = coverage.get(host, {})
            checks = profile.get("checks", {}) if isinstance(profile, dict) else {}
            has_service_baseline = any(
                checks.get(f"nmap:{candidate}", {}).get("status") == "completed"
                for candidate in ("default", "ad_tcp", "top100", "top1000")
            )
            if not has_service_baseline:
                return Decision(
                    next_agent=Role.VALIDATION, objective="enumerate services on discovered range host",
                    action="service_enumeration", target=host,
                    justification=(
                        "The host was discovered in the authorized network but has no service baseline. "
                        "Run the reviewed nmap -sC -sV probe, then adapt from the resulting services."
                    ),
                    tool_parameters=ToolParameters(profile="default"),
                    expected_evidence=["open_ports", "service_inventory"],
                )
        # If -sn produced no host facts, the initial_recon fallback normally
        # records fast. This leaves the next decision to the evidence-driven
        # Supervisor while still carrying the explicit no-host condition.
        return None

    def _react_graph(self, role: Role, state: QAState, instruction: str | None = None,
                     tool_names: list[str] | None = None):
        """Build one specialist's reason -> tools -> reason loop."""
        role_tool_names = {
            Role.VALIDATION: ("check_port", "check_dns_resolution", "ldap_bind", "smb_negotiate",
                              "http_health_check", "nxc_smb_recon", "nxc_ldap_recon",
                              "impacket_rpc_recon", "inspect_routes",
                              "inspect_dns_config"),
            Role.TESTING: ("ad_domain_users", "ad_asrep_roasting", "ad_kerberoasting",
                           "ad_credential_validation", "ad_password_spray", "ad_bloodhound_collection",
                           "nxc_smb_recon", "nxc_ldap_recon", "check_port",
                           "ldap_bind", "smb_negotiate"),
            Role.DEBUGGING: ("inspect_dns_config", "inspect_firewall", "inspect_routes", "inspect_time_sync",
                             "inspect_os_version", "inspect_os_release", "inspect_interfaces", "inspect_open_ports",
                             "inspect_acl", "inspect_local_users", "inspect_domain_users", "inspect_privileges",
                             "inspect_sudo", "check_port", "check_dns_resolution", "ldap_bind", "smb_negotiate",
                             "nxc_smb_recon", "nxc_ldap_recon", "impacket_rpc_recon"),
        }.get(role)
        available = [name for name in (role_tool_names or ()) if name in self.tools.tools]
        # A capability-specific list is preferred. If the Supervisor did not
        # provide one, keep the specialist inside its role tool set instead of
        # exposing every registered command to every specialist.
        selected_names = tool_names
        authorization = None
        if selected_names is None:
            decision = state.get("last_decision")
            capability = get_capability(decision.capability if decision else None)
            if capability:
                selected_names = [name for name in capability.allowed_tools if name in self.tools.tools]
            elif role in {Role.JUDGE, Role.REPORTING}:
                selected_names = []
            else:
                selected_names = available
            grant = state.get("approved_grant")
            if decision and grant and grant.get("decision_fingerprint") == decision_fingerprint(decision):
                authorization = grant
            if authorization is None:
                from .tools import SENSITIVE_TOOL_NAMES
                selected_names = [name for name in selected_names if name not in SENSITIVE_TOOL_NAMES]
        from .tools import SENSITIVE_TOOL_NAMES
        if authorization is None:
            selected_names = [name for name in selected_names if name not in SENSITIVE_TOOL_NAMES]
        decision = state.get("last_decision")
        recovery_mode = bool(state.get("recovery_mode"))
        has_ad_credentials = bool(os.getenv("CYBERQA_AD_DOMAIN") and
                                  os.getenv("CYBERQA_AD_USERNAME") and
                                  os.getenv("CYBERQA_AD_PASSWORD"))
        decision_parameters = decision.tool_parameters.model_dump() if decision else {}
        # Anonymous discovery is read-only and is the deliberate first branch
        # when no domain credential exists. The env flag still allows an
        # operator to disable or explicitly document that mode.
        allow_anonymous_nxc = bool(decision_parameters.get("allow_anonymous_nxc")) or os.getenv(
            "CYBERQA_ALLOW_ANONYMOUS_NXC", "0"
        ) == "1"
        if not recovery_mode and not has_ad_credentials and not allow_anonymous_nxc:
            selected_names = [name for name in selected_names
                              if name not in {"nxc_smb_recon", "nxc_ldap_recon"}]
        if not recovery_mode and not has_ad_credentials and decision and decision.action not in {
            "anonymous_identity_probe", "domain_inventory"
        }:
            # Anonymous LDAP/SMB is a bounded identity-discovery phase only;
            # do not let a later testing/debugging prompt quietly turn it into
            # repeated empty-credential reconnaissance.
            selected_names = [name for name in selected_names
                              if name not in {"ldap_bind", "smb_negotiate",
                                              "nxc_smb_recon", "nxc_ldap_recon"}]
        ldap_bound = any(
            "ldap" in item.source.lower() and item.exit_code in (None, 0)
            and item.facts.get("ok", True) is not False
            for item in state.get("evidence", [])
        )
        if not ldap_bound:
            selected_names = [name for name in selected_names if name != "ad_domain_users"]
        # An analysis-only decision must never expose the generic recon tools.
        # It exists to synthesize durable evidence after a completed probe; if
        # it is allowed to bind tools, the model can fall back to nmap/SMB and
        # recreate the reconnaissance loop this state is meant to prevent.
        if decision and decision.action in {"analyze_existing_evidence", "summarize_evidence"}:
            selected_names = []
        allowed = self.tools.langchain_tools(selected_names, authorization=authorization)
        analysis_only = decision and decision.action in {
            "analyze_existing_evidence", "summarize_evidence", "evaluate_ad_evidence"
        }
        model = (
            self.llm.bind_tools(allowed) if self.llm and allowed else
            self.llm if self.llm and (analysis_only or role in {Role.JUDGE, Role.REPORTING}) else None
        )
        inner = StateGraph(ReactState)

        async def reason(s: dict[str, Any]) -> dict[str, Any]:
            if model is None:
                return {"messages": [AIMessage(content="No model configured; finish with collected facts.")]}
            self.progress("reasoning_start", agent=role.value)
            response = await model.ainvoke([
                SystemMessage(content=(SYSTEM + f"\nYou are the {role.value} specialist. "
                                       "Do not choose another agent or route the workflow. "
                                       "For debugging, explain the observed tool error and choose a "
                                       "non-duplicate diagnostic or alternate target/service. "
                                       "When a tool result has recoverable=true, inspect its complete "
                                       "stderr/stdout and then correct the parameters, choose a different "
                                       "reviewed tool/profile, or pivot to the next justified AD path. "
                                       "Do not ask Human until the recovery budget is exhausted or the "
                                       "result explicitly requires operator context.")),
                HumanMessage(content=json.dumps({
                    "objective": state.get("objective"),
                    "target": state.get("last_decision").target if state.get("last_decision") else "environment",
                    "evidence": [e.model_dump(mode="json") for e in state.get("evidence", [])[-20:]],
                    "evidence_synthesis": state.get("evidence_synthesis", {}),
                    "target_profiles": state.get("target_profiles", {}),
                    "runtime_config": state.get("runtime_config", {}),
                    "ad_knowledge": state.get("ad_knowledge", {}),
                    "recovery_mode": recovery_mode,
                    "operator_instruction": state.get("human_instruction", ""),
                    "operator_instruction_history": state.get("human_directives", [])[-10:],
                    "capabilities": capability_catalog(),
                    "observed_signatures": list(state.get("observation_index", {}).keys())[-50:],
                    "instruction": instruction or (state.get("last_decision").justification if state.get("last_decision") else "Collect useful facts"),
                })),
                *self._react_context(s.get("messages", [])),
            ])
            self.progress("reasoned", agent=role.value,
                          tool_calls=[call.get("name") for call in getattr(response, "tool_calls", [])],
                          has_final_answer=not bool(getattr(response, "tool_calls", [])))
            return {"messages": [response]}

        def after_reason(s: dict[str, Any]) -> str:
            last = s.get("messages", [])[-1] if s.get("messages") else None
            return "tools" if getattr(last, "tool_calls", None) else "done"

        def trailing_tool_messages(s: dict[str, Any]) -> list[ToolMessage]:
            """Return every ToolMessage emitted by the latest ToolNode batch."""
            batch: list[ToolMessage] = []
            for message in reversed(s.get("messages", [])):
                if not isinstance(message, ToolMessage):
                    break
                batch.append(message)
            return list(reversed(batch))

        def tool_payload(message: ToolMessage) -> dict[str, Any] | None:
            try:
                payload = json.loads(message.content) if isinstance(message.content, str) else message.content
            except (TypeError, json.JSONDecodeError):
                return {"needs_human": True, "tool": message.name or "unknown",
                        "error": str(message.content)}
            return payload if isinstance(payload, dict) else None

        def after_inspect(s: dict[str, Any]) -> str:
            batch = trailing_tool_messages(s)
            if batch:
                payloads = [tool_payload(message) for message in batch]
                if any(payload and payload.get("needs_human") for payload in payloads):
                    # Missing executables, approval scope, and other
                    # non-recoverable failures still stop for Human. The
                    # outer specialist will preserve the whole batch.
                    return "done"
                recoverable = [payload for payload in payloads if payload and payload.get("recoverable")]
                if recoverable:
                    failures = s.get("recovery_failures", [])
                    for payload, message in zip(payloads, batch):
                        if payload and payload.get("recoverable"):
                            failure_key = f"{payload.get('tool', message.name)}:{payload.get('error_kind', 'tool_failure')}"
                            if len(failures) >= 3 or failures.count(failure_key) >= 2:
                                return "human"
                    return "reason"
                signatures = s.get("tool_signatures", [])
                for payload, message in zip(payloads, batch):
                    if payload and payload.get("signature") and (
                        payload.get("cached") or signatures.count(payload["signature"]) >= 2
                    ):
                        return "human" if not payload.get("ok", True) else "done"
                return "done"
            return "reason"

        def inspect_tools(s: dict[str, Any]) -> dict[str, Any]:
            batch = trailing_tool_messages(s)
            if not batch:
                return {}
            patch: dict[str, Any] = {}
            failed_signatures = list(s.get("failed_tool_signatures", []))
            recovery_failures = list(s.get("recovery_failures", []))
            tool_signatures = list(s.get("tool_signatures", []))
            for message in batch:
                payload = tool_payload(message)
                if not payload:
                    continue
                if payload.get("needs_human") or payload.get("recoverable"):
                    failure_key = f"{payload.get('tool', message.name or 'unknown')}:{payload.get('error_kind', 'tool_failure')}"
                    failed_signatures.append(failure_key)
                    recovery_failures.append(failure_key)
                if payload.get("signature"):
                    tool_signatures.append(payload["signature"])
            if failed_signatures != s.get("failed_tool_signatures", []):
                patch["failed_tool_signatures"] = failed_signatures
            if recovery_failures != s.get("recovery_failures", []):
                patch["recovery_failures"] = recovery_failures
                patch["recovery_steps"] = s.get("recovery_steps", 0) + len(
                    recovery_failures
                ) - len(s.get("recovery_failures", []))
            if tool_signatures != s.get("tool_signatures", []):
                patch["tool_signatures"] = tool_signatures
            return patch

        async def human(s: dict[str, Any]) -> dict[str, Any]:
            raw = str(s.get("messages", [])[-1].content if s.get("messages") else "")
            problem = await self._human_problem(state, "tool_failure", raw)
            request = {
                "kind": "tool_failure",
                "agent": role.value,
                "problem": problem,
                "question": "請說明要改用哪個診斷方向、目標或工具；也可以直接提供自然語言指示。",
                "options": ["retry_with_correction", "inspect_another_path", "abort"],
                "last_message": raw,
            }
            # Human interaction is handled by the outer graph. Returning a
            # request marker here prevents nested-subgraph interrupts from
            # swallowing stdin and makes CLI resume reliable.
            return {"needs_human": True, "human_request": request,
                    "messages": [AIMessage(content=json.dumps(request))]}

        inner.add_node("reason", reason)
        inner.add_node("tools", ToolNode(allowed))
        inner.add_node("inspect_tools", inspect_tools)
        inner.add_node("human", human)
        inner.add_edge(START, "reason")
        inner.add_conditional_edges("reason", after_reason, {"tools": "tools", "done": END})
        inner.add_edge("tools", "inspect_tools")
        inner.add_conditional_edges("inspect_tools", after_inspect, {"reason": "reason", "human": "human", "done": END})
        inner.add_edge("human", END)
        return inner.compile()

    async def initial_recon(self, state: QAState) -> dict[str, Any]:
        """Run a bounded baseline, then let the supervisor synthesize it."""
        baseline_tools = [
            "inspect_os_version", "inspect_os_release", "inspect_interfaces", "inspect_routes",
            "inspect_dns_config", "inspect_open_ports", "inspect_local_users", "inspect_privileges",
            "check_dns_resolution", "check_port",
        ]
        target = state.get("target", "environment")
        available = [
            name for name in baseline_tools
            if name in self.tools.tools
            # A CIDR is not a DNS name. Resolve discovered host/domain names
            # later; never issue a meaningless `dig CIDR` probe.
            and not (name == "check_dns_resolution" and "/" in target)
        ]
        evidence: list[Evidence] = []
        observation_index = dict(state.get("observation_index", {}))
        proposal: dict[str, Any] = {
            "phase": "initial_recon", "tools": available,
            "deferred": ["ldap_bind", "nxc_smb_recon", "nxc_ldap_recon", "impacket_rpc_recon"],
            "bounded": True,
        }
        for name in available:
            try:
                parameters: dict[str, Any] = {}
                if name == "check_port":
                    # A CIDR starts with host discovery. A single authorized
                    # host starts with a fast scan. Neither starts with the
                    # expensive -sC/-sV probe.
                    parameters = {"profile": "host_discovery" if "/" in target else "fast"}
                result = await self.tools.observe(name, target, "initial_recon", parameters)
                if result.get("evidence"):
                    observed = Evidence.model_validate(result["evidence"])
                    evidence.append(observed)
                    if name == "inspect_interfaces":
                        # The scanner's own interface addresses are context,
                        # never cyber-range targets to probe.
                        for value in re.findall(
                            r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", observed.stdout
                        ):
                            self.tools.target_policy.mark_local(value)
                elif not result.get("ok", False):
                    evidence.append(Evidence(source=f"tool:{name}", action="initial_recon", target=target,
                                             exit_code=-1, stderr=str(result.get("error", "tool failure")),
                                             facts={"ok": False, "tool_result": result}))
                if result.get("signature"):
                    observation_index[result["signature"]] = {
                        "tool": name, "target": target,
                        "action": "initial_recon", "ok": result.get("ok", False),
                        "cached": result.get("cached", False),
                    }
            except Exception as exc:
                evidence.append(Evidence(source=f"tool:{name}", action="initial_recon", target=target,
                                         exit_code=-1, stderr=str(exc), facts={"ok": False}))
        # ICMP discovery can be filtered in a range. Adapt once to a fast TCP
        # discovery scan before handing control to the Supervisor, instead of
        # treating an empty -sn result as an empty environment.
        if "/" in target and "check_port" in available:
            discovered = {
                host for item in evidence
                for host in (item.facts.get("discovered_targets", []) if isinstance(item.facts, dict) else [])
                if not is_local_target(str(host)) and self.tools.target_policy.allows(str(host))
            }
            if not discovered:
                try:
                    fallback = await self.tools.observe(
                        "check_port", target, "initial_recon_fallback", {"profile": "fast"}
                    )
                    if fallback.get("evidence"):
                        evidence.append(Evidence.model_validate(fallback["evidence"]))
                except Exception as exc:
                    evidence.append(Evidence(source="tool:check_port", action="initial_recon_fallback",
                                             target=target, exit_code=-1, stderr=str(exc), facts={"ok": False}))
        proposal["summary_ready"] = True
        event = Event(type="INITIAL_RECON_COMPLETE", run_id=state["run_id"], emitted_by=Role.VALIDATION,
                      target=target, evidence_ids=[item.id for item in evidence], payload=proposal)
        try:
            await self.events.publish(event)
        except Exception:
            pass
        projection = self._project_observations(state, evidence)
        discovered = {
            str(item) for item in state.get("discovered_targets", [])
            if not is_local_target(str(item))
        }
        for item in evidence:
            if not is_local_target(item.target):
                discovered.add(item.target)
            discovered.update(
                str(host) for host in item.facts.get("discovered_targets", [])
                if not is_local_target(str(host)) and self.tools.target_policy.allows(str(host))
            )
        method_history = list(state.get("method_history", []))
        for observed in evidence:
            method_history.append({
                "tool": observed.source,
                "action": observed.action,
                "target": observed.target,
                "outcome": "success" if observed.exit_code in (None, 0) else "failed",
                "exit_code": observed.exit_code,
                "evidence_id": observed.id,
                "argv": (observed.facts or {}).get("argv", []),
                "error_kind": (observed.facts or {}).get("error_kind"),
                "tool_result": (observed.facts or {}).get("tool_result"),
            })
        return {"evidence": evidence, "events": [event], "baseline_complete": True,
                "observation_index": observation_index, "discovered_targets": sorted(discovered),
                "method_history": method_history[-200:],
                **projection,
                "needs_human": bool(proposal.get("needs_human")),
                "human_requests": [proposal["human_request"]] if proposal.get("human_request") else [],
                "messages": [AIMessage(content=f"Initial reconnaissance collected {len(evidence)} evidence item(s). ")]}

    async def supervisor(self, state: QAState) -> dict[str, Any]:
        iteration = state.get("iteration", 0) + 1
        # Human guidance is an executable control input, not merely chat
        # context.  Preserve the frozen decision and its one-shot grant so a
        # subsequent LLM call cannot silently reinterpret "approve AS-REP" as
        # another reconnaissance suggestion.
        if state.get("human_directive") and state.get("last_decision"):
            decision = state["last_decision"]
            self.progress("supervisor_decision",
                          agent=(decision.next_agent.value if isinstance(decision.next_agent, Role) else str(decision.next_agent)),
                          action=decision.action, target=decision.target,
                          source="human_directive")
            return {
                "iteration": iteration,
                "phase": decision.next_agent,
                "last_decision": decision,
                "pending_action": decision.model_dump(),
                "approved_grant": state.get("approved_grant"),
                "human_instruction": "",
                # Keep this marker for the one specialist dispatch. The
                # specialist consumes it after executing the operator command.
                "human_directive": True,
                "needs_human": False,
            }
        if iteration > state.get("max_iterations", 20):
            decision = Decision(next_agent="end", objective="stop", action="end", target="environment", justification="Iteration budget exhausted; human guidance is required to continue.")
            return {"iteration": iteration, "phase": "human_help", "last_decision": decision,
                    "pending_action": decision.model_dump(), "needs_human": True,
                    "human_requests": [{
                        "kind": "iteration_budget",
                        "question": "自主迭代預算已用盡；請提供新的語意約束或輸入 abort。",
                        "reason": decision.justification,
                    }]}
        try:
            result = await self._structured_supervisor(state)
        except Exception as exc:
            decision = Decision(next_agent="end", objective="human_help", action="end", target="environment",
                                justification=f"Supervisor could not produce a valid decision: {exc}")
            return {"iteration": iteration, "phase": "human_help", "last_decision": decision,
                    "pending_action": decision.model_dump(), "needs_human": True,
                    "errors": [str(exc)],
                    "human_requests": [{
                        "kind": "supervisor_error",
                        "question": "Supervisor 無法產生下一步；請提供額外語意或輸入 abort。",
                        "reason": decision.justification,
                    }]}
        # Hard AD prerequisites outrank a free-form model proposal. The model
        # still plans all transitions where the deterministic guard has no
        # stronger conclusion, but it cannot select empty-credential loops or
        # skip an available unauthenticated AS-REP branch.
        result = recommend_ad_method(state) or result
        network_transition = self._network_recon_transition(state)
        if network_transition:
            # Complete the bounded host/service baseline before automatic AD
            # capability selection. A human explicit directive is handled
            # above and bypasses this guard intentionally.
            result = network_transition
        if result.next_agent == "end" and result.objective == "human_help":
            return {"iteration": iteration, "phase": "human_help", "last_decision": result,
                    "pending_action": result.model_dump(), "needs_human": True}
        result = self._redirect_completed_recon(state, result)
        agent = result.next_agent
        action = result.action
        requested_target = result.target
        target = requested_target if requested_target and requested_target != "environment" else state.get("target", "environment")
        # When discovery has produced additional hosts, make the next
        # validation step pay down coverage debt instead of repeatedly
        # returning to the first DC-shaped target.
        uncovered = [
            item for item in state.get("discovered_targets", [])
            if not is_local_target(str(item))
            and self.tools.target_policy.allows(str(item))
            and not self._target_has_completed_recon(state.get("recon_coverage", {}).get(item))
        ]
        if uncovered and result.next_agent == Role.VALIDATION:
            target = uncovered[0]
        if is_local_target(str(target)):
            # A model may mention the runner as context, but never as a
            # remote recon target. Prefer the next authorized host/network.
            target = next(
                (
                    str(item) for item in uncovered
                    if not is_local_target(str(item))
                ),
                next(
                    (
                        str(item) for item in state.get("discovered_targets", [])
                        if not is_local_target(str(item))
                        and self.tools.target_policy.allows(str(item))
                    ),
                    state.get("target", LOCAL_EXECUTION_TARGET),
                ),
            )
        decision = Decision(next_agent=agent, objective=result.objective or state.get("objective", "QA"),
                            action=action, target=target,
                            justification=result.justification or "Resolve the highest-value uncertainty.",
                            expected_information_gain=result.expected_information_gain,
                            approval_required=self.policy.requires_approval(action),
                            capability=result.capability, plan_id=result.plan_id,
                            prerequisites=result.prerequisites,
                            expected_evidence=result.expected_evidence,
                            risk=result.risk, next_options=result.next_options,
                            tool_parameters=result.tool_parameters)
        # Planner output is allowed to vary in wording, but an AS-REP request
        # must still resolve to the reviewed capability. Without this
        # normalization an action like ``ad_asrep_roasting_probe`` could reach
        # the tool layer with capability=None and receive an empty approval
        # grant, which looks like "approved but did nothing" to the operator.
        action_key = decision.action.lower().replace("-", "_")
        if "asrep" in action_key or "as_rep" in action_key:
            decision = decision.model_copy(update={
                "capability": "asrep_roasting_assessment",
                "risk": ADRisk.CREDENTIAL_MATERIAL,
                "approval_required": True,
            })
        if decision.capability:
            decision = decision.model_copy(update={
                "tool_parameters": normalize_capability_parameters(
                    decision.capability, decision.tool_parameters
                )
            })
        capability_check = self.broker.validate(
            decision, target,
            {item.get("signature") for item in state.get("capability_history", [])},
            self._known_prerequisites(state),
        )
        decision.approval_required = decision.approval_required or capability_check.get("requires_approval", False)
        if capability_check.get("missing_prerequisites"):
            missing = ", ".join(capability_check["missing_prerequisites"])
            decision = Decision(
                next_agent=Role.VALIDATION, objective=decision.objective,
                action="collect_prerequisites", target=target,
                justification=f"Blocked capability {result.capability or result.action}; collect: {missing}",
                expected_information_gain=decision.expected_information_gain,
                expected_evidence=decision.expected_evidence,
            )
            capability_check["blocked"] = True
        elif capability_check.get("duplicate"):
            decision = Decision(
                next_agent=Role.DEBUGGING, objective=decision.objective,
                action="choose_alternate_probe", target=target,
                justification="The selected capability was already observed; choose a materially different probe.",
            )
            capability_check["blocked"] = True
        self.progress("supervisor_decision", agent=(decision.next_agent.value if isinstance(decision.next_agent, Role) else str(decision.next_agent)), action=decision.action,
                      target=decision.target)
        signature = json.dumps({
            "capability": decision.capability,
            "action": decision.action,
            "target": decision.target,
            "tool_parameters": decision.tool_parameters.model_dump(mode="json", exclude_none=True),
        }, sort_keys=True)
        rejected_fingerprints = {
            str(item.get("rejected_fingerprint"))
            for item in state.get("human_directives", [])
            if item.get("intent") == "reject_previous" and item.get("rejected_fingerprint")
        }
        if decision_fingerprint(decision) in rejected_fingerprints:
            # A semantic rejection is an instruction to keep going by another
            # justified path, not permission to ask the same question again.
            decision = Decision(
                next_agent=Role.DEBUGGING,
                objective=decision.objective,
                action="choose_alternate_probe",
                target=target,
                justification=(
                    "The operator rejected the previous effective action. Select a materially different "
                    "authorized probe or continue with the next evidence-driven path."
                ),
            )
            signature = json.dumps({
                "capability": decision.capability,
                "action": decision.action,
                "target": decision.target,
                "tool_parameters": decision.tool_parameters.model_dump(mode="json", exclude_none=True),
            }, sort_keys=True)
        history = state.get("action_history", [])
        if history and history[-1] == signature:
            decision = Decision(next_agent="end", objective="stop", action="end", target=decision.target,
                                justification=(
                                    "The same effective decision was already dispatched immediately before. "
                                    "A changed target, credential/source, profile, or explicit operator instruction is required."
                                ))
            return {"iteration": iteration, "phase": "human_help", "last_decision": decision,
                    "pending_action": decision.model_dump(), "action_history": history + [signature],
                    "needs_human": True,
                    "human_requests": [{
                        "kind": "no_progress",
                        "question": "目前沒有新的自主證據路徑；請提供額外語意、目標/服務，或輸入 abort。",
                        "reason": decision.justification,
                    }]}
        capability_history = state.get("capability_history", [])
        capability_history = capability_history + [{**capability_check, "iteration": iteration}]
        reused_grant = None
        if decision.approval_required:
            fingerprint = decision_fingerprint(decision)
            previously_approved = any(
                getattr(item, "status", item.get("status") if isinstance(item, dict) else None) == "approved"
                and getattr(item, "decision_fingerprint", item.get("decision_fingerprint") if isinstance(item, dict) else None) == fingerprint
                for item in state.get("approvals", [])
            )
            if previously_approved:
                reused_grant = {
                    "decision_fingerprint": fingerprint,
                    "target": decision.target,
                    "action": decision.action,
                    "capability": decision.capability,
                    "allowed_tools": approved_tools_for_decision(decision),
                    "tool_parameters": decision.tool_parameters.model_dump(mode="json", exclude_none=True),
                }
                decision = decision.model_copy(update={"approval_required": False})
        return {"iteration": iteration, "phase": decision.next_agent, "last_decision": decision,
                "pending_action": {**decision.model_dump(), "broker": capability_check},
                "action_history": history + [signature], "capability_history": capability_history,
                "approved_grant": reused_grant,
                "human_instruction": ""}

    @staticmethod
    def _target_has_completed_recon(profile: Any) -> bool:
        if not profile:
            return False
        if isinstance(profile, dict) and "checks" in profile:
            return any(item.get("status") == "completed"
                       for item in profile.get("checks", {}).values())
        return bool(profile)

    @staticmethod
    def _completed_recon_decision(state: QAState, decision: Decision) -> bool:
        """Return true when the selected semantic check already completed."""
        text = f"{decision.capability or ''} {decision.action}".lower()
        parameters = decision.tool_parameters.model_dump(mode="json", exclude_none=True)
        category = None
        profile = str(parameters.get("profile") or "default")
        if "nmap" in text or "port" in text or "service_enumeration" in text or "service_detection" in text:
            category = "nmap"
        elif "nxc" in text and "ldap" in text:
            category, profile = "nxc_ldap", profile or "users"
        elif "nxc" in text or "smb" in text:
            category, profile = "nxc_smb", profile or "shares"
        elif "ldap" in text:
            category, profile = "ldap", profile if profile != "default" else "rootdse"
        elif "smb" in text:
            category, profile = "smb", profile if profile != "default" else "anonymous"
        if not category:
            return False
        target_profile = state.get("recon_coverage", {}).get(decision.target, {})
        checks = target_profile.get("checks", {}) if isinstance(target_profile, dict) else {}
        if category == "nxc_ldap" and profile == "default":
            profile = "users"
        if category == "nxc_smb" and profile == "default":
            profile = "shares"
        return checks.get(f"{category}:{profile}", {}).get("status") == "completed"

    @classmethod
    def _redirect_completed_recon(cls, state: QAState, decision: Decision) -> Decision:
        if not cls._completed_recon_decision(state, decision):
            return decision
        return decision.model_copy(update={
            "next_agent": Role.TESTING,
            "capability": None,
            "action": "analyze_existing_evidence",
            "justification": (
                "The selected semantic reconnaissance check is already completed for this target. "
                "Use the accumulated evidence to select an AD testing, trust, ACL, or reporting path."
            ),
            "tool_parameters": ToolParameters(),
        })

    @staticmethod
    def _human_asrep_decision(state: QAState, answer: str) -> tuple[Decision | None, bool, str | None]:
        """Turn an explicit human AS-REP instruction into a frozen decision.

        Returning a Decision here is important: a plain HumanMessage is not a
        command and the next supervisor prompt is free to ignore it.
        """
        text = answer.strip().lower().replace("–", "-")
        is_asrep = any(marker in text for marker in ("as-rep", "asrep", "as_rep", "getnpusers"))
        if not is_asrep:
            return None, False, None
        approved = any(marker in text for marker in (
            "approve", "approved", "allow", "run", "execute", "proceed", "核准", "允許", "執行",
        ))
        target = os.getenv("CYBERQA_AD_DC") or state.get("target", LOCAL_EXECUTION_TARGET)
        prior = state.get("last_decision")
        if prior and prior.target and not is_local_target(prior.target):
            target = prior.target
        if "/" in str(target):
            # GetNPUsers needs one DC address, not a CIDR. Prefer the runtime
            # discovered DC, then the first non-local discovered host.
            for candidate, profile in (state.get("target_profiles", {}) or {}).items():
                if (not is_local_target(str(candidate)) and "/" not in str(candidate)
                        and (profile.get("domain") or profile.get("connectivity") == "reachable")):
                    target = str(candidate)
                    break
            else:
                for candidate in state.get("discovered_targets", []):
                    if "/" not in str(candidate) and not is_local_target(str(candidate)):
                        target = str(candidate)
                        break
        if is_local_target(str(target)):
            target = next(
                (
                    str(candidate) for candidate in state.get("discovered_targets", [])
                    if "/" not in str(candidate) and not is_local_target(str(candidate))
                ),
                LOCAL_EXECUTION_TARGET,
            )
        knowledge = state.get("ad_knowledge") or {}
        if hasattr(knowledge, "model_dump"):
            knowledge = knowledge.model_dump(mode="json")
        parameters: dict[str, Any] = {}
        users_file = Agents._human_users_file(answer) or os.getenv("CYBERQA_AD_USERS_FILE")
        if users_file:
            parameters["users_file"] = users_file
        elif knowledge.get("asrep_candidates"):
            parameters["users"] = list(knowledge["asrep_candidates"][:500])
        elif knowledge.get("users"):
            parameters["users"] = list(knowledge["users"][:500])
        elif prior:
            parameters = prior.tool_parameters.model_dump(mode="json", exclude_none=True)
        if not parameters.get("users") and not parameters.get("users_file"):
            return None, approved, (
                "AS-REP roasting requires a candidate username source. Set CYBERQA_AD_USERS_FILE "
                "or provide a username list in the human response before approving."
            )
        decision = Decision(
            next_agent=Role.TESTING,
            objective=state.get("objective", "Assess authorized AS-REP roasting path"),
            action="asrep_roasting_assessment",
            target=target,
            justification="Human explicitly directed the agent to assess the authorized AS-REP path.",
            approval_required=not approved,
            capability="asrep_roasting_assessment",
            expected_evidence=["asrep_candidates", "ticket_obtained_or_blocked", "credential_validation_status"],
            risk=ADRisk.CREDENTIAL_MATERIAL,
            tool_parameters=ToolParameters.model_validate(parameters),
        )
        return decision, approved, None

    @staticmethod
    def _human_users_file(answer: str) -> str | None:
        """Extract a Linux username-list path from ordinary operator text."""
        patterns = (
            r"(?:cyberqa_ad_users_file|users?_file|username[_ -]?file|username[_ -]?list|使用者清單|帳號清單)"
            r"\s*(?:is|為|是|=|:)?\s*[\"']?([~/][^\s,;\"']+)",
            r"--users-file\s+[\"']?([~/][^\s,;\"']+)",
            r"(?:gain|get|obtain|retrieve|provide|取得|獲取|取得網域)"
            r".{0,80}?(?:by|from|using|透過|使用|用)\s*[\"']?([~/][^\s,;\"']+)",
            r"(?:domain\s+(?:cred|credential)|網域(?:憑證|帳密))"
            r".{0,80}?[\"']?([~/][^\s,;\"']+)",
        )
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                return match.group(1).rstrip(".,，。)")
        return None

    @staticmethod
    def _human_target(state: QAState, answer: str) -> str:
        """Use an explicitly mentioned authorized target, never localhost."""
        matches = re.findall(
            r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?(?![\w.])", answer
        )
        for candidate in matches:
            if not is_local_target(candidate):
                return candidate
        prior = state.get("last_decision")
        if prior and prior.target and prior.target != "environment" and not is_local_target(prior.target):
            return prior.target
        fallback = state.get("target", LOCAL_EXECUTION_TARGET)
        return fallback if not is_local_target(str(fallback)) else LOCAL_EXECUTION_TARGET

    @staticmethod
    def _apply_human_config(answer: str) -> tuple[dict[str, str], str]:
        """Apply safe operator-provided runtime values for this process.

        Only explicitly supported CyberQA settings are accepted. Passwords
        affect the current process but are never put into runtime_config or
        the conversation history in clear text.
        """
        allowed = {
            "CYBERQA_AD_DOMAIN", "CYBERQA_AD_DC", "CYBERQA_AD_BASE_DN",
            "CYBERQA_AD_USERS_FILE", "CYBERQA_AD_USERNAME", "CYBERQA_AD_PASSWORD",
            "CYBERQA_AD_COLLECTION", "CYBERQA_ALLOW_ANONYMOUS_NXC",
        }
        safe: dict[str, str] = {}
        for key, value in re.findall(r"\b(CYBERQA_[A-Z0-9_]+)\s*=\s*([^\s,;]+)", answer):
            if key not in allowed:
                continue
            value = value.strip("'\"")
            os.environ[key] = value
            if key != "CYBERQA_AD_PASSWORD":
                safe[key] = value
        users_file = Agents._human_users_file(answer)
        if users_file:
            os.environ["CYBERQA_AD_USERS_FILE"] = users_file
            safe["CYBERQA_AD_USERS_FILE"] = users_file
        redacted = answer
        for key in ("CYBERQA_AD_PASSWORD", "AD_PASSWORD"):
            redacted = re.sub(
                rf"({re.escape(key)}\s*=\s*)[^\s,;]+", r"\1***REDACTED***", redacted,
                flags=re.IGNORECASE,
            )
        return safe, redacted

    @classmethod
    def _human_explicit_decision(cls, state: QAState, answer: str) -> Decision | None:
        """Map common explicit operator tool instructions to a Decision.

        Unknown natural language remains in ``human_instruction`` and is sent
        to the structured Supervisor. Known reviewed tools are frozen here so
        a later model call cannot discard a direct operator command.
        """
        text = answer.lower().replace("–", "-")
        # Compound guidance belongs to the semantic Supervisor path. Do not
        # freeze only the first tool named in a multi-step instruction.
        if _is_multi_step_instruction(text):
            return None
        target = cls._human_target(state, answer)
        if is_local_target(target):
            return None
        role = Role.VALIDATION
        action = ""
        capability = None
        params: dict[str, Any] = {}
        if "nmap" in text:
            if "-sn" in text or "host discovery" in text or "主機發現" in text:
                action, params = "network_host_discovery", {"profile": "host_discovery"}
            elif re.search(r"(^|\s)-f(\s|$)", text) or "fast scan" in text:
                action, params = "network_fast_discovery", {"profile": "fast"}
            else:
                action, params = "service_enumeration", {"profile": "default"}
        elif "nxc" in text or "netexec" in text:
            role = Role.TESTING
            if "ldap" in text:
                action = "nxc_ldap_recon"
                params = {"profile": "users" if "user" in text else "groups" if "group" in text else "users",
                          "allow_anonymous_nxc": True}
            else:
                action = "nxc_smb_recon"
                params = {"profile": next((name for name in ("shares", "users", "groups", "sessions", "pass-pol")
                                            if name in text), "shares"),
                          "allow_anonymous_nxc": True}
        elif "bloodhound" in text or "sharphound" in text:
            role, action, capability = Role.TESTING, "bloodhound_collection", "bloodhound_collection"
        elif "impacket" in text or "rpcdump" in text:
            action = "impacket_rpc_recon"
        elif "ldap" in text:
            action = "ldap_bind_probe"
        elif "smb" in text:
            action = "smb_negotiate_probe"
        elif "dns" in text and ("resolve" in text or "解析" in text):
            action = "check_dns_resolution"
        elif "http" in text or "curl" in text:
            action = "http_health_check"
        if not action:
            return None
        if role == Role.TESTING and capability:
            spec = get_capability(capability)
            requires_approval = bool(spec and spec.requires_approval) or capability == "bloodhound_collection"
        else:
            requires_approval = False
        return Decision(
            next_agent=role,
            objective=state.get("objective", "Follow the operator instruction"),
            action=action,
            target=target,
            justification=f"Human explicitly instructed: {answer.strip()}",
            approval_required=requires_approval,
            capability=capability,
            tool_parameters=ToolParameters.model_validate(params),
        )

    async def human_help(self, state: QAState) -> dict[str, Any]:
        """Pause the outer workflow when the supervisor detects no progress."""
        decision = state.get("last_decision")
        problem = await self._human_problem(
            state, "no_progress", decision.justification if decision else "No supervisor decision"
        )
        requests = state.get("human_requests") or []
        request = dict(requests[-1] or {}) if requests else {}
        request.update({
            "kind": request.get("kind", "no_progress"),
            "problem": problem,
            "question": request.get(
                "question",
                "請用自然語言指定下一步：檢查哪個目標/服務、如何修正，或是否停止。",
            ),
            "options": request.get("options", ["validation", "testing", "debugging", "abort"]),
            "reason": decision.justification if decision else "No supervisor decision",
            "evidence_summary": "; ".join(problem.splitlines()[-2:]),
        })
        answer = interrupt(request)
        answer_text = str(answer).strip()
        guidance = answer_text.lower()
        runtime_config, safe_answer = self._apply_human_config(answer_text)
        rejected_previous = _is_rejection_instruction(answer_text)
        previous_decision = state.get("last_decision")
        directive_record: dict[str, Any] = {
            "instruction": safe_answer,
            "source": "human",
            "intent": "reject_previous" if rejected_previous else "semantic_guidance",
        }
        if rejected_previous and previous_decision and previous_decision.action != "end":
            directive_record.update({
                "rejected_fingerprint": decision_fingerprint(previous_decision),
                "rejected_action": previous_decision.action,
                "rejected_target": previous_decision.target,
                "rejected_parameters": previous_decision.tool_parameters.model_dump(
                    mode="json", exclude_none=True
                ),
            })
        # A human response is a deliberate change of direction.  Clear the
        # stale decision and let the Supervisor interpret the full meaning of
        # the response.  A short "no" rejects the previous proposal; it does
        # not terminate the whole task or get reinterpreted as a new command.
        patch = {"needs_human": False, "no_progress_count": 0, "action_history": [],
                "last_decision": None, "pending_action": None, "approved_grant": None,
                "human_directive": False,
                "messages": [HumanMessage(content=f"Human guidance for supervisor: {safe_answer}")],
                "human_instruction": safe_answer,
                "human_directives": [directive_record],
                "errors": [] if not _is_abort_instruction(answer_text) else ["Human aborted after no progress"],
                "aborted": _is_abort_instruction(answer_text)}
        human_decision, approved, directive_error = self._human_asrep_decision(state, answer_text)
        if directive_error:
            patch.update({
                "needs_human": True,
                "human_directive": False,
                "human_requests": [{
                    "kind": "human_help",
                    "question": directive_error,
                    "options": ["set CYBERQA_AD_USERS_FILE", "provide usernames", "abort"],
                    "reason": directive_error,
                }],
            })
        elif human_decision:
            grant = None
            if approved:
                grant = {
                    "decision_fingerprint": decision_fingerprint(human_decision),
                    "target": human_decision.target,
                    "action": human_decision.action,
                    "capability": human_decision.capability,
                    "allowed_tools": approved_tools_for_decision(human_decision),
                    "tool_parameters": human_decision.tool_parameters.model_dump(mode="json", exclude_none=True),
                }
                human_decision = human_decision.model_copy(update={"approval_required": False})
            patch.update({
                "last_decision": human_decision,
                "pending_action": human_decision.model_dump(),
                "approved_grant": grant,
                "human_directive": True,
                "needs_human": False,
            })
        else:
            generic_decision = self._human_explicit_decision(state, answer_text)
            if generic_decision:
                human_approved = any(marker in guidance for marker in (
                    "approve", "approved", "allow", "run", "execute", "proceed", "核准", "允許", "執行",
                ))
                grant = None
                if human_approved and generic_decision.approval_required:
                    grant = {
                        "decision_fingerprint": decision_fingerprint(generic_decision),
                        "target": generic_decision.target,
                        "action": generic_decision.action,
                        "capability": generic_decision.capability,
                        "allowed_tools": approved_tools_for_decision(generic_decision),
                        "tool_parameters": generic_decision.tool_parameters.model_dump(mode="json", exclude_none=True),
                    }
                    generic_decision = generic_decision.model_copy(update={"approval_required": False})
                patch.update({
                    "last_decision": generic_decision,
                    "pending_action": generic_decision.model_dump(),
                    "approved_grant": grant,
                    "human_directive": True,
                    "needs_human": False,
                    "human_directives": [{
                        "instruction": safe_answer, "source": "human",
                        "action": generic_decision.action, "target": generic_decision.target,
                    }],
                })
        if runtime_config:
            apply_and_persist_runtime_config(runtime_config)
            patch["runtime_config"] = runtime_config
        if not patch["aborted"] and state.get("iteration", 0) >= state.get("max_iterations", 20):
            patch["max_iterations"] = state.get("iteration", 0) + 5
        return patch

    @staticmethod
    def _grant_for_decision(state: QAState, decision: Decision | None) -> dict[str, Any] | None:
        """Hydrate grants from pre-fix checkpoints without widening scope.

        Older checkpoints stored only the fingerprint, allowed tools, and
        parameters. Adding the frozen target/action fields is safe because
        they are taken from the current supervisor decision; parameters are
        never replaced here. This lets an already-approved AS-REP action
        continue after an upgrade instead of asking for approval again.
        """
        if not decision or not state.get("approved_grant"):
            return None
        grant = dict(state["approved_grant"])
        grant.setdefault("target", decision.target)
        grant.setdefault("action", decision.action)
        grant.setdefault("capability", decision.capability)
        grant["tool_parameters"] = normalize_capability_parameters(
            decision.capability, grant.get("tool_parameters", {})
        ).model_dump(mode="json", exclude_none=True)
        if not grant.get("allowed_tools"):
            grant["allowed_tools"] = approved_tools_for_decision(decision)
        return grant

    @staticmethod
    def _planned_tool_for_action(decision: Decision | None) -> tuple[str, dict[str, Any]] | None:
        """Map explicit probe actions to one reviewed adapter.

        The model may still choose a higher-level capability, but once it
        names a concrete probe the executor must run it even if the model
        returns an empty tool-call list. This closes the "分析完成" no-op path.
        """
        if not decision:
            return None
        text = decision.action.lower().replace("-", "_")
        params = decision.tool_parameters.model_dump(mode="json", exclude_none=True)
        if ("nmap" in text or "check_port" in text or "port_probe" in text
                or "service_enumeration" in text or "service_detection" in text
                or "network_host_discovery" in text or "network_fast_discovery" in text):
            return "check_port", {
                "profile": params.get(
                    "profile",
                    "host_discovery" if "host_discovery" in text else
                    "fast" if "fast_discovery" in text else "default",
                ),
                **({"argv": params["argv"]} if params.get("argv") else {}),
            }
        if "nxc" in text and "ldap" in text:
            return "nxc_ldap_recon", {
                "profile": params.get("profile", "users"),
                "argv": params.get("argv", []),
                "allow_anonymous_nxc": params.get("allow_anonymous_nxc", False),
            }
        if "nxc" in text or "smb_recon" in text:
            return "nxc_smb_recon", {
                "profile": params.get("profile", "shares"),
                "argv": params.get("argv", []),
                "allow_anonymous_nxc": params.get("allow_anonymous_nxc", False),
            }
        if "ldap_bind" in text or text in {"ldap", "ldap_probe"}:
            return "ldap_bind", {
                "profile": params.get("profile", "rootdse"),
                "argv": params.get("argv", []),
            }
        if "smb_negotiate" in text or text in {"smb", "smb_probe"}:
            return "smb_negotiate", {
                "profile": params.get("profile", "anonymous"),
                "argv": params.get("argv", []),
            }
        if "dns" in text and "resolution" in text:
            return "check_dns_resolution", ({"name": params["name"]} if params.get("name") else {})
        return None

    async def _run_react_loop(self, role: Role, state: QAState,
                              instruction: str | None = None) -> tuple[list[Any], dict[str, Any] | None, str | None]:
        """Stream one specialist ReAct loop and return its messages/request.

        Planned capability probes use this same path after a recoverable
        failure, with the failed Evidence already inserted into ``state``.
        That makes repair reasoning see the exact command output instead of a
        prose-only error summary.
        """
        react_messages: list[Any] = []
        human_request: dict[str, Any] | None = None
        error: str | None = None
        try:
            async for update in self._react_graph(role, state, instruction=instruction).astream(
                {"messages": self._conversation_context(state.get("messages", []))},
                stream_mode="updates",
            ):
                for patch in update.values() if isinstance(update, dict) else ():
                    if isinstance(patch, dict):
                        react_messages.extend(patch.get("messages", []))
                        if patch.get("needs_human"):
                            human_request = patch.get("human_request")
        except Exception as exc:
            error = str(exc)
            self.progress("agent_error", agent=role.value, error=error)
        return react_messages, human_request, error

    async def specialist(self, role: Role, state: QAState) -> dict[str, Any]:
        decision = state.get("last_decision")
        target, action = (decision.target, decision.action) if decision else ("environment", "observe")
        evidence = []
        proposal: dict[str, Any] = {}
        new_observation = False
        inner_needs_human = False
        inner_human_request: dict[str, Any] | None = None
        repair_context: str | None = None
        react_messages: list[Any] = []
        react_error: str | None = None
        observation_index = dict(state.get("observation_index", {}))
        decision_grant = self._grant_for_decision(state, decision)
        # A selected capability is an executable workflow obligation, not a
        # suggestion for the ReAct model.  In particular, AS-REP used to be
        # selected in the log and then silently skipped when the model replied
        # with an empty tool-call list. Execute the reviewed adapter once after
        # approval, then let the model reason over its evidence.
        capability_tool = None
        if decision:
            capability_name = (decision.capability or "").lower()
            action_name = action.lower()
            if "asrep" in capability_name or "as-rep" in action_name or "asrep" in action_name:
                capability_tool = "ad_asrep_roasting"
            else:
                capability = get_capability(decision.capability)
                if capability:
                    capability_tool = next(
                        (name for name in capability.allowed_tools
                         if name.startswith("ad_") and name in self.tools.tools),
                        None,
                    )
        planned_calls: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
        blocked_action = False
        if capability_tool:
            capability_parameters = normalize_capability_parameters(
                decision.capability, decision.tool_parameters
            ).model_dump(mode="json", exclude_none=True)
            planned_calls.append((
                capability_tool,
                capability_parameters,
                decision_grant,
            ))
        elif action == "anonymous_identity_probe":
            # This is a bounded phase, not a ReAct playground. Each anonymous
            # identity source is attempted once; successful user discovery
            # becomes the input to the subsequent AS-REP decision.
            prior_sources = {str(item.source).lower() for item in state.get("evidence", [])}
            identity_plan = (
                ("ldap_bind", {}),
                ("smb_negotiate", {}),
                ("nxc_ldap_recon", {"profile": "users", "allow_anonymous_nxc": True}),
            )
            for name, parameters in identity_plan:
                if any(name in source for source in prior_sources):
                    continue
                if name in self.tools.tools:
                    planned_calls.append((name, parameters, None))
        elif not capability_tool:
            action_tool = self._planned_tool_for_action(decision)
            if action_tool and action_tool[0] in self.tools.tools:
                no_credentials = not bool(os.getenv("CYBERQA_AD_USERNAME") and os.getenv("CYBERQA_AD_PASSWORD"))
                anonymous_tool = action_tool[0] in {"ldap_bind", "smb_negotiate",
                                                    "nxc_smb_recon", "nxc_ldap_recon"}
                explicit_human_path = bool(state.get("human_directive"))
                explicit_nxc_anonymous = bool(action_tool[1].get("allow_anonymous_nxc"))
                if no_credentials and anonymous_tool and not explicit_human_path and not explicit_nxc_anonymous and action not in {
                    "anonymous_identity_probe", "domain_inventory"
                }:
                    blocked_action = True
                    inner_needs_human = True
                    proposal = {"action": action, "needs_human": True,
                                "error": "Anonymous identity tools are restricted to the bounded identity phase"}
                    inner_human_request = {
                        "kind": "method_prerequisite", "agent": role.value, "target": target,
                        "question": "目前沒有 domain credential；請提供 username source、指定 bounded identity phase，或停止。",
                    }
                else:
                    planned_calls.append((action_tool[0], action_tool[1], decision_grant))
        if planned_calls:
            for planned_tool, parameters, authorization in planned_calls:
                try:
                    result = await self.tools.observe(
                        planned_tool, target, action, parameters=parameters,
                        authorization=authorization,
                    )
                    proposal = {"tool": planned_tool, "capability": decision.capability if decision else None,
                                "result_ok": result.get("ok", False),
                                "planned_phase": action}
                    if result.get("evidence"):
                        observed = Evidence.model_validate(result["evidence"])
                        evidence.append(observed)
                        new_observation = self._evidence_is_novel(
                            state, observed, bool(result.get("cached", False))
                        ) or new_observation
                        if result.get("signature"):
                            observation_index[result["signature"]] = {
                                "tool": planned_tool, "target": observed.target,
                                "action": observed.action, "ok": result.get("ok", False),
                                "cached": result.get("cached", False),
                                "exit_code": observed.exit_code,
                                "error_kind": result.get("error_kind"),
                                "recoverable": result.get("recoverable", False),
                            }
                    if result.get("needs_human") and action == "anonymous_identity_probe":
                        # Independent anonymous paths are allowed to fail
                        # independently. Continue the bounded phase so an LDAP
                        # bind failure does not prevent SMB/NXC from yielding a
                        # username source.
                        proposal.setdefault("failures", []).append({
                            "tool": planned_tool, "error": result.get("error", "tool failure")
                        })
                        continue
                    if result.get("recoverable") and not result.get("needs_human"):
                        # Keep the failed command as evidence, then let the
                        # specialist's ReAct loop select corrected parameters,
                        # another reviewed adapter, or a different AD path.
                        if not result.get("evidence"):
                            evidence.append(Evidence(
                                source=f"tool:{planned_tool}", action=action, target=target,
                                exit_code=-1, stderr=str(result.get("error", "tool failure")),
                                facts={
                                    "ok": False, "recoverable": True,
                                    "error_kind": result.get("error_kind", "tool_failure"),
                                    "parameters": parameters,
                                    "tool_result": result,
                                },
                            ))
                            new_observation = True
                        repair_context = (
                            f"The planned {planned_tool} call failed but is recoverable. "
                            f"error_kind={result.get('error_kind')}; error={result.get('error', '')}. "
                            "Read the complete failed evidence and choose one materially different "
                            "repair: corrected parameters, an alternate LDAP/SMB/NXC method, "
                            "or the next justified attack-path assessment. Do not repeat the same "
                            "effective command."
                        )
                        proposal.update({
                            "recoverable_failure": True,
                            "error_kind": result.get("error_kind"),
                            "error": result.get("error"),
                        })
                        continue
                    if result.get("needs_human"):
                        inner_needs_human = True
                        proposal.update({"needs_human": True, "error": result.get("error")})
                        if not result.get("evidence"):
                            evidence.append(Evidence(
                                source=f"tool:{planned_tool}", action=action, target=target,
                                exit_code=-1, stderr=str(result.get("error", "tool failure")),
                                facts={"ok": False, "needs_human": True,
                                       "error_kind": result.get("error_kind", "tool_failure")},
                            ))
                        inner_human_request = {
                            "kind": "tool_failure", "agent": role.value,
                            "tool": planned_tool, "target": target,
                            "error": result.get("error", "tool failure"),
                            "question": "請修正目前工具/參數或停止；不會自動重複同一身份探測。",
                        }
                        break
                except Exception as exc:
                    if action == "anonymous_identity_probe":
                        proposal.setdefault("failures", []).append({
                            "tool": planned_tool, "error": str(exc)
                        })
                        continue
                    inner_needs_human = True
                    proposal = {"tool": planned_tool, "planned_phase": action,
                                "needs_human": True, "error": str(exc)}
                    evidence.append(Evidence(source=f"tool:{planned_tool}", action=action,
                                             target=target, exit_code=-1, stderr=str(exc),
                                             facts={"ok": False, "needs_human": True}))
                    inner_human_request = {
                        "kind": "tool_failure", "agent": role.value,
                        "tool": planned_tool, "target": target, "error": str(exc),
                        "question": "請修正目前工具/參數或停止；不會自動重複同一身份探測。",
                    }
                    break
        # A planned LDAP/SMB/NXC call is executed first so that the repair
        # loop starts with the exact stderr/stdout and argv that failed.  The
        # failed result is deliberately added to a projected state instead of
        # being reduced to a prose error string.
        if repair_context and self.llm and not inner_needs_human:
            repair_state = dict(state)
            repair_state["evidence"] = [*state.get("evidence", []), *evidence]
            repair_state.update(self._project_observations(state, evidence))
            repair_state["observation_index"] = observation_index
            repair_state["recovery_mode"] = True
            repair_messages, repair_human_request, react_error = await self._run_react_loop(
                role, repair_state, instruction=repair_context
            )
            react_messages.extend(repair_messages)
            if repair_human_request:
                inner_needs_human = True
                inner_human_request = repair_human_request
        elif repair_context and not self.llm:
            inner_needs_human = True
            proposal["needs_human"] = True
            inner_human_request = {
                "kind": "analysis_engine_unavailable", "agent": role.value,
                "target": target,
                "question": "工具結果可由 Agent 修復，但目前沒有可用的 LLM；請設定模型、提供修正指示，或停止。",
            }
        elif not planned_calls and blocked_action:
            pass
        elif not planned_calls and action == "anonymous_identity_probe":
            proposal = {"planned_phase": action, "phase_complete": True}
            inner_needs_human = True
            inner_human_request = {
                "kind": "identity_phase_unavailable", "agent": role.value,
                "target": target,
                "question": "匿名身份探測沒有可執行的新工具；請提供 username source、修正工具環境，或停止。",
            }
        elif not planned_calls and self.llm and self.tools.tools:
            react_messages, inner_human_request, react_error = await self._run_react_loop(role, state)
            if inner_human_request:
                inner_needs_human = True

        # Consume both normal ReAct results and repair-loop results in one
        # place.  A non-zero exit is not automatically Human: the registry's
        # recoverable flag is the contract that lets the model inspect the
        # failure and attempt a corrected parameter, alternate adapter, or
        # different attack path first.
        if react_error:
            proposal["error"] = react_error
            proposal["needs_human"] = True
            inner_needs_human = True
            evidence.append(Evidence(
                source=f"agent:{role.value}", action=action, target=target,
                exit_code=-1, stderr=react_error,
                facts={"ok": False, "agent_error": True},
            ))

        for message in react_messages:
            if isinstance(message, ToolMessage):
                try:
                    payload = json.loads(message.content) if isinstance(message.content, str) else message.content
                    if not isinstance(payload, dict):
                        continue
                    observed = None
                    if payload.get("evidence"):
                        observed = Evidence.model_validate(payload["evidence"])
                        evidence.append(observed)
                        new_observation = new_observation or self._evidence_is_novel(
                            state, observed, bool(payload.get("cached", False))
                        )
                        if payload.get("signature"):
                            observation_index[payload["signature"]] = {
                                "tool": payload.get("tool", message.name),
                                "target": observed.target,
                                "action": observed.action,
                                "ok": payload.get("ok", True),
                                "cached": payload.get("cached", False),
                                "exit_code": observed.exit_code,
                                "error_kind": payload.get("error_kind"),
                                "recoverable": payload.get("recoverable", False),
                            }

                    failed = bool(payload.get("needs_human")) or bool(
                        observed is not None and observed.exit_code not in (None, 0)
                    )
                    recoverable = bool(payload.get("recoverable")) and not bool(payload.get("needs_human"))
                    if failed and recoverable:
                        proposal.update({
                            "recoverable_failure": True,
                            "error_kind": payload.get("error_kind"),
                            "error": payload.get("error") or (
                                observed.stderr if observed is not None else "tool failed"
                            ),
                        })
                    elif failed:
                        inner_needs_human = True
                        proposal["needs_human"] = True
                        inner_human_request = {
                            "kind": "tool_failure", "agent": role.value,
                            "tool": payload.get("tool", message.name),
                            "target": observed.target if observed is not None else target,
                            "error": payload.get("error") or (
                                observed.stderr if observed is not None else "tool failed"
                            ),
                            "question": "請指定修正方向或停止；自動修復預算已用盡或工具需要操作員權限。",
                        }
                    elif payload.get("recoverable"):
                        # Invalid-argument failures may not have an Evidence
                        # object, but they must still be durable context for
                        # the outer Supervisor.
                        tool_name = payload.get("tool", message.name or "unknown")
                        synthetic = Evidence(
                            source=f"tool:{tool_name}", action=action, target=target,
                            exit_code=-1, stderr=str(payload.get("error", "tool failure")),
                            facts={
                                "ok": False, "recoverable": True,
                                "error_kind": payload.get("error_kind", "tool_failure"),
                                "tool_result": payload,
                            },
                        )
                        evidence.append(synthetic)
                        new_observation = new_observation or not payload.get("cached", False)
                        proposal.update({
                            "recoverable_failure": True,
                            "error_kind": payload.get("error_kind"),
                            "error": payload.get("error", "tool failure"),
                        })
                    if payload.get("signature") and payload.get("signature") not in observation_index:
                        observation_index[payload["signature"]] = {
                            "tool": payload.get("tool", message.name),
                            "target": observed.target if observed is not None else target,
                            "action": action,
                            "ok": payload.get("ok", False),
                            "cached": payload.get("cached", False),
                            "error": payload.get("error", "tool failure"),
                            "error_kind": payload.get("error_kind"),
                            "recoverable": payload.get("recoverable", False),
                        }
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            if isinstance(message, AIMessage) and message.content:
                proposal["summary"] = message.content
        if not planned_calls and not react_messages and self.tools.tools and not (action in {
            "analyze_existing_evidence", "summarize_evidence", "evaluate_ad_evidence"
        } or role in {Role.JUDGE, Role.REPORTING}):
            # Offline mode remains deterministic, but uses the same allow-listed adapter boundary.
            tool_name = role.value if role.value in self.tools.tools else next(iter(self.tools.tools))
            try:
                result = await self.tools.observe(tool_name, target, action)
                if result.get("evidence"):
                    evidence.append(Evidence.model_validate(result["evidence"]))
                elif not result.get("ok", False):
                    evidence.append(Evidence(source=f"tool:{tool_name}", action=action, target=target,
                                             exit_code=-1, stderr=str(result.get("error", "tool failure")),
                                             facts={"ok": False, "tool_result": result}))
                    proposal = {"tool": tool_name, "offline": True, "error": result.get("error"),
                                "needs_human": True}
                    new_observation = True
                    raise RuntimeError(str(result.get("error", "tool failure")))
            except Exception as exc:
                self.progress("tool_result", tool=tool_name, exit_code=-1, stderr=str(exc), stdout="")
                proposal = {"tool": tool_name, "offline": True, "error": str(exc), "needs_human": True}
            else:
                proposal = {"tool": tool_name, "offline": True}
                new_observation = True
        elif action in {"analyze_existing_evidence", "summarize_evidence", "evaluate_ad_evidence"} \
                or role in {Role.JUDGE, Role.REPORTING}:
            # Analysis/reporting is not a licence to execute the first
            # registered command when the LLM is unavailable. Preserve the
            # accumulated evidence and surface the missing analysis engine.
            proposal = {"analysis_only": True, "offline": True,
                        "needs_human": not bool(self.llm)}
            if not self.llm:
                inner_needs_human = True
                inner_human_request = {
                    "kind": "analysis_engine_unavailable", "agent": role.value,
                    "target": target,
                    "question": "目前沒有可用的 LLM 來彙整既有 evidence；請修正模型設定或停止。",
                }
        event_type = {Role.VALIDATION: "SERVICE_VALIDATED", Role.TESTING: "ATTACK_PATH_VALIDATED",
                      Role.DEBUGGING: "REPAIR_COMPLETED", Role.JUDGE: "SCENARIO_EVALUATED",
                      Role.REPORTING: "REPORT_UPDATED"}[role]
        event = Event(type=event_type, run_id=state["run_id"], emitted_by=role, target=target,
                      evidence_ids=[e.id for e in evidence], payload=proposal)
        try:
            await self.events.publish(event)
        except Exception as exc:
            self.progress("event_error", event_type=event.type, error=str(exc))
            proposal.setdefault("event_error", str(exc))
        self.progress("agent_done", agent=role.value, evidence_count=len(evidence), target=target)
        discovered_targets = {
            str(item) for item in state.get("discovered_targets", [])
            if not is_local_target(str(item))
        }
        ad_knowledge = dict(state.get("ad_knowledge", {}))
        ad_knowledge.setdefault("coverage", {})
        for observed in evidence:
            if not is_local_target(observed.target):
                discovered_targets.add(observed.target)
            facts = observed.facts if isinstance(observed.facts, dict) else {}
            discovered_targets.update(
                str(host) for host in facts.get("discovered_targets", [])
                if not is_local_target(str(host)) and self.tools.target_policy.allows(str(host))
            )
            coverage = set()
            for service in facts.get("open_ports", []):
                coverage.add(f"{service.get('protocol', 'tcp')}/{service.get('port')}/{service.get('service')}")
            previous_service_coverage = set(ad_knowledge["coverage"].get(observed.target, []))
            ad_knowledge["coverage"][observed.target] = sorted(previous_service_coverage | coverage | {observed.source})
            for field in ("users", "spns", "asrep_candidates", "credentials_validated", "groups", "acl_edges",
                          "delegation", "adcs_findings", "trusts"):
                values = set(ad_knowledge.get(field, []))
                values.update(str(item) for item in facts.get(field, []))
                ad_knowledge[field] = sorted(values)
            if facts.get("domain_name"):
                ad_knowledge["domain"] = facts["domain_name"]
        projection = self._project_observations(state, evidence)
        method_history = list(state.get("method_history", []))
        for observed in evidence:
            method_history.append({
                "tool": observed.source,
                "action": observed.action,
                "target": observed.target,
                "outcome": "success" if observed.exit_code in (None, 0) else "failed",
                "exit_code": observed.exit_code,
                "evidence_id": observed.id,
                "argv": (observed.facts or {}).get("argv", []),
                "error_kind": (observed.facts or {}).get("error_kind"),
                "tool_result": (observed.facts or {}).get("tool_result"),
            })
        method_history = method_history[-200:]
        patch: dict[str, Any] = {
            "evidence": evidence,
            "events": [event],
            "react_steps": state.get("react_steps", 0) + 1,
            "messages": [AIMessage(content=(
                f"{role.value} completed its current step and collected {len(evidence)} evidence item(s)."
            ))],
            "observation_index": observation_index,
            "method_history": method_history,
            "no_progress_count": 0 if new_observation else state.get("no_progress_count", 0) + 1,
            "discovered_targets": sorted(discovered_targets),
            "ad_knowledge": ad_knowledge,
            "approved_grant": None,
            **projection,
            "needs_human": inner_needs_human or bool(proposal.get("needs_human")),
            "human_requests": [inner_human_request] if inner_human_request else [],
            "human_directive": False,
        }
        if role == Role.DEBUGGING and action == "generate_hypotheses":
            patch["hypotheses"] = [Hypothesis(statement=x, likelihood=.5) for x in proposal.get("hypotheses", [])]
        if role == Role.JUDGE:
            patch["scorecard"] = Scorecard(solvable=True, difficulty="appropriate", scenario_status="evaluated", score=proposal.get("score", 80), findings=proposal.get("findings", []))
        return patch

    async def approval(self, state: QAState) -> dict[str, Any]:
        decision = state["last_decision"]
        fingerprint = decision_fingerprint(decision)
        request = self.policy.request(decision.action, decision.target, decision.justification,
                                      [e.id for e in state.get("evidence", [])[-10:]])
        from uuid import NAMESPACE_URL, uuid5
        request.id = str(uuid5(NAMESPACE_URL, f"{state['run_id']}:{fingerprint}"))
        request.decision_fingerprint = fingerprint
        answer = interrupt({"kind": "approval", "request": request.model_dump(mode="json"),
                            "question": "Approve this action? Reply approve or reject."})
        answer_text = str(answer).strip()
        request.status = "approved" if answer_text.lower() in {"approve", "approved", "yes"} else "rejected"
        # Also accept an explicit combined instruction at an approval prompt,
        # e.g. "approve to as-rep roasting".  This changes the frozen
        # executable decision before the grant is created; it is not left as
        # unstructured chat for the supervisor to reinterpret.
        requested_decision, requested_approved, directive_error = self._human_asrep_decision(state, answer_text)
        if not requested_decision and not directive_error:
            requested_decision = self._human_explicit_decision(state, answer_text)
            requested_approved = requested_decision is not None and any(marker in answer_text.lower() for marker in (
                "approve", "approved", "allow", "run", "execute", "proceed", "核准", "允許", "執行",
            ))
        if requested_decision and requested_approved:
            decision = requested_decision
            fingerprint = decision_fingerprint(decision)
            request.action = decision.action
            request.target = decision.target
            request.reason = decision.justification
            request.decision_fingerprint = fingerprint
            request.status = "approved"
        elif directive_error:
            request.status = "rejected"
        event = Event(type="APPROVAL_DECIDED", run_id=state["run_id"], emitted_by=Role.SUPERVISOR,
                      target=decision.target, payload=request.model_dump())
        try:
            await self.events.publish(event)
        except Exception as exc:
            self.progress("event_error", event_type=event.type, error=str(exc))
        approved_decision = decision.model_copy(update={"approval_required": False})
        grant = {
            "decision_fingerprint": fingerprint,
            "target": decision.target,
            "action": decision.action,
            "capability": decision.capability,
            "allowed_tools": approved_tools_for_decision(decision),
            "tool_parameters": decision.tool_parameters.model_dump(mode="json", exclude_none=True),
        } if request.status == "approved" else None
        return {"approvals": [request], "events": [event], "pending_action": None,
                "last_decision": approved_decision if request.status == "approved" else decision,
                "approved_grant": grant,
                "messages": [HumanMessage(content=f"Human approval result: {request.status}" )],
                "aborted": request.status == "rejected"}
