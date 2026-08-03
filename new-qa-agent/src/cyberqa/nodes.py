from __future__ import annotations

import json
import os
from typing import Any, Annotated, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from .approval import ApprovalPolicy, decision_fingerprint
from .ad_playbooks import capability_catalog, get_capability
from .discovery import (apply_and_persist_runtime_config, build_target_profiles,
                         derive_runtime_config, synthesize_evidence)
from .events import EventBus
from .execution_broker import CapabilityBroker
from .models import Decision, Event, Evidence, Hypothesis, Role, Scorecard, Service, ServiceProtocol
from .state import QAState
from .tools import ToolRegistry


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


class ReactState(TypedDict, total=False):
    """Small private state contract used by each specialist ReAct subgraph."""
    messages: Annotated[list[Any], add_messages]
    failed_tool_signatures: list[str]
    tool_signatures: list[str]
    needs_human: bool
    human_request: dict[str, Any]


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

    def _project_observations(self, state: QAState, new_evidence: list[Evidence]) -> dict[str, Any]:
        """Build one cumulative view used by every future planning decision."""
        all_evidence = [*state.get("evidence", []), *new_evidence]
        old_profiles = state.get("target_profiles", {})
        old_knowledge = state.get("ad_knowledge", {}) or {}
        if hasattr(old_knowledge, "model_dump"):
            old_knowledge = old_knowledge.model_dump()
        profiles = build_target_profiles(all_evidence, old_profiles, old_knowledge.get("domain"))
        synthesis = synthesize_evidence(all_evidence, profiles)
        runtime = derive_runtime_config(all_evidence, profiles, state.get("runtime_config", {}))
        if runtime:
            apply_and_persist_runtime_config(runtime)
        knowledge = dict(old_knowledge)
        for field in ("users", "spns", "asrep_candidates", "groups", "acl_edges",
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
                "runtime_config": runtime, "ad_knowledge": knowledge}

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
        decision = state.get("last_decision")
        if decision and (decision.tool_parameters.get("users") or decision.tool_parameters.get("users_file")):
            known.add("user enumeration")
        if knowledge.get("credentials_validated"):
            known.add("validated domain credential")
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
        model = self.llm.with_structured_output(Decision)
        failures = [
            {
                "source": evidence.source,
                "target": evidence.target,
                "exit_code": evidence.exit_code,
                "stderr": evidence.stderr[-2000:],
                "argv": evidence.facts.get("argv") if isinstance(evidence.facts, dict) else None,
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
            "tool_failures": failures,
            "ad_knowledge": state.get("ad_knowledge", {}),
            "target_profiles": state.get("target_profiles", {}),
            "runtime_config": state.get("runtime_config", {}),
            "approved_tool_parameters": state.get("last_decision").tool_parameters if state.get("last_decision") else {},
            "capabilities": capability_catalog(),
            "instruction": "Reason over the complete evidence synthesis, not only the last result. Advance one or more unresolved findings. Cover all discovered hosts/services and cross-forest candidates. Never repeat an identical effective argv; a different reviewed argv/profile is allowed. Treat LDAP authentication, DNS context, forest mismatch, permissions, and command syntax as different hypotheses. If domain credentials are absent, do not loop over empty-credential SMB/LDAP/NXC probes; use domain discovery, anonymous LDAP only when justified, a supplied users_file for AS-REP assessment, or another evidence-backed path. For check_port choose a profile deliberately; for NXC choose shares/users/groups/sessions/pass-pol/enum deliberately. Return one primary decision plus useful next_options and exact tool_parameters, including users_file when supplied.",
        })
        self.progress("reasoning_start", agent=Role.SUPERVISOR.value)
        response = await model.ainvoke([
                SystemMessage(content=(
                    "You are the workflow supervisor for an authorized cyber-range QA agent. "
                    "Choose dynamically based on the conversation and evidence. Tool failures are "
                    "diagnostic evidence: send them to debugging, do not blindly repeat them. "
                    "Select an AD capability when applicable and fill prerequisites, expected_evidence, "
                    "risk, tool_parameters, and next_options. You may propose a multi-step chain; the execution broker "
                    "will enforce scope and approvals. "
                    "Do not execute tools. Return a Decision object."
            )),
            *self._conversation_context(state.get("messages", [])),
            HumanMessage(content=prompt),
        ])
        return response if isinstance(response, Decision) else Decision.model_validate(response)

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
        has_ad_credentials = bool(os.getenv("CYBERQA_AD_DOMAIN") and
                                  os.getenv("CYBERQA_AD_USERNAME") and
                                  os.getenv("CYBERQA_AD_PASSWORD"))
        allow_anonymous_nxc = bool(decision and decision.tool_parameters.get("allow_anonymous_nxc"))
        if not has_ad_credentials and not allow_anonymous_nxc:
            selected_names = [name for name in selected_names
                              if name not in {"nxc_smb_recon", "nxc_ldap_recon"}]
        allowed = self.tools.langchain_tools(selected_names, authorization=authorization)
        model = self.llm.bind_tools(allowed) if self.llm and allowed else None
        inner = StateGraph(ReactState)

        async def reason(s: dict[str, Any]) -> dict[str, Any]:
            if model is None:
                return {"messages": [AIMessage(content="No model configured; finish with collected facts.")]}
            self.progress("reasoning_start", agent=role.value)
            response = await model.ainvoke([
                SystemMessage(content=(SYSTEM + f"\nYou are the {role.value} specialist. "
                                       "Do not choose another agent or route the workflow. "
                                       "For debugging, explain the observed tool error and choose a "
                                       "non-duplicate diagnostic or alternate target/service.")),
                HumanMessage(content=json.dumps({
                    "objective": state.get("objective"),
                    "target": state.get("last_decision").target if state.get("last_decision") else "environment",
                    "evidence": [e.model_dump(mode="json") for e in state.get("evidence", [])[-20:]],
                    "evidence_synthesis": state.get("evidence_synthesis", {}),
                    "target_profiles": state.get("target_profiles", {}),
                    "runtime_config": state.get("runtime_config", {}),
                    "ad_knowledge": state.get("ad_knowledge", {}),
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

        def after_inspect(s: dict[str, Any]) -> str:
            last = s.get("messages", [])[-1] if s.get("messages") else None
            if isinstance(last, ToolMessage):
                try:
                    result = json.loads(last.content) if isinstance(last.content, str) else last.content
                    if isinstance(result, dict) and result.get("needs_human"):
                        # The failed result is already durable evidence. Stop
                        # this specialist and let the outer Supervisor route
                        # to debugging; retrying here is how identical nmap
                        # commands used to escape the cache in practice.
                        return "done"
                    if isinstance(result, dict) and result.get("signature"):
                        signatures = s.get("tool_signatures", [])
                        current = result["signature"]
                        # A cached/repeated observation has no new information;
                        # stop this sub-agent rather than spending more calls.
                        if result.get("cached") or signatures.count(current) >= 2:
                            return "done"
                except (TypeError, json.JSONDecodeError):
                    return "reason"
            return "reason"

        def inspect_tools(s: dict[str, Any]) -> dict[str, Any]:
            last = s.get("messages", [])[-1] if s.get("messages") else None
            if not isinstance(last, ToolMessage):
                return {}
            try:
                payload = json.loads(last.content) if isinstance(last.content, str) else last.content
            except (TypeError, json.JSONDecodeError):
                payload = {"needs_human": True, "tool": last.name or "unknown", "error": str(last.content)}
            if isinstance(payload, dict) and payload.get("needs_human"):
                signature = f"{payload.get('tool', last.name or 'unknown')}:{payload.get('error', 'unknown tool failure')}"
                patch = {"failed_tool_signatures": s.get("failed_tool_signatures", []) + [signature]}
            else:
                patch = {}
            if isinstance(payload, dict) and payload.get("signature"):
                patch["tool_signatures"] = s.get("tool_signatures", []) + [payload["signature"]]
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
        available = [name for name in baseline_tools if name in self.tools.tools]
        evidence: list[Evidence] = []
        observation_index = dict(state.get("observation_index", {}))
        proposal: dict[str, Any] = {
            "phase": "initial_recon", "tools": available,
            "deferred": ["ldap_bind", "nxc_smb_recon", "nxc_ldap_recon", "impacket_rpc_recon"],
            "bounded": True,
        }
        for name in available:
            try:
                result = await self.tools.observe(name, target, "initial_recon")
                if result.get("evidence"):
                    observed = Evidence.model_validate(result["evidence"])
                    evidence.append(observed)
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
        proposal["summary_ready"] = True
        event = Event(type="INITIAL_RECON_COMPLETE", run_id=state["run_id"], emitted_by=Role.VALIDATION,
                      target=target, evidence_ids=[item.id for item in evidence], payload=proposal)
        try:
            await self.events.publish(event)
        except Exception:
            pass
        projection = self._project_observations(state, evidence)
        discovered = set(state.get("discovered_targets", []))
        for item in evidence:
            discovered.add(item.target)
            discovered.update(item.facts.get("discovered_targets", []))
        return {"evidence": evidence, "events": [event], "baseline_complete": True,
                "observation_index": observation_index, "discovered_targets": sorted(discovered),
                **projection,
                "needs_human": bool(proposal.get("needs_human")),
                "human_requests": [proposal["human_request"]] if proposal.get("human_request") else [],
                "messages": [AIMessage(content=f"Initial reconnaissance collected {len(evidence)} evidence item(s). ")]}

    async def supervisor(self, state: QAState) -> dict[str, Any]:
        iteration = state.get("iteration", 0) + 1
        if state.get("no_progress_count", 0) >= 2:
            decision = Decision(next_agent="end", objective="human_help", action="end",
                                target=state.get("target", "environment"),
                                justification="Two consecutive specialist steps produced no new observations.")
            return {"iteration": iteration, "phase": "human_help", "last_decision": decision,
                    "pending_action": decision.model_dump(), "needs_human": True}
        if iteration > state.get("max_iterations", 20):
            decision = Decision(next_agent="end", objective="stop", action="end", target="environment", justification="Iteration budget exhausted; human guidance is required to continue.")
            return {"iteration": iteration, "phase": "human_help", "last_decision": decision,
                    "pending_action": decision.model_dump(), "needs_human": True}
        try:
            result = await self._structured_supervisor(state)
        except Exception as exc:
            decision = Decision(next_agent="end", objective="human_help", action="end", target="environment",
                                justification=f"Supervisor could not produce a valid decision: {exc}")
            return {"iteration": iteration, "phase": "human_help", "last_decision": decision,
                    "pending_action": decision.model_dump(), "needs_human": True,
                    "errors": [str(exc)]}
        agent = result.next_agent
        action = result.action
        requested_target = result.target
        target = requested_target if requested_target and requested_target != "environment" else state.get("target", "environment")
        # When discovery has produced additional hosts, make the next
        # validation step pay down coverage debt instead of repeatedly
        # returning to the first DC-shaped target.
        uncovered = [
            item for item in state.get("discovered_targets", [])
            if not state.get("recon_coverage", {}).get(item)
        ]
        if uncovered and result.next_agent == Role.VALIDATION:
            target = uncovered[0]
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
        signature = f"{action}:{decision.target}"
        history = state.get("action_history", [])
        if len(history) >= 3 and history[-3:] == [signature] * 3:
            decision = Decision(next_agent="end", objective="stop", action="end", target=decision.target,
                                justification="The same action produced no new information three times.")
            return {"iteration": iteration, "phase": "human_help", "last_decision": decision,
                    "pending_action": decision.model_dump(), "action_history": history + [signature],
                    "needs_human": True}
        capability_history = state.get("capability_history", [])
        capability_history = capability_history + [{**capability_check, "iteration": iteration}]
        return {"iteration": iteration, "phase": decision.next_agent, "last_decision": decision,
                "pending_action": {**decision.model_dump(), "broker": capability_check},
                "action_history": history + [signature], "capability_history": capability_history,
                "approved_grant": None}

    async def human_help(self, state: QAState) -> dict[str, Any]:
        """Pause the outer workflow when the supervisor detects no progress."""
        decision = state.get("last_decision")
        problem = await self._human_problem(
            state, "no_progress", decision.justification if decision else "No supervisor decision"
        )
        request = {"kind": "no_progress", "problem": problem,
                   "question": "請用自然語言指定下一步：檢查哪個目標/服務、如何修正，或是否停止。",
                   "options": ["validation", "testing", "debugging", "abort"],
                   "reason": decision.justification if decision else "No supervisor decision",
                   "evidence_summary": "; ".join(problem.splitlines()[-2:])}
        answer = interrupt(request)
        guidance = str(answer).lower()
        # A human response is a deliberate change of direction.  Clear the
        # guard that caused this pause, otherwise the supervisor immediately
        # interrupts again before it can evaluate the guidance.
        patch = {"needs_human": False, "no_progress_count": 0, "action_history": [],
                "messages": [HumanMessage(content=f"Human guidance for supervisor: {answer}")],
                "errors": [] if guidance != "abort" else ["Human aborted after no progress"],
                "aborted": guidance == "abort"}
        if not patch["aborted"] and state.get("iteration", 0) >= state.get("max_iterations", 20):
            patch["max_iterations"] = state.get("iteration", 0) + 5
        return patch

    async def specialist(self, role: Role, state: QAState) -> dict[str, Any]:
        decision = state.get("last_decision")
        target, action = (decision.target, decision.action) if decision else ("environment", "observe")
        evidence = []
        proposal: dict[str, Any] = {}
        new_observation = False
        inner_needs_human = False
        inner_human_request: dict[str, Any] | None = None
        if self.llm and self.tools.tools:
            react_messages: list[Any] = []
            try:
                async for update in self._react_graph(role, state).astream(
                    {"messages": self._conversation_context(state.get("messages", []))},
                    stream_mode="updates",
                ):
                    for patch in update.values() if isinstance(update, dict) else ():
                        if isinstance(patch, dict):
                            react_messages.extend(patch.get("messages", []))
                            if patch.get("needs_human"):
                                inner_needs_human = True
                                inner_human_request = patch.get("human_request")
            except Exception as exc:
                self.progress("agent_error", agent=role.value, error=str(exc))
                proposal["error"] = str(exc)
                proposal["needs_human"] = True
                from .models import Evidence
                evidence.append(Evidence(source=f"agent:{role.value}", action=action, target=target,
                                         exit_code=-1, stderr=str(exc),
                                         facts={"ok": False, "agent_error": True}))
            observation_index = dict(state.get("observation_index", {}))
            for message in react_messages:
                if isinstance(message, ToolMessage):
                    try:
                        payload = json.loads(message.content) if isinstance(message.content, str) else message.content
                        if not isinstance(payload, dict):
                            continue
                        if payload.get("evidence"):
                            from .models import Evidence
                            observed = Evidence.model_validate(payload["evidence"])
                            evidence.append(observed)
                            new_observation = new_observation or not payload.get("cached", False)
                            if payload.get("signature"):
                                observation_index[payload["signature"]] = {
                                    "tool": payload.get("tool", message.name),
                                    "target": observed.target,
                                    "action": observed.action,
                                    "ok": payload.get("ok", True),
                                    "cached": payload.get("cached", False),
                                    "exit_code": observed.exit_code,
                                }
                        elif payload.get("needs_human"):
                            from .models import Evidence
                            evidence.append(Evidence(
                                source=f"tool:{payload.get('tool', message.name or 'unknown')}",
                                action=action, target=target, exit_code=-1,
                                stderr=str(payload.get("error", "tool failure")),
                                facts={"ok": False, "needs_human": True, "tool_result": payload},
                            ))
                            new_observation = new_observation or not payload.get("cached", False)
                            if payload.get("signature"):
                                observation_index[payload["signature"]] = {
                                    "tool": payload.get("tool", message.name),
                                    "target": target,
                                    "action": action,
                                    "ok": False,
                                    "cached": payload.get("cached", False),
                                    "error": payload.get("error", "tool failure"),
                                }
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                if isinstance(message, AIMessage) and message.content:
                    proposal["summary"] = message.content
        elif self.tools.tools:
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
        discovered_targets = set(state.get("discovered_targets", []))
        recon_coverage = dict(state.get("recon_coverage", {}))
        ad_knowledge = dict(state.get("ad_knowledge", {}))
        ad_knowledge.setdefault("coverage", {})
        for observed in evidence:
            discovered_targets.add(observed.target)
            facts = observed.facts if isinstance(observed.facts, dict) else {}
            discovered_targets.update(facts.get("discovered_targets", []))
            coverage = set(recon_coverage.get(observed.target, []))
            coverage.add(observed.source)
            for service in facts.get("open_ports", []):
                coverage.add(f"{service.get('protocol', 'tcp')}/{service.get('port')}/{service.get('service')}")
            recon_coverage[observed.target] = sorted(coverage)
            for field in ("users", "spns", "asrep_candidates", "groups", "acl_edges",
                          "delegation", "adcs_findings", "trusts"):
                values = set(ad_knowledge.get(field, []))
                values.update(str(item) for item in facts.get(field, []))
                ad_knowledge[field] = sorted(values)
            if facts.get("domain_name"):
                ad_knowledge["domain"] = facts["domain_name"]
            ad_knowledge["coverage"][observed.target] = recon_coverage[observed.target]
        projection = self._project_observations(state, evidence)
        patch: dict[str, Any] = {
            "evidence": evidence,
            "events": [event],
            "react_steps": state.get("react_steps", 0) + 1,
            "messages": [AIMessage(content=(
                f"{role.value} completed its current step and collected {len(evidence)} evidence item(s)."
            ))],
            "observation_index": observation_index,
            "no_progress_count": 0 if new_observation else state.get("no_progress_count", 0) + 1,
            "discovered_targets": sorted(discovered_targets),
            "recon_coverage": recon_coverage,
            "ad_knowledge": ad_knowledge,
            **projection,
            "needs_human": inner_needs_human or bool(proposal.get("needs_human")),
            "human_requests": [inner_human_request] if inner_human_request else [],
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
        answer = interrupt({"kind": "approval", "request": request.model_dump(mode="json"),
                            "question": "Approve this action? Reply approve or reject."})
        request.status = "approved" if str(answer).strip().lower() in {"approve", "approved", "yes"} else "rejected"
        event = Event(type="APPROVAL_DECIDED", run_id=state["run_id"], emitted_by=Role.SUPERVISOR,
                      target=decision.target, payload=request.model_dump())
        try:
            await self.events.publish(event)
        except Exception as exc:
            self.progress("event_error", event_type=event.type, error=str(exc))
        approved_decision = decision.model_copy(update={"approval_required": False})
        grant = {
            "decision_fingerprint": fingerprint,
            "allowed_tools": get_capability(decision.capability).allowed_tools if get_capability(decision.capability) else [],
            "tool_parameters": decision.tool_parameters,
        } if request.status == "approved" else None
        return {"approvals": [request], "events": [event], "pending_action": None,
                "last_decision": approved_decision if request.status == "approved" else decision,
                "approved_grant": grant,
                "messages": [HumanMessage(content=f"Human approval result: {request.status}" )],
                "aborted": request.status == "rejected"}
