from __future__ import annotations

import json
from typing import Any, Annotated, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from .approval import ApprovalPolicy
from .events import EventBus
from .models import Decision, Event, Hypothesis, Role, Scorecard, Service, ServiceProtocol
from .state import QAState
from .tools import ToolRegistry


SYSTEM = """You are a cyber-range QA specialist operating only on authorized targets. Use OODA:
observe facts, orient against the objective and prior evidence, decide one justified action, and act
through the supplied fact-only tools. Inspect every tool result before selecting the next tool. Continue
until the objective is complete. Never invent facts, credentials, vulnerabilities, or successful attacks."""


class ReactState(TypedDict, total=False):
    """Small private state contract used by each specialist ReAct subgraph."""
    messages: Annotated[list[Any], add_messages]
    failed_tool_signatures: list[str]
    tool_signatures: list[str]


class Agents:
    def __init__(self, llm: BaseChatModel | None = None, tools: ToolRegistry | None = None,
                 events: EventBus | None = None, policy: ApprovalPolicy | None = None,
                 on_progress: Callable[[str, dict[str, Any]], None] | None = None):
        self.llm, self.tools, self.events, self.policy = llm, tools or ToolRegistry(), events or EventBus(), policy or ApprovalPolicy()
        self.on_progress = on_progress

    def progress(self, event: str, **data: Any) -> None:
        if self.on_progress:
            self.on_progress(event, data)

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
            "observed_signatures": list(state.get("observation_index", {}).keys())[-50:],
            "available_tools": list(self.tools.tools),
            "no_progress_count": state.get("no_progress_count", 0),
            "tool_failures": failures,
            "instruction": "Choose the highest-value next specialist or end. Do not assume a fixed phase order. Never repeat a cached observation or re-run an identical failed command. If tool_failures are present, route to debugging and use the exact stderr, exit code, and argv to diagnose arguments, credentials, permissions, connectivity, or timeout before choosing a replacement probe.",
        })
        self.progress("reasoning_start", agent=Role.SUPERVISOR.value)
        response = await model.ainvoke([
                SystemMessage(content=(
                    "You are the workflow supervisor for an authorized cyber-range QA agent. "
                    "Choose dynamically based on the conversation and evidence. Tool failures are "
                    "diagnostic evidence: send them to debugging, do not blindly repeat them. "
                    "Do not execute tools. Return a Decision object."
            )),
            *self._conversation_context(state.get("messages", [])),
            HumanMessage(content=prompt),
        ])
        return response if isinstance(response, Decision) else Decision.model_validate(response)

    def _react_graph(self, role: Role, state: QAState):
        """Build one specialist's reason -> tools -> reason loop."""
        role_tool_names = {
            Role.VALIDATION: ("check_port", "check_dns_resolution", "ldap_bind", "kerberos_request_ticket",
                              "smb_negotiate", "winrm_execute_probe", "http_health_check", "database_connectivity"),
            Role.TESTING: ("enumerate_spns", "request_kerberos_tickets", "check_asrep_accounts",
                           "validate_ntlm_relay_prerequisites", "enumerate_adcs_templates",
                           "test_authorized_attack_path", "retrieve_flag"),
            Role.DEBUGGING: ("inspect_dns_config", "inspect_firewall", "inspect_routes", "inspect_time_sync",
                             "inspect_ldap_config", "inspect_replication", "restart_service", "correct_dns",
                             "correct_route", "sync_time"),
        }.get(role)
        available = [name for name in (role_tool_names or ()) if name in self.tools.tools]
        # The registry is the security boundary. Specialists may use any
        # registered fact tool when evidence shows that the original role
        # assumption was wrong; routing remains dynamically controlled by the
        # Supervisor instead of a fixed phase sequence.
        allowed = self.tools.langchain_tools()
        model = self.llm.bind_tools(allowed) if self.llm and allowed else None
        inner = StateGraph(ReactState)

        async def reason(s: dict[str, Any]) -> dict[str, Any]:
            if model is None:
                return {"messages": [AIMessage(content="No model configured; finish with collected facts.")]}
            self.progress("reasoning_start", agent=role.value)
            response = await model.ainvoke([
                SystemMessage(content=(SYSTEM + f"\nYou are the {role.value} specialist. "
                                       "Do not choose another agent or route the workflow.")),
                HumanMessage(content=json.dumps({
                    "objective": state.get("objective"),
                    "target": state.get("last_decision").target if state.get("last_decision") else "environment",
                    "evidence": [e.model_dump(mode="json") for e in state.get("evidence", [])[-20:]],
                    "observed_signatures": list(state.get("observation_index", {}).keys())[-50:],
                    "instruction": state.get("last_decision").justification if state.get("last_decision") else "Collect useful facts",
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
                        signature = s.get("failed_tool_signatures", [])
                        tool_name = result.get("tool", last.name or "unknown")
                        error = result.get("error", "unknown tool failure")
                        current = f"{tool_name}:{error}"
                        repeats = sum(item == current for item in signature)
                        return "human" if repeats >= 2 else "reason"
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
            request = {
                "kind": "tool_failure",
                "agent": role.value,
                "question": "A tool failed or returned an unusable result. Choose a safe next step.",
                "options": ["retry", "inspect_another_path", "abort"],
                "last_message": str(s.get("messages", [])[-1].content if s.get("messages") else ""),
            }
            answer = interrupt(request)
            return {"messages": [HumanMessage(content=f"Human guidance: {answer}")]}

        inner.add_node("reason", reason)
        inner.add_node("tools", ToolNode(allowed))
        inner.add_node("inspect_tools", inspect_tools)
        inner.add_node("human", human)
        inner.add_edge(START, "reason")
        inner.add_conditional_edges("reason", after_reason, {"tools": "tools", "done": END})
        inner.add_edge("tools", "inspect_tools")
        inner.add_conditional_edges("inspect_tools", after_inspect, {"reason": "reason", "human": "human", "done": END})
        inner.add_edge("human", "reason")
        return inner.compile()

    async def supervisor(self, state: QAState) -> dict[str, Any]:
        iteration = state.get("iteration", 0) + 1
        if state.get("no_progress_count", 0) >= 2:
            decision = Decision(next_agent="end", objective="human_help", action="end",
                                target=state.get("target", "environment"),
                                justification="Two consecutive specialist steps produced no new observations.")
            return {"iteration": iteration, "phase": "human_help", "last_decision": decision,
                    "pending_action": decision.model_dump(), "needs_human": True}
        if iteration >= state.get("max_iterations", 20):
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
        decision = Decision(next_agent=agent, objective=result.objective or state.get("objective", "QA"),
                            action=action, target=target,
                            justification=result.justification or "Resolve the highest-value uncertainty.",
                            expected_information_gain=result.expected_information_gain,
                            approval_required=self.policy.requires_approval(action))
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
        return {"iteration": iteration, "phase": decision.next_agent, "last_decision": decision,
                "pending_action": decision.model_dump(), "action_history": history + [signature]}

    async def human_help(self, state: QAState) -> dict[str, Any]:
        """Pause the outer workflow when the supervisor detects no progress."""
        decision = state.get("last_decision")
        request = {"kind": "no_progress", "question": "The workflow made no progress. Choose the next safe direction.",
                   "options": ["retry", "validation", "testing", "debugging", "abort"],
                   "reason": decision.justification if decision else "No supervisor decision"}
        answer = interrupt(request)
        guidance = str(answer).lower()
        # A human response is a deliberate change of direction.  Clear the
        # guard that caused this pause, otherwise the supervisor immediately
        # interrupts again before it can evaluate the guidance.
        return {"needs_human": False, "no_progress_count": 0, "action_history": [],
                "messages": [HumanMessage(content=f"Human guidance for supervisor: {answer}")],
                "errors": [] if guidance != "abort" else ["Human aborted after no progress"],
                "aborted": guidance == "abort"}

    async def specialist(self, role: Role, state: QAState) -> dict[str, Any]:
        decision = state.get("last_decision")
        target, action = (decision.target, decision.action) if decision else ("environment", "observe")
        evidence = []
        proposal: dict[str, Any] = {}
        new_observation = False
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
                evidence.append(await self.tools.get(tool_name).observe(target, action))
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
        patch: dict[str, Any] = {
            "evidence": evidence,
            "events": [event],
            "react_steps": state.get("react_steps", 0) + 1,
            "messages": [AIMessage(content=(
                f"{role.value} completed its current step and collected {len(evidence)} evidence item(s)."
            ))],
            "observation_index": observation_index,
            "no_progress_count": 0 if new_observation else state.get("no_progress_count", 0) + 1,
        }
        if role == Role.DEBUGGING and action == "generate_hypotheses":
            patch["hypotheses"] = [Hypothesis(statement=x, likelihood=.5) for x in proposal.get("hypotheses", [])]
        if role == Role.JUDGE:
            patch["scorecard"] = Scorecard(solvable=True, difficulty="appropriate", scenario_status="evaluated", score=proposal.get("score", 80), findings=proposal.get("findings", []))
        return patch

    async def approval(self, state: QAState) -> dict[str, Any]:
        decision = state["last_decision"]
        request = self.policy.request(decision.action, decision.target, decision.justification,
                                      [e.id for e in state.get("evidence", [])[-10:]])
        event = Event(type="APPROVAL_REQUIRED", run_id=state["run_id"], emitted_by=Role.SUPERVISOR,
                      target=decision.target, payload=request.model_dump())
        try:
            await self.events.publish(event)
        except Exception as exc:
            self.progress("event_error", event_type=event.type, error=str(exc))
        answer = interrupt({"kind": "approval", "request": request.model_dump(mode="json"),
                            "question": "Approve this action? Reply approve or reject."})
        request.status = "approved" if str(answer).lower() in {"approve", "approved", "yes"} else "rejected"
        return {"approvals": [request], "events": [event], "pending_action": None,
                "messages": [HumanMessage(content=f"Human approval result: {request.status}" )],
                "aborted": request.status == "rejected"}
