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


class Agents:
    def __init__(self, llm: BaseChatModel | None = None, tools: ToolRegistry | None = None,
                 events: EventBus | None = None, policy: ApprovalPolicy | None = None,
                 on_progress: Callable[[str, dict[str, Any]], None] | None = None):
        self.llm, self.tools, self.events, self.policy = llm, tools or ToolRegistry(), events or EventBus(), policy or ApprovalPolicy()
        self.on_progress = on_progress

    def progress(self, event: str, **data: Any) -> None:
        if self.on_progress:
            self.on_progress(event, data)

    async def _reason(self, role: Role, state: QAState, instruction: str) -> dict[str, Any]:
        if not self.llm:
            return {"action": "observe", "target": "environment", "justification": "Collect missing facts before changing state."}
        prompt = json.dumps({"objective": state.get("objective"), "phase": state.get("phase"),
                             "target": state.get("target", "environment"),
                             "evidence": [e.model_dump() for e in state.get("evidence", [])[-20:]],
                             "instruction": instruction})
        self.progress("reasoning_start", agent=role.value)
        conversation = state.get("messages", [])[-20:]
        response = await self.llm.ainvoke([
            SystemMessage(content=SYSTEM + "\nYou are the workflow supervisor. Return only a valid decision JSON."),
            *conversation,
            HumanMessage(content=prompt),
        ])
        return json.loads(response.content)

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
        allowed = self.tools.langchain_tools(available or None)
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
                    "instruction": state.get("last_decision").justification if state.get("last_decision") else "Collect useful facts",
                })),
                *s.get("messages", [])[-20:],
            ])
            self.progress("reasoned", agent=role.value,
                          tool_calls=[call.get("name") for call in getattr(response, "tool_calls", [])],
                          has_final_answer=not bool(getattr(response, "tool_calls", [])))
            return {"messages": [response]}

        def after_reason(s: dict[str, Any]) -> str:
            last = s.get("messages", [])[-1] if s.get("messages") else None
            return "tools" if getattr(last, "tool_calls", None) else "done"

        def after_tools(s: dict[str, Any]) -> str:
            last = s.get("messages", [])[-1] if s.get("messages") else None
            if isinstance(last, ToolMessage):
                try:
                    result = json.loads(last.content) if isinstance(last.content, str) else last.content
                    if isinstance(result, dict) and result.get("needs_human"):
                        return "human"
                except (TypeError, json.JSONDecodeError):
                    return "human"
            return "reason"

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
        inner.add_node("human", human)
        inner.add_edge(START, "reason")
        inner.add_conditional_edges("reason", after_reason, {"tools": "tools", "done": END})
        inner.add_conditional_edges("tools", after_tools, {"reason": "reason", "human": "human"})
        inner.add_edge("human", "reason")
        return inner.compile()

    async def supervisor(self, state: QAState) -> dict[str, Any]:
        iteration = state.get("iteration", 0) + 1
        if iteration >= state.get("max_iterations", 20):
            decision = Decision(next_agent="end", objective="stop", action="end", target="environment", justification="Iteration budget exhausted; human guidance is required to continue.")
            return {"iteration": iteration, "phase": "human_help", "last_decision": decision,
                    "pending_action": decision.model_dump(), "needs_human": True}
        try:
            result = await self._reason(Role.SUPERVISOR, state, "Select the highest-value next action. Choose validation, testing, debugging, judge, reporting, approval, or end.")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            decision = Decision(next_agent="end", objective="human_help", action="end", target="environment",
                                justification=f"Supervisor could not produce a valid decision: {exc}")
            return {"iteration": iteration, "phase": "human_help", "last_decision": decision,
                    "pending_action": decision.model_dump(), "needs_human": True,
                    "errors": [str(exc)]}
        agent = result.get("next_agent", "validation")
        action = result.get("action", "observe")
        requested_target = result.get("target")
        target = requested_target if requested_target and requested_target != "environment" else state.get("target", "environment")
        decision = Decision(next_agent=agent, objective=result.get("objective", state.get("objective", "QA")),
                            action=action, target=target,
                            justification=result.get("justification", "Resolve the highest-value uncertainty."),
                            expected_information_gain=float(result.get("expected_information_gain", .5)),
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
        return {"needs_human": False, "action_history": [],
                "messages": [HumanMessage(content=f"Human guidance for supervisor: {answer}")],
                "errors": [] if guidance != "abort" else ["Human aborted after no progress"],
                "aborted": guidance == "abort"}

    async def specialist(self, role: Role, state: QAState) -> dict[str, Any]:
        decision = state.get("last_decision")
        target, action = (decision.target, decision.action) if decision else ("environment", "observe")
        evidence = []
        proposal: dict[str, Any] = {}
        if self.llm and self.tools.tools:
            result = await self._react_graph(role, state).ainvoke(
                {"messages": list(state.get("messages", [])[-20:])}
            )
            for message in result.get("messages", []):
                if isinstance(message, ToolMessage):
                    try:
                        payload = json.loads(message.content) if isinstance(message.content, str) else message.content
                        if payload.get("evidence"):
                            from .models import Evidence
                            evidence.append(Evidence.model_validate(payload["evidence"]))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                if isinstance(message, AIMessage) and message.content:
                    proposal["summary"] = message.content
        elif self.tools.tools:
            # Offline mode remains deterministic, but uses the same allow-listed adapter boundary.
            tool_name = role.value if role.value in self.tools.tools else next(iter(self.tools.tools))
            evidence.append(await self.tools.get(tool_name).observe(target, action))
            proposal = {"tool": tool_name, "offline": True}
        event_type = {Role.VALIDATION: "SERVICE_VALIDATED", Role.TESTING: "ATTACK_PATH_VALIDATED",
                      Role.DEBUGGING: "REPAIR_COMPLETED", Role.JUDGE: "SCENARIO_EVALUATED",
                      Role.REPORTING: "REPORT_UPDATED"}[role]
        event = Event(type=event_type, run_id=state["run_id"], emitted_by=role, target=target,
                      evidence_ids=[e.id for e in evidence], payload=proposal)
        await self.events.publish(event)
        self.progress("agent_done", agent=role.value, evidence_count=len(evidence), target=target)
        patch: dict[str, Any] = {
            "evidence": evidence,
            "events": [event],
            "react_steps": state.get("react_steps", 0) + 1,
            "messages": [AIMessage(content=(
                f"{role.value} completed its current step and collected {len(evidence)} evidence item(s)."
            ))],
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
        await self.events.publish(event)
        answer = interrupt({"kind": "approval", "request": request.model_dump(mode="json"),
                            "question": "Approve this action? Reply approve or reject."})
        request.status = "approved" if str(answer).lower() in {"approve", "approved", "yes"} else "rejected"
        return {"approvals": [request], "events": [event], "pending_action": None,
                "messages": [HumanMessage(content=f"Human approval result: {request.status}" )],
                "aborted": request.status == "rejected"}
