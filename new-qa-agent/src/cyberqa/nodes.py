from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from .approval import ApprovalPolicy
from .events import EventBus
from .models import Decision, Event, Hypothesis, Role, Scorecard, Service, ServiceProtocol
from .state import QAState
from .tools import ToolRegistry


SYSTEM = """You are a cyber-range QA specialist operating only on authorized targets. Use OODA:
observe facts, orient against the objective and prior evidence, decide one justified action, act through
fact-only tools. Return JSON. Do not invent facts, credentials, vulnerabilities, or successful attacks."""


class Agents:
    def __init__(self, llm: BaseChatModel | None = None, tools: ToolRegistry | None = None,
                 events: EventBus | None = None, policy: ApprovalPolicy | None = None):
        self.llm, self.tools, self.events, self.policy = llm, tools or ToolRegistry(), events or EventBus(), policy or ApprovalPolicy()

    async def _reason(self, role: Role, state: QAState, instruction: str) -> dict[str, Any]:
        if not self.llm:
            return {"action": "observe", "target": "environment", "justification": "Collect missing facts before changing state."}
        prompt = json.dumps({"objective": state.get("objective"), "phase": state.get("phase"),
                             "evidence": [e.model_dump() for e in state.get("evidence", [])[-20:]],
                             "instruction": instruction})
        response = await self.llm.ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=prompt)])
        return json.loads(response.content)

    async def supervisor(self, state: QAState) -> dict[str, Any]:
        iteration = state.get("iteration", 0) + 1
        if iteration >= state.get("max_iterations", 20):
            return {"iteration": iteration, "last_decision": Decision(next_agent="end", objective="stop", action="end", target="environment", justification="Iteration budget exhausted.")}
        result = await self._reason(Role.SUPERVISOR, state, "Select the highest-value next action. Choose validation, testing, debugging, judge, reporting, approval, or end.")
        agent = result.get("next_agent", "validation")
        action = result.get("action", "observe")
        decision = Decision(next_agent=agent, objective=result.get("objective", state.get("objective", "QA")),
                            action=action, target=result.get("target", "environment"),
                            justification=result.get("justification", "Resolve the highest-value uncertainty."),
                            expected_information_gain=float(result.get("expected_information_gain", .5)),
                            approval_required=self.policy.requires_approval(action))
        signature = f"{action}:{decision.target}"
        history = state.get("action_history", [])
        if len(history) >= 3 and history[-3:] == [signature] * 3:
            decision = Decision(next_agent="end", objective="stop", action="end", target=decision.target,
                                justification="The same action produced no new information three times.")
        return {"iteration": iteration, "phase": decision.next_agent, "last_decision": decision,
                "pending_action": decision.model_dump(), "action_history": history + [signature]}

    async def specialist(self, role: Role, state: QAState) -> dict[str, Any]:
        decision = state.get("last_decision")
        proposal = await self._reason(role, state, f"Perform your role. Current supervisor action: {decision.model_dump() if decision else {}}")
        target, action = (decision.target, decision.action) if decision else ("environment", "observe")
        tool_name = proposal.get("tool", role.value)
        evidence = []
        if tool_name in self.tools.tools:
            evidence.append(await self.tools.get(tool_name).observe(target, action))
        event_type = {Role.VALIDATION: "SERVICE_VALIDATED", Role.TESTING: "ATTACK_PATH_VALIDATED",
                      Role.DEBUGGING: "REPAIR_COMPLETED", Role.JUDGE: "SCENARIO_EVALUATED",
                      Role.REPORTING: "REPORT_UPDATED"}[role]
        event = Event(type=event_type, run_id=state["run_id"], emitted_by=role, target=target,
                      evidence_ids=[e.id for e in evidence], payload=proposal)
        await self.events.publish(event)
        patch: dict[str, Any] = {"evidence": evidence, "events": [event]}
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
        return {"approvals": [request], "events": [event], "pending_action": None}
