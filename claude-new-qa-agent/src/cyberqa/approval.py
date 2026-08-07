from __future__ import annotations

import hashlib
import json

from .ad_playbooks import get_capability
from .models import Decision
from .models import ApprovalRequest


DESTRUCTIVE_ACTIONS = {
    "reset_credential", "delete_user", "replace_gpo", "rebuild_domain", "rebuild_forest",
    "rebuild_adcs", "destroy_environment", "change_firewall", "correct_dns", "correct_route",
    "restart_service", "install_package", "sync_time",
}
NORMALIZED_DESTRUCTIVE_ACTIONS = {
    item.lower().replace("-", "_") for item in DESTRUCTIVE_ACTIONS
}


class ApprovalPolicy:
    def requires_approval(self, action: str) -> bool:
        action_key = action.lower().replace("-", "_")
        return action_key in NORMALIZED_DESTRUCTIVE_ACTIONS or "asrep" in action_key or "as_rep" in action_key

    def request(self, action: str, target: str, reason: str, evidence_ids: list[str]) -> ApprovalRequest:
        return ApprovalRequest(action=action, target=target, reason=reason,
                               impact="May alter challenge control-plane or identity state",
                               rollback="Restore from the latest environment snapshot",
                               evidence_ids=evidence_ids)


def decision_fingerprint(decision: Decision) -> str:
    """Identify the exact executable portion of an approval decision."""
    payload = {
        "next_agent": decision.next_agent.value if hasattr(decision.next_agent, "value") else decision.next_agent,
        "action": decision.action,
        "target": decision.target,
        "capability": decision.capability,
        "risk": decision.risk.value,
        "tool_parameters": decision.tool_parameters.model_dump(mode="json", exclude_none=True),
        "allowed_tools": approved_tools_for_decision(decision),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def approved_tools_for_decision(decision: Decision) -> list[str]:
    """Return the concrete tools covered by this exact approval decision.

    Older planner responses sometimes named the AS-REP action but omitted the
    capability field. Keep the approval boundary strict while deriving the
    one unambiguous reviewed adapter from that action instead of issuing an
    empty grant that can only fail at execution time.
    """
    capability = get_capability(decision.capability)
    if capability:
        return list(capability.allowed_tools)
    action = decision.action.lower().replace("-", "_")
    if "asrep" in action or "as_rep" in action:
        return ["ad_asrep_roasting"]
    return []
