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


class ApprovalPolicy:
    def requires_approval(self, action: str) -> bool:
        return action in DESTRUCTIVE_ACTIONS

    def request(self, action: str, target: str, reason: str, evidence_ids: list[str]) -> ApprovalRequest:
        return ApprovalRequest(action=action, target=target, reason=reason,
                               impact="May alter challenge control-plane or identity state",
                               rollback="Restore from the latest environment snapshot",
                               evidence_ids=evidence_ids)


def decision_fingerprint(decision: Decision) -> str:
    """Identify the exact executable portion of an approval decision."""
    capability = get_capability(decision.capability)
    payload = {
        "next_agent": decision.next_agent.value if hasattr(decision.next_agent, "value") else decision.next_agent,
        "action": decision.action,
        "target": decision.target,
        "capability": decision.capability,
        "risk": decision.risk.value,
        "tool_parameters": decision.tool_parameters,
        "allowed_tools": capability.allowed_tools if capability else [],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()
