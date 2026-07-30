from __future__ import annotations

from .models import ApprovalRequest


AUTONOMOUS_ACTIONS = {"restart_service", "change_firewall", "correct_dns", "correct_route",
                      "install_package", "sync_time"}


class ApprovalPolicy:
    def requires_approval(self, action: str) -> bool:
        return action not in AUTONOMOUS_ACTIONS

    def request(self, action: str, target: str, reason: str, evidence_ids: list[str]) -> ApprovalRequest:
        return ApprovalRequest(action=action, target=target, reason=reason,
                               impact="May alter challenge control-plane or identity state",
                               rollback="Restore from the latest environment snapshot",
                               evidence_ids=evidence_ids)
