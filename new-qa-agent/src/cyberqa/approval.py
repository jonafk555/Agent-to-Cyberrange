from __future__ import annotations

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
