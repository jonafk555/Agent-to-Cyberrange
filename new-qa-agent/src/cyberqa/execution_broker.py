"""Policy boundary between AD planning and concrete tool execution."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .ad_playbooks import get_capability
from .models import ADRisk, Decision


class CapabilityBroker:
    """Validate a model plan without constraining how the model finds it."""

    def validate(self, decision: Decision, target: str, observed: set[str] | None = None) -> dict[str, Any]:
        capability = get_capability(decision.capability)
        if capability is None:
            return {"ok": True, "capability": None, "signature": self.signature(decision, target)}

        missing = [item for item in capability.prerequisites
                   if not self._prerequisite_known(item, decision)]
        approval = capability.requires_approval or decision.risk in {
            ADRisk.CREDENTIAL_MATERIAL, ADRisk.ACCOUNT_LOCKOUT, ADRisk.CHANGE
        }
        signature = self.signature(decision, target)
        duplicate = signature in (observed or set())
        return {
            "ok": not duplicate and not missing,
            "capability": capability.name,
            "missing_prerequisites": missing,
            "requires_approval": approval,
            "duplicate": duplicate,
            "signature": signature,
            "allowed_tools": capability.allowed_tools,
        }

    @staticmethod
    def _prerequisite_known(value: str, decision: Decision) -> bool:
        text = f"{decision.justification} {' '.join(decision.prerequisites)}".lower()
        markers = {
            "domain_inventory": ("domain", "ldap", "inventory"),
            "user enumeration": ("user", "account", "ldap"),
            "valid ldap access or explicitly allowed anonymous ldap": ("ldap",),
            "lockout_policy": ("lockout", "lock-out"),
            "approved_test_password": ("approved", "password"),
            "human_supplied_or_range_issued_credential": ("credential", "account"),
            "validated domain credential": ("validated", "credential", "bloodhound"),
            "dns resolution": ("dns", "resolve"),
            "bloodhound_collection or equivalent relationship evidence": ("bloodhound", "acl", "relationship"),
        }
        required = markers.get(value)
        return not required or any(token in text for token in required)

    @staticmethod
    def signature(decision: Decision, target: str) -> str:
        payload = {
            "capability": decision.capability or decision.action,
            "target": target,
            "parameters": decision.next_options,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:20]
