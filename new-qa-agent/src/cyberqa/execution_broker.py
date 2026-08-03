"""Policy boundary between AD planning and concrete tool execution."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .ad_playbooks import get_capability
from .models import ADRisk, Decision


class CapabilityBroker:
    """Validate a model plan without constraining how the model finds it."""

    def validate(self, decision: Decision, target: str, observed: set[str] | None = None,
                 known_prerequisites: set[str] | None = None) -> dict[str, Any]:
        capability = get_capability(decision.capability)
        if capability is None:
            return {"ok": True, "capability": None, "signature": self.signature(decision, target)}

        known = known_prerequisites or set()
        missing = [item for item in capability.prerequisites if item not in known]
        approval = capability.requires_approval or capability.risk in {
            ADRisk.CREDENTIAL_MATERIAL, ADRisk.AUTHENTICATION_TEST,
            ADRisk.ACCOUNT_LOCKOUT, ADRisk.CHANGE
        } or decision.risk in {
            ADRisk.CREDENTIAL_MATERIAL, ADRisk.AUTHENTICATION_TEST,
            ADRisk.ACCOUNT_LOCKOUT, ADRisk.CHANGE
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
    def signature(decision: Decision, target: str) -> str:
        payload = {
            "capability": decision.capability or decision.action,
            "target": target,
            "parameters": decision.next_options,
            "tool_parameters": decision.tool_parameters,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:20]
