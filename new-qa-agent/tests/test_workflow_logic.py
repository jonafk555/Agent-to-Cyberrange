from cyberqa.graph import route
from cyberqa.models import Decision, Role
from cyberqa.execution_broker import CapabilityBroker


def test_router_accepts_literal_end_and_approval_values():
    approval = Decision(next_agent="approval", objective="x", action="x", target="t", justification="x")
    assert route({"last_decision": approval}) == "approval"


def test_broker_does_not_accept_prerequisites_from_llm_prose():
    decision = Decision(
        next_agent=Role.TESTING, objective="AD QA", action="collect", target="10.0.0.1",
        justification="domain inventory and validated credential",
        capability="bloodhound_collection",
    )
    result = CapabilityBroker().validate(decision, decision.target)
    assert result["ok"] is False
    assert "domain_inventory" in result["missing_prerequisites"]


def test_broker_marks_credential_capabilities_for_approval_from_spec():
    decision = Decision(
        next_agent=Role.TESTING, objective="AD QA", action="assess", target="10.0.0.1",
        justification="facts collected", capability="asrep_roasting_assessment",
    )
    result = CapabilityBroker().validate(
        decision, decision.target,
        known_prerequisites={"domain_inventory", "user enumeration"},
    )
    assert result["requires_approval"] is True


def test_broker_marks_credential_validation_for_approval():
    decision = Decision(
        next_agent=Role.TESTING, objective="AD QA", action="validate", target="10.0.0.1",
        justification="supplied lab credential", capability="credential_validation",
    )
    result = CapabilityBroker().validate(
        decision, decision.target,
        known_prerequisites={"human_supplied_or_range_issued_credential"},
    )
    assert result["requires_approval"] is True
