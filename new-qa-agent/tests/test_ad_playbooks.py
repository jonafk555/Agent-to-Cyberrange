from cyberqa.ad_playbooks import capability_catalog, get_capability
from cyberqa.execution_broker import CapabilityBroker
from cyberqa.models import ADRisk, Decision, Role


def test_ad_playbook_exposes_multi_step_capabilities():
    names = {item["name"] for item in capability_catalog()}
    assert {"asrep_roasting_assessment", "kerberoasting_assessment",
            "hash_cracking_assessment", "enumerate_domain_users", "bloodhound_collection"} <= names


def test_password_spray_requires_approval_and_scope_prerequisites():
    spec = get_capability("controlled_password_spray_assessment")
    assert spec is not None
    assert spec.risk == ADRisk.ACCOUNT_LOCKOUT
    assert spec.requires_approval is True

    decision = Decision(
        next_agent=Role.TESTING, objective="AD QA", action="assess password reuse",
        target="10.0.0.10", justification="test", capability=spec.name,
    )
    result = CapabilityBroker().validate(decision, decision.target)
    assert result["requires_approval"] is True
    assert result["ok"] is False
    assert "lockout_policy" in result["missing_prerequisites"]


def test_broker_deduplicates_capability_target_pair():
    decision = Decision(
        next_agent=Role.TESTING, objective="AD QA", action="collect relationships",
        target="10.0.0.10", justification="domain inventory and validated credential",
        capability="bloodhound_collection",
        prerequisites=["domain_inventory", "validated domain credential", "DNS resolution"],
    )
    broker = CapabilityBroker()
    known = {"domain_inventory", "validated domain credential", "DNS resolution"}
    first = broker.validate(decision, decision.target, known_prerequisites=known)
    second = broker.validate(decision, decision.target, {first["signature"]}, known)
    assert first["ok"] is True
    assert second["duplicate"] is True
    assert second["ok"] is False
