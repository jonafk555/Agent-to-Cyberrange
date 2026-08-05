from cyberqa.graph import route
from cyberqa.models import Decision, Role
from cyberqa.approval import approved_tools_for_decision
from cyberqa.ad_playbooks import normalize_capability_parameters
from cyberqa.ad_strategy import recommend as recommend_ad_method
from cyberqa.execution_broker import CapabilityBroker


def test_router_accepts_literal_end_and_approval_values():
    approval = Decision(next_agent="approval", objective="x", action="x", target="t", justification="x")
    assert route({"last_decision": approval}) == "approval"


def test_decision_schema_closes_nested_tool_parameters_object():
    schema = Decision.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["ToolParameters"]["additionalProperties"] is False


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


def test_asrep_action_without_capability_maps_to_reviewed_tool():
    decision = Decision(
        next_agent=Role.TESTING, objective="AD QA", action="ad_asrep_roasting_probe",
        target="10.0.0.1", justification="approved lab assessment",
    )
    assert approved_tools_for_decision(decision) == ["ad_asrep_roasting"]


def test_asrep_parameters_drop_generic_nmap_fields_before_execution():
    normalized = normalize_capability_parameters(
        "asrep_roasting_assessment",
        {"profile": "ad_tcp", "argv": ["-Pn"], "users_file": "/tmp/users.txt"},
    )
    assert normalized.profile is None
    assert normalized.argv == []
    assert normalized.users_file == "/tmp/users.txt"


def test_ad_strategy_prioritizes_asrep_when_source_exists(monkeypatch):
    monkeypatch.delenv("CYBERQA_AD_DOMAIN", raising=False)
    monkeypatch.delenv("CYBERQA_AD_USERNAME", raising=False)
    monkeypatch.delenv("CYBERQA_AD_PASSWORD", raising=False)
    decision = recommend_ad_method({
        "target": "10.0.0.1",
        "ad_knowledge": {"domain": "corp.local", "users": ["alice"]},
        "evidence": [], "method_history": [], "target_profiles": {},
    })
    assert decision is not None
    assert decision.capability == "asrep_roasting_assessment"
    assert decision.tool_parameters.users == ["alice"]


def test_ad_strategy_stops_after_bounded_identity_phase():
    evidence = [
        {"source": "kali:ldap_bind", "action": "anonymous_identity_probe", "target": "10.0.0.1", "exit_code": 1},
        {"source": "kali:smb_negotiate", "action": "anonymous_identity_probe", "target": "10.0.0.1", "exit_code": 1},
        {"source": "kali:nxc_ldap_recon", "action": "anonymous_identity_probe", "target": "10.0.0.1", "exit_code": 0, "facts": {}},
    ]
    from cyberqa.models import Evidence
    decision = recommend_ad_method({
        "target": "10.0.0.1",
        "ad_knowledge": {"domain": "corp.local", "users": []},
        "evidence": [Evidence.model_validate(item) for item in evidence],
        "method_history": [], "target_profiles": {},
    })
    assert decision is not None
    assert decision.action == "provide_asrep_username_source"


def test_ad_strategy_retries_asrep_after_stale_approval_scope_rejection():
    from cyberqa.models import Evidence

    decision = recommend_ad_method({
        "target": "10.0.0.1",
        "last_decision": Decision(
            next_agent=Role.TESTING, objective="AD QA", action="asrep_roasting_assessment",
            target="10.0.0.1", justification="approved lab assessment",
            capability="asrep_roasting_assessment",
            tool_parameters={"users": ["alice"]},
        ),
        "ad_knowledge": {"domain": "corp.local", "users": []},
        "evidence": [Evidence(
            source="tool:ad_asrep_roasting", action="asrep_roasting_assessment",
            target="10.0.0.1", exit_code=-1,
            facts={"error_kind": "approval_scope"},
        )],
        "method_history": [{
            "tool": "tool:ad_asrep_roasting", "action": "asrep_roasting_assessment",
            "error_kind": "approval_scope",
        }],
        "target_profiles": {},
    })
    assert decision is not None
    assert decision.capability == "asrep_roasting_assessment"


def test_ad_strategy_sends_completed_asrep_to_evidence_judge():
    from cyberqa.models import Evidence

    decision = recommend_ad_method({
        "target": "10.0.0.1",
        "ad_knowledge": {"domain": "corp.local", "users": ["alice"]},
        "evidence": [Evidence(
            source="ad-capability:ad_asrep_roasting", action="asrep_roasting_assessment",
            target="10.0.0.1", exit_code=1,
            facts={"ticket_obtained_or_blocked": "none_observed"},
        )],
        "method_history": [{
            "tool": "ad-capability:ad_asrep_roasting", "action": "asrep_roasting_assessment",
            "error_kind": "nonzero_exit",
        }],
        "target_profiles": {},
    })
    assert decision is not None
    assert decision.next_agent == Role.JUDGE
    assert decision.action == "evaluate_ad_evidence"
