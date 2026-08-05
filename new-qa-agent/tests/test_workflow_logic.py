from cyberqa.graph import entry_route, route
from cyberqa.models import Decision, Evidence, Role
from cyberqa.approval import approved_tools_for_decision
from cyberqa.ad_playbooks import normalize_capability_parameters
from cyberqa.ad_strategy import recommend as recommend_ad_method
from cyberqa.execution_broker import CapabilityBroker
from cyberqa.nodes import Agents
from cyberqa.tools import TargetPolicy, ToolRegistry, build_kali_registry


class InterfaceOnlyTool:
    name = "inspect_interfaces"

    def __init__(self, calls):
        self.calls = calls

    async def observe(self, target, action, **kwargs):
        self.calls.append((target, action))
        return Evidence(
            source="runner:interfaces", action=action, target=target,
            stdout='[{"addr_info":[{"local":"10.0.0.42"}]}]',
        )


def test_router_accepts_literal_end_and_approval_values():
    approval = Decision(next_agent="approval", objective="x", action="x", target="t", justification="x")
    assert route({"last_decision": approval}) == "approval"


def test_router_keeps_supervisor_replans_inside_supervisor():
    replan = Decision(
        next_agent=Role.SUPERVISOR, objective="recon", action="replan_after_duplicate",
        target="10.0.0.1", justification="effective command already observed",
    )
    assert route({"last_decision": replan}) == "supervisor"


def test_graph_bootstraps_runner_identity_before_supervisor():
    assert entry_route({"baseline_complete": False}) == "runner_identity"
    assert entry_route({"baseline_complete": True}) == "supervisor"


def test_human_semantics_keep_compound_guidance_for_supervisor():
    assert Agents._human_users_file("gain domain cred by ~/Desktop/username.txt") == "~/Desktop/username.txt"
    assert Agents._human_explicit_decision(
        {"target": "10.0.0.0/24"},
        "nmap 10.0.0.1, then inspect SMB and continue with the next host",
    ) is None


def test_network_transition_excludes_runner_name():
    agents = Agents(tools=build_kali_registry(allowed_targets=["10.0.0.0/24"]))
    decision = agents._network_recon_transition({
        "target": "10.0.0.0/24",
        "discovered_targets": ["10.0.0.0/24", "local-kali", "10.0.0.1"],
        "recon_coverage": {
            "10.0.0.0/24": {"checks": {"nmap:host_discovery": {"status": "completed"}}},
            "10.0.0.1": {"checks": {}},
        },
    })

    assert decision is not None
    assert decision.target == "10.0.0.1"


def test_runner_identity_only_collects_kali_ip_before_remote_recon():
    import asyncio

    calls = []
    policy = TargetPolicy(["10.0.0.0/24"])
    registry = ToolRegistry(
        tools={"inspect_interfaces": InterfaceOnlyTool(calls)},
        target_policy=policy,
    )
    result = asyncio.run(Agents(tools=registry).runner_identity({
        "run_id": "runner-test",
        "target": "10.0.0.0/24",
        "runner_ips": [],
        "discovered_targets": ["10.0.0.0/24"],
        "evidence": [],
        "target_profiles": {},
        "ad_knowledge": {},
        "runtime_config": {},
        "method_history": [],
        "observation_index": {},
    }))

    assert calls == [("environment", "runner_identity")]
    assert "10.0.0.42" in result["runner_ips"]
    assert not registry.target_policy.allows("10.0.0.42")
    assert result["discovered_targets"] == ["10.0.0.0/24"]


def test_human_help_handles_missing_request_and_rejects_previous_decision(monkeypatch):
    monkeypatch.setenv("CYBERQA_OBSERVATION_DB", ":memory:")
    monkeypatch.setattr("cyberqa.nodes.interrupt", lambda request: "no")

    import asyncio

    result = asyncio.run(Agents().human_help({
        "target": "10.0.0.0/24",
        "evidence": [],
        "human_requests": [],
        "human_directives": [],
        "action_history": ["already-executed-command"],
        "last_decision": Decision(
            next_agent=Role.VALIDATION, objective="recon", action="service_enumeration",
            target="local-kali", justification="bad target",
        ),
        "iteration": 1,
        "max_iterations": 8,
        "needs_human": True,
    }))

    assert result["last_decision"] is None
    assert result["human_directives"][0]["intent"] == "reject_previous"
    assert result["action_history"] == ["already-executed-command"]


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


def test_ad_strategy_continues_identity_after_completed_asrep():
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
    assert decision.next_agent == Role.VALIDATION
    assert decision.action == "anonymous_identity_probe"


def test_ad_strategy_does_not_use_missing_username_file(monkeypatch):
    monkeypatch.setenv("CYBERQA_AD_DOMAIN", "corp.local")
    monkeypatch.setenv("CYBERQA_AD_USERS_FILE", "/definitely/missing/users.txt")
    decision = recommend_ad_method({
        "target": "10.0.0.1",
        "ad_knowledge": {"domain": "corp.local", "users": []},
        "evidence": [], "method_history": [], "target_profiles": {},
    })
    assert decision is not None
    assert decision.action == "anonymous_identity_probe"


def test_completion_gate_requires_identity_after_asrep():
    from cyberqa.models import Evidence

    agents = Agents(tools=build_kali_registry(allowed_targets=["10.0.0.1"]))
    base = {
        "target": "10.0.0.1",
        "recon_coverage": {"10.0.0.1": {"checks": {"nmap:default": {"status": "completed"}}}},
        "ad_knowledge": {"domain": "corp.local", "users": ["alice"]},
        "evidence": [Evidence(
            source="ad-capability:ad_asrep_roasting", action="asrep_roasting_assessment",
            target="10.0.0.1", exit_code=1,
        )],
        "method_history": [{"tool": "ad-capability:ad_asrep_roasting", "action": "asrep_roasting_assessment"}],
    }
    assert not agents._completion_gate_open(base)
    base["evidence"].extend([
        Evidence(source="kali:ldap_bind", action="anonymous_identity_probe", target="10.0.0.1", exit_code=1),
        Evidence(source="kali:smb_negotiate", action="anonymous_identity_probe", target="10.0.0.1", exit_code=1),
        Evidence(source="kali:nxc_ldap_recon", action="anonymous_identity_probe", target="10.0.0.1", exit_code=1),
    ])
    assert agents._completion_gate_open(base)


def test_judge_only_creates_scorecard_after_supervisor_authorization():
    import asyncio

    state = {
        "run_id": "judge-test",
        "target": "10.0.0.1",
        "last_decision": Decision(
            next_agent=Role.JUDGE,
            objective="evaluate",
            action="evaluate_ad_evidence",
            target="10.0.0.1",
            justification="evaluate accumulated evidence",
        ),
        "evidence": [], "events": [], "method_history": [], "observation_index": {},
        "ad_knowledge": {}, "discovered_targets": [], "target_profiles": {},
        "recon_coverage": {}, "messages": [], "judge_authorized": False,
    }
    agents = Agents(tools=ToolRegistry(target_policy=TargetPolicy(["10.0.0.1"])))
    result = asyncio.run(agents.specialist(Role.JUDGE, state))
    assert "scorecard" not in result

    state["judge_authorized"] = True
    result = asyncio.run(agents.specialist(Role.JUDGE, state))
    assert result["scorecard_authorized"] is True


def test_ad_guard_does_not_replace_a_safe_supervisor_path():
    safe = Decision(
        next_agent=Role.VALIDATION, objective="remote recon", action="service_enumeration",
        target="10.0.0.1", justification="inspect a newly discovered remote host",
    )
    guard = Decision(
        next_agent=Role.TESTING, objective="assess AD", action="asrep_roasting_assessment",
        target="10.0.0.1", justification="candidate username source is available",
    )
    terminal = safe.model_copy(update={"next_agent": Role.JUDGE, "action": "evaluate_ad_evidence"})
    assert Agents._should_apply_ad_guard(safe, guard) is False
    assert Agents._should_apply_ad_guard(terminal, guard) is True


def test_supervisor_replans_when_model_stops_without_a_blocker():
    import asyncio

    class StopModel:
        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            return Decision(
                next_agent="end", objective="human_help", action="end",
                target="10.0.0.1", justification="no pipeline found",
            )

    state = {
        "target": "10.0.0.1", "objective": "continue QA", "iteration": 0,
        "evidence": [], "method_history": [], "target_profiles": {},
        "recon_coverage": {}, "discovered_targets": ["10.0.0.1"],
        "runner_ips": [], "ad_knowledge": {}, "runtime_config": {},
        "human_directives": [], "messages": [], "observation_index": {},
        "action_history": [], "replan_count": 0,
        "autonomous_replan_count": 0, "autonomous_continuation_required": False,
    }
    agents = Agents(
        llm=StopModel(),
        tools=ToolRegistry(target_policy=TargetPolicy(["10.0.0.1"])),
    )
    result = asyncio.run(agents.supervisor(state))
    assert result["phase"] == Role.SUPERVISOR
    assert result["autonomous_replan_count"] == 1
    assert result["needs_human"] is False
