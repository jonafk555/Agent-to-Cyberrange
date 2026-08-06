import asyncio

from cyberqa.ad_capability_tools import ADCapabilityTool
from cyberqa.ad_strategy import derive_context
from cyberqa.intent import parse_human_intent
from cyberqa.models import (
    Decision,
    Evidence,
    EvidenceAnalysis,
    EvidenceLevel,
    EvidenceOpportunity,
    HumanIntent,
    Role,
    ToolParameters,
    VisibilityMode,
)
from cyberqa.main import build_task_assertions, graph_recursion_limit, write_initial_recon_report
from cyberqa.nodes import Agents
from cyberqa.qa_assessment import (
    build_bootstrap_assertions,
    infer_evidence_level,
    parse_visibility_mode,
    refresh_assessment,
)
from cyberqa.security import redact_facts, redact_output
from cyberqa.tools import TargetPolicy, ToolRegistry


def test_compound_intent_hydrates_asrep_users_from_previous_evidence():
    intent = parse_human_intent(
        "nxc ldap 10.0.0.1 --users，取得帳號後繼續 AS-REP 評估",
        {"target": "10.0.0.1"},
    ).model_copy(update={"current_step": 1, "step_statuses": ["completed", "pending"]})

    decision = Agents._intent_decision(
        {
            "target": "10.0.0.1",
            "ad_knowledge": {"domain": "corp.local", "users": ["alice", "bob"]},
        },
        intent,
    )

    assert decision is not None
    assert decision.tool_parameters.users == ["alice", "bob"]


def test_negated_remote_target_is_excluded_but_replacement_is_selected():
    intent = parse_human_intent(
        "不要再掃描 10.0.0.1，改掃 10.0.0.2 的 SMB",
        {"target": "10.0.0.1"},
    )

    assert "10.0.0.1" in intent.excluded_targets
    assert "10.0.0.2" not in intent.excluded_targets
    decision = Agents._intent_decision(
        {"target": "10.0.0.1", "discovered_targets": ["10.0.0.1", "10.0.0.2"]},
        intent,
    )
    assert decision is not None
    assert decision.target == "10.0.0.2"
    assert Agents._target_is_excluded(intent, "10.0.0.1:445")


def test_failed_human_step_is_not_marked_completed_or_retried_frozen():
    intent = HumanIntent(
        raw_instruction="nxc ldap 10.0.0.1 --users，取得帳號後繼續 AS-REP 評估",
        ordered_steps=["nxc_ldap_recon", "ad_asrep_roasting"],
        step_parameters={"nxc_ldap_recon": {"profile": "users"}, "ad_asrep_roasting": {}},
        step_statuses=["pending", "pending"],
    )
    updated, plan = Agents._advance_human_intent(
        {"human_intent": intent.model_dump(mode="json")},
        [Evidence(
            source="kali:nxc_ldap_recon", action="nxc_ldap_recon", target="10.0.0.1",
            exit_code=1, stderr="connection failed",
        )],
    )

    assert updated["current_step"] == 0
    assert updated["step_statuses"] == ["failed", "pending"]
    assert plan["steps"][0]["status"] == "failed"
    assert Agents._intent_decision({"target": "10.0.0.1"}, HumanIntent.model_validate(updated)) is None


def test_unsupported_nmap_argument_is_reported_instead_of_dropped():
    intent = parse_human_intent(
        "nmap --script default -sV 10.0.0.1",
        {"target": "10.0.0.1"},
    )

    assert intent.parsing_errors
    assert "--script" in intent.parsing_errors[0]
    assert intent.step_statuses == ["blocked"]
    assert Agents._human_explicit_decision(
        {"target": "10.0.0.1"}, "nmap --script default -sV 10.0.0.1"
    ) is None


def test_unsupported_later_step_stays_blocked_after_earlier_step_completes():
    intent = parse_human_intent(
        "先 nxc ldap 10.0.0.1 --users，完成後 nmap --script default -sV 10.0.0.1",
        {"target": "10.0.0.1"},
    )

    assert intent.step_statuses == ["pending", "blocked"]
    next_step = intent.model_copy(update={"current_step": 1})
    assert Agents._intent_decision({"target": "10.0.0.1"}, next_step) is None


def test_redaction_covers_hash_labels_and_short_password_flags():
    rendered = redact_output("hash=abcdef --password PlainSecret -p PlainSecret")

    assert "abcdef" not in rendered
    assert "PlainSecret" not in rendered
    assert "-p [REDACTED]" in rendered
    facts = redact_facts({
        "hash": "abcdef",
        "hash_cracked": True,
        "argv": ["nxc", "smb", "10.0.0.1", "-p", "PlainSecret"],
    })
    assert facts["hash"] == "[REDACTED]"
    assert facts["hash_cracked"] is True
    assert facts["argv"][-1] == "[REDACTED]"
    safe_argv = ADCapabilityTool._safe_argv(["nxc", "smb", "10.0.0.1", "-p", "PlainSecret"])
    assert safe_argv[-1] == "***REDACTED***"


def test_runner_ip_is_excluded_from_intent_target_selection():
    intent = parse_human_intent(
        "對 10.0.0.42 做 SMB 檢查，若不可用改查 10.0.0.10",
        {"target": "10.0.0.42"},
    )
    decision = Agents._intent_decision(
        {
            "target": "10.0.0.42",
            "runner_ips": ["10.0.0.42"],
            "discovered_targets": ["10.0.0.42", "10.0.0.10"],
        },
        intent,
    )

    assert decision is not None
    assert decision.target == "10.0.0.10"


def test_runner_ip_is_excluded_from_network_transition():
    registry = ToolRegistry(target_policy=TargetPolicy(["10.0.0.0/24"]))
    registry.target_policy.mark_local("10.0.0.42")
    agents = Agents(tools=registry)
    decision = agents._network_recon_transition({
        "target": "10.0.0.0/24",
        "runner_ips": ["10.0.0.42"],
        "discovered_targets": ["10.0.0.42", "10.0.0.10"],
        "recon_coverage": {
            "10.0.0.0/24": {"checks": {"nmap:host_discovery": {"status": "completed"}}},
            "10.0.0.42": {"checks": {}},
            "10.0.0.10": {"checks": {}},
        },
    })

    assert decision is not None
    assert decision.target == "10.0.0.10"


def test_cache_hit_still_produces_evidence_analysis_for_supervisor():
    class Probe:
        name = "check_port"

        async def observe(self, target, action, **kwargs):
            return Evidence(
                source="probe:check_port", action=action, target=target,
                facts={"open_ports": [{"port": "445", "protocol": "tcp", "service": "microsoft-ds"}]},
            )

    registry = ToolRegistry(
        tools={"check_port": Probe()},
        target_policy=TargetPolicy(["10.0.0.10"]),
    )
    agents = Agents(tools=registry)
    decision = Decision(
        next_agent=Role.VALIDATION, objective="recon", action="service_enumeration",
        target="10.0.0.10", justification="probe", tool_parameters=ToolParameters(profile="default"),
    )
    base = {
        "run_id": "cache-analysis",
        "last_decision": decision,
        "target": "10.0.0.10",
        "evidence": [], "events": [], "method_history": [], "observation_index": {},
        "discovered_targets": [], "target_profiles": {}, "ad_knowledge": {},
        "runtime_config": {}, "react_steps": 0, "no_progress_count": 0,
    }
    first = asyncio.run(agents.specialist(Role.VALIDATION, base))
    second_state = {**base, "evidence": first["evidence"], "observation_index": first["observation_index"]}
    second = asyncio.run(agents.specialist(Role.VALIDATION, second_state))

    assert first["evidence_analyses"]
    assert second["evidence_analyses"]


def test_malformed_open_ports_fact_does_not_crash_projection():
    agents = Agents()
    projection = agents._project_observations(
        {
            "target": "10.0.0.10", "evidence": [], "target_profiles": {},
            "ad_knowledge": {}, "runtime_config": {}, "runner_ips": [],
        },
        [Evidence(
            source="probe", action="service_enumeration", target="10.0.0.10",
            facts={
                "open_ports": [445, "malformed", {"port": "445", "protocol": "tcp", "service": "smb"}],
                "discovered_targets": None, "users": None, "argv": None,
            },
        )],
    )

    assert "10.0.0.10" in projection["recon_coverage"]


def test_ad_strategy_never_selects_runner_ip_as_remote_target():
    context = derive_context({
        "target": "10.0.0.0/24",
        "runner_ips": ["10.0.0.42"],
        "discovered_targets": ["10.0.0.42", "10.0.0.10"],
        "target_profiles": {},
        "ad_knowledge": {"domain": "corp.local", "users": ["alice"]},
        "evidence": [], "method_history": [],
    })

    assert context.target == "10.0.0.10"


def test_supervisor_continues_to_hash_cracking_after_completed_asrep(tmp_path, monkeypatch):
    class StopModel:
        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            return Decision(
                next_agent="end", objective="human_help", action="end",
                target="10.0.0.10", justification="no pipeline found",
            )

    hash_file = tmp_path / "asrep.hash"
    wordlist = tmp_path / "wordlist.txt"
    hash_file.write_text("protected-artifact\n", encoding="utf-8")
    wordlist.write_text("candidate\n", encoding="utf-8")
    monkeypatch.setenv("CYBERQA_AD_DOMAIN", "corp.local")
    monkeypatch.setenv("CYBERQA_AD_WORDLIST", str(wordlist))
    state = {
        "target": "10.0.0.10", "objective": "continue QA", "iteration": 2,
        "evidence": [Evidence(
            source="ad-capability:ad_asrep_roasting", action="asrep_roasting_assessment",
            target="10.0.0.10", exit_code=0,
            facts={"asrep_hash_file": str(hash_file), "asrep_hash_count": 1},
        )],
        "method_history": [], "target_profiles": {}, "recon_coverage": {},
        "discovered_targets": ["10.0.0.10"], "runner_ips": [],
        "ad_knowledge": {"domain": "corp.local", "asrep_hash_file": str(hash_file)},
        "runtime_config": {}, "human_directives": [], "messages": [],
        "observation_index": {}, "action_history": [], "replan_count": 0,
        "autonomous_replan_count": 0, "autonomous_continuation_required": False,
        "human_intent": HumanIntent(
            ordered_steps=["ad_asrep_roasting"], current_step=1,
            step_statuses=["completed"], completed=True,
        ).model_dump(mode="json"),
    }
    agents = Agents(
        llm=StopModel(),
        tools=ToolRegistry(target_policy=TargetPolicy(["10.0.0.10"])),
    )

    result = asyncio.run(agents.supervisor(state))

    assert result["last_decision"].action == "hash_cracking_assessment"
    assert result["last_decision"].tool_parameters.hash_file == str(hash_file)


def test_evidence_reasoning_finds_smb_follow_up_without_asrep_context():
    analysis = Agents._fallback_evidence_analysis(
        {},
        Evidence(
            source="probe:check_port",
            action="service_enumeration",
            target="10.0.0.10",
            facts={
                "open_ports": [
                    {"port": "445", "protocol": "tcp", "service": "microsoft-ds"},
                ],
            },
        ),
        ["check_port", "smb_negotiate", "nxc_smb_recon"],
    )

    assert any(tool in analysis.candidate_tools for tool in ("smb_negotiate", "nxc_smb_recon"))
    assert any("SMB" in item or "445" in item for item in analysis.useful_content)
    assert analysis.opportunities
    assert any(item.tool in {"smb_negotiate", "nxc_smb_recon"} for item in analysis.opportunities)


def test_evidence_reasoning_finds_http_follow_up_from_service_facts():
    analysis = Agents._fallback_evidence_analysis(
        {},
        Evidence(
            source="probe:check_port",
            action="service_enumeration",
            target="10.0.0.20",
            facts={
                "open_ports": [
                    {"port": 443, "protocol": "tcp", "service": "https"},
                ],
                "http_headers": {"server": "IIS"},
            },
        ),
        ["http_health_check", "check_port"],
    )

    assert "http_health_check" in analysis.candidate_tools
    assert any(any(marker in item.lower() for marker in ("http", "https", "443"))
               for item in analysis.useful_content)


def test_evidence_reasoning_finds_ldap_follow_up_from_domain_service_facts():
    analysis = Agents._fallback_evidence_analysis(
        {},
        Evidence(
            source="probe:check_port",
            action="service_enumeration",
            target="10.0.0.1",
            facts={
                "domain_name": "corp.local",
                "open_ports": [
                    {"port": "389", "protocol": "tcp", "service": "ldap"},
                ],
            },
        ),
        ["ldap_bind", "nxc_ldap_recon", "check_port"],
    )

    assert any(tool in analysis.candidate_tools for tool in ("ldap_bind", "nxc_ldap_recon"))
    assert any("LDAP" in item or "corp.local" in item for item in analysis.useful_content)


def test_non_ad_service_result_does_not_force_asrep_or_open_completion_early():
    registry = ToolRegistry(
        tools={"http_health_check": object()},
        target_policy=TargetPolicy(["10.0.0.20"]),
    )
    agents = Agents(tools=registry)
    analysis = Agents._fallback_evidence_analysis(
        {},
        Evidence(
            source="probe:check_port",
            action="service_enumeration",
            target="10.0.0.20",
            facts={"open_ports": [{"port": "443", "protocol": "tcp", "service": "https"}]},
        ),
        ["http_health_check", "ad_asrep_roasting"],
    )
    state = {
        "target": "10.0.0.20",
        "target_profiles": {},
        "runner_ips": [],
        "discovered_targets": ["10.0.0.20"],
        "recon_coverage": {"10.0.0.20": {"checks": {"nmap:default": {"status": "completed"}}}},
        "evidence_opportunities": [item.model_dump(mode="json") for item in analysis.opportunities],
        "method_history": [],
        "ad_knowledge": {},
        "evidence": [],
    }

    assert "ad_asrep_roasting" not in analysis.candidate_tools
    assert not agents._completion_gate_open(state)

    state["method_history"] = [{
        "tool": "probe:http_health_check", "action": "http_health_check", "target": "10.0.0.20",
    }]
    assert agents._completion_gate_open(state)


def test_authenticated_directory_result_projects_multiple_ad_options_without_fixed_order():
    analysis = Agents._fallback_evidence_analysis(
        {},
        Evidence(
            source="ad-capability:ad_credential_validation",
            action="credential_validation",
            target="10.0.0.1",
            facts={"credentials_validated": ["alice"], "groups": ["Domain Users"]},
        ),
        ["ad_domain_users", "ad_bloodhound_collection", "ad_asrep_roasting"],
    )

    assert {"ad_domain_users", "ad_bloodhound_collection"}.issubset(analysis.candidate_tools)
    assert "ad_asrep_roasting" not in analysis.candidate_tools
    assert any("validated domain credential" in item.prerequisites_met for item in analysis.opportunities)


def test_supervisor_prompt_receives_evidence_opportunities_as_decision_memory():
    class CaptureModel:
        def __init__(self):
            self.prompt = ""

        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            self.prompt = messages[-1].content
            return Decision(
                next_agent=Role.VALIDATION,
                objective="inspect web service",
                action="http_health_check",
                target="10.0.0.20",
                justification="Use the observed HTTPS service opportunity.",
            )

    model = CaptureModel()
    agents = Agents(
        llm=model,
        tools=ToolRegistry(
            tools={"http_health_check": object()},
            target_policy=TargetPolicy(["10.0.0.20"]),
        ),
    )
    decision = asyncio.run(agents._structured_supervisor({
        "objective": "continue authorized QA",
        "target": "10.0.0.20",
        "evidence": [],
        "evidence_analyses": [],
        "evidence_opportunities": [{
            "tool": "http_health_check", "target": "10.0.0.20",
            "reason": "HTTPS observed", "prerequisites_missing": [],
        }],
        "visibility_mode": "black_box",
        "qa_assertions": [{"id": "web-functional", "statement": "HTTPS works", "target": "10.0.0.20", "required_evidence_level": "C3"}],
        "evidence_sufficiency": [{"assertion_id": "web-functional", "current_level": "C2", "required_level": "C3", "sufficient": False}],
        "messages": [],
        "human_intent": {}, "task_plan": {}, "human_directives": [],
        "method_history": [], "observation_index": {}, "runner_ips": [],
        "discovered_targets": ["10.0.0.20"], "recon_coverage": {},
        "target_profiles": {}, "runtime_config": {}, "ad_knowledge": {},
    }))

    assert decision.action == "http_health_check"
    assert "evidence_opportunities" in model.prompt
    assert "HTTPS observed" in model.prompt
    assert "web-functional" in model.prompt
    assert "C3" in model.prompt
    assert "Do not wait for a named pipeline" in model.prompt


def test_evidence_opportunity_tools_have_direct_executor_mapping():
    http = Agents._planned_tool_for_action(Decision(
        next_agent=Role.VALIDATION, objective="web", action="http_health_check",
        target="10.0.0.20", justification="HTTPS evidence",
    ))
    rpc = Agents._planned_tool_for_action(Decision(
        next_agent=Role.VALIDATION, objective="rpc", action="impacket_rpc_recon",
        target="10.0.0.30", justification="RPC evidence",
    ))

    assert http == ("http_health_check", {})
    assert rpc == ("impacket_rpc_recon", {})


def test_repeated_model_stop_does_not_escalate_while_reviewed_path_remains():
    class StopModel:
        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            return Decision(
                next_agent="end", objective="human_help", action="end",
                target="10.0.0.20", justification="no pipeline found",
            )

    agents = Agents(
        llm=StopModel(),
        tools=ToolRegistry(
            tools={"http_health_check": object()},
            target_policy=TargetPolicy(["10.0.0.20"]),
        ),
    )
    state = {
        "target": "10.0.0.20", "objective": "continue QA", "iteration": 0,
        "evidence": [], "method_history": [], "target_profiles": {},
        "recon_coverage": {"10.0.0.20": {"checks": {"nmap:default": {"status": "completed"}}}},
        "discovered_targets": ["10.0.0.20"], "runner_ips": [], "ad_knowledge": {},
        "evidence_opportunities": [{
            "tool": "http_health_check", "target": "10.0.0.20", "reason": "HTTPS observed",
        }],
        "messages": [], "human_directives": [], "human_intent": {}, "task_plan": {},
        "observation_index": {}, "autonomous_replan_count": 0,
        "autonomous_continuation_required": False, "replan_count": 0,
        "runtime_config": {},
    }

    for _ in range(3):
        result = asyncio.run(agents.supervisor(state))
        state = {**state, **result}

    assert state["autonomous_replan_count"] == 3
    assert state["needs_human"] is False
    assert state["last_decision"].action == "autonomous_replan_after_stop"


def test_assertion_sufficiency_selects_functional_depth_after_enumeration():
    assertions = build_bootstrap_assertions(
        "Validate LDAP service functionality", "10.0.0.1", VisibilityMode.BLACK_BOX
    )
    functional = next(item for item in assertions if item.id == "functional-validation")
    enumerated = Evidence(
        source="probe:check_port", action="service_enumeration", target="10.0.0.1",
        facts={"open_ports": [{"port": 389, "protocol": "tcp", "service": "ldap"}]},
    )
    _, first = refresh_assessment([functional], [enumerated], [])
    assert first[0]["current_level"] == EvidenceLevel.C2.value
    assert first[0]["sufficient"] is False

    functional_result = Evidence(
        source="probe:ldap_bind", action="ldap_bind_probe", target="10.0.0.1",
        facts={"functional": True, "protocol_verified": True, "domain_name": "corp.local"},
    )
    _, second = refresh_assessment([functional], [enumerated, functional_result], [])
    assert second[0]["current_level"] == EvidenceLevel.C3.value
    assert second[0]["sufficient"] is True


def test_wrong_service_functional_evidence_cannot_satisfy_ldap_assertion():
    assertions = build_bootstrap_assertions(
        "Validate LDAP service functionality", "10.0.0.1", VisibilityMode.BLACK_BOX
    )
    functional = next(item for item in assertions if item.id == "functional-validation")
    wrong_service = Evidence(
        source="probe:http_health_check", action="http_health_check", target="10.0.0.1",
        facts={"functional": True, "protocol_verified": True},
    )

    _, sufficiency = refresh_assessment([functional], [wrong_service], [])

    assert sufficiency[0]["current_level"] == EvidenceLevel.C0.value
    assert sufficiency[0]["sufficient"] is False


def test_asrep_configuration_assertion_does_not_require_hash_cracking():
    assertions = build_bootstrap_assertions(
        "Validate AS-REP configuration", "10.0.0.1", VisibilityMode.BLACK_BOX
    )
    evidence = [Evidence(
        source="probe:ldap", action="directory_enumeration", target="10.0.0.1",
        facts={"domain_name": "corp.local", "asrep_candidates": ["alice"]},
    )]
    _, sufficiency = refresh_assessment(assertions, evidence, [])
    assert all(item["sufficient"] for item in sufficiency)

    agents = Agents(tools=ToolRegistry(target_policy=TargetPolicy(["10.0.0.1"])))
    state = {
        "target": "10.0.0.1", "target_profiles": {}, "runner_ips": [],
        "discovered_targets": ["10.0.0.1"],
        "recon_coverage": {"10.0.0.1": {"checks": {"nmap:default": {"status": "completed"}}}},
        "qa_assertions": [item.model_dump(mode="json") for item in assertions],
        "evidence": evidence, "evidence_opportunities": [], "method_history": [],
        "ad_knowledge": {"domain": "corp.local"},
    }
    assert agents._completion_gate_open(state)


def test_sufficient_assertion_blocks_unnecessary_hash_cracking_escalation():
    class EscalatingModel:
        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            return Decision(
                next_agent=Role.TESTING,
                objective="crack the observed AS-REP material",
                action="ad_hash_cracking",
                target="10.0.0.1",
                justification="The tool is available.",
            )

    assertions = build_bootstrap_assertions(
        "Validate AS-REP configuration", "10.0.0.1", VisibilityMode.BLACK_BOX
    )
    evidence = [Evidence(
        source="probe:ldap", action="directory_enumeration", target="10.0.0.1",
        facts={"domain_name": "corp.local", "asrep_candidates": ["alice"]},
    )]
    agents = Agents(
        llm=EscalatingModel(),
        tools=ToolRegistry(
            tools={"ad_hash_cracking": object()},
            target_policy=TargetPolicy(["10.0.0.1"]),
        ),
    )
    state = {
        "target": "10.0.0.1", "objective": "Validate AS-REP configuration", "iteration": 0,
        "evidence": evidence, "method_history": [], "target_profiles": {},
        "recon_coverage": {"10.0.0.1": {"checks": {"nmap:default": {"status": "completed"}}}},
        "discovered_targets": ["10.0.0.1"], "runner_ips": [],
        "ad_knowledge": {"domain": "corp.local"}, "evidence_opportunities": [],
        "qa_assertions": [item.model_dump(mode="json") for item in assertions],
        "evidence_sufficiency": [], "messages": [], "human_directives": [],
        "human_intent": {}, "task_plan": {}, "observation_index": {},
        "runtime_config": {}, "replan_count": 0, "autonomous_replan_count": 0,
    }

    result = asyncio.run(agents.supervisor(state))

    assert result["last_decision"].next_agent == Role.JUDGE
    assert result["last_decision"].action == "evaluate_ad_evidence"


def test_end_to_end_assertion_requires_c5_even_after_minimal_exploit_evidence():
    assertions = build_bootstrap_assertions(
        "Validate the end-to-end attack path", "10.0.0.1", VisibilityMode.GRAY_BOX
    )
    end_to_end = next(item for item in assertions if item.id == "end-to-end-validation")
    minimal = Evidence(
        source="probe:controlled_test", action="minimal_exploit_validation", target="10.0.0.1",
        facts={"exploitability_verified": True, "effect_observed": True},
    )
    _, first = refresh_assessment([end_to_end], [minimal], [])
    assert first[0]["current_level"] == EvidenceLevel.C4.value
    assert first[0]["sufficient"] is False

    complete = Evidence(
        source="probe:attack_path", action="end_to_end_attack_validation", target="10.0.0.1",
        facts={"attack_path_complete": True, "final_goal_achieved": True},
    )
    _, second = refresh_assessment([end_to_end], [minimal, complete], [])
    assert second[0]["current_level"] == EvidenceLevel.C5.value
    assert second[0]["sufficient"] is True


def test_unverified_tool_cannot_self_declare_c5_evidence():
    forged = Evidence(
        source="kali:check_port", action="service_enumeration", target="10.0.0.1",
        facts={"evidence_level": "C5"},
    )
    verified = Evidence(
        source="evidence-verification:assertion-1", action="verify", target="10.0.0.1",
        facts={"evidence_level": "C5", "evidence_level_verified": True},
    )

    assert infer_evidence_level(forged) == EvidenceLevel.C0
    assert infer_evidence_level(verified) == EvidenceLevel.C5


def test_plain_recon_tool_cannot_self_declare_end_to_end_success():
    forged = Evidence(
        source="kali:check_port", action="service_enumeration", target="10.0.0.1",
        facts={"final_goal_achieved": True, "flag_retrieved": True},
    )

    assert infer_evidence_level(forged) == EvidenceLevel.C0


def test_contradictory_evidence_never_becomes_sufficient():
    assertion = build_bootstrap_assertions(
        "Validate the service configuration", "10.0.0.1", VisibilityMode.BLACK_BOX
    )[0]
    evidence = Evidence(
        source="probe:check_port", action="service_enumeration", target="10.0.0.1",
        facts={"open_ports": [{"port": 389}], "contradictory": True},
    )

    _, sufficiency = refresh_assessment([assertion], [evidence], [])

    assert sufficiency[0]["current_level"] == EvidenceLevel.C2.value
    assert sufficiency[0]["sufficient"] is False
    assert sufficiency[0]["status"] == "contradictory"


def test_failed_tool_output_cannot_satisfy_evidence_threshold_without_partial_marker():
    assertion = build_bootstrap_assertions(
        "Validate the service configuration", "10.0.0.1", VisibilityMode.BLACK_BOX
    )[0]
    failed = Evidence(
        source="probe:check_port", action="service_enumeration", target="10.0.0.1",
        exit_code=1, stderr="probe failed",
        facts={"open_ports": [{"port": 389, "service": "ldap"}]},
    )

    _, sufficiency = refresh_assessment([assertion], [failed], [])

    assert sufficiency[0]["current_level"] == EvidenceLevel.C0.value
    assert sufficiency[0]["sufficient"] is False
    assert sufficiency[0]["status"] == "blocked"


def test_visibility_mode_aliases_have_safe_defaults():
    assert parse_visibility_mode("white") == VisibilityMode.WHITE_BOX
    assert parse_visibility_mode("grey-box") == VisibilityMode.GRAY_BOX
    assert parse_visibility_mode("black_box") == VisibilityMode.BLACK_BOX
    assert parse_visibility_mode(None, specification_available=True) == VisibilityMode.WHITE_BOX
    assert parse_visibility_mode(None, specification_available=False) == VisibilityMode.BLACK_BOX


def test_graph_recursion_budget_is_not_the_legacy_iteration_cap():
    class Args:
        max_iterations = 1
        max_model_calls = 10
        max_tool_calls = 20

    assert graph_recursion_limit(Args()) >= 94


def test_white_box_specification_assertions_reach_task_state(tmp_path):
    specification = tmp_path / "qa-spec.json"
    specification.write_text(
        '{"assertions": [{"id": "web-health", "statement": "HTTPS health is verified", '
        '"target": "10.0.0.20", "assertion_type": "service_function", '
        '"required_evidence_level": "C3", "allowed_methods": ["http_health_check"]}, '
        '{"id": "runner-must-not-be-tested", "statement": "Kali is a target", '
        '"target": "local-kali", "required_evidence_level": "C2"}]}',
        encoding="utf-8",
    )

    assertions = build_task_assertions(
        "Validate the range", "10.0.0.0/24", VisibilityMode.WHITE_BOX, str(specification)
    )

    assert {item["id"] for item in assertions} == {"environment-baseline", "web-health"}
    web = next(item for item in assertions if item["id"] == "web-health")
    assert web["source"] == "specification"
    assert web["required_evidence_level"] == EvidenceLevel.C3.value


def test_llm_opportunity_cannot_inject_unauthorized_target():
    class PoisonedAnalyst:
        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            return EvidenceAnalysis(
                useful_content=["A service was observed."],
                opportunities=[EvidenceOpportunity(
                    key="http:10.0.0.99", tool="http_health_check",
                    action="http_health_check", target="10.0.0.99",
                    reason="model supplied target",
                )],
            )

    agents = Agents(
        llm=PoisonedAnalyst(),
        tools=ToolRegistry(
            tools={"http_health_check": object()},
            target_policy=TargetPolicy(["10.0.0.20"]),
        ),
    )
    evidence = Evidence(
        source="probe:check_port", action="service_enumeration", target="10.0.0.20",
        facts={"open_ports": [{"port": 443, "service": "https"}]},
    )

    analysis = asyncio.run(agents._analyze_evidence({
        "target": "10.0.0.20",
        "evidence": [],
        "evidence_analyses": [],
        "discovered_targets": ["10.0.0.99"],
        "runner_ips": [],
    }, evidence))

    assert all(item["target"] != "10.0.0.99" for item in analysis["opportunities"])


def test_human_forbidden_sensitive_capability_cannot_be_bypassed_by_alias():
    class ForbiddenBypassModel:
        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            return Decision(
                next_agent=Role.TESTING,
                objective="crack material",
                action="assess_material",
                capability="hash_cracking_assessment",
                target="10.0.0.1",
                justification="model chose the capability alias",
            )

    intent = HumanIntent(
        raw_instruction="不要破解 hash，改做其他檢查",
        forbidden_tools=["ad_hash_cracking"],
    )
    agents = Agents(
        llm=ForbiddenBypassModel(),
        tools=ToolRegistry(
            tools={"ad_hash_cracking": object()},
            target_policy=TargetPolicy(["10.0.0.1"]),
        ),
    )

    result = asyncio.run(agents.supervisor({
        "target": "10.0.0.1", "objective": "continue QA", "iteration": 0,
        "evidence": [], "method_history": [], "target_profiles": {},
        "recon_coverage": {}, "discovered_targets": ["10.0.0.1"], "runner_ips": [],
        "ad_knowledge": {}, "runtime_config": {}, "human_directives": [],
        "human_intent": intent.model_dump(mode="json"), "messages": [],
        "observation_index": {}, "action_history": [], "replan_count": 0,
        "autonomous_replan_count": 0,
    }))

    assert result["last_decision"].action == "replan_after_human_constraint"


def test_runner_identity_evidence_is_kept_in_initial_report(tmp_path, monkeypatch):
    monkeypatch.setenv("CYBERQA_REPORT_DIR", str(tmp_path))
    report = write_initial_recon_report({
        "scenario_id": "runner-report",
        "target": "10.0.0.0/24",
        "runner_ips": ["10.0.0.42"],
        "evidence": [Evidence(
            source="runner:interfaces", action="runner_identity", target="environment",
            stdout="10.0.0.42",
        )],
    }, "runner-report")

    assert "runner:interfaces" in open(report, encoding="utf-8").read()
