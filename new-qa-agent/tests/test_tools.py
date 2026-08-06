import asyncio
import os

import pytest

from cyberqa.ad_capability_tools import ADCapabilityTool, _store_asrep_hashes
from cyberqa.memory import ObservationStore
from cyberqa.models import Evidence
from cyberqa.tools import KaliTool, TargetPolicy, ToolRegistry, build_kali_registry, summarize_output


class FakeProcess:
    returncode = 0

    async def communicate(self):
        return b"", b""


@pytest.mark.asyncio
async def test_netexec_places_target_after_protocol_and_deduplicates(monkeypatch):
    calls = []

    async def fake_create_subprocess_exec(*argv, **kwargs):
        calls.append(argv)
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    registry = build_kali_registry(allowed_targets=["10.0.0.1"])

    await asyncio.gather(
        registry.langchain_tools()[5].ainvoke({"target": "10.0.0.1", "action": "recon"}),
        registry.langchain_tools()[5].ainvoke({"target": "10.0.0.1", "action": "different wording"}),
    )
    await registry.get("nxc_ldap_recon").observe("10.0.0.1", "recon")

    assert calls.count(("nxc", "smb", "10.0.0.1", "--shares")) == 1
    assert calls[-1][:3] == ("nxc", "ldap", "10.0.0.1")


def test_target_index_is_validated():
    tool = KaliTool("bad", "tool", ("subcommand",), target_index=3,
                    target_policy=TargetPolicy(["10.0.0.1"]))

    with pytest.raises(ValueError, match="target_index"):
        asyncio.run(tool.observe("10.0.0.1", "check"))


def test_nmap_default_and_nxc_profiles_are_reviewed_but_selectable():
    registry = build_kali_registry(allowed_targets=["10.0.0.1"])
    assert registry.get("check_port").build_argv("10.0.0.1") == ["nmap", "-sC", "-sV", "10.0.0.1"]
    assert registry.get("check_port").build_argv("10.0.0.1", {"profile": "ad_tcp"})[0] == "nmap"
    assert registry.get("nxc_smb_recon").build_argv("10.0.0.1", {"profile": "sessions"}) == [
        "nxc", "smb", "10.0.0.1", "--sessions"
    ]


def test_semantic_probe_aliases_share_effective_command_signature(monkeypatch):
    monkeypatch.setenv("CYBERQA_OBSERVATION_DB", ":memory:")
    registry = build_kali_registry(allowed_targets=["10.0.0.1"])

    semantic = registry.command_signature(
        "check_port", "10.0.0.1", "service_enumeration", {"profile": "default"}
    )
    direct = registry.command_signature(
        "check_port", "10.0.0.1", "nmap_service_detection", {"profile": "default"}
    )

    assert semantic == direct


def test_observation_namespace_separates_tasks_with_same_effective_command(monkeypatch):
    monkeypatch.setenv("CYBERQA_OBSERVATION_DB", ":memory:")
    first = build_kali_registry(allowed_targets=["10.0.0.1"])
    second = build_kali_registry(allowed_targets=["10.0.0.1"])
    first.set_cache_namespace("scenario:run-one")
    second.set_cache_namespace("scenario:run-two")

    assert first.command_signature("check_port", "10.0.0.1", "service_enumeration", {"profile": "default"}) != second.command_signature(
        "check_port", "10.0.0.1", "service_enumeration", {"profile": "default"}
    )


@pytest.mark.asyncio
async def test_observation_namespace_prevents_cross_task_cache_hit(tmp_path, monkeypatch):
    calls = []

    class Probe:
        name = "probe"

        async def observe(self, target, action, **kwargs):
            calls.append((target, action))
            return Evidence(source="probe", action=action, target=target)

    monkeypatch.setenv("CYBERQA_OBSERVATION_DB", str(tmp_path / "observations.sqlite3"))
    first = ToolRegistry({"probe": Probe()}, TargetPolicy(["10.0.0.1"]))
    second = ToolRegistry({"probe": Probe()}, TargetPolicy(["10.0.0.1"]))
    first.set_cache_namespace("scenario:run-one")
    second.set_cache_namespace("scenario:run-two")

    await first.observe("probe", "10.0.0.1", "probe")
    await second.observe("probe", "10.0.0.1", "probe")

    assert calls == [("10.0.0.1", "probe"), ("10.0.0.1", "probe")]


@pytest.mark.asyncio
async def test_per_task_tool_budget_is_a_real_boundary(monkeypatch):
    class Probe:
        name = "probe"

        async def observe(self, target, action, **kwargs):
            return Evidence(source="probe", action=action, target=target)

    monkeypatch.setenv("CYBERQA_OBSERVATION_DB", ":memory:")
    registry = ToolRegistry(
        tools={"probe": Probe()},
        target_policy=TargetPolicy(["10.0.0.1"]),
    )
    registry.begin_run("budget-test", 1)

    first = await registry.observe("probe", "10.0.0.1", "one")
    second = await registry.observe("probe", "10.0.0.1", "two")

    assert first["ok"] is True
    assert second["error_kind"] == "resource_budget"
    assert second["needs_human"] is True


@pytest.mark.asyncio
async def test_generic_tool_output_is_redacted_before_registry_evidence(monkeypatch):
    class SecretTool:
        name = "generic_probe"

        async def observe(self, target, action, **kwargs):
            return Evidence(
                source="generic_probe", action=action, target=target,
                stdout="username=alice password=PlainSecret $krb5asrep$23$alice@corp.local:ticket",
                facts={"password": "PlainSecret", "safe": "corp.local"},
            )

    monkeypatch.setenv("CYBERQA_OBSERVATION_DB", ":memory:")
    registry = ToolRegistry(
        tools={"generic_probe": SecretTool()},
        target_policy=TargetPolicy(["10.0.0.1"]),
    )
    result = await registry.observe("generic_probe", "10.0.0.1", "probe")

    rendered = str(result)
    assert "PlainSecret" not in rendered
    assert "$krb5asrep$" not in rendered
    assert result["evidence"]["facts"]["password"] == "[REDACTED]"


def test_observation_store_can_clear_durable_entries():
    store = ObservationStore(":memory:")
    store.put("one", {"ok": True})
    store.put("two", {"ok": False})

    assert store.clear("one") == 1
    assert store.get("one") is None
    assert store.get("two") is not None
    assert store.clear() == 1
    assert store.get("two") is None


def test_asrep_hashes_are_stored_as_restricted_artifacts_not_facts(tmp_path, monkeypatch):
    monkeypatch.setenv("CYBERQA_CREDENTIAL_MATERIAL_DIR", str(tmp_path))
    path, count = _store_asrep_hashes(
        "$krb5asrep$23$alice@corp.local:opaque-material\n"
    )
    assert path is not None
    assert count == 1
    assert "opaque-material" in open(path, encoding="utf-8").read()


def test_hash_cracking_promotes_only_process_local_credential(monkeypatch, tmp_path):
    monkeypatch.setenv("CYBERQA_AD_PASSWORD", "")
    output = tmp_path / "cracked.out"
    output.write_text("$krb5asrep$23$alice@corp.local:PlainSecret\n", encoding="utf-8")
    tool = ADCapabilityTool(
        "ad_hash_cracking", "hash_cracking_assessment", TargetPolicy(["10.0.0.1"])
    )
    facts = tool._hash_cracking_facts(
        ["hashcat", "--outfile", str(output)], 0
    )
    assert facts["hash_cracked"] is True
    assert facts["cracked_users"] == ["alice"]
    assert "PlainSecret" not in str(facts)
    assert os.getenv("CYBERQA_AD_PASSWORD") == "PlainSecret"


def test_hashcat_no_match_is_evidence_not_human_blocker(monkeypatch):
    class NoMatchTool:
        name = "ad_hash_cracking"

        def command_identity(self, target, parameters):
            return {"target": target, "parameters": parameters}

        async def observe(self, target, action, **kwargs):
            return Evidence(
                source="ad-capability:ad_hash_cracking", action=action, target=target,
                exit_code=1, facts={
                    "hash_cracking_attempted": True,
                    "hash_cracked": False,
                    "crack_status": "not_found",
                },
            )

    registry = ToolRegistry(
        tools={"ad_hash_cracking": NoMatchTool()},
        target_policy=TargetPolicy(["10.0.0.1"]),
    )
    result = asyncio.run(registry.observe(
        "ad_hash_cracking", "10.0.0.1", "hash_cracking_assessment",
        parameters={"hash_file": "a", "wordlist": "b"},
        authorization={
            "target": "10.0.0.1", "allowed_tools": ["ad_hash_cracking"],
            "tool_parameters": {"hash_file": "a", "wordlist": "b"},
        },
    ))
    assert result["ok"] is True
    assert result["expected_result"] == "hash_not_found"
    assert result.get("needs_human") is not True


def test_nmap_and_nxc_accept_reviewed_dynamic_argv_fragments():
    registry = build_kali_registry(allowed_targets=["10.0.0.1"])
    assert registry.get("check_port").build_argv("10.0.0.1", {
        "argv": ["-Pn", "-sV", "-p", "53,88,389,445", "-T", "3"],
    }) == ["nmap", "-Pn", "-sV", "-p", "53,88,389,445", "-T", "3", "10.0.0.1"]
    assert registry.get("nxc_ldap_recon").build_argv("10.0.0.1", {
        "argv": ["--users", "--groups", "--threads", "4"],
    }) == ["nxc", "ldap", "10.0.0.1", "--users", "--groups", "--threads", "4"]


def test_ldap_and_smb_expose_only_reviewed_repair_profiles():
    registry = build_kali_registry(allowed_targets=["10.0.0.1"])
    assert registry.get("ldap_bind").build_argv("10.0.0.1", {
        "profile": "starttls_rootdse",
    }) == ["ldapsearch", "-H", "ldap://10.0.0.1", "-x", "-ZZ", "-s", "base", "-b", ""]
    assert registry.get("smb_negotiate").build_argv("10.0.0.1", {
        "profile": "smb3",
    }) == ["smbclient", "-L", "//10.0.0.1", "-N", "-m", "SMB3"]
    with pytest.raises(ValueError, match="ldap argv"):
        registry.get("ldap_bind").build_argv("10.0.0.1", {"argv": ["-H", "10.0.0.2"]})


@pytest.mark.asyncio
async def test_recoverable_nonzero_result_is_evidence_before_human(monkeypatch):
    monkeypatch.setenv("CYBERQA_OBSERVATION_DB", ":memory:")

    class FailureProcess:
        returncode = 1

        async def communicate(self):
            return b"", b"Can't contact LDAP server"

    async def fake_create_subprocess_exec(*argv, **kwargs):
        return FailureProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    registry = build_kali_registry(allowed_targets=["10.0.0.1"])
    result = await registry.observe("ldap_bind", "10.0.0.1", "anonymous_identity_probe")

    assert result["ok"] is False
    assert result["error_kind"] == "nonzero_exit"
    assert result["recoverable"] is True
    assert result["needs_human"] is False
    assert result["evidence"]["facts"]["recoverable"] is True


def test_dynamic_argv_cannot_inject_a_command_or_target():
    registry = build_kali_registry(allowed_targets=["10.0.0.1"])
    with pytest.raises(ValueError, match="Unsupported nmap argv option"):
        registry.get("check_port").build_argv("10.0.0.1", {"argv": ["--script", "default"]})
    with pytest.raises(ValueError, match="Unsupported nxc argv option"):
        registry.get("nxc_smb_recon").build_argv("10.0.0.1", {"argv": ["10.0.0.2"]})


def test_probe_parameter_shapes_and_host_port_allowlist():
    registry = build_kali_registry(allowed_targets=["10.0.0.1"])
    assert registry.get("check_dns_resolution").build_argv(
        "10.0.0.1", {"name": "example.local"}
    ) == ["dig", "+short", "example.local"]
    assert registry.get("inspect_open_ports").fixed_args == ("-lntup",)
    assert registry.target_policy.allows("10.0.0.1:5985")


def test_runner_names_are_never_remote_recon_targets():
    policy = TargetPolicy(["10.0.0.0/24", "local-kali"])

    assert policy.is_local("local-kali")
    assert not policy.allows("local-kali")
    assert not policy.allows("environment")


@pytest.mark.asyncio
async def test_runner_interface_lookup_is_labeled_as_execution_context(monkeypatch):
    monkeypatch.setenv("CYBERQA_OBSERVATION_DB", ":memory:")

    async def fake_create_subprocess_exec(*argv, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    registry = build_kali_registry(allowed_targets=["10.0.0.0/24"])
    result = await registry.observe(
        "inspect_interfaces", "10.0.0.0/24", "runner_identity", force_refresh=True
    )

    assert result["evidence"]["target"] == "environment"


@pytest.mark.asyncio
async def test_non_identity_runner_inspection_is_blocked(monkeypatch):
    monkeypatch.setenv("CYBERQA_OBSERVATION_DB", ":memory:")
    registry = build_kali_registry(allowed_targets=["10.0.0.0/24"])

    result = await registry.observe(
        "inspect_os_version", "10.0.0.0/24", "initial_recon", force_refresh=True
    )

    assert result["error_kind"] == "runner_context_only"
    assert result["needs_human"] is False


def test_empty_tool_selection_is_not_expanded_to_registry():
    registry = build_kali_registry(allowed_targets=["10.0.0.1"])
    assert registry.langchain_tools([]) == []


@pytest.mark.asyncio
async def test_evidence_keeps_full_output_and_adds_summary(monkeypatch):
    output = "\n".join([
        "noise line 1", "53/tcp open domain", "noise line 3",
        "445/tcp open microsoft-ds", "domain: corp.local",
    ])

    class OutputProcess:
        returncode = 0

        async def communicate(self):
            return output.encode(), b""

    async def fake_create_subprocess_exec(*argv, **kwargs):
        return OutputProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    tool = KaliTool("check_port", "nmap", target_policy=TargetPolicy(["10.0.0.1"]))
    evidence = await tool.observe("10.0.0.1", "recon")

    assert evidence.stdout == output
    assert evidence.facts["stdout_lines"] == 5
    assert "53/tcp open domain" in evidence.facts["stdout_summary"]
    assert "domain: corp.local" in summarize_output(output)
