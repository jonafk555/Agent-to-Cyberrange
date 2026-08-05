import asyncio

import pytest

from cyberqa.tools import KaliTool, TargetPolicy, build_kali_registry, summarize_output


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


def test_nmap_and_nxc_accept_reviewed_dynamic_argv_fragments():
    registry = build_kali_registry(allowed_targets=["10.0.0.1"])
    assert registry.get("check_port").build_argv("10.0.0.1", {
        "argv": ["-Pn", "-sV", "-p", "53,88,389,445", "-T", "3"],
    }) == ["nmap", "-Pn", "-sV", "-p", "53,88,389,445", "-T", "3", "10.0.0.1"]
    assert registry.get("nxc_ldap_recon").build_argv("10.0.0.1", {
        "argv": ["--users", "--groups", "--threads", "4"],
    }) == ["nxc", "ldap", "10.0.0.1", "--users", "--groups", "--threads", "4"]


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
