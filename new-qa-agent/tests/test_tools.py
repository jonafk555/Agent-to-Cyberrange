import asyncio

import pytest

from cyberqa.tools import KaliTool, TargetPolicy, build_kali_registry


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
