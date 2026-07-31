from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from langchain_core.tools import BaseTool, tool

from .models import Evidence


class FactTool(Protocol):
    name: str
    async def observe(self, target: str, action: str, **kwargs: Any) -> Evidence: ...


@dataclass
class DryRunTool:
    """Safe adapter used by tests/examples; real adapters preserve this contract."""
    name: str
    facts: dict[str, Any]

    async def observe(self, target: str, action: str, **kwargs: Any) -> Evidence:
        return Evidence(source=self.name, action=action, target=target, facts=self.facts)


class CommandTool:
    """Interface boundary for SSH, PowerShell, WinRM, Nmap and operator tooling."""
    def __init__(self, name: str, runner: Any):
        self.name, self.runner = name, runner

    async def observe(self, target: str, action: str, **kwargs: Any) -> Evidence:
        result = await self.runner(target=target, action=action, **kwargs)
        return Evidence(source=self.name, action=action, target=target,
                        exit_code=result.get("exit_code"), stdout=result.get("stdout", ""),
                        stderr=result.get("stderr", ""), facts=result.get("facts", {}))


@dataclass
class KaliTool:
    """Run one fixed, allow-listed Kali command without invoking a shell."""
    name: str
    executable: str
    fixed_args: tuple[str, ...] = ()
    target_arg: bool = True
    target_prefix: str = ""
    timeout: float = 30.0
    requires_target: bool = True
    on_event: Callable[[str, dict[str, Any]], None] | None = None

    async def observe(self, target: str, action: str, **kwargs: Any) -> Evidence:
        if self.requires_target and not _target_allowed(target):
            raise PermissionError(f"Target is not in CYBERQA_ALLOWED_TARGETS: {target}")
        if not self.requires_target:
            target = "local-kali"
        argv = [self.executable, *self.fixed_args]
        if self.target_arg:
            argv.append(f"{self.target_prefix}{target}")
        if kwargs.get("args"):
            raise ValueError("This fixed Kali adapter does not accept arbitrary command arguments")
        if self.on_event:
            self.on_event("tool_start", {"tool": self.name, "argv": argv})
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except FileNotFoundError as exc:
            if self.on_event:
                self.on_event("tool_result", {"tool": self.name, "exit_code": -1,
                                               "stderr": str(exc), "stdout": ""})
            raise RuntimeError(f"Kali executable is not installed: {self.executable}") from exc
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            if self.on_event:
                self.on_event("tool_result", {"tool": self.name, "exit_code": -1,
                                               "stderr": str(exc), "stdout": ""})
            raise TimeoutError(f"{self.name} timed out after {self.timeout}s") from exc
        evidence = Evidence(
            source=f"kali:{self.name}", action=action, target=target,
            exit_code=process.returncode, stdout=stdout.decode(errors="replace")[-12000:],
            stderr=stderr.decode(errors="replace")[-12000:],
            facts={"argv": [shlex.quote(x) for x in argv], "returncode": process.returncode},
        )
        if self.on_event:
            self.on_event("tool_result", {"tool": self.name, "exit_code": process.returncode,
                                           "stdout": evidence.stdout, "stderr": evidence.stderr})
        return evidence


def _target_allowed(target: str) -> bool:
    configured = os.getenv("CYBERQA_ALLOWED_TARGETS", "127.0.0.1,localhost,::1")
    return target in {item.strip() for item in configured.split(",") if item.strip()}


TOOL_NAMES = ("ssh", "powershell", "winrm", "nmap", "bloodhound", "sharphound", "impacket",
              "certipy", "netexec", "metasploit", "ansible", "terraform", "vmware", "proxmox")


class ToolRegistry:
    def __init__(self, tools: dict[str, FactTool] | None = None):
        self.tools = tools or {}

    def register(self, tool: FactTool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> FactTool:
        if name not in self.tools:
            raise KeyError(f"No tool registered: {name}")
        return self.tools[name]

    def langchain_tools(self, names: list[str] | tuple[str, ...] | None = None) -> list[BaseTool]:
        """Expose allow-listed fact tools to a ReAct agent.

        The wrapper deliberately returns facts only.  It also converts adapter
        failures into a tool result so the agent can inspect the failure and
        either try a safer next step or ask a human for help.
        """
        selected = names or tuple(self.tools)
        wrapped: list[BaseTool] = []
        for tool_name in selected:
            if tool_name not in self.tools:
                continue
            adapter = self.tools[tool_name]

            def make_probe(bound_adapter: FactTool, bound_name: str) -> BaseTool:
                @tool(f"{bound_name}_probe")
                async def probe(target: str, action: str,
                                parameters: dict[str, Any] | None = None) -> dict[str, Any]:
                    """Run one authorized, fact-only probe against the cyber-range target."""
                    try:
                        evidence = await bound_adapter.observe(target, action, **(parameters or {}))
                        return {"ok": True, "tool": bound_name, "evidence": evidence.model_dump(mode="json")}
                    except Exception as exc:  # surfaced to ReAct; never silently treated as evidence
                        return {"ok": False, "tool": bound_name, "error": str(exc), "needs_human": True}

                return probe

            wrapped.append(make_probe(adapter, tool_name))
        return wrapped


def build_kali_registry(on_event: Callable[[str, dict[str, Any]], None] | None = None) -> ToolRegistry:
    """Create the standard fixed-command Kali tool set for ReAct agents."""
    registry = ToolRegistry()
    specs = [
        KaliTool("check_port", "nmap", ("-Pn", "-T3", "--top-ports", "100"), on_event=on_event),
        KaliTool("check_dns_resolution", "dig", ("+short",), on_event=on_event),
        KaliTool("http_health_check", "curl", ("--fail", "--silent", "--show-error", "--max-time", "15"), target_prefix="http://", on_event=on_event),
        KaliTool("ldap_bind", "ldapsearch", ("-x", "-H"), target_prefix="ldap://", on_event=on_event),
        KaliTool("smb_negotiate", "smbclient", ("-L", "-N"), target_prefix="//", on_event=on_event),
        KaliTool("inspect_routes", "ip", ("route",), target_arg=False, requires_target=False, on_event=on_event),
        KaliTool("inspect_dns_config", "cat", ("/etc/resolv.conf",), target_arg=False, requires_target=False, on_event=on_event),
        KaliTool("inspect_firewall", "nft", ("list", "ruleset"), target_arg=False, requires_target=False, on_event=on_event),
        KaliTool("inspect_time_sync", "timedatectl", ("show-timesync",), target_arg=False, requires_target=False, on_event=on_event),
    ]
    for spec in specs:
        registry.register(spec)
    return registry
