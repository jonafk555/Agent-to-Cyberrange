from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

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
