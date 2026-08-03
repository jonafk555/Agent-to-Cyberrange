from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import shlex
import shutil
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from langchain_core.tools import BaseTool, tool

from .ad_capability_tools import build_ad_capability_tools
from .memory import ObservationStore
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


class TargetPolicy:
    """Runtime target authorization with CIDR and discovery support."""

    def __init__(self, entries: list[str] | tuple[str, ...] | None = None):
        configured = entries or [item.strip() for item in os.getenv(
            "CYBERQA_ALLOWED_TARGETS", "127.0.0.1,localhost,::1"
        ).split(",") if item.strip()]
        self.entries: set[str] = set(configured)

    def allows(self, target: str) -> bool:
        if target in self.entries:
            return True
        try:
            address = ipaddress.ip_address(target)
            return any(address in ipaddress.ip_network(entry, strict=False)
                       for entry in self.entries if "/" in entry)
        except ValueError:
            return False

    def add(self, target: str) -> bool:
        if target and target not in self.entries:
            self.entries.add(target)
            return True
        return False

    def snapshot(self) -> list[str]:
        return sorted(self.entries)


@dataclass
class KaliTool:
    """Run one fixed, allow-listed Kali command without invoking a shell."""
    name: str
    executable: str
    fixed_args: tuple[str, ...] = ()
    tail_args: tuple[str, ...] = ()
    executable_candidates: tuple[str, ...] = ()
    target_arg: bool = True
    target_prefix: str = ""
    target_index: int | None = None
    timeout: float = 30.0
    requires_target: bool = True
    required_env: tuple[str, ...] = ()
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    target_policy: TargetPolicy | None = None

    async def observe(self, target: str, action: str, **kwargs: Any) -> Evidence:
        policy = self.target_policy or TargetPolicy()
        if self.requires_target and not policy.allows(target):
            raise PermissionError(f"Target is not in CYBERQA_ALLOWED_TARGETS: {target}")
        missing_env = [name for name in self.required_env if not os.getenv(name)]
        if missing_env:
            raise RuntimeError(
                f"{self.name} requires AD configuration: set {', '.join(missing_env)} "
                "in the same runtime environment as cyberqa"
            )
        if not self.requires_target:
            target = "local-kali"
        executable = self._resolve_executable()
        argv = [executable, *self.fixed_args]
        if self.target_arg:
            target_arg = f"{self.target_prefix}{target}"
            if self.target_index is None:
                argv.append(target_arg)
            else:
                if not 0 <= self.target_index <= len(self.fixed_args):
                    raise ValueError(f"target_index is out of range for {self.name}")
                argv.insert(1 + self.target_index, target_arg)
        argv.extend(self.tail_args)
        if kwargs.get("args"):
            raise ValueError("This fixed Kali adapter does not accept arbitrary command arguments")
        safe_argv = self._redact_argv(argv)
        if self.on_event:
            self.on_event("tool_start", {"tool": self.name, "argv": safe_argv})
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except FileNotFoundError as exc:
            if self.on_event:
                self.on_event("tool_result", {"tool": self.name, "exit_code": -1,
                                               "stderr": str(exc), "stdout": ""})
            raise RuntimeError(self._missing_executable_message()) from exc
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
            facts={"argv": [shlex.quote(x) for x in safe_argv], "returncode": process.returncode},
        )
        if self.executable == "nmap":
            evidence.facts["discovered_targets"] = sorted(_discover_ip_addresses(evidence.stdout))
            evidence.facts["open_ports"] = _discover_open_ports(evidence.stdout)
        if self.on_event:
            self.on_event("tool_result", {"tool": self.name, "exit_code": process.returncode,
                                           "stdout": evidence.stdout, "stderr": evidence.stderr})
        if self.name == "check_port" or self.executable == "nmap":
            for discovered in _discover_ip_addresses(evidence.stdout):
                if policy.add(discovered) and self.on_event:
                    self.on_event("target_discovered", {"target": discovered,
                                                         "allowed_targets": policy.snapshot()})
        return evidence

    def _resolve_executable(self) -> str:
        candidates = (self.executable, *self.executable_candidates)
        for candidate in candidates:
            if shutil.which(candidate):
                return candidate
        # Preserve the primary name for mocked runners and let subprocess
        # produce the same diagnostic if PATH changes between checks.
        return self.executable

    def _missing_executable_message(self) -> str:
        candidates = ", ".join((self.executable, *self.executable_candidates))
        return (f"No executable found for {self.name}. Tried: {candidates}. "
                "Check that the Kali package is installed in the same runtime "
                "where cyberqa is running and that its bin directory is on PATH.")

    @staticmethod
    def _redact_argv(argv: list[str]) -> list[str]:
        secrets = {value for name in ("CYBERQA_AD_PASSWORD", "AD_PASSWORD")
                   if (value := os.getenv(name))}
        return ["***REDACTED***" if item in secrets else item for item in argv]


def _discover_ip_addresses(output: str) -> set[str]:
    candidates = set(re.findall(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", output))
    discovered: set[str] = set()
    for candidate in candidates:
        try:
            ipaddress.ip_address(candidate)
            discovered.add(candidate)
        except ValueError:
            continue
    return discovered


def _discover_open_ports(output: str) -> list[dict[str, str]]:
    """Extract the small, stable service inventory needed for QA planning."""
    ports: list[dict[str, str]] = []
    for line in output.splitlines():
        match = re.match(r"\s*(\d+)/(tcp|udp)\s+open\s+(\S+)", line)
        if match:
            ports.append({"port": match.group(1), "protocol": match.group(2),
                          "service": match.group(3)})
    return ports


TOOL_NAMES = ("ssh", "powershell", "winrm", "nmap", "bloodhound", "sharphound", "impacket",
              "certipy", "netexec", "metasploit", "ansible", "terraform", "vmware", "proxmox")


class ToolRegistry:
    def __init__(self, tools: dict[str, FactTool] | None = None,
                 target_policy: TargetPolicy | None = None):
        self.tools = tools or {}
        self.target_policy = target_policy or TargetPolicy()
        self.observations = ObservationStore()
        # ToolNode may schedule identical tool calls concurrently.  A cache
        # lookup alone is not enough in that case: both calls can miss before
        # either result is stored.  One lock per signature makes each probe a
        # single-flight operation, including failed probes.
        self._observation_locks: dict[str, asyncio.Lock] = {}

    def register(self, tool: FactTool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> FactTool:
        if name not in self.tools:
            raise KeyError(f"No tool registered: {name}")
        return self.tools[name]

    def _signature(self, adapter: FactTool, name: str, target: str, action: str,
                   parameters: dict[str, Any]) -> str:
        # For fixed adapters the LLM's prose action is not a new command.
        # This prevents the same nmap probe being rerun under a new action
        # description.
        signature_action = "fixed-command" if isinstance(adapter, KaliTool) else action
        return self.observations.signature(name, target, signature_action, parameters)

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
                    parameters = parameters or {}
                    signature = self._signature(bound_adapter, bound_name, target, action, parameters)
                    cached = self.observations.get(signature)
                    if cached is not None:
                        return {**cached, "signature": signature, "cached": True}
                    lock = self._observation_locks.setdefault(signature, asyncio.Lock())
                    async with lock:
                        # Re-check after waiting: another concurrent caller
                        # may have completed this exact probe.
                        cached = self.observations.get(signature)
                        if cached is not None:
                            return {**cached, "signature": signature, "cached": True}
                        try:
                            evidence = await bound_adapter.observe(target, action, **(parameters or {}))
                            evidence_data = evidence.model_dump(mode="json")
                            if evidence.exit_code not in (None, 0):
                                result = {
                                    "ok": False, "tool": bound_name,
                                    "error": _tool_error_message(evidence),
                                    "error_kind": _tool_error_kind(evidence),
                                    "needs_human": True, "evidence": evidence_data,
                                }
                            else:
                                result = {"ok": True, "tool": bound_name,
                                          "evidence": evidence_data}
                        except Exception as exc:  # surfaced to ReAct; never silently treated as evidence
                            result = {"ok": False, "tool": bound_name,
                                      "error": str(exc), "error_kind": type(exc).__name__,
                                      "needs_human": True}
                        # Cache failures too. Retrying the same broken command
                        # is not debugging and can create noisy/infinite loops.
                        self.observations.put(signature, result)
                        return {**result, "signature": signature, "cached": False}

                return probe

            wrapped.append(make_probe(adapter, tool_name))
        return wrapped


def _tool_error_kind(evidence: Evidence) -> str:
    text = f"{evidence.stderr}\n{evidence.stdout}".lower()
    if "usage:" in text or "unrecognized arguments" in text or "required" in text:
        return "invalid_arguments"
    if "permission denied" in text or "access denied" in text:
        return "permission_denied"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "connection refused" in text or "no route to host" in text:
        return "connectivity"
    return "nonzero_exit"


def _tool_error_message(evidence: Evidence) -> str:
    detail = (evidence.stderr or evidence.stdout or "tool exited with a non-zero status").strip()
    argv = evidence.facts.get("argv") if isinstance(evidence.facts, dict) else None
    return f"exit={evidence.exit_code}; argv={argv}; {detail[-2000:]}"


def build_kali_registry(on_event: Callable[[str, dict[str, Any]], None] | None = None,
                        allowed_targets: list[str] | tuple[str, ...] | None = None) -> ToolRegistry:
    """Create the standard fixed-command Kali tool set for ReAct agents."""
    policy = TargetPolicy(allowed_targets)
    registry = ToolRegistry(target_policy=policy)
    bloodhound_args = (
        "-c", os.getenv("CYBERQA_AD_COLLECTION", "DCOnly"),
        "-d", os.getenv("CYBERQA_AD_DOMAIN", ""),
        "-u", os.getenv("CYBERQA_AD_USERNAME", ""),
        "-p", os.getenv("CYBERQA_AD_PASSWORD", ""),
        "-ns",
    )
    specs = [
        KaliTool("check_port", "nmap", ("-Pn", "-T3", "--top-ports", "100"), on_event=on_event, target_policy=policy),
        KaliTool("check_dns_resolution", "dig", ("+short",), on_event=on_event, target_policy=policy),
        KaliTool("http_health_check", "curl", ("--fail", "--silent", "--show-error", "--max-time", "15"), target_prefix="http://", on_event=on_event, target_policy=policy),
        KaliTool("ldap_bind", "ldapsearch", ("-x", "-H"), tail_args=("-s", "base", "-b", ""), target_prefix="ldap://", on_event=on_event, target_policy=policy),
        # smbclient's -L consumes the server argument immediately after it:
        # ``smbclient -L //TARGET -N``.  Appending TARGET after -N makes
        # smbclient parse the option stream as a service/password command.
        KaliTool("smb_negotiate", "smbclient", ("-L", "-N"), target_prefix="//",
                 target_index=1, on_event=on_event, target_policy=policy),
        # NetExec expects the target immediately after the protocol module:
        # ``nxc smb TARGET --shares ...``.  Keeping the insertion point in the
        # fixed adapter prevents the generic argv builder from producing the
        # invalid ``nxc smb --shares ... TARGET`` form.
        KaliTool("nxc_smb_recon", "nxc", ("smb", "--shares", "-u", "", "-p", ""), target_index=1, on_event=on_event, target_policy=policy),
        KaliTool("nxc_ldap_recon", "nxc", ("ldap", "-u", "", "-p", ""), target_index=1, on_event=on_event, target_policy=policy),
        KaliTool("impacket_rpc_recon", "impacket-rpcdump", fixed_args=(),
                 executable_candidates=("rpcdump.py", "rpcdump"),
                 on_event=on_event, target_policy=policy),
        KaliTool("bloodhound_recon", "bloodhound-python", fixed_args=bloodhound_args,
                 target_index=len(bloodhound_args),
                 required_env=("CYBERQA_AD_DOMAIN", "CYBERQA_AD_USERNAME", "CYBERQA_AD_PASSWORD"),
                 on_event=on_event, target_policy=policy),
        KaliTool("inspect_routes", "ip", ("route",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_dns_config", "cat", ("/etc/resolv.conf",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_firewall", "nft", ("list", "ruleset"), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_time_sync", "timedatectl", ("show-timesync",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_os_version", "uname", ("-a",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_os_release", "cat", ("/etc/os-release",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_interfaces", "ip", ("-j", "addr"), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_open_ports", "ss", ("-lntup"), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_acl", "getfacl", ("-p", "/etc", "/opt", "/srv"), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_local_users", "getent", ("passwd",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_domain_users", "wbinfo", ("-u",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_privileges", "id", (), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_sudo", "sudo", ("-n", "-l"), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_suid_files", "find", ("/", "-xdev", "-type", "f", "-perm", "/6000", "-print"), target_arg=False, requires_target=False, timeout=90.0, on_event=on_event, target_policy=policy),
        KaliTool("inspect_range_config", "find", ("/etc", "/opt", "/srv", "-maxdepth", "3", "-type", "f", "-print"), target_arg=False, requires_target=False, timeout=45.0, on_event=on_event, target_policy=policy),
    ]
    for spec in specs:
        registry.register(spec)
    for capability_tool in build_ad_capability_tools(policy):
        registry.register(capability_tool)
    return registry
