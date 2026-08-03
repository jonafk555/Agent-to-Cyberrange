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
        argv = self.build_argv(target, kwargs)
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

    def build_argv(self, target: str, parameters: dict[str, Any] | None = None) -> list[str]:
        """Build the reviewed argv used both for execution and cache identity."""
        parameters = parameters or {}
        supported = {"profile"} if self.name in {"check_port", "nxc_smb_recon", "nxc_ldap_recon"} else set()
        unknown = set(parameters) - supported
        if unknown:
            raise ValueError(
                f"Unsupported parameters for {self.name}: {', '.join(sorted(unknown))}"
            )
        executable = self._resolve_executable()
        fixed_args = self.fixed_args
        if self.name == "check_port":
            profiles = {
                "default": ("-sC", "-sV"),
                "top100": ("-Pn", "-T3", "--top-ports", "100"),
                "top1000": ("-Pn", "-T3", "--top-ports", "1000"),
                "ad_tcp": ("-Pn", "-T3", "-sV", "-p",
                           "53,80,88,135,139,389,443,445,464,593,636,3268,3269,3389,5985,5986,9389"),
            }
            profile = str(parameters.get("profile", "default"))
            if profile not in profiles:
                raise ValueError(f"Unsupported check_port profile: {profile}")
            fixed_args = profiles[profile]
        elif self.name in {"nxc_smb_recon", "nxc_ldap_recon"}:
            module = "smb" if self.name == "nxc_smb_recon" else "ldap"
            profiles = {
                "shares": ("--shares",),
                "users": ("--users",),
                "groups": ("--groups",),
                "sessions": ("--sessions",),
                "pass-pol": ("--pass-pol",),
                "enum": ("--shares", "--sessions"),
            }
            default_profile = "shares" if module == "smb" else "users"
            profile = str(parameters.get("profile", default_profile))
            if profile not in profiles:
                raise ValueError(f"Unsupported {self.name} profile: {profile}")
            credentials = []
            username = os.getenv("CYBERQA_AD_USERNAME", "")
            password = os.getenv("CYBERQA_AD_PASSWORD", "")
            if username and password:
                credentials = ["-u", username, "-p", password]
            fixed_args = (module, *profiles[profile], *credentials)
            self_target_index = 1
            argv = [executable, *fixed_args]
            argv.insert(self_target_index + 1, f"{self.target_prefix}{target}")
            argv.extend(self.tail_args)
            return argv
        argv = [executable, *fixed_args]
        if self.target_arg:
            target_arg = f"{self.target_prefix}{target}"
            if self.target_index is None:
                argv.append(target_arg)
            else:
                if not 0 <= self.target_index <= len(fixed_args):
                    raise ValueError(f"target_index is out of range for {self.name}")
                argv.insert(1 + self.target_index, target_arg)
        argv.extend(self.tail_args)
        return argv

    def command_identity(self, target: str, parameters: dict[str, Any]) -> dict[str, Any]:
        effective_target = "local-kali" if not self.requires_target else target
        return {"argv": self._redact_argv(self.build_argv(effective_target, parameters))}

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
        signature_action = "effective-command" if hasattr(adapter, "command_identity") else action
        identity = (
            adapter.command_identity(target, parameters)  # type: ignore[attr-defined]
            if hasattr(adapter, "command_identity") else parameters
        )
        return self.observations.signature(name, target, signature_action, identity)

    async def observe(self, name: str, target: str, action: str,
                      parameters: dict[str, Any] | None = None,
                      force_refresh: bool = False,
                      authorization: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute one probe through the same durable cache used by ToolNode."""
        parameters = parameters or {}
        adapter = self.get(name)
        if name in SENSITIVE_TOOL_NAMES:
            allowed = (authorization or {}).get("allowed_tools", [])
            expected = (authorization or {}).get("tool_parameters", {})
            if name not in allowed or parameters != expected:
                return {
                    "ok": False, "tool": name,
                    "error": "Sensitive tool requires an exact approved capability and parameters",
                    "error_kind": "approval_required", "needs_human": True,
                }
        try:
            signature = self._signature(adapter, name, target, action, parameters)
        except Exception as exc:
            # Invalid reviewed parameters are still durable diagnostic
            # evidence; they must not escape ToolNode as an uncaught graph
            # exception or be retried forever.
            signature = self.observations.signature(
                name, target, "invalid-command", {"parameters": parameters, "error": str(exc)}
            )
            result = {"ok": False, "tool": name, "error": str(exc),
                      "error_kind": "invalid_arguments", "needs_human": True}
            self.observations.put(signature, result)
            return {**result, "signature": signature, "cached": False}
        cached = None if force_refresh else self.observations.get(signature)
        if cached is not None:
            if isinstance(adapter, KaliTool) and adapter.on_event:
                adapter.on_event("tool_cached", {"tool": name, "target": target,
                                                  "signature": signature})
            return self._cache_hit(cached, signature, action)
        lock = self._observation_locks.setdefault(signature, asyncio.Lock())
        async with lock:
            cached = None if force_refresh else self.observations.get(signature)
            if cached is not None:
                return self._cache_hit(cached, signature, action)
            try:
                evidence = await adapter.observe(target, action, **parameters)
                evidence_data = evidence.model_dump(mode="json")
                if evidence.exit_code not in (None, 0):
                    result = {
                        "ok": False, "tool": name,
                        "error": _tool_error_message(evidence),
                        "error_kind": _tool_error_kind(evidence),
                        "needs_human": True, "evidence": evidence_data,
                    }
                else:
                    result = {"ok": True, "tool": name, "evidence": evidence_data}
            except Exception as exc:
                result = {"ok": False, "tool": name, "error": str(exc),
                          "error_kind": type(exc).__name__, "needs_human": True}
            self.observations.put(signature, result)
            return {**result, "signature": signature, "cached": False}

    @staticmethod
    def _cache_hit(cached: dict[str, Any], signature: str, action: str) -> dict[str, Any]:
        result = {**cached, "signature": signature, "cached": True}
        if isinstance(cached.get("evidence"), dict):
            evidence = dict(cached["evidence"])
            facts = dict(evidence.get("facts") or {})
            facts["cache_hit"] = True
            facts["original_action"] = evidence.get("action")
            evidence["action"] = action
            evidence["facts"] = facts
            result["evidence"] = evidence
        return result

    def langchain_tools(self, names: list[str] | tuple[str, ...] | None = None,
                        authorization: dict[str, Any] | None = None) -> list[BaseTool]:
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
                                parameters: dict[str, Any] | None = None,
                                force_refresh: bool = False) -> dict[str, Any]:
                    """Run one authorized, fact-only probe against the cyber-range target."""
                    return await self.observe(bound_name, target, action, parameters, force_refresh, authorization)

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


SENSITIVE_TOOL_NAMES = frozenset({
    "ad_asrep_roasting", "ad_kerberoasting", "ad_credential_validation",
    "ad_password_spray", "ad_bloodhound_collection", "bloodhound_recon",
})


def build_kali_registry(on_event: Callable[[str, dict[str, Any]], None] | None = None,
                        allowed_targets: list[str] | tuple[str, ...] | None = None) -> ToolRegistry:
    """Create the standard fixed-command Kali tool set for ReAct agents."""
    policy = TargetPolicy(allowed_targets)
    registry = ToolRegistry(target_policy=policy)
    specs = [
        KaliTool("check_port", "nmap", (), timeout=180.0, on_event=on_event, target_policy=policy),
        KaliTool("check_dns_resolution", "dig", ("+short",), on_event=on_event, target_policy=policy),
        KaliTool("http_health_check", "curl", ("--fail", "--silent", "--show-error", "--max-time", "15"), target_prefix="http://", on_event=on_event, target_policy=policy),
        KaliTool("ldap_bind", "ldapsearch", ("-x", "-H"), tail_args=("-s", "base", "-b", ""), target_prefix="ldap://", on_event=on_event, target_policy=policy),
        # smbclient's -L consumes the server argument immediately after it:
        # ``smbclient -L //TARGET -N``.  Appending TARGET after -N makes
        # smbclient parse the option stream as a service/password command.
        KaliTool("smb_negotiate", "smbclient", ("-L", "-N"), target_prefix="//",
                 target_index=1, on_event=on_event, target_policy=policy),
        # NetExec expects the target immediately after the protocol module;
        # build_argv selects a reviewed profile and inserts it accordingly.
        KaliTool("nxc_smb_recon", "nxc", (), target_index=1, on_event=on_event, target_policy=policy),
        KaliTool("nxc_ldap_recon", "nxc", (), target_index=1, on_event=on_event, target_policy=policy),
        KaliTool("impacket_rpc_recon", "impacket-rpcdump", fixed_args=(),
                 executable_candidates=("rpcdump.py", "rpcdump"),
                 on_event=on_event, target_policy=policy),
        KaliTool("inspect_routes", "ip", ("route",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_dns_config", "cat", ("/etc/resolv.conf",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_firewall", "nft", ("list", "ruleset"), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_time_sync", "timedatectl", ("show-timesync",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_os_version", "uname", ("-a",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_os_release", "cat", ("/etc/os-release",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_interfaces", "ip", ("-j", "addr"), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
        KaliTool("inspect_open_ports", "ss", ("-lntup",), target_arg=False, requires_target=False, on_event=on_event, target_policy=policy),
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
