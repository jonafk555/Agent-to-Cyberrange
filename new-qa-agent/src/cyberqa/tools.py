from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import shlex
import shutil
import socket
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from langchain_core.tools import BaseTool, tool

from .ad_capability_tools import build_ad_capability_tools, output_facts, summarize_output
from .memory import ObservationStore
from .models import Evidence


# These values describe the runner, not an authorized cyber-range host.  They
# must never become a remote reconnaissance target or be passed to nmap/nxc.
LOCAL_EXECUTION_TARGET = "environment"
LOCAL_TARGET_NAMES = frozenset({
    "environment", "local-kali", "local_kali", "local-runtime", "local_runtime",
    "local-runner", "local_runner", "runner", "kali",
})


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
            "CYBERQA_ALLOWED_TARGETS", ""
        ).split(",") if item.strip()]
        self.entries: set[str] = set(configured)
        self.local_hosts: set[str] = {"localhost", "127.0.0.1", "::1"}
        self.local_hosts.update(item.strip() for item in os.getenv(
            "CYBERQA_LOCAL_IPS", ""
        ).split(",") if item.strip())
        # Resolve only the local runner's own hostname. These addresses are
        # excluded when a supplied CIDR also contains the Kali interface.
        try:
            self.local_hosts.update(
                info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None)
                if info[4] and info[4][0]
            )
        except OSError:
            pass

    def mark_local(self, target: str) -> None:
        host = _target_host(target)
        if host:
            self.local_hosts.add(host)

    def is_local(self, target: str) -> bool:
        host = _target_host(target)
        if host.lower() in LOCAL_TARGET_NAMES:
            return True
        if host in self.local_hosts:
            return True
        if "/" in host:
            try:
                network = ipaddress.ip_network(host, strict=False)
                if network.is_loopback or network.is_unspecified:
                    return True
                if network.num_addresses == 1:
                    return self.is_local(str(network.network_address))
                return False
            except ValueError:
                return False
        try:
            address = ipaddress.ip_address(host)
            return address.is_loopback or address.is_unspecified
        except ValueError:
            return host.lower() in {"localhost", "localhost.localdomain"}

    def allows(self, target: str) -> bool:
        host = _target_host(target)
        if self.is_local(target):
            return False
        if target in self.entries or host in self.entries:
            return True
        try:
            address = ipaddress.ip_address(host)
            return any(address in ipaddress.ip_network(entry, strict=False)
                       for entry in self.entries if "/" in entry)
        except ValueError:
            return False

    def add(self, target: str) -> bool:
        # Discovery may expand an authorized CIDR, but it may not expand the
        # policy to an unrelated address or to the scanner itself.
        if target and not self.is_local(target) and self.allows(target) and target not in self.entries:
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
            # Local inspection is execution context, never a scan target.
            target = LOCAL_EXECUTION_TARGET
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
        stdout_text = stdout.decode(errors="replace")
        stderr_text = stderr.decode(errors="replace")
        evidence = Evidence(
            source=f"kali:{self.name}", action=action, target=target,
            exit_code=process.returncode, stdout=stdout_text, stderr=stderr_text,
            facts={"argv": [shlex.quote(x) for x in safe_argv], "returncode": process.returncode,
                   **output_facts(stdout_text, stderr_text)},
        )
        if self.executable == "nmap":
            evidence.facts["discovered_targets"] = sorted(_discover_ip_addresses(evidence.stdout))
            evidence.facts["open_ports"] = _discover_open_ports(evidence.stdout)
        if self.name == "nxc_ldap_recon" or self.executable == "ldapsearch":
            users = _discover_user_names(evidence.stdout)
            if users:
                evidence.facts["users"] = sorted(users)
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
        policy = self.target_policy or TargetPolicy()
        if self.name in {"check_port", "nxc_smb_recon", "nxc_ldap_recon"}:
            supported = {"profile"}
            if self.name.startswith("nxc_"):
                supported.add("allow_anonymous_nxc")
            supported.add("argv")
        elif self.name == "check_dns_resolution":
            supported = {"name"}
        elif self.name in {"ldap_bind", "smb_negotiate"}:
            # These are intentionally narrow correction surfaces. The model
            # may change protocol/profile flags, but it never supplies the
            # executable, URL, SMB service path, or an arbitrary shell string.
            supported = {"profile", "argv"}
        else:
            supported = set()
        unknown = set(parameters) - supported
        if unknown:
            raise ValueError(
                f"Unsupported parameters for {self.name}: {', '.join(sorted(unknown))}"
            )
        executable = self._resolve_executable()
        fixed_args = self.fixed_args
        if self.name == "check_dns_resolution":
            query = str(parameters.get("name") or target)
            return [executable, "+short", query, *self.tail_args]
        if self.name == "ldap_bind":
            profiles = {
                "rootdse": ("-x", "-s", "base", "-b", ""),
                "subtree": ("-x", "-s", "sub", "-b", ""),
                "starttls_rootdse": ("-x", "-ZZ", "-s", "base", "-b", ""),
                "gssapi_rootdse": ("-Y", "GSSAPI", "-s", "base", "-b", ""),
            }
            profile = str(parameters.get("profile", "rootdse"))
            if profile not in profiles:
                raise ValueError(f"Unsupported ldap_bind profile: {profile}")
            custom_argv = parameters.get("argv") or []
            selected = tuple(_validated_option_argv(
                custom_argv,
                flags={"-x", "-LLL", "-v", "-ZZ", "-Z"},
                value_flags={"-b", "-d", "-E", "-o", "-s", "-Y"},
                label="ldap",
            )) if custom_argv else profiles[profile]
            # Always force the target through -H. A model can select a base,
            # scope, StartTLS, or SASL mode, but cannot smuggle another host.
            if "-H" in selected:
                raise ValueError("ldap argv cannot override the reviewed target")
            if "-x" not in selected and "-Y" not in selected:
                selected = ("-x", *selected)
            return [executable, "-H", f"ldap://{target}", *selected]
        if self.name == "smb_negotiate":
            profiles = {
                "anonymous": ("-L", "-N"),
                "smb2": ("-L", "-N", "-m", "SMB2"),
                "smb3": ("-L", "-N", "-m", "SMB3"),
                "port445": ("-L", "-N", "-p", "445"),
            }
            profile = str(parameters.get("profile", "anonymous"))
            if profile not in profiles:
                raise ValueError(f"Unsupported smb_negotiate profile: {profile}")
            custom_argv = parameters.get("argv") or []
            selected = tuple(_validated_option_argv(
                custom_argv,
                flags={"-L", "-N", "-g"},
                value_flags={"-m", "-p", "-W"},
                label="smb",
            )) if custom_argv else profiles[profile]
            if "-L" not in selected:
                selected = ("-L", *selected)
            # smbclient requires the server immediately after -L. Insert the
            # reviewed target there regardless of the model's flag ordering.
            list_args = list(selected)
            list_args.insert(list_args.index("-L") + 1, f"//{target}")
            return [executable, *list_args]
        if self.name == "check_port":
            profiles = {
                # Network reconnaissance starts with host discovery/fast
                # discovery. Full service detection is a later adaptive step.
                "host_discovery": ("-sn",),
                "fast": ("-F",),
                "default": ("-sC", "-sV"),
                "top100": ("-Pn", "-T3", "--top-ports", "100"),
                "top1000": ("-Pn", "-T3", "--top-ports", "1000"),
                "ad_tcp": ("-Pn", "-T3", "-sV", "-p",
                           "53,80,88,135,139,389,443,445,464,593,636,3268,3269,3389,5985,5986,9389"),
            }
            profile = str(parameters.get("profile", "default"))
            if profile not in profiles:
                raise ValueError(f"Unsupported check_port profile: {profile}")
            custom_argv = parameters.get("argv") or []
            if "/" in target and profile == "default":
                # Service/version scripts against a whole CIDR are not the
                # first step. Keep the tool adaptive and safe even if a model
                # accidentally requests the default profile too early.
                profile = "host_discovery"
            if "/" in target and custom_argv and any(flag in custom_argv for flag in ("-sC", "-sV")):
                raise ValueError("Run nmap -sn or -F against a CIDR first; use -sC/-sV on a discovered host")
            fixed_args = (
                tuple(_validated_option_argv(
                    custom_argv,
                    flags={"-6", "-F", "-n", "-Pn", "-sC", "-sS", "-sT", "-sU", "-sV",
                           "-T0", "-T1", "-T2", "-T3", "-T4", "-T5",
                           "--open", "--reason", "--traceroute", "--version-light"},
                    value_flags={"-T", "-p", "--host-timeout", "--max-retries", "--max-rate",
                                 "--min-rate", "--top-ports", "--version-intensity"},
                    label="nmap",
                )) if custom_argv else profiles[profile]
            )
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
            custom_argv = parameters.get("argv") or []
            selected_args = tuple(_validated_option_argv(
                custom_argv,
                flags={"--computers", "--continue-on-success", "--groups", "--local-auth",
                       "--loggedon-users", "--pass-pol", "--rid-brute", "--sessions",
                       "--shares", "--users"},
                value_flags={"-t", "--threads", "--timeout"},
                label="nxc",
            )) if custom_argv else profiles[profile]
            credentials = []
            username = os.getenv("CYBERQA_AD_USERNAME", "")
            password = os.getenv("CYBERQA_AD_PASSWORD", "")
            if username and password:
                credentials = ["-u", username, "-p", password]
            fixed_args = (module, *selected_args, *credentials)
            self_target_index = 1
            argv = [executable, *fixed_args]
            argv.insert(self_target_index + 1, f"{self.target_prefix}{target}")
            argv.extend(self.tail_args)
            return argv
        if self.name == "check_port" and "/" in target:
            # The supplied CIDR may contain the Kali runner's own interface.
            # Exclude those addresses in the Nmap command itself; merely
            # omitting them from the discovered-target list would still scan
            # the local machine.
            try:
                network = ipaddress.ip_network(_target_host(target), strict=False)
                excluded = []
                for host in policy.local_hosts:
                    if not host or "/" in host:
                        continue
                    try:
                        address = ipaddress.ip_address(_target_host(host))
                    except ValueError:
                        continue
                    if address in network:
                        excluded.append(str(address))
                excluded.sort()
            except ValueError:
                excluded = []
            if excluded:
                fixed_args = (*fixed_args, "--exclude", ",".join(excluded))
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
        effective_target = LOCAL_EXECUTION_TARGET if not self.requires_target else target
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
        return [
            next((item.replace(secret, "***REDACTED***") for secret in secrets if secret in item), item)
            for item in argv
        ]


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


def _target_host(target: str) -> str:
    """Normalize host:port and URL targets for allowlist checks."""
    value = str(target).strip()
    if "://" in value:
        return urlsplit(value).hostname or value
    if value.count(":") == 1:
        host, port = value.rsplit(":", 1)
        if port.isdigit():
            return host
    return value


def is_local_target(target: str) -> bool:
    """Return true for loopback/unspecified/local-runner target values."""
    return TargetPolicy().is_local(target)


def _validated_option_argv(
    values: Any,
    flags: set[str],
    value_flags: set[str],
    label: str,
) -> list[str]:
    """Validate model-selected option fragments without accepting a command."""
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ValueError(f"{label} argv must be a list of strings")
    result: list[str] = []
    expecting: str | None = None
    for item in values:
        if not item or any(char in item for char in "\x00\r\n;|&><`$"):
            raise ValueError(f"Invalid {label} argv token")
        if expecting:
            if item.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.,:/+%-]+", item):
                raise ValueError(f"Invalid value for {expecting} in {label} argv")
            result.append(item)
            expecting = None
        elif item in value_flags:
            result.append(item)
            expecting = item
        elif item in flags:
            result.append(item)
        else:
            raise ValueError(f"Unsupported {label} argv option: {item}")
    if expecting:
        raise ValueError(f"Missing value for {expecting} in {label} argv")
    return result


def _discover_open_ports(output: str) -> list[dict[str, str]]:
    """Extract the small, stable service inventory needed for QA planning."""
    ports: list[dict[str, str]] = []
    for line in output.splitlines():
        match = re.match(r"\s*(\d+)/(tcp|udp)\s+open\s+(\S+)", line)
        if match:
            ports.append({"port": match.group(1), "protocol": match.group(2),
                          "service": match.group(3)})
    return ports


def _discover_user_names(output: str) -> set[str]:
    """Extract conservative username facts from anonymous LDAP/NXC output."""
    users: set[str] = set()
    patterns = (
        r"\bsAMAccountName\s*:\s*([A-Za-z0-9_.$-]+)",
        r"\b(?:username|user)\s*:\s*([A-Za-z0-9_.$-]+)",
    )
    for pattern in patterns:
        users.update(match.group(1) for match in re.finditer(pattern, output, re.IGNORECASE))
    return {user for user in users if len(user) <= 128 and user.lower() not in {"user", "username"}}


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

    @staticmethod
    def _execution_target(adapter: FactTool, target: str) -> str:
        """Return a target label suitable for evidence and cache events.

        Adapters without a target (the ``inspect_*`` family) run on the
        runner.  Keeping their logical target separate prevents a CIDR or a
        discovered host from being polluted by local runtime observations.
        """
        return target if getattr(adapter, "requires_target", True) else LOCAL_EXECUTION_TARGET

    async def observe(self, name: str, target: str, action: str,
                      parameters: dict[str, Any] | None = None,
                      force_refresh: bool = False,
                      authorization: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute one probe through the same durable cache used by ToolNode."""
        parameters = parameters or {}
        adapter = self.get(name)
        execution_target = self._execution_target(adapter, target)
        # An approved grant freezes the target and concrete tool set for this
        # dispatch. Apply the boundary to every tool in the approved branch,
        # not only credential-material adapters.
        if authorization:
            approved_target = authorization.get("target")
            allowed = authorization.get("allowed_tools", [])
            if approved_target and target != approved_target:
                return {
                    "ok": False, "tool": name,
                    "error": f"Approved action is scoped to target {approved_target}, not {target}",
                    "error_kind": "approval_scope", "needs_human": True,
                }
            if name not in allowed:
                return {
                    "ok": False, "tool": name,
                    "error": "Tool is not part of the approved capability",
                    "error_kind": "approval_scope", "needs_human": True,
                }
        if name in SENSITIVE_TOOL_NAMES:
            allowed = (authorization or {}).get("allowed_tools", [])
            expected = (authorization or {}).get("tool_parameters", {})
            # LangChain may omit Pydantic default fields from a tool call;
            # compare the meaningful reviewed values, not serialization shape.
            normalize = lambda value: {
                key: item for key, item in (value or {}).items()
                if item not in (None, "", [], False)
            }
            if name not in allowed or normalize(parameters) != normalize(expected):
                return {
                    "ok": False, "tool": name,
                    "error": "Sensitive tool requires an exact approved capability and parameters",
                    "error_kind": "approval_required", "needs_human": True,
                }
        try:
            signature = self._signature(adapter, name, execution_target, action, parameters)
        except Exception as exc:
            # Invalid reviewed parameters are still durable diagnostic
            # evidence; they must not escape ToolNode as an uncaught graph
            # exception or be retried forever.
            signature = self.observations.signature(
                name, execution_target, "invalid-command", {"parameters": parameters, "error": str(exc)}
            )
            recoverable = _recoverable_tool_error(name, "invalid_arguments", str(exc))
            result = {"ok": False, "tool": name, "error": str(exc),
                      "error_kind": "invalid_arguments", "recoverable": recoverable,
                      "retryable": recoverable, "needs_human": not recoverable}
            self.observations.put(signature, result)
            return {**result, "signature": signature, "cached": False}
        cached = None if force_refresh else self.observations.get(signature)
        if cached is not None:
            on_event = getattr(adapter, "on_event", None)
            if on_event:
                on_event("tool_cached", {"tool": name, "target": execution_target,
                                          "signature": signature})
            return self._cache_hit(cached, signature, action, name, execution_target)
        lock = self._observation_locks.setdefault(signature, asyncio.Lock())
        async with lock:
            cached = None if force_refresh else self.observations.get(signature)
            if cached is not None:
                on_event = getattr(adapter, "on_event", None)
                if on_event:
                    on_event("tool_cached", {"tool": name, "target": execution_target,
                                              "signature": signature})
                return self._cache_hit(cached, signature, action, name, execution_target)
            try:
                evidence = await adapter.observe(execution_target, action, **parameters)
                evidence_data = evidence.model_dump(mode="json")
                if evidence.exit_code not in (None, 0):
                    error_kind = _tool_error_kind(evidence)
                    recoverable = _recoverable_tool_error(
                        name, error_kind, f"{evidence.stderr}\n{evidence.stdout}"
                    )
                    # Keep the classification beside the raw command facts.
                    # Specialists later receive Evidence rather than the
                    # transient registry result, so without this metadata a
                    # future Supervisor turn could not distinguish a
                    # recoverable LDAP/SMB/NXC failure from an exhausted one.
                    evidence_data["facts"] = {
                        **(evidence_data.get("facts") or {}),
                        "error_kind": error_kind,
                        "recoverable": recoverable,
                    }
                    result = {
                        "ok": False, "tool": name,
                        "error": _tool_error_message(evidence),
                        "error_kind": error_kind,
                        "recoverable": recoverable,
                        "retryable": recoverable,
                        "needs_human": not recoverable, "evidence": evidence_data,
                    }
                else:
                    result = {"ok": True, "tool": name, "evidence": evidence_data}
            except Exception as exc:
                error_kind = "invalid_target" if isinstance(exc, PermissionError) else type(exc).__name__
                recoverable = _recoverable_tool_error(name, error_kind, str(exc))
                result = {"ok": False, "tool": name, "error": str(exc),
                          "error_kind": error_kind, "recoverable": recoverable,
                          "retryable": recoverable, "needs_human": not recoverable}
            self.observations.put(signature, result)
            return {**result, "signature": signature, "cached": False}

    @staticmethod
    def _cache_hit(cached: dict[str, Any], signature: str, action: str,
                   tool_name: str | None = None, target: str | None = None) -> dict[str, Any]:
        result = {**cached, "signature": signature, "cached": True}
        if (not result.get("ok", True)
                and ("recoverable" not in result or result.get("error_kind") == "PermissionError")):
            cached_kind = str(result.get("error_kind", "nonzero_exit"))
            if cached_kind == "PermissionError" and "target is not" in str(result.get("error", "")).lower():
                cached_kind = "invalid_target"
            recoverable = _recoverable_tool_error(
                tool_name or str(result.get("tool", "")),
                cached_kind,
                str(result.get("error", "")),
            )
            result.update({"error_kind": cached_kind, "recoverable": recoverable, "retryable": recoverable,
                           "needs_human": not recoverable})
        if isinstance(cached.get("evidence"), dict):
            evidence = dict(cached["evidence"])
            facts = dict(evidence.get("facts") or {})
            if not result.get("ok", True):
                facts.setdefault("error_kind", result.get("error_kind", "nonzero_exit"))
                facts.setdefault("recoverable", result.get("recoverable", False))
            facts["cache_hit"] = True
            facts["original_action"] = evidence.get("action")
            evidence["action"] = action
            if target is not None:
                evidence["target"] = target
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
        # An explicit empty list means "no tools". Do not turn an
        # analysis-only specialist back into an unrestricted tool user.
        selected = tuple(self.tools) if names is None else names
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
                    # A model-visible tool has no refresh switch: the
                    # durable observation ledger is authoritative. A fresh
                    # probe remains an explicit operator/API concern through
                    # ToolRegistry.observe(force_refresh=True).
                    return await self.observe(bound_name, target, action, parameters, False, authorization)

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


def _recoverable_tool_error(tool_name: str, error_kind: str, detail: str = "") -> bool:
    """Classify failures that a read-only ReAct specialist can repair.

    The command result remains evidence in every case. Recoverable failures
    are returned to the model as normal tool results; non-recoverable failures
    retain the Human-in-the-loop boundary.
    """
    name = str(tool_name).lower()
    read_only = name in {
        "ldap_bind", "smb_negotiate", "nxc_smb_recon", "nxc_ldap_recon", "check_port",
        "check_dns_resolution", "http_health_check", "impacket_rpc_recon",
    } or name.startswith("inspect_")
    if not read_only:
        return False
    if error_kind in {"approval_scope", "approval_required", "PermissionError", "RuntimeError",
                      "FileNotFoundError", "invalid_arguments"}:
        # Invalid reviewed arguments are exactly what the ReAct model can
        # correct. Runtime/executable/approval failures need operator context.
        return error_kind == "invalid_arguments"
    if error_kind == "invalid_target":
        # The planner can select another authorized host; only an exhausted
        # target set should reach the human boundary.
        return True
    if error_kind in {"connectivity", "timeout", "permission_denied", "nonzero_exit"}:
        return True
    lowered = detail.lower()
    return any(marker in lowered for marker in (
        "can't contact ldap", "operations error", "stronger authentication",
        "not enough", "logon failure", "connection refused", "timed out",
        "no route to host", "unrecognized argument", "usage:",
    ))


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
    for capability_tool in build_ad_capability_tools(policy, on_event=on_event):
        registry.register(capability_tool)
    return registry
