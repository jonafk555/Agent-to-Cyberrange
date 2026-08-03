"""Concrete, allow-listed AD capability adapters.

The planner can select a capability by name, but it cannot provide an
arbitrary shell command. Each adapter below builds one reviewed argv shape.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

from .models import Evidence


def _first_available(names: tuple[str, ...]) -> str:
    for name in names:
        if shutil.which(name):
            return name
    return names[0]


def _redact(text: str) -> str:
    if os.getenv("CYBERQA_KEEP_CREDENTIAL_MATERIAL") == "1":
        return text
    lines = []
    for line in text.splitlines():
        if "$krb5asrep$" in line or "$krb5tgs$" in line or "password" in line.lower():
            lines.append("[credential material redacted]")
        else:
            lines.append(line)
    return "\n".join(lines)


@dataclass
class ADCapabilityTool:
    name: str
    capability: str
    target_policy: Any
    timeout: float = 60.0

    def command_identity(self, target: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Stable effective argv identity without persisting credentials or temp paths."""
        password = os.getenv("CYBERQA_AD_PASSWORD", "")
        return {
            "capability": self.capability,
            "target": target,
            "domain": os.getenv("CYBERQA_AD_DOMAIN", ""),
            "username": os.getenv("CYBERQA_AD_USERNAME", ""),
            "base_dn": os.getenv("CYBERQA_AD_BASE_DN", ""),
            "credential_fingerprint": hashlib.sha256(password.encode()).hexdigest()[:12]
            if password else "",
            "parameters": parameters,
        }

    async def observe(self, target: str, action: str, **kwargs: Any) -> Evidence:
        if not self.target_policy.allows(target):
            raise PermissionError(f"Target is not in CYBERQA_ALLOWED_TARGETS: {target}")
        parameters = kwargs or {}
        argv = self._argv(target, parameters)
        temporary_users_file = None
        if self.capability == "asrep_roasting_assessment" and not parameters.get("users_file"):
            temporary_users_file = next((item for index, item in enumerate(argv)
                                         if index and argv[index - 1] == "-usersfile"), None)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"{self.name} timed out after {self.timeout}s")
        finally:
            if temporary_users_file:
                try:
                    os.unlink(temporary_users_file)
                except FileNotFoundError:
                    pass
        stdout_text = _redact(stdout.decode(errors="replace")[-12000:])
        stderr_text = _redact(stderr.decode(errors="replace")[-12000:])
        return Evidence(
            source=f"ad-capability:{self.name}", action=action, target=target,
            exit_code=process.returncode, stdout=stdout_text, stderr=stderr_text,
            facts={"capability": self.capability, "argv": self._safe_argv(argv),
                   "returncode": process.returncode},
        )

    def _argv(self, target: str, parameters: dict[str, Any]) -> list[str]:
        domain = os.getenv("CYBERQA_AD_DOMAIN", "")
        username = os.getenv("CYBERQA_AD_USERNAME", "")
        password = os.getenv("CYBERQA_AD_PASSWORD", "")
        base_dn = os.getenv("CYBERQA_AD_BASE_DN", "")
        if self.capability == "enumerate_domain_users":
            if not base_dn:
                raise RuntimeError("Set CYBERQA_AD_BASE_DN for LDAP user enumeration")
            argv = [_first_available(("ldapsearch",)), "-x", "-LLL", "-H", f"ldap://{target}",
                    "-b", base_dn, "(objectClass=user)", "sAMAccountName",
                    "servicePrincipalName", "userAccountControl"]
            if username and password:
                argv[2:2] = ["-D", f"{domain}\\{username}", "-w", password]
            return argv
        if self.capability == "asrep_roasting_assessment":
            users = parameters.get("users", [])
            users_file = parameters.get("users_file") or os.getenv("CYBERQA_AD_USERS_FILE", "")
            if not domain or (not users and not users_file):
                raise RuntimeError(
                    "AS-REP assessment requires CYBERQA_AD_DOMAIN and a candidate username source; "
                    "set CYBERQA_AD_USERS_FILE or provide tool_parameters.users"
                )
            if users_file:
                users_path = os.path.abspath(os.path.expanduser(str(users_file)))
                if not os.path.isfile(users_path):
                    raise RuntimeError(f"AS-REP users_file does not exist: {users_path}")
                if os.path.getsize(users_path) > 1024 * 1024:
                    raise RuntimeError("AS-REP users_file exceeds the 1 MiB limit")
                with open(users_path, "rb") as handle:
                    if b"\x00" in handle.read(1024 * 1024):
                        raise RuntimeError("AS-REP users_file is not a text file")
                users_file = users_path
            else:
                with tempfile.NamedTemporaryFile("w", prefix="cyberqa-users-", delete=False) as handle:
                    handle.write("\n".join(str(user) for user in users) + "\n")
                    users_file = handle.name
            return [_first_available(("impacket-GetNPUsers", "GetNPUsers.py")),
                    f"{domain}/", "-dc-ip", target, "-usersfile", users_file,
                    "-no-pass", "-format", "hashcat"]
        if self.capability == "kerberoasting_assessment":
            if not domain or not username or not password:
                raise RuntimeError("Kerberoasting assessment requires AD domain credentials")
            return [_first_available(("impacket-GetUserSPNs", "GetUserSPNs.py")),
                    f"{domain}/{username}:{password}", "-dc-ip", target, "-request"]
        if self.capability == "credential_validation":
            if not username or not password:
                raise RuntimeError("Credential validation requires CYBERQA_AD_USERNAME/PASSWORD")
            return [_first_available(("nxc",)), "smb", target, "-u", username, "-p", password]
        if self.capability == "controlled_password_spray_assessment":
            raise PermissionError(
                "Password spraying is blocked until approval, lockout policy, bounded users, "
                "and an approved test password are supplied"
            )
        if self.capability == "bloodhound_collection":
            collection = os.getenv("CYBERQA_AD_COLLECTION", "DCOnly")
            if not domain or not username or not password:
                raise RuntimeError("BloodHound collection requires AD domain credentials")
            return [_first_available(("bloodhound-python",)), "-c", collection, "-d", domain,
                    "-u", username, "-p", password, "-ns", target]
        raise ValueError(f"Unsupported AD capability: {self.capability}")

    @staticmethod
    def _safe_argv(argv: list[str]) -> list[str]:
        secrets = {value for name in ("CYBERQA_AD_PASSWORD", "AD_PASSWORD")
                   if (value := os.getenv(name))}
        return ["***REDACTED***" if item in secrets else item for item in argv]


def build_ad_capability_tools(target_policy: Any) -> list[ADCapabilityTool]:
    specs = (
        ("ad_domain_users", "enumerate_domain_users"),
        ("ad_asrep_roasting", "asrep_roasting_assessment"),
        ("ad_kerberoasting", "kerberoasting_assessment"),
        ("ad_credential_validation", "credential_validation"),
        ("ad_password_spray", "controlled_password_spray_assessment"),
        ("ad_bloodhound_collection", "bloodhound_collection"),
    )
    return [ADCapabilityTool(name, capability, target_policy) for name, capability in specs]
