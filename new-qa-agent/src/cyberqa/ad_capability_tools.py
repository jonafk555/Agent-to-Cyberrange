"""Concrete, allow-listed AD capability adapters.

The planner can select a capability by name, but it cannot provide an
arbitrary shell command. Each adapter below builds one reviewed argv shape.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import Evidence
from .ad_playbooks import CAPABILITY_PARAMETER_FIELDS


def summarize_output(text: str, max_lines: int = 24, max_chars: int = 3200) -> str:
    """Create a useful preview without replacing the stored full stream."""
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    useful_markers = (
        "[+]", "[-]", "[*]", "open", "domain", "user", "group", "spn", "signing",
        "status", "error", "failed", "success", "warning", "port", "service", "kerberos",
        "ldap", "smb", "dns", "anonymous", "access", "permission",
    )
    selected = [line for line in lines if any(marker in line.lower() for marker in useful_markers)] or lines
    if len(selected) > max_lines:
        selected = selected[: max_lines // 2] + [
            f"... {len(lines) - max_lines} more output lines stored in evidence ..."
        ] + selected[-max_lines // 2:]
    preview = "\n".join(selected)
    if len(preview) > max_chars:
        preview = preview[:max_chars] + "\n... preview truncated; full output remains in evidence ..."
    return preview


def output_facts(stdout: str, stderr: str) -> dict[str, Any]:
    """Record output size and useful previews while retaining full streams."""
    return {
        "stdout_lines": len([line for line in stdout.splitlines() if line.strip()]),
        "stderr_lines": len([line for line in stderr.splitlines() if line.strip()]),
        "stdout_summary": summarize_output(stdout),
        "stderr_summary": summarize_output(stderr),
    }


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


def _account_names(text: str) -> set[str]:
    patterns = (
        r"\bsAMAccountName\s*:\s*([A-Za-z0-9_.$-]+)",
        r"\b(?:username|user)\s*:\s*([A-Za-z0-9_.$-]+)",
    )
    values: set[str] = set()
    for pattern in patterns:
        values.update(match.group(1) for match in re.finditer(pattern, text, re.IGNORECASE))
    return {value for value in values if value.lower() not in {"user", "username"}}


def _asrep_hash_tokens(text: str) -> list[str]:
    """Extract complete AS-REP hash tokens without returning them as evidence."""

    return sorted(set(re.findall(r"\$krb5asrep\$\d+\$[^\s]+", text, re.IGNORECASE)))


def _credential_material_dir() -> Path:
    path = Path(os.getenv("CYBERQA_CREDENTIAL_MATERIAL_DIR", ".cyberqa/credential-material"))
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def _store_asrep_hashes(text: str) -> tuple[str | None, int]:
    """Store hashes in a mode-600 local artifact and expose only its reference."""

    hashes = _asrep_hash_tokens(text)
    if not hashes:
        return None, 0
    digest = hashlib.sha256("\n".join(hashes).encode()).hexdigest()[:20]
    path = _credential_material_dir() / f"asrep-{digest}.hash"
    if not path.exists():
        path.write_text("\n".join(hashes) + "\n", encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return str(path), len(hashes)


def _hash_cracking_paths(hash_file: str, wordlist: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(f"{hash_file}\0{wordlist}".encode()).hexdigest()[:20]
    root = _credential_material_dir()
    return root / f"cracked-{digest}.out", root / f"hashcat-{digest}.potfile"


def _account_from_hash(value: str) -> str | None:
    match = re.search(r"\$krb5asrep\$\d+\$([^:\s]+)", value, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).split("@", 1)[0]


def _capability_facts(capability: str, text: str, returncode: int | None,
                      domain: str, username: str) -> dict[str, Any]:
    """Project useful AD facts without storing ticket/password material."""
    facts: dict[str, Any] = {}
    if domain:
        facts["domain_name"] = domain
    if capability == "enumerate_domain_users":
        users = _account_names(text)
        if users:
            facts["users"] = sorted(users)
        spns = sorted(set(re.findall(r"\bservicePrincipalName\s*:\s*([^\s]+)", text, re.IGNORECASE)))
        if spns:
            facts["spns"] = spns
    elif capability == "asrep_roasting_assessment":
        matches = re.findall(r"\$krb5asrep\$\d+\$([^:\s]+)", text, re.IGNORECASE)
        candidates = sorted({item.split("@", 1)[0] for item in matches})
        facts["asrep_candidates"] = candidates
        facts["ticket_material_observed"] = bool(matches)
        facts["ticket_obtained_or_blocked"] = "ticket_material_observed" if matches else "none_observed"
    elif capability == "hash_cracking_assessment":
        facts["hash_cracking_attempted"] = True
    elif capability == "kerberoasting_assessment":
        matches = re.findall(r"\$krb5tgs\$\d+\$[^$]*\$[^$]*\$([^:\s]+)", text, re.IGNORECASE)
        facts["spn_accounts"] = sorted({item.split("@", 1)[0] for item in matches})
        facts["ticket_material_observed"] = bool(matches)
        facts["ticket_obtained_or_blocked"] = "ticket_material_observed" if matches else "none_observed"
    elif capability == "credential_validation":
        lowered = text.lower()
        failure_markers = (
            "logon_failure", "status_access_denied", "invalid credentials",
            "bad password", "authentication failure", "login failed",
        )
        success_markers = ("[+]", "pwn3d", "authenticated", "valid credentials")
        authentication_success = (
            returncode in (None, 0)
            and not any(marker in lowered for marker in failure_markers)
            and any(marker in lowered for marker in success_markers)
        )
        facts["authentication_success"] = authentication_success
        if username and authentication_success:
            facts["credentials_validated"] = [username]
    return facts


@dataclass
class ADCapabilityTool:
    name: str
    capability: str
    target_policy: Any
    timeout: float = 60.0
    on_event: Callable[[str, dict[str, Any]], None] | None = None

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        """Publish progress without allowing a logger to break execution."""
        if self.on_event:
            try:
                self.on_event(event, data)
            except Exception:
                pass

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
        try:
            argv = self._argv(target, parameters)
        except Exception as exc:
            self._emit("tool_result", {"tool": self.name, "exit_code": -1,
                                        "stderr": str(exc), "stdout": ""})
            raise
        safe_argv = self._safe_argv(argv)
        self._emit("tool_start", {"tool": self.name, "argv": safe_argv})
        temporary_users_file = None
        if self.capability == "asrep_roasting_assessment" and not parameters.get("users_file"):
            temporary_users_file = next((item for index, item in enumerate(argv)
                                         if index and argv[index - 1] == "-usersfile"), None)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except FileNotFoundError as exc:
            self._emit("tool_result", {"tool": self.name, "exit_code": -1,
                                        "stderr": str(exc), "stdout": ""})
            raise RuntimeError(f"{self.name} executable was not found: {argv[0]}") from exc
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            self._emit("tool_result", {"tool": self.name, "exit_code": -1,
                                        "stderr": f"{self.name} timed out after {self.timeout}s",
                                        "stdout": ""})
            raise TimeoutError(f"{self.name} timed out after {self.timeout}s")
        finally:
            if temporary_users_file:
                try:
                    os.unlink(temporary_users_file)
                except FileNotFoundError:
                    pass
        stdout_text = _redact(stdout.decode(errors="replace"))
        stderr_text = _redact(stderr.decode(errors="replace"))
        raw_output = f"{stdout.decode(errors='replace')}\n{stderr.decode(errors='replace')}"
        self._emit("tool_result", {"tool": self.name, "exit_code": process.returncode,
                                    "stdout": stdout_text, "stderr": stderr_text})
        facts = {
            "capability": self.capability,
            "argv": self._safe_argv(argv),
            "returncode": process.returncode,
            **output_facts(stdout_text, stderr_text),
            **_capability_facts(
                self.capability, raw_output, process.returncode,
                os.getenv("CYBERQA_AD_DOMAIN", ""), os.getenv("CYBERQA_AD_USERNAME", ""),
            ),
        }
        if self.capability == "asrep_roasting_assessment":
            hash_file, hash_count = _store_asrep_hashes(raw_output)
            if hash_file:
                facts.update({
                    "credential_material_ref": hash_file,
                    "asrep_hash_file": hash_file,
                    "asrep_hash_count": hash_count,
                })
        elif self.capability == "hash_cracking_assessment":
            facts.update(self._hash_cracking_facts(argv, process.returncode))
        return Evidence(
            source=f"ad-capability:{self.name}", action=action, target=target,
            exit_code=process.returncode, stdout=stdout_text, stderr=stderr_text,
            facts=facts,
        )

    def _argv(self, target: str, parameters: dict[str, Any]) -> list[str]:
        meaningful = {
            key: value for key, value in parameters.items()
            if value not in (None, "", [], False)
        }
        allowed_parameters = CAPABILITY_PARAMETER_FIELDS.get(self.capability, frozenset())
        unknown = set(meaningful) - allowed_parameters
        if unknown:
            raise ValueError(
                f"Unsupported parameters for {self.name}: {', '.join(sorted(unknown))}"
            )
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
        if self.capability == "hash_cracking_assessment":
            hash_file = parameters.get("hash_file") or os.getenv("CYBERQA_AD_HASH_FILE", "")
            wordlist = parameters.get("wordlist") or os.getenv("CYBERQA_AD_WORDLIST", "")
            if not hash_file or not wordlist:
                raise RuntimeError(
                    "Hash cracking requires an AS-REP hash artifact and CYBERQA_AD_WORDLIST"
                )
            hash_path = Path(hash_file).expanduser()
            wordlist_path = Path(wordlist).expanduser()
            if not hash_path.is_file():
                raise RuntimeError(f"AS-REP hash artifact does not exist: {hash_path}")
            if not wordlist_path.is_file():
                raise RuntimeError(f"Cracking wordlist does not exist: {wordlist_path}")
            if hash_path.stat().st_size > 16 * 1024 * 1024:
                raise RuntimeError("AS-REP hash artifact exceeds the 16 MiB limit")
            if wordlist_path.stat().st_size > 4 * 1024 * 1024 * 1024:
                raise RuntimeError("Cracking wordlist exceeds the 4 GiB limit")
            output_path, potfile_path = _hash_cracking_paths(str(hash_path), str(wordlist_path))
            return [
                _first_available(("hashcat",)), "-m", "18200", "-a", "0",
                str(hash_path), str(wordlist_path),
                "--outfile", str(output_path), "--outfile-format", "2",
                "--potfile-path", str(potfile_path), "--quiet",
            ]
        if self.capability == "bloodhound_collection":
            collection = os.getenv("CYBERQA_AD_COLLECTION", "DCOnly")
            if not domain or not username or not password:
                raise RuntimeError("BloodHound collection requires AD domain credentials")
            return [_first_available(("bloodhound-python",)), "-c", collection, "-d", domain,
                    "-u", username, "-p", password, "-ns", target]
        raise ValueError(f"Unsupported AD capability: {self.capability}")

    def _hash_cracking_facts(self, argv: list[str], returncode: int | None) -> dict[str, Any]:
        """Promote one cracked account to process-local credentials, never evidence."""

        try:
            output_path = Path(argv[argv.index("--outfile") + 1])
        except (ValueError, IndexError):
            return {"hash_cracking_attempted": True, "hash_cracked": False, "crack_status": "invalid_output"}
        rows: list[tuple[str, str]] = []
        try:
            if output_path.is_file():
                for line in output_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if ":" not in line:
                        continue
                    hash_value, secret = line.rsplit(":", 1)
                    account = _account_from_hash(hash_value)
                    if account and secret:
                        rows.append((account, secret))
        finally:
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
        users = sorted({account for account, _ in rows})
        facts: dict[str, Any] = {
            "hash_cracking_attempted": True,
            "hash_cracked": bool(rows),
            "cracked_users": users,
            "crack_status": "cracked" if rows else "not_found",
        }
        if rows:
            # The secret stays process-local and is never copied into facts,
            # stdout, reports, prompts, or the observation cache.
            username, password = rows[0]
            os.environ["CYBERQA_AD_USERNAME"] = username
            os.environ["CYBERQA_AD_PASSWORD"] = password
            facts["credential_source"] = "asrep_hash_cracking"
        return facts

    @staticmethod
    def _safe_argv(argv: list[str]) -> list[str]:
        secrets = {value for name in ("CYBERQA_AD_PASSWORD", "AD_PASSWORD")
                   if (value := os.getenv(name))}
        result: list[str] = []
        redact_next = False
        secret_flags = {"-p", "-P", "-w", "-W", "--password", "--pass", "--secret", "/password"}
        for item in argv:
            if redact_next:
                result.append("***REDACTED***")
                redact_next = False
                continue
            if item in secret_flags:
                result.append(item)
                redact_next = True
                continue
            result.append(next(
                (item.replace(secret, "***REDACTED***") for secret in secrets if secret in item),
                item,
            ))
        return result


def build_ad_capability_tools(
    target_policy: Any,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[ADCapabilityTool]:
    specs = (
        ("ad_domain_users", "enumerate_domain_users"),
        ("ad_asrep_roasting", "asrep_roasting_assessment"),
        ("ad_hash_cracking", "hash_cracking_assessment"),
        ("ad_kerberoasting", "kerberoasting_assessment"),
        ("ad_credential_validation", "credential_validation"),
        ("ad_password_spray", "controlled_password_spray_assessment"),
        ("ad_bloodhound_collection", "bloodhound_collection"),
    )
    return [ADCapabilityTool(name, capability, target_policy, on_event=on_event)
            for name, capability in specs]
