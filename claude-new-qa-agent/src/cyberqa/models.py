from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    SUPERVISOR = "supervisor"
    VALIDATION = "validation"
    TESTING = "testing"
    DEBUGGING = "debugging"
    JUDGE = "judge"
    REPORTING = "reporting"


class ADRisk(str, Enum):
    READ_ONLY = "read_only"
    CREDENTIAL_MATERIAL = "credential_material"
    AUTHENTICATION_TEST = "authentication_test"
    ACCOUNT_LOCKOUT = "account_lockout"
    CHANGE = "change"


class ServiceProtocol(str, Enum):
    DNS = "dns"; LDAP = "ldap"; LDAPS = "ldaps"; KERBEROS = "kerberos"; SMB = "smb"
    WINRM = "winrm"; ADCS = "adcs"; IIS = "iis"; MSSQL = "mssql"; SSH = "ssh"
    HTTP = "http"; HTTPS = "https"; SAMBA = "samba"; POSTGRES = "postgres"; MYSQL = "mysql"
    DOCKER = "docker"


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    action: str
    target: str
    observed_at: datetime = Field(default_factory=utcnow)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    facts: dict[str, Any] = Field(default_factory=dict)
    redacted: bool = True


class Service(BaseModel):
    host: str
    protocol: ServiceProtocol
    port: int
    running: bool | None = None
    reachable: bool | None = None
    functional: bool | None = None
    banner: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class Host(BaseModel):
    name: str
    platform: Literal["windows", "linux", "unknown"] = "unknown"
    address: str | None = None
    services: list[Service] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)


class AttackPath(BaseModel):
    name: str
    expected_steps: list[str]
    observed_steps: list[str] = Field(default_factory=list)
    result: Literal[
        "not_run", "passed", "failed", "degraded", "blocked",
        "capability_verified", "out_of_scope", "simulated", "inconclusive",
    ] = "not_run"
    evidence_ids: list[str] = Field(default_factory=list)
    alternatives: list[list[str]] = Field(default_factory=list)


class Hypothesis(BaseModel):
    statement: str
    likelihood: float = Field(ge=0, le=1)
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    verified: bool | None = None


class ApprovalRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    action: str
    target: str
    reason: str
    impact: str
    rollback: str
    evidence_ids: list[str] = Field(default_factory=list)
    decision_fingerprint: str | None = None
    status: Literal["pending", "approved", "rejected"] = "pending"


class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    run_id: str
    emitted_by: Role
    timestamp: datetime = Field(default_factory=utcnow)
    target: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolParameters(BaseModel):
    """Closed schema for model-selected, reviewed tool variations."""
    model_config = ConfigDict(extra="forbid")

    profile: str | None = None
    name: str | None = None
    # Optional reviewed argv fragments for Nmap/NXC. Adapters validate the
    # allowed flags and always inject the authorized target/module themselves.
    argv: list[str] = Field(default_factory=list)
    users_file: str | None = None
    users: list[str] = Field(default_factory=list)
    allow_anonymous_nxc: bool = False


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    next_agent: Role | Literal["approval", "end"]
    objective: str
    action: str
    target: str
    justification: str
    expected_information_gain: float = Field(default=0.0, ge=0, le=1)
    approval_required: bool = False
    capability: str | None = None
    plan_id: str | None = None
    prerequisites: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    risk: ADRisk = ADRisk.READ_ONLY
    next_options: list[str] = Field(default_factory=list)
    tool_parameters: ToolParameters = Field(default_factory=ToolParameters)


class CapabilitySpec(BaseModel):
    name: str
    purpose: str
    prerequisites: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    risk: ADRisk = ADRisk.READ_ONLY
    requires_approval: bool = False
    notes: str = ""


class ADKnowledge(BaseModel):
    domain: str | None = None
    domains: list[str] = Field(default_factory=list)
    forests: list[str] = Field(default_factory=list)
    target_domains: dict[str, str] = Field(default_factory=dict)
    cross_forest_targets: list[str] = Field(default_factory=list)
    hosts: list[str] = Field(default_factory=list)
    users: list[str] = Field(default_factory=list)
    spns: list[str] = Field(default_factory=list)
    asrep_candidates: list[str] = Field(default_factory=list)
    credentials_validated: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    acl_edges: list[str] = Field(default_factory=list)
    delegation: list[str] = Field(default_factory=list)
    adcs_findings: list[str] = Field(default_factory=list)
    trusts: list[str] = Field(default_factory=list)
    coverage: dict[str, list[str]] = Field(default_factory=dict)


class WeaknessVerdict(BaseModel):
    """Per-weakness / per-expectation verdict grounded in collected evidence.

    The judge emits one of these for every expected capability or baseline
    fact it was asked to confirm, so a scorecard can be audited item-by-item
    instead of collapsing into a single opaque number.
    """
    id: str
    expected: str
    status: Literal[
        "present", "absent", "unverifiable", "out_of_scope", "inconclusive"
    ] = "inconclusive"
    reasoning: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class Scorecard(BaseModel):
    solvable: bool
    difficulty: Literal["too_easy", "appropriate", "too_hard", "broken", "unknown"]
    scenario_status: str
    findings: list[str] = Field(default_factory=list)
    score: float = Field(ge=0, le=100)
    weaknesses: list[WeaknessVerdict] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    expected_baseline_met: bool | None = None
    grounded: bool = False


class QAExpectations(BaseModel):
    """Configured ground-truth baseline the scenario is expected to expose.

    This is *data*, not hardcoded control flow: it is loaded from the
    environment (and can later be sourced from a scenario file) and handed to
    the judge/supervisor as reference so their reasoning about solvability and
    difficulty is measured against a declared baseline instead of an implicit
    constant.
    """
    local_users: list[str] = Field(default_factory=list)
    domain_users: list[str] = Field(default_factory=list)
    privileged_groups: list[str] = Field(default_factory=list)
    open_ports: list[str] = Field(default_factory=list)
    networks: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not any((self.local_users, self.domain_users, self.privileged_groups,
                        self.open_ports, self.networks))

    @classmethod
    def from_env(cls, getenv=None) -> "QAExpectations":
        import os
        getenv = getenv or os.getenv

        def _split(value: str | None) -> list[str]:
            if not value:
                return []
            return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]

        return cls(
            local_users=_split(getenv("CYBERQA_EXPECTED_LOCAL_USERS")),
            domain_users=_split(getenv("CYBERQA_EXPECTED_DOMAIN_USERS")),
            privileged_groups=_split(getenv("CYBERQA_EXPECTED_PRIVILEGED_GROUPS")),
            open_ports=_split(getenv("CYBERQA_EXPECTED_OPEN_PORTS")),
            networks=_split(getenv("CYBERQA_EXPECTED_NETWORKS")),
        )
