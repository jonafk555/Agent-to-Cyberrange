from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


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
    result: Literal["not_run", "passed", "failed", "degraded", "blocked"] = "not_run"
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


class Decision(BaseModel):
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
    tool_parameters: dict[str, Any] = Field(default_factory=dict)


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


class Scorecard(BaseModel):
    solvable: bool
    difficulty: Literal["too_easy", "appropriate", "too_hard", "broken"]
    scenario_status: str
    findings: list[str] = Field(default_factory=list)
    score: float = Field(ge=0, le=100)
