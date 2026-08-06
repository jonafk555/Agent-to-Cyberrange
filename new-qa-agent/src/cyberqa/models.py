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


class VisibilityMode(str, Enum):
    WHITE_BOX = "white_box"
    GRAY_BOX = "gray_box"
    BLACK_BOX = "black_box"


class EvidenceLevel(str, Enum):
    C0 = "C0"  # Unknown
    C1 = "C1"  # Inferred
    C2 = "C2"  # Enumerated
    C3 = "C3"  # Functionally verified
    C4 = "C4"  # Exploitability verified
    C5 = "C5"  # End-to-end verified


class AssertionStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


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


class EvidenceOpportunity(BaseModel):
    """A reviewed next-step candidate derived from observed evidence.

    This is planning memory, not an instruction to execute.  The Supervisor
    still checks authorization, duplicate history, target policy, and risk
    before selecting a Decision.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    tool: str
    action: str
    target: str
    reason: str
    evidence_fields: list[str] = Field(default_factory=list)
    prerequisites_met: list[str] = Field(default_factory=list)
    prerequisites_missing: list[str] = Field(default_factory=list)
    capability: str | None = None
    risk: ADRisk = ADRisk.READ_ONLY
    expected_evidence: list[str] = Field(default_factory=list)


class EvidenceAnalysis(BaseModel):
    """A compact, evidence-backed interpretation of one tool result.

    This is deliberately not a chain-of-thought field.  It records only the
    useful facts, open questions, and reviewed tools that the Supervisor may
    consider next.  Secret material must never be copied into this model.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = ""
    source: str = ""
    target: str = ""
    useful_content: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    candidate_tools: list[str] = Field(default_factory=list)
    opportunities: list[EvidenceOpportunity] = Field(default_factory=list)
    recommended_action: str | None = None
    recommended_target: str | None = None
    reason: str = ""
    no_new_information: bool = False


class QAAssertion(BaseModel):
    """A machine-checkable QA question with an explicit evidence threshold."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    statement: str
    target: str
    assertion_type: str = "custom"
    source: Literal["specification", "operator", "discovery", "system"] = "system"
    visibility: VisibilityMode = VisibilityMode.BLACK_BOX
    required_evidence_level: EvidenceLevel = EvidenceLevel.C2
    allowed_methods: list[str] = Field(default_factory=list)
    status: AssertionStatus = AssertionStatus.NOT_STARTED
    evidence_ids: list[str] = Field(default_factory=list)
    conclusion: str | None = None
    unknown_reason: str | None = None


class EvidenceSufficiency(BaseModel):
    """Evidence threshold evaluation; it is not itself a QA conclusion."""

    model_config = ConfigDict(extra="forbid")

    assertion_id: str
    current_level: EvidenceLevel = EvidenceLevel.C0
    required_level: EvidenceLevel = EvidenceLevel.C2
    sufficient: bool = False
    status: Literal["sufficient", "insufficient", "contradictory", "blocked"] = "insufficient"
    evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    next_methods: list[str] = Field(default_factory=list)
    reason: str = ""


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
    # Paths to controlled local artifacts; their contents never enter the
    # model-visible decision or evidence payload.
    hash_file: str | None = None
    wordlist: str | None = None
    allow_anonymous_nxc: bool = False


class HumanIntent(BaseModel):
    """Machine-checkable operator intent kept alongside the raw instruction."""

    model_config = ConfigDict(extra="forbid")

    raw_instruction: str = ""
    goals: list[str] = Field(default_factory=list)
    ordered_steps: list[str] = Field(default_factory=list)
    step_parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    step_statuses: list[str] = Field(default_factory=list)
    parsing_errors: list[str] = Field(default_factory=list)
    current_step: int = 0
    requested_targets: list[str] = Field(default_factory=list)
    excluded_targets: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    has_ordering: bool = False
    continue_autonomously: bool = True
    rejected_previous: bool = False
    completed: bool = False


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
    asrep_hash_file: str | None = None
    asrep_hash_count: int = 0
    hash_cracking_attempted: bool = False
    hash_cracked: bool = False
    cracked_users: list[str] = Field(default_factory=list)
    crack_status: str | None = None
    credential_source: str | None = None
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
