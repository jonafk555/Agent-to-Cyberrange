"""AD QA capabilities and planning knowledge.

This module describes what an agent may investigate; it deliberately does not
hard-code a single command for every capability. Adapters can be swapped for
NetExec, Impacket, ldapsearch, BloodHound, PowerShell, or range-specific APIs.
"""
from __future__ import annotations

from .models import ADRisk, CapabilitySpec, ToolParameters


AD_CAPABILITIES: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        name="domain_inventory",
        purpose="Discover domain, hosts, services, DNS, LDAP, Kerberos, SMB, and trust boundaries.",
        expected_evidence=["domain_name", "host_inventory", "service_inventory", "dns_or_ldap_root"],
        allowed_tools=["check_port", "check_dns_resolution", "ldap_bind", "nxc_ldap_recon", "smb_negotiate"],
    ),
    CapabilitySpec(
        name="enumerate_domain_users",
        purpose="Enumerate users, groups, SPNs, delegation flags, and account controls.",
        # The reviewed adapter performs the LDAP bind and query in one
        # bounded operation when a credential is supplied. Anonymous user
        # discovery is handled by the separate nxc_ldap read-only phase.
        prerequisites=["domain_inventory", "validated domain credential"],
        expected_evidence=["users", "groups", "spns", "preauth_flags", "delegation_flags"],
        allowed_tools=["ad_domain_users"],
    ),
    CapabilitySpec(
        name="asrep_roasting_assessment",
        purpose="Identify accounts without Kerberos pre-authentication and assess the authorized QA path.",
        # A domain credential is intentionally not a prerequisite. AS-REP is
        # the unauthenticated branch; it needs a domain/DC and a candidate
        # username source (range file, anonymous enumeration, or evidence).
        prerequisites=["domain_inventory", "candidate username source"],
        expected_evidence=["asrep_candidates", "ticket_obtained_or_blocked", "credential_validation_status"],
        allowed_tools=["ad_asrep_roasting"],
        risk=ADRisk.CREDENTIAL_MATERIAL,
        notes="Do not claim a password or crack result without evidence; protect ticket material.",
    ),
    CapabilitySpec(
        name="kerberoasting_assessment",
        purpose="Identify service accounts with SPNs and assess ticket exposure under the range policy.",
        prerequisites=["domain_inventory", "user enumeration", "authorized identity or approved anonymous path"],
        expected_evidence=["spn_accounts", "ticket_obtained_or_blocked", "credential_validation_status"],
        allowed_tools=["ad_kerberoasting"],
        risk=ADRisk.CREDENTIAL_MATERIAL,
        notes="Use a lab output directory and redact ticket/password material from logs.",
    ),
    CapabilitySpec(
        name="credential_validation",
        purpose="Validate a supplied lab credential against an explicitly scoped host/service.",
        prerequisites=["human_supplied_or_range_issued_credential"],
        expected_evidence=["authentication_success_or_failure", "protocol", "target"],
        allowed_tools=["ad_credential_validation"],
        risk=ADRisk.AUTHENTICATION_TEST,
    ),
    CapabilitySpec(
        name="controlled_password_spray_assessment",
        purpose="Assess password reuse only after lockout policy and scope are known.",
        prerequisites=["domain_inventory", "user enumeration", "lockout_policy", "approved_test_password"],
        expected_evidence=["scope", "rate", "lockout_policy", "authentication_results"],
        allowed_tools=["ad_password_spray"],
        risk=ADRisk.ACCOUNT_LOCKOUT,
        requires_approval=True,
        notes="Never spray by default. Require explicit approval, low rate, bounded accounts, and one pass.",
    ),
    CapabilitySpec(
        name="bloodhound_collection",
        purpose="Collect AD relationships using a validated identity and analyze attack-path coverage.",
        prerequisites=["domain_inventory", "validated domain credential", "DNS resolution"],
        expected_evidence=["users", "groups", "sessions", "acl_edges", "delegation", "trusts"],
        allowed_tools=["ad_bloodhound_collection"],
        risk=ADRisk.READ_ONLY,
    ),
    CapabilitySpec(
        name="privilege_path_assessment",
        purpose="Assess ACL, delegation, LAPS/GPO, AD CS, trust, and local-admin paths from evidence.",
        prerequisites=["bloodhound_collection or equivalent relationship evidence"],
        expected_evidence=["prerequisite_edges", "affected_principal", "impact", "blocked_or_proven"],
        allowed_tools=["ad_bloodhound_collection", "ldap_bind"],
    ),
)

CAPABILITY_INDEX = {item.name: item for item in AD_CAPABILITIES}

# These adapters do not accept arbitrary command-line arguments. Keep their
# executable parameter contract separate from the generic Nmap/NXC contract.
CAPABILITY_PARAMETER_FIELDS: dict[str, frozenset[str]] = {
    "enumerate_domain_users": frozenset(),
    "asrep_roasting_assessment": frozenset({"users", "users_file"}),
    "kerberoasting_assessment": frozenset(),
    "credential_validation": frozenset(),
    "controlled_password_spray_assessment": frozenset(),
    "bloodhound_collection": frozenset(),
}


def normalize_capability_parameters(
    capability: str | None,
    parameters: ToolParameters | dict | None,
) -> ToolParameters:
    """Strip generic planner fields unsupported by a concrete AD adapter."""
    if parameters is None:
        raw: dict = {}
    elif hasattr(parameters, "model_dump"):
        raw = parameters.model_dump(mode="json", exclude_none=True)
    else:
        raw = dict(parameters)
    allowed = CAPABILITY_PARAMETER_FIELDS.get(capability or "")
    if allowed is not None:
        raw = {key: value for key, value in raw.items() if key in allowed}
    return ToolParameters.model_validate(raw)


def capability_catalog() -> list[dict]:
    """Return a JSON-friendly catalog for the LLM prompt."""
    return [item.model_dump(mode="json") for item in AD_CAPABILITIES]


def get_capability(name: str | None) -> CapabilitySpec | None:
    return CAPABILITY_INDEX.get(name or "")
