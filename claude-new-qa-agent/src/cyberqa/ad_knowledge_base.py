"""Senior-level Windows Active Directory QA knowledge base.

This module encodes what an experienced AD security/QA engineer actually checks
when validating a Windows Active Directory environment: the components, the
settings, and the configuration items worth verifying, grouped into concrete QA
checks. It exists so the agent stops behaving like a generic port scanner and
starts behaving like a senior AD reviewer.

It is used two ways:

1. ``qa_checklist_prompt`` renders the checklist into the planner/specialist
   system prompt, so the LLM reasons with senior AD expertise (which functions,
   components, and configuration settings to QA) instead of generic recon
   instincts.
2. ``qa_coverage`` compares the checklist against accumulated evidence to
   compute which checks are done, which remain, and what to do next. This is the
   backbone that turns scattered observations into a single, stable QA progress
   view and stops the agent from re-running work it has already completed.

The checklist is intentionally evidence-driven and read-only: it describes what
to *verify*, never how to exploit. Each check names the AD component and the
concrete settings/configuration a senior reviewer would inspect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ADQACheck:
    """One senior AD QA check.

    ``knowledge_keys`` / ``source_markers`` are the signals that mark the check
    complete; ``verifies`` is the reviewer-facing list of settings/config items
    the check is responsible for; ``requires_credential`` / ``service_marker``
    describe when the check is applicable.
    """

    id: str
    category: str
    component: str
    objective: str
    verifies: tuple[str, ...]
    knowledge_keys: tuple[str, ...] = ()
    source_markers: tuple[str, ...] = ()
    capability: str | None = None
    requires_credential: bool = False
    service_marker: tuple[str, ...] = ()
    priority: int = 50
    notes: str = ""


# Priority: lower number = check earlier. Foundation before credentialed depth.
AD_QA_CHECKS: tuple[ADQACheck, ...] = (
    ADQACheck(
        id="domain_dns_foundation",
        category="Foundation",
        component="DNS / Domain naming",
        objective="Confirm the domain/forest naming context and that DC DNS resolves.",
        verifies=(
            "defaultNamingContext / rootDomainNamingContext",
            "forest vs domain FQDN",
            "SRV records for _ldap._tcp / _kerberos._tcp",
            "DC A/PTR records and forwarders",
        ),
        knowledge_keys=("domain", "domains", "forests"),
        source_markers=("dig", "dns", "ldap_bind", "ldapsearch", "nxc_ldap"),
        priority=10,
        notes="Everything downstream depends on a resolvable, correctly named domain.",
    ),
    ADQACheck(
        id="service_inventory",
        category="Foundation",
        component="DC service surface",
        objective="Inventory the AD service ports exposed by each DC/member.",
        verifies=(
            "LDAP 389 / LDAPS 636 / GC 3268-3269",
            "Kerberos 88, kpasswd 464",
            "SMB 445, RPC 135, WinRM 5985/5986",
            "AD CS web enrollment 80/443, ADWS 9389",
        ),
        source_markers=("nmap", "check_port"),
        priority=15,
    ),
    ADQACheck(
        id="ldap_channel_hygiene",
        category="Directory",
        component="LDAP / LDAPS",
        objective="Verify LDAP signing/channel-binding posture and anonymous exposure.",
        verifies=(
            "LDAP signing requirement (LdapEnforceChannelBinding)",
            "LDAPS certificate presence and validity",
            "anonymous bind / RootDSE exposure",
            "dsHeuristics anonymous-op bit",
        ),
        source_markers=("ldap_bind", "ldapsearch", "nxc_ldap"),
        priority=20,
    ),
    ADQACheck(
        id="smb_signing_shares",
        category="File services",
        component="SMB",
        objective="Check SMB signing enforcement and unauthenticated share exposure.",
        verifies=(
            "SMB signing required (server + DC)",
            "SMBv1 disabled",
            "null-session share/enum exposure",
            "sensitive share ACLs (SYSVOL/NETLOGON and custom)",
        ),
        source_markers=("smb_negotiate", "smbclient", "nxc_smb"),
        priority=25,
    ),
    ADQACheck(
        id="anonymous_identity",
        category="Identity",
        component="Anonymous enumeration",
        objective="Derive a bounded username source when no credential is held.",
        verifies=(
            "anonymous LDAP user disclosure",
            "RID cycling / SAMR exposure",
            "pass-pol via null session",
        ),
        knowledge_keys=("users",),
        source_markers=("nxc_ldap", "nxc_smb", "ldap_bind", "smb_negotiate", "anonymous_identity"),
        priority=30,
        notes="Only the bounded read-only identity phase; never guess accounts.",
    ),
    ADQACheck(
        id="password_lockout_policy",
        category="Policy",
        component="Account / password policy",
        objective="Record the password and lockout policy before any auth testing.",
        verifies=(
            "min length / complexity / history / max age",
            "lockout threshold / window / duration",
            "fine-grained password policies (PSOs)",
        ),
        knowledge_keys=(),
        source_markers=("pass-pol", "password_policy", "lockout"),
        priority=32,
        notes="Lockout policy is a prerequisite for any spray assessment.",
    ),
    ADQACheck(
        id="asrep_preauth",
        category="Kerberos",
        component="Pre-authentication",
        objective="Identify accounts with Kerberos pre-auth disabled (AS-REP roastable).",
        verifies=(
            "userAccountControl DONT_REQ_PREAUTH flag",
            "AS-REP ticket exposure for candidate users",
        ),
        knowledge_keys=("asrep_candidates",),
        source_markers=("asrep", "ad_asrep"),
        capability="asrep_roasting_assessment",
        priority=35,
        notes="Unauthenticated branch: needs domain/DC + a candidate username source.",
    ),
    ADQACheck(
        id="credential_validation",
        category="Identity",
        component="Domain credential",
        objective="Validate the supplied range credential against a scoped service.",
        verifies=(
            "authentication success/failure",
            "credential -> protocol -> target mapping",
        ),
        knowledge_keys=("credentials_validated",),
        source_markers=("ad_credential_validation", "credential_validation"),
        capability="credential_validation",
        priority=38,
    ),
    ADQACheck(
        id="user_spn_enumeration",
        category="Identity",
        component="Directory objects",
        objective="Enumerate users, groups, SPNs, delegation and UAC flags (authenticated).",
        verifies=(
            "sAMAccountName / servicePrincipalName inventory",
            "userAccountControl flags (DES, no-preauth, trusted-for-delegation)",
            "adminCount / protected accounts",
            "group memberships",
        ),
        knowledge_keys=("spns", "groups"),
        source_markers=("ad_domain_users", "enumerate_domain_users"),
        capability="enumerate_domain_users",
        requires_credential=True,
        priority=40,
    ),
    ADQACheck(
        id="kerberoast_spn",
        category="Kerberos",
        component="Service accounts / SPNs",
        objective="Assess service-account ticket exposure for accounts with SPNs.",
        verifies=(
            "SPN-bearing accounts and their crypto (RC4 vs AES)",
            "TGS ticket exposure under range policy",
            "service account password age / gMSA usage",
        ),
        knowledge_keys=(),
        source_markers=("kerberoast", "ad_kerberoasting"),
        capability="kerberoasting_assessment",
        requires_credential=True,
        priority=45,
    ),
    ADQACheck(
        id="relationship_graph",
        category="Attack paths",
        component="ACL / delegation graph",
        objective="Collect the AD relationship graph for path analysis.",
        verifies=(
            "ACL edges (GenericAll/WriteDACL/etc.)",
            "sessions and local-admin edges",
            "delegation (unconstrained/constrained/RBCD)",
        ),
        knowledge_keys=("acl_edges", "delegation", "groups"),
        source_markers=("bloodhound", "ad_bloodhound"),
        capability="bloodhound_collection",
        requires_credential=True,
        priority=50,
    ),
    ADQACheck(
        id="delegation_review",
        category="Attack paths",
        component="Kerberos delegation",
        objective="Review delegation configuration for dangerous settings.",
        verifies=(
            "unconstrained delegation on non-DC hosts",
            "constrained delegation targets (msDS-AllowedToDelegateTo)",
            "resource-based constrained delegation (msDS-AllowedToActOnBehalfOfOtherIdentity)",
        ),
        knowledge_keys=("delegation",),
        source_markers=("bloodhound", "delegation"),
        requires_credential=True,
        priority=55,
    ),
    ADQACheck(
        id="adcs_review",
        category="Certificate services",
        component="AD CS",
        objective="Review certificate templates and CA configuration for misissuance.",
        verifies=(
            "enrollee-supplies-subject templates (ESC1)",
            "dangerous EKUs / any-purpose (ESC2)",
            "vulnerable enrollment agent / CA ACLs (ESC3-ESC8)",
            "web enrollment / NTLM relay to CA",
        ),
        knowledge_keys=("adcs_findings",),
        source_markers=("certipy", "adcs", "ad_cs"),
        service_marker=("adcs", "9389", "443"),
        requires_credential=True,
        priority=60,
    ),
    ADQACheck(
        id="trust_review",
        category="Trusts",
        component="Domain / forest trusts",
        objective="Enumerate trusts and review SID filtering / transitivity.",
        verifies=(
            "trust direction and transitivity",
            "SID filtering / SID history posture",
            "cross-forest reachable targets",
        ),
        knowledge_keys=("trusts",),
        source_markers=("trust", "bloodhound", "nxc_ldap"),
        requires_credential=True,
        priority=62,
    ),
    ADQACheck(
        id="privileged_access",
        category="Privileged access",
        component="Tier-0 / admin hygiene",
        objective="Assess privileged group hygiene and admin exposure.",
        verifies=(
            "Domain/Enterprise/Schema Admins membership",
            "AdminSDHolder / adminCount consistency",
            "Protected Users group usage",
            "stale/duplicate privileged accounts",
        ),
        knowledge_keys=("groups", "acl_edges"),
        source_markers=("bloodhound", "ad_domain_users"),
        requires_credential=True,
        priority=65,
    ),
    ADQACheck(
        id="laps_gpo_review",
        category="Endpoint / policy",
        component="LAPS / GPO",
        objective="Review LAPS deployment and GPO delegation exposure.",
        verifies=(
            "ms-Mcs-AdmPwd / LAPS readers",
            "GPO edit rights delegation",
            "SYSVOL scripts and cpassword remnants",
        ),
        knowledge_keys=(),
        source_markers=("laps", "gpo", "sysvol"),
        requires_credential=True,
        priority=70,
    ),
)


CATEGORY_ORDER = (
    "Foundation", "Directory", "File services", "Identity", "Policy",
    "Kerberos", "Attack paths", "Certificate services", "Trusts",
    "Privileged access", "Endpoint / policy",
)


def qa_checklist_prompt() -> str:
    """Render the checklist as compact senior-reviewer guidance for the prompt."""
    lines = [
        "Senior Windows AD QA checklist. Work these components/settings in roughly",
        "this order; each line is: [id] component -> what to verify.",
    ]
    for check in sorted(AD_QA_CHECKS, key=lambda c: c.priority):
        gate = ""
        if check.requires_credential:
            gate = " (needs validated credential)"
        verifies = "; ".join(check.verifies)
        lines.append(f"[{check.id}] {check.component}{gate}: {verifies}")
    lines.append(
        "Verify configuration and prerequisites; state proven / blocked / unknown. "
        "Never invent credentials or claim exploitation without evidence."
    )
    return "\n".join(lines)


def _flatten_sources(evidence_sources: Any, method_history: Any) -> set[str]:
    sources: set[str] = set()
    for item in evidence_sources or ():
        sources.add(str(item).lower())
    for record in method_history or ():
        if isinstance(record, dict):
            sources.add(str(record.get("tool", "")).lower())
            sources.add(str(record.get("action", "")).lower())
            sources.add(str(record.get("capability", "")).lower())
    return {source for source in sources if source}


def qa_coverage(
    knowledge: dict[str, Any] | None,
    evidence_sources: Any = (),
    method_history: Any = (),
    has_credential: bool = False,
    open_services: Any = (),
) -> dict[str, Any]:
    """Compute done / pending / blocked QA checks from accumulated memory.

    This is deliberately conservative: a check is "done" only when it produced
    durable knowledge or a matching successful observation source. A credentialed
    check with no validated credential is "blocked" (with a reason), never
    silently skipped, so the planner knows the prerequisite instead of looping.
    """
    knowledge = knowledge or {}
    sources = _flatten_sources(evidence_sources, method_history)
    services = {str(item).lower() for item in (open_services or ())}
    have_credential = bool(has_credential or knowledge.get("credentials_validated"))

    checks: list[dict[str, Any]] = []
    for check in sorted(AD_QA_CHECKS, key=lambda c: c.priority):
        done = any(knowledge.get(key) for key in check.knowledge_keys) or any(
            marker in source for marker in check.source_markers for source in sources
        )
        if done:
            status, reason = "done", ""
        elif check.requires_credential and not have_credential:
            status, reason = "blocked", "needs a validated domain credential"
        elif check.service_marker and not any(
            marker in service for marker in check.service_marker for service in services
        ):
            status, reason = "blocked", "gating service not observed yet"
        else:
            status, reason = "pending", ""
        checks.append({
            "id": check.id,
            "category": check.category,
            "component": check.component,
            "objective": check.objective,
            "verifies": list(check.verifies),
            "capability": check.capability,
            "status": status,
            "reason": reason,
            "priority": check.priority,
        })

    completed = [c["id"] for c in checks if c["status"] == "done"]
    pending = [c for c in checks if c["status"] == "pending"]
    blocked = [c for c in checks if c["status"] == "blocked"]
    recommended = [
        {"id": c["id"], "component": c["component"], "objective": c["objective"],
         "capability": c["capability"]}
        for c in pending[:5]
    ]
    total = len(checks)
    coverage_pct = round(100.0 * len(completed) / total, 1) if total else 0.0
    return {
        "checks": checks,
        "completed": completed,
        "pending": [c["id"] for c in pending],
        "blocked": [{"id": c["id"], "reason": c["reason"]} for c in blocked],
        "recommended_next": recommended,
        "coverage_pct": coverage_pct,
    }
