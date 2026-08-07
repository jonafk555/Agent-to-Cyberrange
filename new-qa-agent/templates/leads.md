# AD reasoning leads

Non-prescriptive reference for the planner. Format: `see X -> may enable Y -> reviewed method Z1/Z2`.
These are hints for orientation, NOT a mandatory sequence. Always prefer the least-invasive evidence
that answers the current assertion, respect scope, and never treat a hash as a password.

- see open 88/tcp (Kerberos) -> domain is reachable -> nxc/ldap domain discovery, enum users
- see username list, no creds -> AS-REP roastable accounts possible -> AS-REP assessment (no pre-auth)
- see AS-REP hash material -> offline crack candidate -> reviewed local hash-cracking (never inline)
- see cracked credential -> authenticated context -> credential-validation, authenticated LDAP/SMB enum
- see SPN accounts -> Kerberoast candidate -> reviewed Kerberoast assessment when prerequisites met
- see LDAP anonymous bind -> low-priv enumeration -> anonymous LDAP user/group enum
- see delegation / ACL edges -> privilege path possible -> BloodHound collection, attack-path analysis
- see AD CS endpoint -> certificate abuse possible -> AD CS misconfiguration assessment
- see domain trust -> cross-forest reach possible -> trust enumeration
- see SMB signing disabled -> relay possible -> record as blocked unless assertion requires validation
