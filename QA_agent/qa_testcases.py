"""
qa_testcases.py
===============
QA test case 定義 —— 每個弱點對應一個「受約束的驗證方法」。

設計原則 (constrained tool calling):
  - 每個 test case 是**參數化**的,不是自由指令。LLM 只能填來自 manifest 的參數。
  - 目標 IP 一律鎖在 range 網段 (由 tool 層強制,見 qa_agent.py 的 _assert_in_range)。
  - 判讀分兩層: 這裡的 `verdict()` 做**確定性** pass/fail (regex/pattern);
                LLM 只做編排與摘要,不做底層判定。

每個 TestCase:
  vuln_id     對應弱點編號
  phase       validate / test
  tool        用哪個工具 (nxc / impacket / ldapsearch / nmap / bloodhound)
  build_cmd   (manifest, params) -> argv list  (組指令; 目標鎖 range)
  verdict     (returncode, stdout, stderr) -> (status, evidence)
  needs_creds 是否需要 QA 犧牲帳號
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal


@dataclass
class TestResult:
    status: Literal["pass", "fail", "error", "skip"]
    evidence: str


@dataclass
class TestCase:
    vuln_id: int
    name: str
    phase: Literal["validate", "test"]
    tool: str
    build_cmd: Callable[[dict, dict], list[str]]   # (manifest_dict, params) -> argv
    verdict: Callable[[int, str, str], TestResult]  # (rc, out, err) -> result
    needs_creds: bool = True
    description: str = ""


# ── 輔助 ────────────────────────────────────────────────────────────────────
def _dc_ip(mf: dict) -> str:
    dc = next((m for m in mf["machines"] if m["role"] == "dc"), None)
    return dc["ip"] if dc else ""


def _dc_fqdn(mf: dict) -> str:
    dc = next((m for m in mf["machines"] if m["role"] == "dc"), None)
    return dc["fqdn"] if dc else ""


def _qa(mf: dict) -> tuple[str, str]:
    c = mf.get("qa_credential") or {}
    return c.get("username", ""), c.get("password", "")


def _host_ip(mf: dict, hostname: str) -> str:
    m = next((x for x in mf["machines"] if x["hostname"] == hostname), None)
    return m["ip"] if m else ""


def _found(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


# ══ 基礎設施 validate（不需弱點,先跑）═════════════════════════════════════════
def _tc_dns():
    return TestCase(
        vuln_id=0, name="DNS resolution", phase="validate", tool="nslookup",
        needs_creds=False,
        build_cmd=lambda mf, p: ["nslookup", mf["ad_domain"], _dc_ip(mf)],
        verdict=lambda rc, o, e: TestResult(
            "pass" if _dc_ip(_LAST_MF) and _dc_ip(_LAST_MF) in o else "fail",
            o[-300:] or e[-300:]),
        description="DC 能解析網域名稱",
    )


def _tc_ldap_reachable():
    return TestCase(
        vuln_id=0, name="LDAP reachable", phase="validate", tool="nxc",
        needs_creds=False,
        build_cmd=lambda mf, p: ["nxc", "ldap", _dc_ip(mf)],
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"LDAP", o) else "fail", (o + e)[-300:]),
        description="LDAP 服務可連線",
    )


def _tc_smb_reachable():
    return TestCase(
        vuln_id=0, name="SMB reachable", phase="validate", tool="nxc",
        needs_creds=False,
        build_cmd=lambda mf, p: ["nxc", "smb", _dc_ip(mf)],
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"SMB.*\[\*\]", o) else "fail", (o + e)[-300:]),
        description="SMB 服務可連線並回傳主機資訊",
    )


def _tc_kerberos_reachable():
    return TestCase(
        vuln_id=0, name="Kerberos reachable", phase="validate", tool="nmap",
        needs_creds=False,
        build_cmd=lambda mf, p: ["nmap", "-p", "88", "-Pn", _dc_ip(mf)],
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"88/tcp\s+open", o) else "fail", o[-300:]),
        description="Kerberos (88) 開放",
    )


# ══ 弱點 test（需 QA 帳號）════════════════════════════════════════════════════
def _tc_asrep():
    def build(mf, p):
        u, _ = _qa(mf)
        return ["impacket-GetNPUsers", f"{mf['ad_domain']}/", "-no-pass",
                "-usersfile", "/tmp/qa_users.txt", "-dc-ip", _dc_ip(mf),
                "-format", "hashcat"]
    return TestCase(
        vuln_id=1, name="AS-REP Roasting", phase="test", tool="impacket",
        build_cmd=build,
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"\$krb5asrep\$", o) else "fail", (o + e)[-400:]),
        description="種下的帳號可被 AS-REP roast (拿到 $krb5asrep$ hash)",
    )


def _tc_kerberoast():
    def build(mf, p):
        u, pw = _qa(mf)
        return ["impacket-GetUserSPNs", f"{mf['ad_domain']}/{u}:{pw}",
                "-dc-ip", _dc_ip(mf), "-request"]
    def verdict(rc, o, e):
        if _found(r"\$krb5tgs\$", o):
            return TestResult("pass", o[-400:])
        return TestResult("fail", (o + e)[-400:])
    return TestCase(vuln_id=2, name="Kerberoasting (RC4)", phase="test",
                    tool="impacket", build_cmd=build, verdict=verdict,
                    description="服務帳號可被 kerberoast (拿到 $krb5tgs$ hash)")


def _tc_kerberoast_aes():
    tc = _tc_kerberoast()
    tc.vuln_id = 3
    tc.name = "Kerberoasting (AES)"
    return tc


def _tc_unconstrained():
    def build(mf, p):
        u, pw = _qa(mf)
        return ["nxc", "ldap", _dc_ip(mf), "-u", u, "-p", pw, "--find-delegation"]
    return TestCase(
        vuln_id=15, name="Unconstrained Delegation", phase="test", tool="nxc",
        build_cmd=build,
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"unconstrained", o) else "fail", (o + e)[-400:]),
        description="存在非約束委派的機器")


def _tc_rbcd():
    def build(mf, p):
        u, pw = _qa(mf)
        target = p.get("target_computer", "")
        return ["impacket-rbcd", f"{mf['ad_domain']}/{u}:{pw}", "-dc-ip", _dc_ip(mf),
                "-action", "read", "-delegate-to", f"{target}$"]
    return TestCase(
        vuln_id=20, name="RBCD", phase="test", tool="impacket", build_cmd=build,
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"Accounts able to delegate|msDS-AllowedToActOnBehalf", o)
            else "fail", (o + e)[-400:]),
        description="目標機器的 RBCD 設定存在")


def _tc_smb_signing():
    return TestCase(
        vuln_id=136, name="SMB Signing Disabled", phase="test", tool="nxc",
        needs_creds=False,
        build_cmd=lambda mf, p: ["nxc", "smb", _dc_ip(mf)],
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"signing:\s*False", o) else "fail", (o + e)[-300:]),
        description="SMB 簽章已關閉 (relay 前置)")


def _tc_maq():
    def build(mf, p):
        u, pw = _qa(mf)
        return ["nxc", "ldap", _dc_ip(mf), "-u", u, "-p", pw, "-M", "maq"]
    return TestCase(
        vuln_id=160, name="MachineAccountQuota", phase="test", tool="nxc", build_cmd=build,
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"MachineAccountQuota:\s*[1-9]", o) else "fail", (o + e)[-300:]),
        description="MachineAccountQuota 非 0")


def _tc_pass_pol():
    def build(mf, p):
        u, pw = _qa(mf)
        return ["nxc", "smb", _dc_ip(mf), "-u", u, "-p", pw, "--pass-pol"]
    return TestCase(
        vuln_id=161, name="Weak Password Policy", phase="test", tool="nxc", build_cmd=build,
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"Complexity.*Disabled|Minimum password length:\s*[0-5]\b", o)
            else "fail", (o + e)[-400:]),
        description="密碼策略已弱化")


def _tc_gpp():
    def build(mf, p):
        u, pw = _qa(mf)
        return ["nxc", "smb", _dc_ip(mf), "-u", u, "-p", pw, "-M", "gpp_password"]
    return TestCase(
        vuln_id=80, name="GPP cpassword", phase="test", tool="nxc", build_cmd=build,
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"password|cpassword|Found", o) else "fail", (o + e)[-300:]),
        description="SYSVOL 中存在 GPP cpassword")


def _tc_desc():
    def build(mf, p):
        u, pw = _qa(mf)
        return ["nxc", "ldap", _dc_ip(mf), "-u", u, "-p", pw, "-M", "get-desc-users"]
    return TestCase(
        vuln_id=77, name="Password in Description", phase="test", tool="nxc", build_cmd=build,
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"description", o) and _found(r"pass|pwd|:", o) else "fail",
            (o + e)[-400:]),
        description="有帳號的 description 欄位含密碼")


def _tc_anon_ldap():
    def build(mf, p):
        parts = mf["ad_domain"].split(".")
        base = ",".join(f"DC={x}" for x in parts)
        return ["ldapsearch", "-x", "-H", f"ldap://{_dc_ip(mf)}", "-b", base,
                "(objectClass=user)", "samaccountname"]
    return TestCase(
        vuln_id=243, name="Anonymous LDAP Bind", phase="test", tool="ldapsearch",
        needs_creds=False, build_cmd=build,
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"sAMAccountName:", o) else "fail", (o + e)[-300:]),
        description="匿名 LDAP 綁定可讀取使用者")


def _tc_wdigest():
    def build(mf, p):
        u, pw = _qa(mf)
        host = p.get("computer", "")
        ip = _host_ip(mf, host) or _dc_ip(mf)
        return ["nxc", "smb", ip, "-u", u, "-p", pw, "-x",
                r"reg query HKLM\SYSTEM\CurrentControlSet\Control\SecurityProviders\WDigest /v UseLogonCredential"]
    return TestCase(
        vuln_id=74, name="WDigest Cleartext", phase="test", tool="nxc", build_cmd=build,
        verdict=lambda rc, o, e: TestResult(
            "pass" if _found(r"UseLogonCredential.*0x1", o) else "fail", (o + e)[-300:]),
        description="WDigest 已啟用明文憑證快取")


# ── BloodHound 驗證 (ACL 邊) — 由 qa_agent 特別處理 (需先採集再查 Cypher) ──────
BH_VULNS = {25, 37, 66}  # GenericAll / ForceChangePassword / DCSync


# ══ 註冊表 ════════════════════════════════════════════════════════════════════
_LAST_MF: dict = {}  # nslookup verdict 需要,執行前由 qa_agent 設定


VALIDATE_CASES = [
    _tc_dns(), _tc_ldap_reachable(), _tc_smb_reachable(), _tc_kerberos_reachable(),
]

# vuln_id -> TestCase (有 nxc/impacket/ldapsearch 驗證的)
TEST_CASES: dict[int, TestCase] = {
    tc.vuln_id: tc for tc in [
        _tc_asrep(), _tc_kerberoast(), _tc_kerberoast_aes(), _tc_unconstrained(),
        _tc_rbcd(), _tc_smb_signing(), _tc_maq(), _tc_pass_pol(), _tc_gpp(),
        _tc_desc(), _tc_anon_ldap(), _tc_wdigest(),
    ]
}


def get_test_case(vuln_id: int) -> TestCase | None:
    return TEST_CASES.get(vuln_id)


def has_direct_test(vuln_id: int) -> bool:
    return vuln_id in TEST_CASES


if __name__ == "__main__":
    print("Validate cases:", [t.name for t in VALIDATE_CASES])
    print("Test cases (direct tool):", sorted(TEST_CASES))
    print("BloodHound-verified:", sorted(BH_VULNS))
    # 示範組指令
    mf = {"ad_domain": "lab.local",
          "machines": [{"hostname": "DC01", "role": "dc", "ip": "10.3.10.10",
                        "fqdn": "dc01.lab.local"}],
          "qa_credential": {"username": "qa_tester", "password": "QaTester!2026"}}
    for vid in (1, 2, 136, 243):
        tc = TEST_CASES[vid]
        print(f"\n#{vid} {tc.name}:")
        print("  ", " ".join(tc.build_cmd(mf, {})))
