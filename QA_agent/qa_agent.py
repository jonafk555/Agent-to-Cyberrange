"""
qa_agent.py
===========
QA Agent —— 靶場部署後的驗證/測試/除錯代理 (constrained tool calling)。

讀取 range_manifest.json,對種下的弱點逐一驗證「真的可利用」。

設計:
  - LLM 負責**編排** (下一步測什麼、失敗要不要 debug) 與**摘要** (寫人類可讀報告)。
  - 工具負責**確定性判定** (regex verdict)。LLM 不做底層 pass/fail。
  - 每個工具**參數化且鎖定 range 網段** (_assert_in_range),LLM 不能自由指定目標。
  - HITL: 測試前確認範圍、debug 前確認、破壞性操作前確認。

黑箱 / 白箱:
  --mode blackbox  只用 QA 犧牲帳號 (模擬攻擊者初始存取),測「打得通嗎」。
  --mode whitebox  額外允許讀 manifest 全部細節 + 透過 MCP 登入 DC 檢查實際設定值,
                   用於 debug 與精確根因分析。

無 manifest 模式:
  若 --manifest 指到的檔案不存在 (例如對外部/未知靶場做稽核,沒有 cyberrange worker
  產生的 range_manifest.json),agent 改用 --network-cidr 對網段做 nmap 探索找出 DC
  (88/389/445 皆開),並嘗試以匿名 LDAP rootDSE 查詢猜出網域名稱。找到 DC 後,不再依賴
  "種下的弱點清單",改用 generic_ad_audit 對 qa_testcases 中**全部**已知檢測項目跑一輪
  基線稽核 (黑箱)。若有提供 --qa-user/--qa-pass,連需要憑證的項目也會測;沒提供則那些
  項目回 skip。

依賴:
  pip install "langgraph>=1.0" langchain pydantic
  地端 LLM: langchain-ollama 或 langchain-openai
  工具 (在 attack host 上): impacket, netexec(nxc), nmap, ldapsearch, bloodhound-python
  同資料夾: range_manifest.py, qa_testcases.py, ad_vuln_catalog.py, ad_vuln_catalog.json

⚠️ 只對隔離 range 網段執行。工具層強制目標在 range CIDR 內,拒絕範圍外目標。
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import subprocess
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

import qa_testcases as tcs
from range_manifest import load_manifest, RangeManifest


# ═══════════════════════════════════════════════════════════════════════════
# 全域: 載入的 manifest + 模式 + range 網段 (工具讀取)
# ═══════════════════════════════════════════════════════════════════════════
_MF: dict = {}
_MODE: str = "blackbox"
_RANGE_NET: Optional[ipaddress.IPv4Network] = None
_NO_MANIFEST: bool = False
_TIMEOUT = int(os.environ.get("QA_TOOL_TIMEOUT", "180"))


def _load(path: str, discover_cidr: Optional[str] = None, ad_domain: Optional[str] = None,
          dc_ip: Optional[str] = None, qa_user: Optional[str] = None,
          qa_pass: Optional[str] = None) -> None:
    """載入 manifest。若 path 不存在,改用 discover_cidr 做無 manifest 探索。"""
    global _MF, _RANGE_NET, _NO_MANIFEST
    if path and os.path.exists(path):
        mf = load_manifest(path)
        _MF = mf.model_dump()
        _NO_MANIFEST = False
    else:
        _NO_MANIFEST = True
        _MF = _discover_manifest(discover_cidr, ad_domain, dc_ip, qa_user, qa_pass)
    try:
        _RANGE_NET = ipaddress.ip_network(_MF.get("network_cidr", "10.2.10.0/24"), strict=False)
    except ValueError:
        _RANGE_NET = None
    tcs._LAST_MF = _MF  # nslookup verdict 需要


# ═══════════════════════════════════════════════════════════════════════════
# 無 manifest 探索: 掃網段找 DC + 猜網域名,組出最小可用的 in-memory manifest
# ═══════════════════════════════════════════════════════════════════════════
def _discover_dc(cidr: str) -> Optional[str]:
    """nmap 掃 cidr,找出同時開 88(kerberos)/389(ldap)/445(smb) 的主機,視為 DC。"""
    rc, out, err = _run(["nmap", "-p", "88,389,445", "--open", "-Pn", cidr])
    if not out:
        return None
    current_ip, open_ports = None, set()

    def _flush():
        if current_ip and {"88", "389", "445"} <= open_ports:
            return current_ip
        return None

    for line in out.splitlines():
        m = re.match(r"Nmap scan report for (?:\S+ )?\(?([\d.]+)\)?", line)
        if m:
            hit = _flush()
            if hit:
                return hit
            current_ip, open_ports = m.group(1), set()
        pm = re.match(r"(\d+)/tcp\s+open", line)
        if pm:
            open_ports.add(pm.group(1))
    return _flush()


def _discover_domain(dc_ip: str) -> Optional[str]:
    """匿名 LDAP rootDSE 查 defaultNamingContext,轉成網域名。查不到就放棄 (人可用 --ad-domain 補)。"""
    rc, out, err = _run(["ldapsearch", "-x", "-H", f"ldap://{dc_ip}", "-s", "base",
                         "defaultNamingContext"])
    m = re.search(r"defaultNamingContext:\s*(.+)", out)
    if not m:
        return None
    parts = re.findall(r"DC=([^,]+)", m.group(1).strip(), re.IGNORECASE)
    return ".".join(parts) if parts else None


def _discover_manifest(cidr: Optional[str], ad_domain: Optional[str], dc_ip: Optional[str],
                        qa_user: Optional[str], qa_pass: Optional[str]) -> dict:
    if not dc_ip and not cidr:
        raise SystemExit(
            "[QA] 找不到 range_manifest.json,且未給 --network-cidr / --dc-ip,"
            "無法探索目標,中止。")
    if not dc_ip:
        print(f"[QA] 無 manifest,對 {cidr} 做 nmap 探索找 DC ...")
        dc_ip = _discover_dc(cidr)
        if not dc_ip:
            raise SystemExit(
                f"[QA] 在 {cidr} 內找不到同時開 88/389/445 的主機 (DC)。"
                "改用 --dc-ip 明確指定目標。")
        print(f"[QA] 探索到疑似 DC: {dc_ip}")
    if not ad_domain:
        ad_domain = _discover_domain(dc_ip) or "unknown.local"
        print(f"[QA] 探索到網域: {ad_domain}")
    net = cidr or f"{'.'.join(dc_ip.split('.')[:3])}.0/24"
    machine = {"hostname": "DC-DISCOVERED", "role": "dc", "template": "", "ip": dc_ip,
               "vlan": 10, "os_version": "", "fqdn": f"dc.{ad_domain}",
               "services": ["ldap", "kerberos", "smb", "dns"], "credentials": []}
    qa_cred = None
    if qa_user and qa_pass:
        qa_cred = {"username": qa_user, "password": qa_pass, "domain": ad_domain,
                   "kind": "qa_sacrificial", "sensitive": True,
                   "note": "供自 CLI (--qa-user/--qa-pass),非來自 manifest"}
    return {
        "schema_version": "1.0", "range_id": "discovered-no-manifest", "ad_domain": ad_domain,
        "ad_version": "", "network_cidr": net, "machines": [machine],
        "planted_vulns": [], "endpoints": [], "qa_credential": qa_cred,
        "notes": "無 range_manifest.json,由 QA agent 自行探索組出。無種植弱點清單,"
                 "用 generic_ad_audit 對已知檢測項目跑基線稽核。",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 安全: 目標鎖定 range 網段
# ═══════════════════════════════════════════════════════════════════════════
def _assert_in_range(argv: list[str]) -> Optional[str]:
    """檢查指令中的 IP 是否都落在 range 網段。回傳錯誤字串,或 None 表示通過。"""
    if _RANGE_NET is None:
        return None
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", " ".join(argv))
    for ip in ips:
        try:
            if ipaddress.ip_address(ip) not in _RANGE_NET:
                return f"[BLOCKED] target {ip} is outside range network {_RANGE_NET}"
        except ValueError:
            continue
    return None


def _run(argv: list[str]) -> tuple[int, str, str]:
    """執行指令 (已通過 range 檢查)。工具不存在時回錯而非 crash。"""
    if not shutil.which(argv[0]):
        return 127, "", f"tool not found: {argv[0]} (install it on the attack host)"
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=_TIMEOUT)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {_TIMEOUT}s"
    except OSError as e:
        return 1, "", str(e)


# ═══════════════════════════════════════════════════════════════════════════
# 受約束的工具 (LLM 只能填 manifest 內的參數;目標鎖 range)
# ═══════════════════════════════════════════════════════════════════════════
@tool
def list_planted_vulns() -> str:
    """列出 manifest 中所有種下的弱點及其測試座標。回傳 JSON。
    先呼叫這個以了解要驗證什麼。若沒有 manifest (無種植弱點清單),回傳空清單提示,
    改呼叫 generic_ad_audit 做基線稽核。"""
    if _NO_MANIFEST:
        return json.dumps({"status": "no_manifest",
                           "evidence": "沒有 range_manifest.json,無種植弱點清單可比對。"
                                       "改呼叫 generic_ad_audit 對已知檢測項目跑一輪基線稽核。"},
                          ensure_ascii=False)
    out = [{"id": v["id"], "name": v["name"], "kind": v["kind"],
            "target": v["target"], "test_case_id": v["test_case_id"],
            "has_direct_test": tcs.has_direct_test(v["id"]),
            "bh_verified": v["id"] in tcs.BH_VULNS}
           for v in _MF.get("planted_vulns", [])]
    return json.dumps(out, ensure_ascii=False)


@tool
def generic_ad_audit() -> str:
    """[無 manifest 模式專用] 沒有 range_manifest.json、不知道種了哪些弱點時,
    對已探索到的 DC 跑 qa_testcases 中**全部**已知 AD 弱點檢測項目 (AS-REP roast、
    kerberoast、無簽章 SMB、匿名 LDAP、弱密碼策略、GPP cpassword...等),當成黑箱基線稽核。
    沒有提供 --qa-user/--qa-pass 的項目會回 skip (該項目需要驗證身分才能測)。
    回傳每項 pass/fail/skip + evidence。manifest 存在時不要呼叫這個,改用
    list_planted_vulns + test_vuln。"""
    if not _NO_MANIFEST:
        return json.dumps({"status": "skip",
                           "evidence": "manifest 存在,請改用 list_planted_vulns + test_vuln"},
                          ensure_ascii=False)
    have_creds = bool((_MF.get("qa_credential") or {}).get("password"))
    results = []
    for vid, tc in sorted(tcs.TEST_CASES.items()):
        if tc.needs_creds and not have_creds:
            results.append({"vuln_id": vid, "name": tc.name, "status": "skip",
                            "evidence": "no credential supplied (--qa-user/--qa-pass)"})
            continue
        argv = tc.build_cmd(_MF, {})
        blocked = _assert_in_range(argv)
        if blocked:
            results.append({"vuln_id": vid, "name": tc.name, "status": "error", "evidence": blocked})
            continue
        rc, out, err = _run(argv)
        r = tc.verdict(rc, out, err)
        results.append({"vuln_id": vid, "name": tc.name, "status": r.status,
                        "evidence": r.evidence[:300]})
    return json.dumps(results, ensure_ascii=False)


@tool
def get_range_info() -> str:
    """取得 range 基本資訊: 網域、DC IP、機器清單、服務端點。回傳 JSON。
    白箱模式含更多細節。"""
    dc = next((m for m in _MF["machines"] if m["role"] == "dc"), None)
    info = {
        "ad_domain": _MF["ad_domain"],
        "dc_ip": dc["ip"] if dc else None,
        "network_cidr": _MF["network_cidr"],
        "machines": [{"hostname": m["hostname"], "role": m["role"], "ip": m["ip"]}
                     for m in _MF["machines"]],
        "mode": _MODE,
        "no_manifest": _NO_MANIFEST,
    }
    if _MODE == "whitebox":
        info["endpoints"] = _MF.get("endpoints", [])
    return json.dumps(info, ensure_ascii=False)


@tool
def run_validation() -> str:
    """跑基礎設施驗證 (DNS/LDAP/SMB/Kerberos 可連線)。應在測試弱點前先跑。
    回傳每項的 pass/fail + evidence。"""
    results = []
    for tc in tcs.VALIDATE_CASES:
        argv = tc.build_cmd(_MF, {})
        blocked = _assert_in_range(argv)
        if blocked:
            results.append({"check": tc.name, "status": "error", "evidence": blocked})
            continue
        rc, out, err = _run(argv)
        r = tc.verdict(rc, out, err)
        results.append({"check": tc.name, "status": r.status,
                        "evidence": r.evidence[:200]})
    return json.dumps(results, ensure_ascii=False)


@tool
def test_vuln(vuln_id: int) -> str:
    """對單一種下的弱點執行驗證測試,確認它真的可被利用。
    vuln_id 必須是 manifest 中已種下的弱點編號。
    工具會用固定的驗證指令 (impacket/nxc/ldapsearch) 並鎖定 range 目標。
    回傳 status (pass=可利用 / fail=種了但打不通 / error / skip) 與 evidence。"""
    # 確認是已種弱點
    planted_ids = {v["id"] for v in _MF.get("planted_vulns", [])}
    if vuln_id not in planted_ids:
        return json.dumps({"status": "skip",
                           "evidence": f"vuln #{vuln_id} not in manifest"}, ensure_ascii=False)
    # BloodHound 類單獨處理
    if vuln_id in tcs.BH_VULNS:
        return json.dumps({"status": "skip",
                           "evidence": f"#{vuln_id} needs bloodhound_verify (ACL edge)"},
                          ensure_ascii=False)
    tc = tcs.get_test_case(vuln_id)
    if tc is None:
        return json.dumps({"status": "skip",
                           "evidence": f"#{vuln_id} has no direct tool test (template LLM-drafted; verify manually)"},
                          ensure_ascii=False)
    # 需要犧牲帳號但沒有 → skip
    if tc.needs_creds and not (_MF.get("qa_credential") or {}).get("password"):
        return json.dumps({"status": "skip",
                           "evidence": "no QA sacrificial credential in manifest"},
                          ensure_ascii=False)
    params = next((v["params"] for v in _MF["planted_vulns"] if v["id"] == vuln_id), {})
    argv = tc.build_cmd(_MF, params)
    blocked = _assert_in_range(argv)
    if blocked:
        return json.dumps({"status": "error", "evidence": blocked}, ensure_ascii=False)
    rc, out, err = _run(argv)
    r = tc.verdict(rc, out, err)
    return json.dumps({"status": r.status, "tool": tc.tool,
                       "cmd": " ".join(argv), "evidence": r.evidence[:400]},
                      ensure_ascii=False)


@tool
def bloodhound_verify(vuln_id: int) -> str:
    """用 BloodHound 驗證 ACL 類弱點 (GenericAll/ForceChangePassword/DCSync) 的攻擊路徑是否存在。
    會先用 QA 帳號採集,再對種下的弱點下對應 Cypher。
    回傳採集狀態與該弱點的 Cypher (供在 BloodHound CE 執行確認)。"""
    if vuln_id not in tcs.BH_VULNS:
        return json.dumps({"status": "skip",
                           "evidence": f"#{vuln_id} not a BloodHound-verified vuln"},
                          ensure_ascii=False)
    v = next((x for x in _MF["planted_vulns"] if x["id"] == vuln_id), None)
    if not v:
        return json.dumps({"status": "skip", "evidence": "not planted"}, ensure_ascii=False)
    dc = next((m for m in _MF["machines"] if m["role"] == "dc"), None)
    u, pw = (_MF.get("qa_credential") or {}).get("username", ""), \
            (_MF.get("qa_credential") or {}).get("password", "")
    argv = ["bloodhound-python", "-d", _MF["ad_domain"], "-u", u, "-p", pw,
            "-ns", dc["ip"] if dc else "", "-c", "All", "--zip"]
    blocked = _assert_in_range(argv)
    if blocked:
        return json.dumps({"status": "error", "evidence": blocked}, ensure_ascii=False)
    rc, out, err = _run(argv)
    collected = "found" in (out + err).lower() or rc == 0
    return json.dumps({
        "status": "collected" if collected else "collect_failed",
        "cypher": v.get("bh_cypher"),
        "note": "在 BloodHound CE 執行 cypher 確認邊存在。空結果=種植失敗。",
        "evidence": (out + err)[-300:],
    }, ensure_ascii=False)


@tool
def whitebox_check_setting(vuln_id: int) -> str:
    """[白箱限定] 透過登入 DC 直接檢查某弱點的實際設定值 (根因分析用)。
    黑箱模式下拒絕。用於 debug: 當 test_vuln 回 fail 時,查為什麼沒種成功。"""
    if _MODE != "whitebox":
        return json.dumps({"status": "denied",
                           "evidence": "whitebox mode required"}, ensure_ascii=False)
    # 這裡回傳「應檢查什麼」而非實際登入 (實際登入需 DA 憑證 / MCP run_command_on_host)
    checks = {
        1: "Get-ADUser <user> -Properties DoesNotRequirePreAuth",
        2: "Get-ADUser <svc> -Properties ServicePrincipalNames,msDS-SupportedEncryptionTypes",
        15: "Get-ADComputer <host> -Properties TrustedForDelegation",
        66: "dsacls '<domain DN>' | Select-String 'Replicating Directory Changes'",
        136: r"reg query HKLM\SYSTEM\...\LanManServer\Parameters /v RequireSecuritySignature",
        160: "Get-ADObject <domain DN> -Properties ms-DS-MachineAccountQuota",
    }
    return json.dumps({"status": "whitebox",
                       "check_command": checks.get(vuln_id, "(no whitebox check defined)"),
                       "note": "透過 MCP run_command_on_host 或 DA 憑證執行以確認實際值"},
                      ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# debug 工具 (HITL 保護 — 會修改 range)
# ═══════════════════════════════════════════════════════════════════════════
@tool
def suggest_fix(vuln_id: int, reason: str) -> str:
    """當 test_vuln 回 fail 時,提出修復建議 (重跑對應的 Ansible task)。
    這只是**建議**,不直接執行。實際重跑需操作者確認並透過 cyberrange agent / ludus-mcp。
    reason: 你判斷失敗的原因。"""
    v = next((x for x in _MF["planted_vulns"] if x["id"] == vuln_id), None)
    name = v["name"] if v else f"#{vuln_id}"
    return json.dumps({
        "vuln_id": vuln_id, "name": name, "diagnosis": reason,
        "suggested_action": f"re-run Ansible task tagged 'id-{vuln_id}' on DC via ludus-mcp",
        "requires_human_approval": True,
    }, ensure_ascii=False)


QA_TOOLS = [list_planted_vulns, get_range_info, run_validation, test_vuln,
            bloodhound_verify, whitebox_check_setting, suggest_fix, generic_ad_audit]

# 需要 HITL 確認的工具 (破壞性或會登入機器)
HITL_TOOLS = {"suggest_fix"}  # test/validate 是唯讀驗證,不需逐次確認


# ═══════════════════════════════════════════════════════════════════════════
# 報告 schema (LLM 最終結構化輸出)
# ═══════════════════════════════════════════════════════════════════════════
class VulnQAResult(BaseModel):
    vuln_id: int
    name: str
    status: Literal["exploitable", "planted_but_broken", "not_tested", "error"]
    evidence: str = ""


class QAReport(BaseModel):
    range_id: str
    mode: str
    infra_ok: bool = Field(description="基礎設施驗證是否全數通過")
    results: list[VulnQAResult]
    summary: str = Field(description="人類可讀的總結,含建議")


# ═══════════════════════════════════════════════════════════════════════════
# Agent 建構
# ═══════════════════════════════════════════════════════════════════════════
QA_SYSTEM = """你是靶場 QA 工程師。你的任務是驗證一個 Windows AD 訓練靶場:
1. 先呼叫 list_planted_vulns 了解種了哪些弱點。
2. 呼叫 run_validation 確認基礎設施 (DNS/LDAP/SMB/Kerberos) 正常。若基礎設施 fail,先報告,不要繼續測弱點。
3. 對每個種下的弱點呼叫 test_vuln (ACL 類用 bloodhound_verify) 確認「真的可被利用」。
4. 對 fail 的弱點,呼叫 suggest_fix 提出診斷 (這需要人工確認才會實際修)。
5. 最後輸出結構化 QA 報告。

重要:
- 工具已經做了確定性的 pass/fail 判定,你直接採信工具回傳的 status,不要自己改判。
- status=pass 代表弱點可利用;fail 代表種了但打不通 (需 debug);skip 代表無直接測試方法。
- 你只負責編排順序、判斷失敗原因、寫報告。不要嘗試自由組指令。
- 全程只針對 manifest 內的 range 目標。
"""

QA_SYSTEM_NO_MANIFEST = """你是靶場 QA 工程師。這次**沒有 range_manifest.json**
(可能是對外部/未知環境做稽核,或 manifest 遺失),已由前置步驟對目標網段做 nmap 探索,
找出 DC 並猜出網域名 (見 get_range_info)。你不知道這個環境「種了哪些弱點」,
所以改成對已知 AD 弱點清單做一輪黑箱基線稽核:
1. 先呼叫 get_range_info 確認探索到的 DC / 網域是否合理。
2. 呼叫 run_validation 確認基礎設施 (DNS/LDAP/SMB/Kerberos) 正常。若基礎設施 fail,先報告,不要繼續。
3. 呼叫 list_planted_vulns 只是形式確認 (會回 no_manifest),不要期待有種植清單。
4. 呼叫 generic_ad_audit 對所有已知檢測項目跑一輪。有些項目因為沒有帳密會 skip,如實回報,
   不要假裝測過。
5. 最後輸出結構化 QA 報告,status 用 exploitable(pass 且有風險)/not_tested(skip)/error 表示;
   沒有 planted_but_broken 這個狀態的意義 (沒有「種植」動作),不要用它。

重要:
- 工具已經做了確定性的 pass/fail 判定,你直接採信工具回傳的 status,不要自己改判。
- 你只負責編排順序、判斷結果、寫報告。不要嘗試自由組指令。
- 全程只針對探索到的 range 目標,不得對網段外主機下手。
"""


def build_llm():
    backend = os.environ.get("CYBERRANGE_LLM_BACKEND", "ollama")
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""),
                          model=os.environ.get("LLM_MODEL", "gpt-4o"),
                          temperature=0)
    if backend == "local-openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                          api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
                          model=os.environ.get("LLM_MODEL", "llama4:scout"), temperature=0)
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.environ.get("LLM_MODEL", "llama4:scout"),
                          base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                          temperature=0)
    return init_chat_model(os.environ.get("CYBERRANGE_MODEL", "anthropic:claude-sonnet-4-6"),
                           temperature=0)


def build_qa_agent(observer=None):
    """用 create_agent + HumanInTheLoopMiddleware 建 QA agent。
    若傳入 observer (ObserverAgent),會將 QA_TOOLS 包裹成帶觀察的版本。"""
    from langchain.agents import create_agent
    try:
        from langchain.agents.middleware import HumanInTheLoopMiddleware
        mw = [HumanInTheLoopMiddleware(interrupt_on={t: True for t in HITL_TOOLS})]
    except Exception:
        mw = []  # 舊版無 middleware 時退化為無 HITL (仍安全,因 suggest_fix 不直接執行)

    tools = QA_TOOLS
    if observer is not None:
        tools = observer.wrap_qa_tools(QA_TOOLS)

    agent = create_agent(
        model=build_llm(),
        tools=tools,
        system_prompt=QA_SYSTEM_NO_MANIFEST if _NO_MANIFEST else QA_SYSTEM,
        middleware=mw,
        checkpointer=InMemorySaver(),
    )
    return agent


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def run(a) -> None:
    global _MODE
    _MODE = a.mode
    os.environ.setdefault("CYBERRANGE_LLM_BACKEND", a.backend)
    if a.model:
        os.environ["LLM_MODEL"] = a.model
    if a.ollama_base_url:
        os.environ["OLLAMA_BASE_URL"] = a.ollama_base_url

    try:
        _load(a.manifest, discover_cidr=a.network_cidr, ad_domain=a.ad_domain,
              dc_ip=a.dc_ip, qa_user=a.qa_user, qa_pass=a.qa_pass)
    except SystemExit as e:
        print(str(e))
        return

    if _NO_MANIFEST:
        print(f"[QA] 無 range_manifest.json,已探索組出: {_MF['ad_domain']} / mode={_MODE}")
        print(f"[QA] range network (target-locked): {_RANGE_NET}")
        print(f"[QA] 無種植弱點清單 — 將對已知檢測項目跑基線稽核 (generic_ad_audit)")
    else:
        print(f"[QA] loaded manifest: {_MF['range_id']} / {_MF['ad_domain']} / mode={_MODE}")
        print(f"[QA] range network (target-locked): {_RANGE_NET}")
        print(f"[QA] planted vulns: {[v['id'] for v in _MF.get('planted_vulns', [])]}")

    # HITL: 測試前確認範圍
    if not a.yes:
        ans = input(f"\n即將對 {_RANGE_NET} 內的 range 目標執行唯讀驗證測試。繼續? [y/N] ")
        if ans.strip().lower() != "y":
            print("已取消。")
            return

    # Observer Agent 初始化
    observer = None
    if getattr(a, "observe", False):
        from observer_agent import ObserverAgent
        observer = ObserverAgent(
            backend=getattr(a, "observer_backend", None) or a.backend,
            model=getattr(a, "observer_model", None) or a.model,
        )
        print(f"[Observer] 觀察員已啟動 (backend={observer.backend}, model={observer.model_name})")

    agent = build_qa_agent(observer=observer)
    config = {"configurable": {"thread_id": a.thread_id}}
    if _NO_MANIFEST:
        task = (f"對探索到的 range (網域 {_MF['ad_domain']}, DC {_MF['machines'][0]['ip']}) "
                f"做基線稽核。沒有種植弱點清單,先確認基礎設施,再對已知 AD 弱點檢測項目 "
                f"跑 generic_ad_audit,最後給我一份 QA 報告。")
    else:
        task = (f"驗證 range '{_MF['range_id']}' (網域 {_MF['ad_domain']})。"
                f"先看種了哪些弱點、跑基礎設施驗證,再逐一測試每個弱點是否可利用,"
                f"最後給我一份 QA 報告。")

    from langgraph.types import Command
    result = agent.invoke({"messages": [{"role": "user", "content": task}]}, config)

    # 處理 HITL 中斷 (suggest_fix)
    while "__interrupt__" in result:
        intr = result["__interrupt__"][0].value
        print("\n[HITL] agent 想執行需確認的動作:")
        print(json.dumps(intr, ensure_ascii=False, indent=2)[:800])
        ans = input("approve / reject> ").strip().lower()
        decision = {"type": "approve"} if ans == "approve" else {"type": "reject"}
        result = agent.invoke(Command(resume={"decisions": [decision]}), config)

    # 輸出最終訊息
    print("\n" + "=" * 72)
    print("=== QA 報告 ===")
    final = result["messages"][-1].content
    print(final if isinstance(final, str) else json.dumps(final, ensure_ascii=False, indent=2))

    # Observer 報告輸出
    if observer:
        output_dir = getattr(a, "observer_output", None) or "."
        observer.generate_report(
            range_id=_MF.get("range_id", "unknown"), mode=_MODE,
            infra_ok=True, output_dir=output_dir,
        )
        print(f"[Observer] 觀察報告已儲存")


def build_parser():
    p = argparse.ArgumentParser(prog="qa_agent",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest", default="range_manifest.json",
                   help="range manifest 路徑;檔案不存在時自動轉入無 manifest 探索模式"
                        " (需搭配 --network-cidr 或 --dc-ip)")
    p.add_argument("--mode", choices=["blackbox", "whitebox"], default="blackbox",
                   help="blackbox=只用 QA 帳號測可利用性; whitebox=額外允許根因檢查")
    # 無 manifest 探索模式
    p.add_argument("--network-cidr", default=None,
                   help="[無 manifest 模式] 要探索的網段,如 10.2.10.0/24;找 DC 用")
    p.add_argument("--ad-domain", default=None,
                   help="[無 manifest 模式] 已知網域名;不給則嘗試匿名 LDAP 探索")
    p.add_argument("--dc-ip", default=None,
                   help="[無 manifest 模式] 已知 DC IP;給了就跳過 nmap 探索")
    p.add_argument("--qa-user", default=None,
                   help="[無 manifest 模式] 供已驗證檢測項目使用的帳號")
    p.add_argument("--qa-pass", default=None,
                   help="[無 manifest 模式] 供已驗證檢測項目使用的密碼")
    p.add_argument("--backend", choices=["openai", "ollama", "local-openai", "cloud"], default="ollama")
    p.add_argument("--model", default=None, help="LLM 模型 (建議 tool-calling 強的,如 llama4:scout)")
    p.add_argument("--ollama-base-url", default=None)
    p.add_argument("--thread-id", default="qa-run-001")
    p.add_argument("--yes", action="store_true", help="跳過測試前的範圍確認")
    # Observer Agent
    p.add_argument("--observe", action="store_true",
                   help="啟用觀察員 Agent,詳細記錄每步檢測的指令、結果、排查方式、因果推理")
    p.add_argument("--observer-output", default=None,
                   help="觀察員報告輸出目錄 (預設當前目錄)")
    p.add_argument("--observer-backend", default=None,
                   help="觀察員 LLM 後端 (預設與 --backend 相同)")
    p.add_argument("--observer-model", default=None,
                   help="觀察員 LLM 模型 (預設與 --model 相同)")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
