"""
ad_vuln_catalog.py
==================
AD 靶場弱點目錄 —— 來源: 使用者提供的 AD_RedTeam_Vulnerability_Index.md (248 條),
以 parse_catalog.py 解析成 ad_vuln_catalog.json (逐條對應,無杜撰)。

本模組在原始 248 條之上,附加三種「靶場觀點」的中繼資料:
  1) kind  : 這一條在**建靶時**要怎麼處理
       - "plant"          可直接把弱設定「種」進 AD (產 Ansible 佈建腳本)
       - "precondition"   開啟某個脆弱服務/設定,讓某攻擊技術得以成立 (產 Ansible)
       - "technique"      攻擊者在 Kali 上用現成工具執行的動作,建靶時不種、只記錄給受訓者
       - "default_present" Windows/AD 預設就存在 (LOLBin、預設可採集),不需佈建
  2) TEMPLATES : 少數高價值 plant 項的實際 Ansible 佈建模板 (PowerShell/RSAT)
  3) BH_CYPHER : 對應的 BloodHound 確認查詢,用於部署後驗證「路徑真的種進去了」

⚠️ 界線: 本模組只產「把弱點種進隔離靶場」的佈建設定 (與 GOAD 的 Ansible role 同性質),
   不產出攻擊/利用工具本身。攻擊由受訓者用 Rubeus/Certipy/Impacket/BloodHound 等既有工具執行。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_JSON = os.path.join(_HERE, "ad_vuln_catalog.json")


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    """讀取 248 條原始目錄 (id/category/category_name/name/weakness)。"""
    with open(_JSON, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. 分類: 類別預設 + 逐條覆寫
# ---------------------------------------------------------------------------
_CATEGORY_DEFAULT = {
    "A": "plant",          # Kerberos 協定弱設定 (票證偽造類另覆寫為 technique)
    "B": "plant",          # 委派設定 (coercion 組合 / S4U 覆寫為 technique)
    "C": "plant",          # ACL / ACE 授權
    "D": "plant",          # 特權群組成員
    "E": "technique",      # 憑證竊取多為攻擊動作 (可種的前置設定逐條覆寫)
    "F": "precondition",   # 強制驗證: 開啟脆弱服務 = 前置條件, coercion 本身是 technique
    "G": "plant",          # ADCS 憑證範本
    "H": "plant",          # (g/d)MSA
    "I": "plant",          # LAPS
    "J": "technique",      # 持久化多為後滲透 (少數前置設定覆寫)
    "K": "precondition",   # 橫向移動: 開啟服務 = 前置條件
    "L": "plant",          # 偵察/資訊洩漏 = 可種的弱設定
    "M": "plant",          # 憑證保護弱設定
    "N": "plant",          # SCCM (需先部署 SCCM,見 note)
    "O": "plant",          # 網域信任
    "P": "plant",          # ADIDNS / 名稱解析
    "Q": "technique",      # 2025-26 新面 (BadSuccessor 前置覆寫為 plant)
    "R": "default_present",# LOLBin: Windows 內建,不需佈建
    "S": "technique",      # DNS 隧道/外洩工具
    "T": "plant",          # 環境/預設帳號
}

_OVERRIDE: dict[int, str] = {
    # A: 票證偽造 / 記憶體竊取類 = technique
    4: "technique", 5: "technique", 6: "technique", 7: "technique",
    8: "technique", 9: "technique", 10: "technique", 12: "technique",
    # B: coercion 組合 & S4U 濫用 = technique (前置的 unconstrained host 是 #15 plant)
    16: "technique", 17: "technique", 21: "technique", 22: "technique",
    # E: 可種的前置設定
    52: "precondition", 53: "precondition", 54: "precondition",
    59: "plant", 62: "precondition", 63: "precondition", 64: "precondition",
    65: "precondition", 66: "plant", 74: "plant", 76: "plant", 77: "plant",
    78: "plant", 79: "plant", 80: "plant", 81: "plant",
    # L: BloodHound 採集為預設可行
    157: "default_present",
    # Q: BadSuccessor 前置可種; RC4 混用可種
    198: "plant", 201: "plant",
    # P: LLMNR/NBT/mDNS 未關閉 = 預設存在 (需明確關閉才算修好)
    195: "default_present", 196: "default_present", 197: "default_present",
}


def kind_of(item: dict) -> str:
    return _OVERRIDE.get(item["id"], _CATEGORY_DEFAULT.get(item["category"], "technique"))


def enrich(item: dict) -> dict:
    e = dict(item)
    e["kind"] = kind_of(item)
    e["buildable"] = e["kind"] in ("plant", "precondition")  # 需產 Ansible?
    e["has_template"] = item["id"] in TEMPLATES
    e["bh_cypher"] = BH_CYPHER.get(item["id"])
    return e


def get(item_id: int) -> dict:
    for it in load_catalog():
        if it["id"] == item_id:
            return enrich(it)
    raise KeyError(f"catalog id {item_id} not found")


def resolve_ids(ids: list[int]) -> list[dict]:
    return [get(i) for i in ids]


# 便利: 依類別/類型過濾,供 master_agent 呈現清單
def by_category() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for it in load_catalog():
        out.setdefault(it["category"], []).append(enrich(it))
    return out


# ---------------------------------------------------------------------------
# 2. 預設 profile (一鍵挑一組經典弱點,對應 GOAD 風格路徑)
# ---------------------------------------------------------------------------
PROFILES: dict[str, list[int]] = {
    # GOAD 風格: AS-REP + Kerberoast + 非約束委派 + RBCD + ACL + DCSync + ESC1 + ESC8
    "goad-like": [1, 2, 15, 20, 25, 37, 66, 89, 96, 136, 160, 161],
    # 憑證濫用主題 (ADCS)
    "adcs": [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
    # 中繼/強制驗證主題
    "relay-coercion": [52, 53, 54, 55, 82, 83, 84, 96, 136, 137, 138],
    # 憑證竊取前置 (讓 dumping/relay 成立的設定)
    "cred-weakness": [59, 62, 63, 66, 74, 76, 77, 78, 79, 80, 81, 171],
}


# ---------------------------------------------------------------------------
# 3. 可種植項的 Ansible 佈建模板 (高價值子集; 其餘由 config_dialog 以 LLM 起草)
#    每個模板: params 需要操作者填的參數, tasks 為 Ansible YAML 片段 (在 DC 上執行)
#    佈建 = 設定弱點條件, 非攻擊工具。
# ---------------------------------------------------------------------------
TEMPLATES: dict[int, dict] = {
    1: {  # AS-REP Roasting
        "params": ["user"],
        "tasks": """- name: "[#1 AS-REP] {user} 停用 Kerberos 預先驗證"
  ansible.windows.win_shell: >
    Set-ADAccountControl -Identity '{user}' -DoesNotRequirePreAuth $true
  tags: [vuln, asrep, "id-1"]""",
    },
    2: {  # Kerberoasting (RC4)
        "params": ["svc_user", "password", "spn"],
        "tasks": """- name: "[#2 Kerberoast/RC4] 建立弱密碼服務帳號 {svc_user}"
  ansible.windows.win_shell: |
    $sec = ConvertTo-SecureString '{password}' -AsPlainText -Force
    if (-not (Get-ADUser -Filter "SamAccountName -eq '{svc_user}'" -ErrorAction SilentlyContinue)) {{
      New-ADUser -Name '{svc_user}' -SamAccountName '{svc_user}' -AccountPassword $sec -Enabled $true -PasswordNeverExpires $true
    }}
    Set-ADUser -Identity '{svc_user}' -ServicePrincipalNames @{{Add='{spn}'}}
    # 允許 RC4 (刻意弱化以利 Kerberoast 破解演練)
    Set-ADUser -Identity '{svc_user}' -Replace @{{'msDS-SupportedEncryptionTypes'=4}}
  tags: [vuln, kerberoast, "id-2"]""",
    },
    3: {  # Kerberoasting (AES only)
        "params": ["svc_user", "password", "spn"],
        "tasks": """- name: "[#3 Kerberoast/AES] 建立服務帳號 {svc_user} (AES only)"
  ansible.windows.win_shell: |
    $sec = ConvertTo-SecureString '{password}' -AsPlainText -Force
    if (-not (Get-ADUser -Filter "SamAccountName -eq '{svc_user}'" -ErrorAction SilentlyContinue)) {{
      New-ADUser -Name '{svc_user}' -SamAccountName '{svc_user}' -AccountPassword $sec -Enabled $true -PasswordNeverExpires $true
    }}
    Set-ADUser -Identity '{svc_user}' -ServicePrincipalNames @{{Add='{spn}'}}
    Set-ADUser -Identity '{svc_user}' -Replace @{{'msDS-SupportedEncryptionTypes'=24}}
  tags: [vuln, kerberoast, "id-3"]""",
    },
    15: {  # Unconstrained Delegation
        "params": ["computer"],
        "tasks": """- name: "[#15 非約束委派] 於 {computer} 設定 TrustedForDelegation"
  ansible.windows.win_shell: >
    Set-ADComputer -Identity '{computer}' -TrustedForDelegation $true
  tags: [vuln, delegation, unconstrained, "id-15"]""",
    },
    20: {  # RBCD
        "params": ["target_computer", "allowed_account"],
        "tasks": """- name: "[#20 RBCD] 允許 {allowed_account} 代理至 {target_computer}"
  ansible.windows.win_shell: >
    Set-ADComputer -Identity '{target_computer}'
    -PrincipalsAllowedToDelegateToAccount (Get-ADUser -Identity '{allowed_account}')
  tags: [vuln, delegation, rbcd, "id-20"]""",
    },
    25: {  # GenericAll on User
        "params": ["grantee", "target_user"],
        "tasks": """- name: "[#25 GenericAll/User] 授予 {grantee} 對 {target_user} 完全控制"
  ansible.windows.win_shell: >
    $t = (Get-ADUser '{target_user}').DistinguishedName;
    dsacls "$t" /G "{grantee}:GA"
  tags: [vuln, acl, genericall, "id-25"]""",
    },
    37: {  # ForceChangePassword
        "params": ["grantee", "target_user"],
        "tasks": """- name: "[#37 ForceChangePassword] {grantee} 可重設 {target_user} 密碼"
  ansible.windows.win_shell: >
    $t = (Get-ADUser '{target_user}').DistinguishedName;
    dsacls "$t" /G "{grantee}:CA;Reset Password"
  tags: [vuln, acl, forcechangepassword, "id-37"]""",
    },
    66: {  # DCSync rights
        "params": ["grantee"],
        "tasks": """- name: "[#66 DCSync] 授予 {grantee} 網域複寫權限 (DS-Replication-Get-Changes*)"
  ansible.windows.win_shell: |
    $dn = (Get-ADDomain).DistinguishedName
    dsacls "$dn" /G "{grantee}:CA;Replicating Directory Changes"
    dsacls "$dn" /G "{grantee}:CA;Replicating Directory Changes All"
  tags: [vuln, dcsync, "id-66"]""",
    },
    74: {  # WDigest cleartext
        "params": ["computer"],
        "tasks": """- name: "[#74 WDigest] 於 {computer} 啟用 LSASS 明文憑證快取"
  ansible.windows.win_shell: >
    reg add "HKLM\\SYSTEM\\CurrentControlSet\\Control\\SecurityProviders\\WDigest"
    /v UseLogonCredential /t REG_DWORD /d 1 /f
  delegate_to: "{computer}"
  tags: [vuln, wdigest, "id-74"]""",
    },
    77: {  # Password in description
        "params": ["target_user", "fake_password"],
        "tasks": """- name: "[#77 Description 洩密] 於 {target_user} description 埋入密碼"
  ansible.windows.win_shell: >
    Set-ADUser -Identity '{target_user}' -Description 'Do not share: {fake_password}'
  tags: [vuln, infoleak, description, "id-77"]""",
    },
    80: {  # GPP cpassword
        "params": [],
        "tasks": """- name: "[#80 GPP cpassword] 於 SYSVOL 放置含 cpassword 的 Groups.xml (骨架)"
  ansible.windows.win_copy:
    dest: 'C:\\Windows\\SYSVOL\\sysvol\\{{{{ ad_domain }}}}\\Policies\\gpp-vuln-Groups.xml'
    content: |
      <?xml version="1.0" encoding="utf-8"?>
      <Groups clsid="{{{{31B2F340-016D-11D2-945F-00C04FB984F9}}}}">
        <User name="lab-gpp" image="2" changed="2024-01-01"
          cpassword="edBSHOwhZLTjt/QS9FeIcJ83mjWA98gw9guKOhJOdcqh+ZGMeXOsQbCpZ3xUjTLfCuNH8pG5aSVYdYw/NglVmQ"/>
      </Groups>
  tags: [vuln, gpp, cpassword, "id-80"]""",
    },
    136: {  # SMB signing disabled
        "params": [],
        "tasks": """- name: "[#136 SMB Signing 關閉] 建立/連結 GPO 停用 SMB 簽章 (relay 前置)"
  ansible.windows.win_shell: >
    reg add "HKLM\\SYSTEM\\CurrentControlSet\\Services\\LanManServer\\Parameters"
    /v RequireSecuritySignature /t REG_DWORD /d 0 /f
  tags: [vuln, smb, signing, relay, "id-136"]""",
    },
    160: {  # MachineAccountQuota
        "params": ["quota"],
        "tasks": """- name: "[#160 MachineAccountQuota] 設定為 {quota} (預設 10; RBCD/shadow-cred 前置)"
  ansible.windows.win_shell: >
    Set-ADObject (Get-ADDomain).DistinguishedName
    -Replace @{{'ms-DS-MachineAccountQuota'={quota}}}
  tags: [vuln, maq, "id-160"]""",
    },
    161: {  # Weak password policy
        "params": [],
        "tasks": """- name: "[#161 弱密碼策略] 關閉複雜度、降低長度"
  ansible.windows.win_shell: >
    Set-ADDefaultDomainPasswordPolicy -Identity (Get-ADDomain).DNSRoot
    -ComplexityEnabled $false -MinPasswordLength 5
  tags: [vuln, passwordpolicy, "id-161"]""",
    },
    243: {  # Anonymous LDAP bind
        "params": [],
        "tasks": """- name: "[#243 匿名 LDAP] 開啟 dsHeuristics 允許匿名綁定"
  ansible.windows.win_shell: |
    $cfg = (Get-ADRootDSE).configurationNamingContext
    $dn = "CN=Directory Service,CN=Windows NT,CN=Services,$cfg"
    Set-ADObject -Identity $dn -Replace @{{dsHeuristics='0000002'}}
  tags: [vuln, ldap, anonymous, "id-243"]""",
    },
}


# ---------------------------------------------------------------------------
# 4. BloodHound 驗證查詢 (部署後確認種下的路徑存在; BloodHound CE 屬性/邊)
# ---------------------------------------------------------------------------
BH_CYPHER: dict[int, str] = {
    1:  "MATCH (u:User {dontreqpreauth:true}) RETURN u.name",
    2:  "MATCH (u:User {hasspn:true}) RETURN u.name",
    3:  "MATCH (u:User {hasspn:true}) RETURN u.name",
    15: "MATCH (c:Computer {unconstraineddelegation:true}) RETURN c.name",
    20: "MATCH (n)-[:AllowedToAct]->(c:Computer) RETURN n.name, c.name",
    25: "MATCH p=(n)-[:GenericAll]->(u:User) RETURN n.name, u.name LIMIT 50",
    37: "MATCH p=(n)-[:ForceChangePassword]->(u:User) RETURN n.name, u.name LIMIT 50",
    66: "MATCH p=(n)-[:DCSync|GetChanges|GetChangesAll]->(d:Domain) RETURN n.name, d.name",
    89: "MATCH p=(n)-[:ADCSESC1]->(d:Domain) RETURN n.name, d.name",
    96: "MATCH p=(n)-[:ADCSESC8]->(d:Domain) RETURN n.name, d.name",
    160: "MATCH (d:Domain) RETURN d.name, d.machineaccountquota",
}


if __name__ == "__main__":
    # 快速統計 (自我檢查)
    cats = by_category()
    from collections import Counter
    kinds = Counter(kind_of(it) for it in load_catalog())
    print("總數:", len(load_catalog()))
    print("分類統計:", dict(kinds))
    print("有 Ansible 模板:", sorted(TEMPLATES))
    print("有 BH 驗證查詢:", sorted(BH_CYPHER))
    print("profiles:", {k: len(v) for k, v in PROFILES.items()})
