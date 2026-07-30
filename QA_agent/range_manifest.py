"""
range_manifest.py
=================
Range Manifest —— 靶場部署後的完整清單,作為 QA agent 的輸入契約。

由 cyberrange agent 的 worker 在部署後產生 (range_manifest.json),
QA agent 讀取後即知道:有哪些機器、用哪組帳號測、種了哪些弱點、對應哪個 test case。

⚠️ 敏感檔案: 內含明文帳密。應設檔案權限 600、勿外流、勿進版控。
   建議部署後加密或存入 secrets manager,QA 執行時才解密。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

import ad_vuln_catalog as cat


# ═══════════════════════════════════════════════════════════════════════════
# 資料模型
# ═══════════════════════════════════════════════════════════════════════════
class Credential(BaseModel):
    """一組帳密。sensitive=True 者在日誌/報告中應遮蔽。"""
    username: str
    password: Optional[str] = None
    ntlm_hash: Optional[str] = None
    domain: Optional[str] = None
    kind: Literal["qa_sacrificial", "local_admin", "domain_admin",
                  "service_account", "standard_user"] = "standard_user"
    sensitive: bool = False
    note: str = ""


class Machine(BaseModel):
    hostname: str
    role: Literal["dc", "server", "workstation", "kali", "wazuh", "router"]
    template: str
    ip: str
    vlan: int = 10
    os_version: str = ""
    fqdn: str = ""
    services: list[str] = Field(default_factory=list)  # ["ldap","kerberos","smb"...]
    credentials: list[Credential] = Field(default_factory=list)


class PlantedVuln(BaseModel):
    """一個種下的弱點,連同它的測試座標。"""
    id: int
    name: str
    category: str
    kind: str                       # plant / precondition
    target: dict = Field(default_factory=dict)   # 被種在哪 (host/account/params)
    params: dict = Field(default_factory=dict)   # config_dialog 收集的參數
    has_template: bool = False
    bh_cypher: Optional[str] = None
    test_case_id: Optional[str] = None           # 對應 qa_testcases 的 id


class ServiceEndpoint(BaseModel):
    name: str                       # "ldap" / "kerberos" / "wazuh_api" / "bloodhound"
    host: str
    ip: str
    port: int
    scheme: str = ""                # "ldap" / "https" / ...


class RangeManifest(BaseModel):
    schema_version: str = "1.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    range_id: str = "range"
    ad_domain: str = "range.local"
    ad_version: str = ""
    network_cidr: str = "10.2.10.0/24"
    machines: list[Machine] = Field(default_factory=list)
    planted_vulns: list[PlantedVuln] = Field(default_factory=list)
    endpoints: list[ServiceEndpoint] = Field(default_factory=list)
    qa_credential: Optional[Credential] = None   # QA 犧牲帳號 (經身分驗證測試用)
    notes: str = ""

    # 便利存取
    def dc(self) -> Optional[Machine]:
        return next((m for m in self.machines if m.role == "dc"), None)

    def dc_ip(self) -> str:
        d = self.dc()
        return d.ip if d else ""

    def windows_hosts(self) -> list[Machine]:
        return [m for m in self.machines if m.role in ("dc", "server", "workstation")]


# ═══════════════════════════════════════════════════════════════════════════
# 產生器: 從 topology + 已種弱點 + 參數 組出 manifest
# ═══════════════════════════════════════════════════════════════════════════
def _base_octet(cidr: str) -> int:
    m = re.match(r"\d+\.(\d+)\.", cidr or "")
    return int(m.group(1)) if m else 2


def build_manifest(
    topology: dict,
    buildable_vulns: list[dict],
    vuln_params: dict,
    wazuh_ip: str = "",
    qa_user: str = "qa_tester",
    qa_pass: str = "QaTester!2026",
    range_id: str = "range",
) -> RangeManifest:
    """
    topology         : master 產出的拓撲 (含 ad_domain / machines...)
    buildable_vulns  : plan 階段確認的可種植弱點 (enriched dicts)
    vuln_params      : config_dialog 收集的參數 {str(id): {...}}
    """
    domain = topology.get("ad_domain", "range.local")
    seg = _base_octet(topology.get("network_cidr", "10.2.10.0/24"))
    vlan_base = 10

    # ---- 機器 ----
    machines: list[Machine] = []
    for m in topology.get("machines", []):
        ip = f"10.{seg}.{m.get('vlan', vlan_base)}.{m['ip_last_octet']}"
        fqdn = f"{m['hostname']}.{domain}".lower()
        services, creds = [], []
        if m["role"] == "dc":
            services = ["ldap", "ldaps", "kerberos", "smb", "dns", "global-catalog"]
            creds = [
                Credential(username="Administrator", domain=domain,
                           kind="domain_admin", sensitive=True,
                           note="DA — QA 不應使用,除非白箱驗證"),
                Credential(username=qa_user, password=qa_pass, domain=domain,
                           kind="qa_sacrificial", sensitive=True,
                           note="QA 犧牲帳號 (標準使用者權限)"),
            ]
        elif m["role"] in ("server", "workstation"):
            services = ["smb", "rpc", "winrm", "wazuh-agent"]
            creds = [Credential(username="Administrator", kind="local_admin",
                                sensitive=True, note="本機 admin")]
        elif m["role"] == "wazuh":
            services = ["wazuh-api", "wazuh-dashboard"]
        elif m["role"] == "kali":
            services = ["attack-host"]
        machines.append(Machine(
            hostname=m["hostname"], role=m["role"], template=m["template"],
            ip=ip, vlan=m.get("vlan", vlan_base), os_version=m.get("os", ""),
            fqdn=fqdn, services=services, credentials=creds,
        ))

    # 若拓撲沒有 wazuh 機器但有給 wazuh_ip,補一台
    if wazuh_ip and not any(m.role == "wazuh" for m in machines):
        machines.append(Machine(
            hostname="WAZUH", role="wazuh", template="debian-12-x64-server",
            ip=wazuh_ip, vlan=vlan_base, services=["wazuh-api", "wazuh-dashboard"],
        ))

    # ---- 種下的弱點 (連結測試座標) ----
    dc = next((m for m in machines if m.role == "dc"), None)
    planted: list[PlantedVuln] = []
    for v in buildable_vulns:
        p = vuln_params.get(str(v["id"]), {})
        # 推斷「種在哪」
        target: dict = {}
        for key in ("user", "svc_user", "target_user", "grantee",
                    "computer", "target_computer", "allowed_account"):
            if key in p:
                target[key] = p[key]
        if not target and dc:
            target["host"] = dc.hostname  # 預設種在 DC
        planted.append(PlantedVuln(
            id=v["id"], name=v["name"], category=v["category"], kind=v["kind"],
            target=target, params=p, has_template=v.get("has_template", False),
            bh_cypher=v.get("bh_cypher"),
            test_case_id=f"tc-{v['id']}",
        ))

    # ---- 服務端點 ----
    endpoints: list[ServiceEndpoint] = []
    if dc:
        endpoints += [
            ServiceEndpoint(name="ldap", host=dc.fqdn, ip=dc.ip, port=389, scheme="ldap"),
            ServiceEndpoint(name="ldaps", host=dc.fqdn, ip=dc.ip, port=636, scheme="ldaps"),
            ServiceEndpoint(name="kerberos", host=dc.fqdn, ip=dc.ip, port=88, scheme="kerberos"),
            ServiceEndpoint(name="smb", host=dc.fqdn, ip=dc.ip, port=445, scheme="smb"),
        ]
    wz = next((m for m in machines if m.role == "wazuh"), None)
    if wz:
        endpoints.append(ServiceEndpoint(name="wazuh_api", host=wz.hostname,
                                         ip=wz.ip, port=55000, scheme="https"))

    return RangeManifest(
        range_id=range_id, ad_domain=domain,
        ad_version=topology.get("ad_version", ""),
        network_cidr=topology.get("network_cidr", "10.2.10.0/24"),
        machines=machines, planted_vulns=planted, endpoints=endpoints,
        qa_credential=Credential(username=qa_user, password=qa_pass, domain=domain,
                                 kind="qa_sacrificial", sensitive=True),
        notes="Sensitive: contains plaintext credentials. chmod 600. Do not commit.",
    )


def save_manifest(manifest: RangeManifest, path: str = "range_manifest.json") -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, ensure_ascii=False, indent=2)
    try:
        os.chmod(path, 0o600)  # 敏感檔,鎖權限
    except OSError:
        pass
    return path


def load_manifest(path: str = "range_manifest.json") -> RangeManifest:
    with open(path, encoding="utf-8") as f:
        return RangeManifest.model_validate(json.load(f))


def redact(manifest: RangeManifest) -> dict:
    """回傳遮蔽敏感欄位的 dict (供顯示/日誌)。"""
    d = manifest.model_dump()
    def _mask_creds(creds):
        for c in creds:
            if c.get("sensitive"):
                if c.get("password"):
                    c["password"] = "***REDACTED***"
                if c.get("ntlm_hash"):
                    c["ntlm_hash"] = "***REDACTED***"
    for m in d.get("machines", []):
        _mask_creds(m.get("credentials", []))
    if d.get("qa_credential"):
        _mask_creds([d["qa_credential"]])
    return d


if __name__ == "__main__":
    # 自我測試
    topo = {
        "ad_domain": "lab.local", "ad_version": "Windows Server 2019",
        "network_cidr": "10.3.10.0/24",
        "machines": [
            {"hostname": "DC01", "role": "dc", "template": "win2019-server-x64",
             "vlan": 10, "ip_last_octet": 10},
            {"hostname": "WS01", "role": "workstation",
             "template": "win11-22h2-x64-enterprise", "vlan": 10, "ip_last_octet": 21},
        ],
    }
    vulns = [cat.get(1), cat.get(2), cat.get(66)]
    params = {"1": {"user": "svc_backup"},
              "2": {"svc_user": "svc_sql", "password": "Welcome1", "spn": "MSSQLSvc/DC01:1433"},
              "66": {"grantee": "svc_backup"}}
    mf = build_manifest(topo, vulns, params, wazuh_ip="10.3.10.250")
    print(json.dumps(redact(mf), ensure_ascii=False, indent=2)[:1500])
    print("...")
    print("machines:", [m.hostname for m in mf.machines])
    print("dc_ip:", mf.dc_ip())
    print("planted:", [(v.id, v.test_case_id, v.target) for v in mf.planted_vulns])
