"""
cyberrange_mcp_server.py
========================
Cyberrange MCP Server (FastMCP) —— 把弱點目錄 + manifest + QA agent 暴露成 MCP tools,
並提供 Prefab web dashboard。可被 Claude Desktop / Cursor 連接,或獨立跑 web 預覽。

暴露的能力:
  【弱點目錄】 list_vulns / get_vuln / search_vulns / list_profiles
  【建靶規劃】 generate_vuln_role / build_range_manifest
  【QA】       qa_validate / qa_test_vuln / qa_full (呼叫 qa_agent 的受約束工具)
  【Dashboard】range_dashboard / vuln_catalog_app / qa_report_app (Prefab UI, app=True)

與 ludus-mcp 的關係:
  本 server 專注「AD 弱點知識 + QA 驗證」。基礎設施操作 (deploy/snapshot/power)
  由 ludus-mcp 負責。可用 FastMCP proxy 掛載 ludus-mcp,讓 client 只連一個 server:
      from fastmcp import FastMCP
      proxy = FastMCP.as_proxy("ludus-mcp-config.json")
      mcp.mount(proxy, prefix="ludus")

依賴:
  pip install fastmcp                    # 3.2+ (含 Prefab apps: pip install fastmcp[apps])
  同資料夾: ad_vuln_catalog.py/.json, range_manifest.py, qa_testcases.py, qa_agent.py

執行:
  stdio (給 MCP client):   python cyberrange_mcp_server.py
  http (獨立/web):          fastmcp run cyberrange_mcp_server.py --transport http --port 8000
  web 預覽 dashboard:       fastmcp dev apps cyberrange_mcp_server.py

⚠️ QA 工具只對 manifest 內的 range 網段執行 (qa_agent 層強制)。
"""

from __future__ import annotations

import json
import os
from typing import Optional

from fastmcp import FastMCP

import ad_vuln_catalog as cat
import range_manifest as rmf
import qa_testcases as tcs

mcp = FastMCP("cyberrange-ad")


# ═══════════════════════════════════════════════════════════════════════════
# 弱點目錄 tools
# ═══════════════════════════════════════════════════════════════════════════
@mcp.tool
def list_vulns(category: Optional[str] = None, kind: Optional[str] = None) -> str:
    """列出 AD 弱點目錄 (248 條)。可依 category (A-T) 或 kind (plant/precondition/
    technique/default_present) 過濾。回傳 JSON 陣列 (id/name/category/kind/has_template)。"""
    items = []
    for it in cat.load_catalog():
        e = cat.enrich(it)
        if category and e["category"] != category.upper():
            continue
        if kind and e["kind"] != kind:
            continue
        items.append({"id": e["id"], "name": e["name"], "category": e["category"],
                      "kind": e["kind"], "has_template": e["has_template"]})
    return json.dumps(items, ensure_ascii=False)


@mcp.tool
def get_vuln(vuln_id: int) -> str:
    """取得單一弱點的完整資訊 (含 weakness 描述、kind、是否有 Ansible 模板、BloodHound cypher)。"""
    try:
        e = cat.get(vuln_id)
    except KeyError:
        return json.dumps({"error": f"vuln #{vuln_id} not found"}, ensure_ascii=False)
    if vuln_id in cat.TEMPLATES:
        e["template_params"] = cat.TEMPLATES[vuln_id]["params"]
    return json.dumps(e, ensure_ascii=False)


@mcp.tool
def search_vulns(keyword: str) -> str:
    """以關鍵字搜尋弱點 (比對 name 與 weakness)。回傳符合的 id/name。"""
    kw = keyword.lower()
    hits = [{"id": it["id"], "name": it["name"]}
            for it in cat.load_catalog()
            if kw in it["name"].lower() or kw in it["weakness"].lower()]
    return json.dumps(hits, ensure_ascii=False)


@mcp.tool
def list_profiles() -> str:
    """列出預建弱點 profile (goad-like / adcs / relay-coercion / cred-weakness) 及其含的弱點 id。"""
    return json.dumps({k: v for k, v in cat.PROFILES.items()}, ensure_ascii=False)


@mcp.tool
def generate_vuln_role(vuln_ids: list[int], params: dict) -> str:
    """為選定的可種植弱點產生 Ansible role tasks (roles/ludus_ad_vulns)。
    params: {"<id>": {"param": "value"}}。回傳 YAML 字串。
    有模板者用模板填值;無模板者回傳 TODO 骨架 (需 LLM 起草或人工補)。"""
    blocks = ["---", "# roles/ludus_ad_vulns/tasks/main.yml (generated via MCP)"]
    for vid in vuln_ids:
        try:
            it = cat.get(vid)
        except KeyError:
            continue
        p = params.get(str(vid), {})
        if vid in cat.TEMPLATES:
            tpl = cat.TEMPLATES[vid]["tasks"]
            safe = {k: p.get(k, f"CHANGE_ME_{k}") for k in cat.TEMPLATES[vid]["params"]}
            blocks.append(tpl.format(**safe))
        else:
            blocks.append(f"- name: \"[#{vid} {it['name']}] TODO(operator) — {it['weakness']}\"\n"
                          f"  ansible.builtin.debug:\n    msg: \"draft me\"\n"
                          f"  tags: [vuln, \"id-{vid}\"]")
    return "\n".join(blocks) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Manifest tools
# ═══════════════════════════════════════════════════════════════════════════
@mcp.tool
def build_range_manifest(topology: dict, vuln_ids: list[int], params: dict,
                         wazuh_ip: str = "", qa_user: str = "qa_tester",
                         qa_pass: str = "QaTester!2026",
                         save_path: str = "range_manifest.json") -> str:
    """從拓撲 + 選定弱點 + 參數,產生 range manifest 並存檔 (QA agent 的輸入契約)。
    回傳遮蔽敏感欄位後的 manifest 摘要。實體檔案含明文帳密,權限設 600。"""
    buildable = [cat.get(v) for v in vuln_ids
                 if cat.get(v)["buildable"]]
    mf = rmf.build_manifest(topology, buildable, params, wazuh_ip=wazuh_ip,
                            qa_user=qa_user, qa_pass=qa_pass)
    rmf.save_manifest(mf, save_path)
    return json.dumps({"saved": save_path,
                       "manifest": rmf.redact(mf)}, ensure_ascii=False)


@mcp.tool
def read_manifest(path: str = "range_manifest.json") -> str:
    """讀取 range manifest (遮蔽敏感欄位)。"""
    try:
        mf = rmf.load_manifest(path)
    except FileNotFoundError:
        return json.dumps({"error": f"{path} not found"}, ensure_ascii=False)
    return json.dumps(rmf.redact(mf), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# QA tools (呼叫 qa_agent 的受約束驗證; 目標鎖 range)
# ═══════════════════════════════════════════════════════════════════════════
def _load_qa(manifest_path: str, mode: str = "blackbox"):
    import qa_agent as qa
    qa._MODE = mode
    qa._load(manifest_path)
    return qa


@mcp.tool
def qa_validate(manifest_path: str = "range_manifest.json") -> str:
    """跑基礎設施驗證 (DNS/LDAP/SMB/Kerberos 可連線)。回傳每項 pass/fail。"""
    qa = _load_qa(manifest_path)
    return qa.run_validation.func()  # 直接呼叫底層函式


@mcp.tool
def qa_test_vuln(vuln_id: int, manifest_path: str = "range_manifest.json",
                 mode: str = "blackbox") -> str:
    """對單一種下的弱點做可利用性驗證 (受約束、目標鎖 range)。
    回傳 status: pass=可利用 / fail=種了打不通 / skip / error。"""
    qa = _load_qa(manifest_path, mode)
    if vuln_id in tcs.BH_VULNS:
        return qa.bloodhound_verify.func(vuln_id)
    return qa.test_vuln.func(vuln_id)


@mcp.tool
def qa_full(manifest_path: str = "range_manifest.json", mode: str = "blackbox") -> str:
    """對 manifest 中所有種下的弱點跑完整 QA (validate + 逐一 test)。回傳彙整結果。
    這是確定性執行 (不經 LLM 編排),適合排程/CI。互動式編排請用 qa_agent CLI。"""
    qa = _load_qa(manifest_path, mode)
    report = {"infra": json.loads(qa.run_validation.func()), "vulns": []}
    for v in qa._MF.get("planted_vulns", []):
        vid = v["id"]
        if vid in tcs.BH_VULNS:
            res = json.loads(qa.bloodhound_verify.func(vid))
        else:
            res = json.loads(qa.test_vuln.func(vid))
        report["vulns"].append({"id": vid, "name": v["name"], **res})
    passed = sum(1 for x in report["vulns"] if x.get("status") == "pass")
    report["summary"] = f"{passed}/{len(report['vulns'])} vulns verified exploitable"
    return json.dumps(report, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# Prefab dashboard apps (app=True → 在 MCP client 對話中渲染互動 UI)
# 需要 pip install fastmcp[apps]; 若未安裝,這些 tool 會回退為 JSON。
# ═══════════════════════════════════════════════════════════════════════════
try:
    from prefab_ui.components import DataTable, DataTableColumn, Column, Heading, Text, Badge
    _HAS_PREFAB = True
except Exception:
    _HAS_PREFAB = False


if _HAS_PREFAB:
    @mcp.tool(app=True)
    def range_dashboard(manifest_path: str = "range_manifest.json"):
        """[UI] 顯示靶場機器清單 dashboard (機器/角色/IP/服務)。"""
        try:
            mf = rmf.load_manifest(manifest_path)
        except FileNotFoundError:
            return Column(children=[Heading("No manifest"), Text(f"{manifest_path} not found")])
        rows = [{"hostname": m.hostname, "role": m.role, "ip": m.ip,
                 "services": ", ".join(m.services)} for m in mf.machines]
        return Column(children=[
            Heading(f"Range: {mf.range_id} ({mf.ad_domain})"),
            Text(f"Network: {mf.network_cidr} | Machines: {len(mf.machines)}"),
            DataTable(
                columns=[DataTableColumn(key="hostname", header="Host", sortable=True),
                         DataTableColumn(key="role", header="Role", sortable=True),
                         DataTableColumn(key="ip", header="IP"),
                         DataTableColumn(key="services", header="Services")],
                rows=rows),
        ])

    @mcp.tool(app=True)
    def vuln_catalog_app(category: str = ""):
        """[UI] 弱點目錄互動表 (可排序/搜尋)。"""
        rows = []
        for it in cat.load_catalog():
            e = cat.enrich(it)
            if category and e["category"] != category.upper():
                continue
            rows.append({"id": e["id"], "name": e["name"], "category": e["category"],
                         "kind": e["kind"], "template": "yes" if e["has_template"] else "-"})
        return DataTable(
            columns=[DataTableColumn(key="id", header="ID", sortable=True),
                     DataTableColumn(key="name", header="Vulnerability", sortable=True),
                     DataTableColumn(key="category", header="Cat", sortable=True),
                     DataTableColumn(key="kind", header="Kind", sortable=True),
                     DataTableColumn(key="template", header="Template")],
            rows=rows)

    @mcp.tool(app=True)
    def qa_report_app(manifest_path: str = "range_manifest.json", mode: str = "blackbox"):
        """[UI] 跑完整 QA 並以 dashboard 呈現每個弱點的 pass/fail。"""
        result = json.loads(qa_full.fn(manifest_path, mode)) if hasattr(qa_full, "fn") \
            else json.loads(qa_full(manifest_path, mode))
        rows = [{"id": v["id"], "name": v["name"], "status": v.get("status", "?"),
                 "evidence": (v.get("evidence", "") or "")[:80]} for v in result["vulns"]]
        return Column(children=[
            Heading("QA Report"),
            Text(result.get("summary", "")),
            DataTable(
                columns=[DataTableColumn(key="id", header="ID", sortable=True),
                         DataTableColumn(key="name", header="Vuln", sortable=True),
                         DataTableColumn(key="status", header="Status", sortable=True),
                         DataTableColumn(key="evidence", header="Evidence")],
                rows=rows),
        ])


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 預設 stdio (給 MCP client);要 http 用: fastmcp run cyberrange_mcp_server.py --transport http
    mcp.run()
