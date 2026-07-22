"""
windows_ad_cyberrange_agent.py
================================
Windows AD Cyber Range AI Agent — LangGraph LLM + MCP 版

LLM 用途:
  1. master  : 自然語言 → 結構化靶場拓撲
  2. dialog  : 多輪對話逐弱點問參數 → LLM 萃取自然語言回答
  3. drafting: 沒模板的可種植項 → LLM 起草 Ansible tasks

MCP 整合:
  deploy_node 透過 langchain-mcp-adapters 呼叫 ludus-mcp (190+ tools),
  取代直接 subprocess 呼叫 ludus CLI。支援兩種模式:
    --ludus-mcp-cmd   stdio 模式 (本機啟動 ludus-mcp subprocess)
    --ludus-mcp-url   HTTP 模式 (連到已執行的 ludus-mcp server)
  未設定 MCP 時,自動 fallback 到 ludus CLI。

流程:
  START → master → plan → config_dialog (多輪自迴圈) → worker → deploy → validate → END

操作指令 (所有 HITL 關卡通用):
  approve / Enter   確認繼續
  back              退回上一步
  (其他)            依該關卡的說明處理

依賴:
  pip install "langgraph>=1.0" langchain pydantic requests pyyaml
  pip install langchain-mcp-adapters          # MCP 整合
  地端 LLM: pip install langchain-openai 或 langchain-ollama
  ludus-mcp: pip install ludus-fastmcp 或 npx -y @badsectorlabs/ludus-mcp
  同資料夾: ad_vuln_catalog.py + ad_vuln_catalog.json

⚠️ 僅於隔離靶場種植刻意弱設定供防禦演練。不產出攻擊工具本身。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
from typing import Annotated, Any, Literal, TypedDict

import requests
import yaml
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

import ad_vuln_catalog as cat


# ═══════════════════════════════════════════════════════════════════════════
# 0-a. Ludus MCP client (lazy init)
# ═══════════════════════════════════════════════════════════════════════════
_MCP_CLIENT = None
_MCP_TOOLS: dict[str, Any] = {}  # name -> LangChain BaseTool


async def _init_mcp_client():
    """
    用 langchain-mcp-adapters 連接 ludus-mcp。
    支援 stdio (本機 subprocess) 和 http (遠端/daemon) 兩種 transport。
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    mcp_cmd = os.environ.get("LUDUS_MCP_CMD")     # e.g. "ludus-fastmcp" or "npx -y @badsectorlabs/ludus-mcp"
    mcp_url = os.environ.get("LUDUS_MCP_URL")      # e.g. "http://localhost:8000/mcp"
    mcp_args = os.environ.get("LUDUS_MCP_ARGS", "")  # extra args for stdio cmd

    if not mcp_cmd and not mcp_url:
        return None

    config: dict[str, Any] = {}
    if mcp_url:
        config["ludus"] = {"url": mcp_url, "transport": "http"}
    elif mcp_cmd:
        parts = mcp_cmd.split()
        extra = mcp_args.split() if mcp_args else []
        config["ludus"] = {
            "command": parts[0],
            "args": parts[1:] + extra,
            "transport": "stdio",
        }
        # 傳遞 Ludus API key / URL 給 ludus-mcp subprocess
        env = dict(os.environ)
        if os.environ.get("LUDUS_API_KEY"):
            env["LUDUS_API_KEY"] = os.environ["LUDUS_API_KEY"]
        if os.environ.get("LUDUS_URL"):
            env["LUDUS_URL"] = os.environ["LUDUS_URL"]
        config["ludus"]["env"] = env

    client = MultiServerMCPClient(config)
    return client


def _get_mcp_tools_sync() -> dict[str, Any]:
    """同步取得 MCP tools dict (name -> BaseTool)。快取以避免重複連線。"""
    global _MCP_CLIENT, _MCP_TOOLS
    if _MCP_TOOLS:
        return _MCP_TOOLS
    if _MCP_CLIENT is None:
        _MCP_CLIENT = asyncio.get_event_loop().run_until_complete(_init_mcp_client())
    if _MCP_CLIENT is None:
        return {}
    tools = asyncio.get_event_loop().run_until_complete(_MCP_CLIENT.get_tools())
    _MCP_TOOLS = {t.name: t for t in tools}
    return _MCP_TOOLS


def _call_mcp_tool(name: str, args: dict | None = None) -> str:
    """呼叫單一 MCP tool,回傳結果字串。"""
    tools = _get_mcp_tools_sync()
    if name not in tools:
        return f"[MCP] tool '{name}' not found. Available: {sorted(tools)[:20]}..."
    try:
        result = tools[name].invoke(args or {})
        return str(result)
    except Exception as e:
        return f"[MCP] {name} failed: {e}"


def _has_mcp() -> bool:
    """MCP 是否已設定 (lazy: 只看環境變數,不觸發連線)。"""
    return bool(os.environ.get("LUDUS_MCP_CMD") or os.environ.get("LUDUS_MCP_URL"))


# ═══════════════════════════════════════════════════════════════════════════
# 0. LLM
# ═══════════════════════════════════════════════════════════════════════════
def build_llm():
    backend = os.environ.get("CYBERRANGE_LLM_BACKEND", "local-openai")
    if backend == "local-openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
            model=os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            temperature=0,
        )
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.environ.get("LLM_MODEL", "qwen2.5:32b"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    return init_chat_model(
        os.environ.get("CYBERRANGE_MODEL", "anthropic:claude-sonnet-4-6"), temperature=0
    )


_LLM = None


def get_llm():
    global _LLM
    if _LLM is None:
        _LLM = build_llm()
    return _LLM


def _safe_str(s: str) -> str:
    """清除 surrogate 字元,避免 httpx UTF-8 encode 爆掉。"""
    return s.encode("utf-8", errors="replace").decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# 1. 導航: back 指令與節點順序
# ═══════════════════════════════════════════════════════════════════════════
# 定義流程順序,每個 stage 對應的節點名稱,以及「上一步」的目標節點。
# config_dialog 的 back 分兩種:已問到第 2+ 個弱點 → 回到上一個弱點 (自迴圈);
#                              第 1 個弱點 → 回到 plan。
STAGE_TO_NODE = {
    "confirm_topology": "master",
    "select_vulns": "master",
    "confirm_plan": "plan",
    "config_dialog": "config_dialog",
    "confirm_deploy": "worker",
}

# stage → 退回到哪個節點 (None = 最前面,不再退)
BACK_TARGET = {
    "confirm_topology": None,       # 已是第一步
    "select_vulns": "master",       # 回到確認拓撲 (重跑 master)
    "confirm_plan": "master",       # 回到選弱點 (重跑 master)
    "config_dialog": "plan",        # 回到確認計畫 (或上一個弱點)
    "confirm_deploy": "config_dialog",  # 回到對話
}


def _is_back(response: Any) -> bool:
    return isinstance(response, str) and response.strip().lower() == "back"


# ═══════════════════════════════════════════════════════════════════════════
# 2. 資料模型
# ═══════════════════════════════════════════════════════════════════════════
class MachineSpec(BaseModel):
    hostname: str = Field(description="VM hostname e.g. DC01")
    role: Literal["dc", "server", "workstation", "kali", "router"]
    template: str = Field(description="Ludus template e.g. win2019-server-x64")
    vlan: int = 10
    ip_last_octet: int
    ram_gb: int = 4
    cpus: int = 2


class Topology(BaseModel):
    ad_domain: str = Field(description="AD domain FQDN e.g. range.local")
    ad_version: str = Field(description="Windows Server version e.g. 2019")
    network_cidr: str = Field(description="Range network CIDR e.g. 10.2.10.0/24")
    machines: list[MachineSpec]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Graph 狀態
# ═══════════════════════════════════════════════════════════════════════════
class CyberRangeState(TypedDict):
    messages: Annotated[list, add_messages]
    topology: dict
    selected_ids: list[int]
    buildable: list[dict]
    documented: list[dict]
    vuln_params: dict
    dialog_progress: int
    vuln_role_yaml: str
    trainee_playbook: str
    range_config_yaml: str
    wazuh_manager_ip: str
    deploy_result: str
    validation_report: str


# ═══════════════════════════════════════════════════════════════════════════
# 4. master_agent
# ═══════════════════════════════════════════════════════════════════════════
MASTER_SYS = (
    "You are a cyber range architect. Convert the user's natural language request "
    "into a structured Windows AD range topology (JSON). Use reasonable Ludus template "
    "names like win2019-server-x64, win11-22h2-x64-enterprise, kali-x64-desktop-template. "
    "Do not fabricate fields the user did not mention."
)


def extract_topology(user_req: str) -> Topology:
    messages = [{"role": "system", "content": _safe_str(MASTER_SYS)},
                {"role": "user", "content": _safe_str(user_req)}]
    llm = get_llm()
    method = os.environ.get("LLM_STRUCTURED_METHOD")
    # 第一次嘗試 structured output
    error_msg = None
    try:
        binder = (llm.with_structured_output(Topology, method=method)
                  if method else llm.with_structured_output(Topology))
        return binder.invoke(messages)
    except Exception as e1:
        error_msg = str(e1)
    # 第二次嘗試 JSON fallback
    try:
        schema = json.dumps(Topology.model_json_schema(), ensure_ascii=True)
        raw = llm.invoke(messages + [{"role": "user",
              "content": _safe_str(
                  "Output ONLY valid JSON matching this schema, no explanation, no ```:\n" + schema
              )}]).content.strip()
        for fence in ("```json", "```"):
            raw = raw.removeprefix(fence)
        return Topology.model_validate_json(raw.removesuffix("```").strip())
    except Exception as e2:
        error_msg = f"structured: {error_msg} | json: {e2}"
    # 兩次都失敗 → 回傳 None + 錯誤訊息,由呼叫端決定怎麼處理
    raise RuntimeError(error_msg)


def _catalog_menu() -> str:
    lines = ["弱點目錄 (輸入編號多選,可用逗號/範圍/類別/profile):"]
    cats = cat.by_category()
    for c in sorted(cats):
        name = cats[c][0]["category_name"]
        lines.append(f"\n== {c} -- {name} ==")
        for it in cats[c]:
            flag = {"plant": "種", "precondition": "前置",
                    "technique": "技", "default_present": "預設"}[it["kind"]]
            tpl = "*" if it["has_template"] else " "
            lines.append(f"  {it['id']:>3}{tpl}[{flag}] {it['name']}")
    lines.append("\n(* = has Ansible template; back = go back)")
    lines.append("profiles: " + ", ".join(cat.PROFILES.keys()))
    lines.append("example: '1,2,15-20,66' or 'profile:goad-like' or 'G:all,136,160'")
    return "\n".join(lines)


def parse_selection(text: str) -> list[int]:
    ids: set[int] = set()
    by_cat = cat.by_category()
    for tok in re.split(r"[,\s]+", text.strip()):
        if not tok:
            continue
        low = tok.lower()
        if low.startswith("profile:"):
            low = low.split(":", 1)[1]
        if low in cat.PROFILES:
            ids.update(cat.PROFILES[low])
            continue
        m = re.fullmatch(r"([A-Za-z]):all", tok)
        if m:
            ids.update(it["id"] for it in by_cat.get(m.group(1).upper(), []))
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            ids.update(range(int(m.group(1)), int(m.group(2)) + 1))
            continue
        if tok.isdigit():
            ids.add(int(tok))
    valid = {it["id"] for it in cat.load_catalog()}
    return sorted(i for i in ids if i in valid)


def master_node(state: CyberRangeState) -> Command[Literal["master", "plan"]]:
    user_req = state["messages"][-1].content

    # 嘗試 LLM 解析; 失敗時讓操作者手動補拓撲
    try:
        topo = extract_topology(user_req).model_dump()
    except RuntimeError as e:
        # 錯誤容錯: 不中斷,讓操作者手動輸入 JSON
        fix = interrupt({
            "stage": "confirm_topology",
            "prompt": (f"LLM 解析拓撲失敗: {e}\n"
                       "請直接貼上 topology JSON,或輸入 'back' 重新描述需求。"),
            "error": str(e),
        })
        if _is_back(fix):
            return Command(goto="master")
        try:
            topo = json.loads(fix) if isinstance(fix, str) else fix
        except (json.JSONDecodeError, TypeError):
            return Command(goto="master")  # 解析再失敗就重跑

    # HITL: 確認拓撲 (支援 back)
    d1 = interrupt({
        "stage": "confirm_topology",
        "prompt": "確認/修改靶場拓撲。approve=繼續 | back=重新輸入需求 | 或貼修改後 JSON",
        "topology": topo,
    })
    if _is_back(d1):
        return Command(goto="master")
    if isinstance(d1, dict):
        topo = d1
    elif isinstance(d1, str) and d1.strip().lower() != "approve":
        try:
            topo = json.loads(d1)
        except json.JSONDecodeError:
            pass

    # HITL: 選弱點 (支援 back → 回到確認拓撲,即重跑 master)
    d2 = interrupt({
        "stage": "select_vulns",
        "prompt": "選擇要種進靶場的弱點 (編號多選)。back=回到拓撲確認",
        "menu": _catalog_menu(),
    })
    if _is_back(d2):
        return Command(update={"topology": topo}, goto="master")
    selected = parse_selection(d2 if isinstance(d2, str) else ",".join(map(str, d2 or [])))
    return Command(update={"topology": topo, "selected_ids": selected}, goto="plan")


# ═══════════════════════════════════════════════════════════════════════════
# 5. plan_agent
# ═══════════════════════════════════════════════════════════════════════════
def plan_node(state: CyberRangeState) -> Command[Literal["master", "config_dialog"]]:
    resolved = cat.resolve_ids(state["selected_ids"])
    buildable = [r for r in resolved if r["buildable"]]
    documented = [r for r in resolved if not r["buildable"]]
    summary = {
        "total_selected": len(resolved),
        "to_build": [{"id": r["id"], "name": r["name"], "kind": r["kind"],
                      "has_template": r["has_template"]} for r in buildable],
        "documentation_only": [{"id": r["id"], "name": r["name"],
                                "kind": r["kind"]} for r in documented],
    }
    d = interrupt({
        "stage": "confirm_plan",
        "prompt": "確認計畫。approve=繼續 | back=回到選弱點 | 或回傳要保留的 id list (JSON)",
        "plan": summary,
    })
    if _is_back(d):
        return Command(goto="master")
    if isinstance(d, list):
        keep = set(d)
        buildable = [r for r in buildable if r["id"] in keep]
        documented = [r for r in documented if r["id"] in keep]
    elif isinstance(d, str) and d.strip().lower() not in ("", "approve"):
        try:
            keep = set(json.loads(d))
            buildable = [r for r in buildable if r["id"] in keep]
            documented = [r for r in documented if r["id"] in keep]
        except json.JSONDecodeError:
            pass
    return Command(update={"buildable": buildable, "documented": documented,
                           "dialog_progress": 0, "vuln_params": {}},
                   goto="config_dialog")


# ═══════════════════════════════════════════════════════════════════════════
# 6. config_dialog — 多輪 + back + 錯誤容錯
# ═══════════════════════════════════════════════════════════════════════════
DIALOG_ASK_SYS = _safe_str(
    "你是 Windows AD 靶場設定助手。操作者正在為隔離訓練靶場佈建刻意的弱設定。\n"
    "規則: 用繁體中文提問,說明弱點用途與攻擊路徑,列出需要的參數附預設值建議。\n"
    "若不需額外參數,告訴操作者按 Enter。不要輸出程式碼。"
)

DIALOG_EXTRACT_SYS = _safe_str(
    "Extract structured params from operator's answer. Return ONLY a JSON object. "
    "If operator said 'use default' or empty, return {}. No ``` or explanation."
)

DRAFT_SYS = _safe_str(
    "Write Ansible tasks for an isolated Windows AD training range to plant a deliberate "
    "misconfiguration (same nature as GOAD lab setup, NOT an attack tool). "
    "Use ansible.windows.win_shell with PowerShell/RSAT on DC. "
    "Mark unknowns with # TODO(operator). Each task needs tags: [vuln, \"id-<N>\"]."
)


def _llm_ask_question(item: dict, topo: dict) -> str:
    params = cat.TEMPLATES[item["id"]]["params"] if item["id"] in cat.TEMPLATES else []
    prompt = _safe_str(
        f"Vuln: #{item['id']} {item['name']}\n"
        f"Description: {item['weakness']}\n"
        f"Required params: {params if params else '(none)'}\n"
        f"Topology: domain={topo.get('ad_domain','range.local')}, "
        f"machines={[m['hostname'] for m in topo.get('machines',[])]}\n"
        f"Generate question in Traditional Chinese."
    )
    try:
        return get_llm().invoke([
            {"role": "system", "content": DIALOG_ASK_SYS},
            {"role": "user", "content": prompt}
        ]).content.strip()
    except Exception as e:
        # 錯誤容錯: LLM 失敗時回傳靜態提問 + 錯誤提示
        fallback = f"[LLM 暫時無法生成提問: {e}]\n"
        if params:
            fallback += (f"弱點 #{item['id']} {item['name']} 需要以下參數:\n"
                         + "\n".join(f"  - {p}" for p in params)
                         + "\n請逐一提供,或按 Enter 使用預設,或輸入 back 退回。")
        else:
            fallback += f"弱點 #{item['id']} {item['name']}: 無需額外參數,按 Enter 繼續。"
        return fallback


def _llm_extract_params(item: dict, human_answer: str) -> dict:
    if not human_answer.strip():
        return {}
    params = cat.TEMPLATES[item["id"]]["params"] if item["id"] in cat.TEMPLATES else []
    prompt = _safe_str(
        f"Vuln: #{item['id']} {item['name']}\n"
        f"Expected params: {params}\n"
        f"Operator answer: {human_answer}\n"
        f"Extract JSON."
    )
    try:
        raw = get_llm().invoke([
            {"role": "system", "content": DIALOG_EXTRACT_SYS},
            {"role": "user", "content": prompt}
        ]).content.strip()
        for fence in ("```json", "```"):
            raw = raw.removeprefix(fence)
        return json.loads(raw.removesuffix("```").strip())
    except Exception:
        try:
            return json.loads(human_answer)
        except json.JSONDecodeError:
            return {"_raw": human_answer}


def _render_templated(item: dict, params: dict) -> str:
    tpl = cat.TEMPLATES[item["id"]]["tasks"]
    safe = {p: params.get(p, f"CHANGE_ME_{p}")
            for p in cat.TEMPLATES[item["id"]]["params"]}
    return tpl.format(**safe)


def _llm_draft_tasks(item: dict, params: dict) -> str:
    prompt = _safe_str(
        f"Vuln: #{item['id']} {item['name']} -- {item['weakness']}\n"
        f"Operator params: {json.dumps(params, ensure_ascii=True)}\n"
    )
    try:
        out = get_llm().invoke([
            {"role": "system", "content": DRAFT_SYS},
            {"role": "user", "content": prompt}
        ]).content.strip()
        for fence in ("```yaml", "```"):
            out = out.removeprefix(fence)
        return out.removesuffix("```").strip()
    except Exception as e:
        return (f"# LLM draft failed ({e}); manually add #{item['id']} {item['name']}\n"
                f"- name: \"[#{item['id']} {item['name']}] TODO\"\n"
                f"  ansible.builtin.debug:\n"
                f"    msg: \"{item['weakness']}\"\n"
                f"  tags: [vuln, \"id-{item['id']}\"]")


def config_dialog_node(state: CyberRangeState) -> Command[
        Literal["config_dialog", "worker", "plan"]]:
    """
    多輪對話 + back 支援:
      - back 且 progress==0 → 回到 plan
      - back 且 progress>0  → 回到上一個弱點 (dialog_progress - 1)
    """
    buildable = state["buildable"]
    progress = state.get("dialog_progress", 0)
    params = dict(state.get("vuln_params") or {})
    topo = state["topology"]

    if progress < len(buildable):
        item = buildable[progress]
        question = _llm_ask_question(item, topo)
        human = interrupt({
            "stage": "config_dialog",
            "prompt": question + "\n\n(back=退回上一步)",
            "current_vuln": {"id": item["id"], "name": item["name"]},
            "progress": f"{progress + 1}/{len(buildable)}",
        })

        # back 處理
        if _is_back(human):
            if progress == 0:
                return Command(goto="plan")      # 回到確認計畫
            else:
                return Command(                   # 回到上一個弱點
                    update={"dialog_progress": progress - 1},
                    goto="config_dialog"
                )

        # LLM 萃取參數
        extracted = _llm_extract_params(item, human if isinstance(human, str) else "")
        params[str(item["id"])] = extracted

        if progress + 1 < len(buildable):
            return Command(
                update={"vuln_params": params, "dialog_progress": progress + 1},
                goto="config_dialog"
            )

    # 全部問完 → 組 role
    domain = topo.get("ad_domain", "range.local")
    blocks = [
        "---",
        "# roles/ludus_ad_vulns/tasks/main.yml",
        f"# ad_domain: {{{{ ad_domain | default('{domain}') }}}}",
    ]
    for r in buildable:
        p = params.get(str(r["id"]), {})
        if r["id"] in cat.TEMPLATES:
            blocks.append(_render_templated(r, p))
        else:
            blocks.append(_llm_draft_tasks(r, p))
    role_yaml = "\n".join(blocks) + "\n"

    pb = ["# Trainee Playbook", ""]
    for r in state.get("documented", []):
        pb.append(f"- **#{r['id']} {r['name']}** ({r['kind']}): {r['weakness']}")
    playbook = "\n".join(pb) + "\n"

    return Command(
        update={"vuln_params": params, "dialog_progress": progress + 1,
                "vuln_role_yaml": role_yaml, "trainee_playbook": playbook},
        goto="worker"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. worker (Ludus config + Wazuh) + back
# ═══════════════════════════════════════════════════════════════════════════
def _base_octet(cidr: str) -> int:
    m = re.match(r"\d+\.(\d+)\.", cidr or "")
    return int(m.group(1)) if m else 2


def _render_ludus_config(topo: dict, wazuh_mgr_octet: int) -> tuple[str, str]:
    vlan_base = 10
    seg = _base_octet(topo.get("network_cidr", "10.2.10.0/24"))
    wazuh_ip = f"10.{seg}.{vlan_base}.{wazuh_mgr_octet}"
    vms = []
    for m in topo.get("machines", []):
        vm: dict[str, Any] = {
            "vm_name": "{{ range_id }}-" + m["hostname"],
            "hostname": "{{ range_id }}-" + m["hostname"],
            "template": m["template"],
            "vlan": m.get("vlan", vlan_base),
            "ip_last_octet": m["ip_last_octet"],
            "ram_gb": m.get("ram_gb", 4),
            "cpus": m.get("cpus", 2),
        }
        if m["role"] == "dc":
            vm["windows"] = {"install_additional_tools": True}
            vm["domain"] = {"fqdn": topo["ad_domain"], "role": "primary-dc"}
            vm["roles"] = ["ludus_ad_vulns"]
            vm["role_vars"] = {"ad_domain": topo["ad_domain"]}
        elif m["role"] in ("server", "workstation"):
            vm["windows"] = {}
            vm["domain"] = {"fqdn": topo["ad_domain"], "role": "member"}
            vm["roles"] = ["wazuh_agent"]
            vm["role_vars"] = {"wazuh_manager_ip": wazuh_ip}
        elif m["role"] == "kali":
            vm["linux"] = True
        vms.append(vm)
    vms.append({
        "vm_name": "{{ range_id }}-WAZUH",
        "hostname": "{{ range_id }}-WAZUH",
        "template": "debian-12-x64-server",
        "vlan": vlan_base, "ip_last_octet": wazuh_mgr_octet,
        "ram_gb": 8, "cpus": 4, "linux": True,
        "roles": ["wazuh_manager"],
    })
    doc = {"# yaml-language-server": "$schema=https://docs.ludus.cloud/schemas/range-config.json",
           "ludus": vms}
    hdr = f"# Ludus range config (lab use only). Wazuh @ {wazuh_ip}\n"
    return hdr + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), wazuh_ip


def worker_node(state: CyberRangeState) -> Command[
        Literal["deploy", "config_dialog", "__end__"]]:
    wazuh_octet = int(os.environ.get("WAZUH_MANAGER_OCTET", "250"))
    range_yaml, wazuh_ip = _render_ludus_config(state["topology"], wazuh_octet)
    role_dir = os.path.join(os.environ.get("LUDUS_ROLES_DIR", "./roles"),
                            "ludus_ad_vulns", "tasks")
    try:
        os.makedirs(role_dir, exist_ok=True)
        with open(os.path.join(role_dir, "main.yml"), "w", encoding="utf-8") as f:
            f.write(state.get("vuln_role_yaml", ""))
    except OSError:
        pass

    d = interrupt({
        "stage": "confirm_deploy",
        "prompt": "最終產物。deploy=部署 | cancel=取消 | back=回到參數對話",
        "range_config_yaml": range_yaml,
        "vuln_role_yaml": state.get("vuln_role_yaml", ""),
        "wazuh_manager_ip": wazuh_ip,
        "trainee_playbook": state.get("trainee_playbook", ""),
    })
    update = {"range_config_yaml": range_yaml, "wazuh_manager_ip": wazuh_ip}
    if _is_back(d):
        # 回到 config_dialog 重填參數 (從第一個弱點開始)
        return Command(update={**update, "dialog_progress": 0}, goto="config_dialog")
    if isinstance(d, str) and d.strip().lower() == "deploy":
        return Command(update=update, goto="deploy")
    return Command(update={**update, "deploy_result": "cancelled by human"}, goto=END)


# ═══════════════════════════════════════════════════════════════════════════
# 8. deploy (MCP → ludus-mcp tools; fallback → ludus CLI)
# ═══════════════════════════════════════════════════════════════════════════
def _ludus(args_list: list[str]) -> list[str]:
    base = ["ludus"]
    if os.environ.get("LUDUS_USER"):
        base += ["--user", os.environ["LUDUS_USER"]]
    return base + args_list


def _deploy_via_mcp(range_config_yaml: str) -> str:
    """透過 ludus-mcp 的 MCP tools 部署。"""
    logs = []
    # 1. update_range_config
    r1 = _call_mcp_tool("update_range_config", {"config": range_config_yaml})
    logs.append(f"[MCP] update_range_config: {r1[:200]}")
    if "error" in r1.lower():
        return "\n".join(logs) + "\n[STOPPED] config update failed"
    # 2. validate_config
    r2 = _call_mcp_tool("validate_config", {})
    logs.append(f"[MCP] validate_config: {r2[:200]}")
    # 3. deploy_range
    r3 = _call_mcp_tool("deploy_range", {})
    logs.append(f"[MCP] deploy_range: {r3[:300]}")
    # 4. configure_wazuh (如果 tool 存在)
    if "configure_wazuh" in _get_mcp_tools_sync():
        r4 = _call_mcp_tool("configure_wazuh", {})
        logs.append(f"[MCP] configure_wazuh: {r4[:200]}")
    return "\n".join(logs)


def _deploy_via_cli(range_config_yaml: str) -> str:
    """Fallback: 直接呼叫 ludus CLI。"""
    cfg = "/tmp/ludus-range-config.yml"
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(range_config_yaml)
    cmds = [_ludus(["range", "config", "set", "-f", cfg]), _ludus(["range", "deploy"])]
    logs = []
    for c in cmds:
        p = subprocess.run(c, capture_output=True, text=True)
        logs.append(f"$ {' '.join(c)}\n{p.stdout}\n{p.stderr}")
        if p.returncode != 0:
            logs.append(f"[FAILED rc={p.returncode}]")
            break
    return "\n".join(logs)


def deploy_node(state: CyberRangeState) -> dict:
    range_yaml = state["range_config_yaml"]
    if os.environ.get("CYBERRANGE_DRY_RUN") == "1":
        method = "MCP" if _has_mcp() else "CLI"
        return {"deploy_result": f"[DRY-RUN via {method}] would deploy range config"}

    if _has_mcp():
        result = _deploy_via_mcp(range_yaml)
    else:
        result = _deploy_via_cli(range_yaml)
    return {"deploy_result": result}


# ═══════════════════════════════════════════════════════════════════════════
# 9. validate
# ═══════════════════════════════════════════════════════════════════════════
def validate_node(state: CyberRangeState) -> dict:
    if state.get("deploy_result", "").startswith("cancelled"):
        return {"validation_report": "skipped (deploy cancelled)"}
    topo = state["topology"]
    dc = next((m for m in topo["machines"] if m["role"] == "dc"), None)
    seg = _base_octet(topo.get("network_cidr", "10.2.10.0/24"))
    dc_ip = f"10.{seg}.10.{dc['ip_last_octet']}" if dc else "DC_IP"
    collect_cmd = (f"bloodhound-python -d {topo['ad_domain']} -u <user> -p <pass> "
                   f"-ns {dc_ip} -c All --zip")
    lines = ["# BloodHound Validation", "",
             f"## Collection\n```\n{collect_cmd}\n```", "",
             "## Per-vuln Cypher queries"]
    for r in state.get("buildable", []):
        q = r.get("bh_cypher")
        if q:
            lines.append(f"\n### #{r['id']} {r['name']}\n```cypher\n{q}\n```")
        else:
            lines.append(f"\n### #{r['id']} {r['name']}\n"
                         f"> No built-in Cypher; verify manually: {r['weakness']}")
    report = "\n".join(lines) + "\n"
    if os.environ.get("CYBERRANGE_DRY_RUN") != "1" and os.environ.get("BH_COLLECT_CREDS"):
        u, p = os.environ["BH_COLLECT_CREDS"].split(":", 1)
        cmd = ["bloodhound-python", "-d", topo["ad_domain"], "-u", u, "-p", p,
               "-ns", dc_ip, "-c", "All", "--zip"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            report += f"\n## Output\n$ {' '.join(cmd[:6])} ...\n{r.stdout}\n{r.stderr}\n"
        except (subprocess.SubprocessError, OSError) as e:
            report += f"\n## Failed\n{e}\n"
    return {"validation_report": report}


# ═══════════════════════════════════════════════════════════════════════════
# 10. 組圖
# ═══════════════════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(CyberRangeState)
    g.add_node("master", master_node)
    g.add_node("plan", plan_node)
    g.add_node("config_dialog", config_dialog_node)
    g.add_node("worker", worker_node)
    g.add_node("deploy", deploy_node)
    g.add_node("validate", validate_node)
    g.add_edge(START, "master")
    g.add_edge("deploy", "validate")
    g.add_edge("validate", END)
    return g.compile(checkpointer=InMemorySaver())


# ═══════════════════════════════════════════════════════════════════════════
# 11. CLI
# ═══════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="windows_ad_cyberrange_agent",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-r", "--request", help="Range requirement (natural language)")
    p.add_argument("--vulns", help="Pre-select vuln IDs e.g. '1,2,15-20'")
    p.add_argument("--profile", choices=list(cat.PROFILES))
    g = p.add_argument_group("LLM backend")
    g.add_argument("--backend", choices=["local-openai", "ollama", "cloud"],
                   default=os.environ.get("CYBERRANGE_LLM_BACKEND", "local-openai"))
    g.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    g.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    g.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"))
    g.add_argument("--ollama-base-url", default=os.environ.get("OLLAMA_BASE_URL"))
    g.add_argument("--cloud-model", default=os.environ.get("CYBERRANGE_MODEL"))
    g.add_argument("--structured-method",
                   choices=["json_schema", "function_calling", "json_mode"],
                   default=os.environ.get("LLM_STRUCTURED_METHOD"))
    g2 = p.add_argument_group("Ludus")
    g2.add_argument("--ludus-user", default=os.environ.get("LUDUS_USER"))
    g2.add_argument("--ludus-mcp-cmd",
                    default=os.environ.get("LUDUS_MCP_CMD"),
                    help="ludus-mcp command for stdio transport (e.g. 'ludus-fastmcp' "
                         "or 'npx -y @badsectorlabs/ludus-mcp')")
    g2.add_argument("--ludus-mcp-url",
                    default=os.environ.get("LUDUS_MCP_URL"),
                    help="ludus-mcp HTTP URL (e.g. http://localhost:8000/mcp)")
    g2.add_argument("--ludus-mcp-args",
                    default=os.environ.get("LUDUS_MCP_ARGS", ""),
                    help="Extra args for ludus-mcp stdio cmd (e.g. '--url https://ludus:8080 --api-key KEY')")
    g2.add_argument("--ludus-api-key", default=os.environ.get("LUDUS_API_KEY"),
                    help="Ludus API key (passed to ludus-mcp subprocess)")
    g2.add_argument("--ludus-url", default=os.environ.get("LUDUS_URL"),
                    help="Ludus server URL (passed to ludus-mcp subprocess)")
    g2.add_argument("--thread-id", default="range-demo-001")
    # dry-run 預設 False (真跑); 要 dry-run 才加 --dry-run
    g2.add_argument("--dry-run", action="store_true",
                    help="Only print Ludus commands, do not actually deploy")
    g2.add_argument("--auto-approve", action="store_true",
                    help="Auto-approve all gates (requires --dry-run)")
    return p


def args_to_env(a) -> None:
    mp = {"CYBERRANGE_LLM_BACKEND": a.backend, "LLM_BASE_URL": a.base_url,
          "LLM_MODEL": a.model, "LLM_API_KEY": a.api_key,
          "OLLAMA_BASE_URL": a.ollama_base_url, "CYBERRANGE_MODEL": a.cloud_model,
          "LLM_STRUCTURED_METHOD": a.structured_method, "LUDUS_USER": a.ludus_user,
          "LUDUS_MCP_CMD": a.ludus_mcp_cmd, "LUDUS_MCP_URL": a.ludus_mcp_url,
          "LUDUS_MCP_ARGS": a.ludus_mcp_args, "LUDUS_API_KEY": a.ludus_api_key,
          "LUDUS_URL": a.ludus_url}
    for k, v in mp.items():
        if v is not None:
            os.environ[k] = v
    if a.dry_run:
        os.environ["CYBERRANGE_DRY_RUN"] = "1"
    # 不再讀環境變數當 dry_run 預設值 — 預設就是真跑


def _auto_reply(stage: str, a) -> str:
    if stage == "confirm_topology":
        return "approve"
    if stage == "select_vulns":
        return (a.profile and f"profile:{a.profile}") or a.vulns or "profile:goad-like"
    if stage == "confirm_plan":
        return "approve"
    if stage == "config_dialog":
        return ""  # Enter = 用預設
    if stage == "confirm_deploy":
        return "deploy"
    return "approve"


def run(a) -> None:
    args_to_env(a)
    if a.auto_approve and not a.dry_run:
        raise SystemExit("[refused] --auto-approve requires --dry-run")

    graph = build_graph()
    config = {"configurable": {"thread_id": a.thread_id}}
    user_request = a.request or input("描述你要的 Windows AD 靶場需求:\n> ")
    result = graph.invoke({"messages": [{"role": "user", "content": user_request}]}, config)

    while "__interrupt__" in result:
        intr = result["__interrupt__"][0].value
        stage = intr.get("stage")
        print("\n" + "=" * 72)
        print(f"[{stage}]")
        if "progress" in intr:
            print(f"(progress: {intr['progress']})")
        if "error" in intr:
            print(f"[ERROR] {intr['error']}")
        print(intr.get("prompt", ""))
        for k in ("topology", "menu", "plan", "current_vuln",
                  "range_config_yaml", "vuln_role_yaml",
                  "trainee_playbook", "wazuh_manager_ip"):
            if k in intr:
                v = intr[k]
                print(f"\n--- {k} ---\n" + (v if isinstance(v, str)
                      else json.dumps(v, ensure_ascii=False, indent=2)))
        print("=" * 72)
        human = _auto_reply(stage, a) if a.auto_approve else input("回覆 (back=退回)> ")
        if a.auto_approve:
            print(f"[auto] {human}")
        result = graph.invoke(Command(resume=human), config)

    print("\n=== deploy result ===\n" + result.get("deploy_result", "(none)"))
    print("\n=== validation report ===\n" + result.get("validation_report", "(none)"))


if __name__ == "__main__":
    run(build_parser().parse_args())
