"""
observer_agent.py
=================
觀察員 AI Agent —— 被動記錄型代理,詳細記錄 QA Agent 每步檢測。

功能:
  - 攔截並記錄每個工具呼叫的指令、參數、原始結果
  - 用 LLM 對每步做因果推理分析
  - 產出結構化觀察報告 (JSON + Markdown)

記錄內容:
  ① 指令 (tool name + args)
  ② 原始結果 (tool output)
  ③ 排查方式 (為何執行此測試、預期結果)
  ④ 因果推理 (結果→原因→影響→下一步建議)
  ⑤ 時間戳

設計:
  - Observer 不執行任何操作,只觀察與分析
  - 可獨立設定 LLM 後端 (預設與 QA Agent 共用)
  - 支援兩種整合方式:
    1. wrap_qa_tools(): 包裹 QA 工具,LLM Agent 模式自動記錄
    2. record_step(): 手動記錄,適合確定性 pipeline 模式

依賴:
  pip install "langgraph>=1.0" langchain pydantic
  地端 LLM: langchain-ollama 或 langchain-openai
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# 結構化資料模型
# ═══════════════════════════════════════════════════════════════════════════
class CausalAnalysis(BaseModel):
    """因果推理結構。"""
    observation: str = Field(description="觀察到的現象 (工具回傳了什麼)")
    root_cause: str = Field(description="推測的根因 (為什麼出現這個結果)")
    impact: str = Field(description="影響 (這個結果對靶場驗證有什麼影響)")
    next_step: str = Field(description="建議的下一步 (基於此結果應該做什麼)")


class ObservationRecord(BaseModel):
    """單一步驟的觀察紀錄。"""
    step_id: int = Field(description="步驟編號 (從 1 開始)")
    timestamp: str = Field(description="ISO 格式時間戳")
    tool_name: str = Field(description="工具名稱")
    tool_args: dict = Field(default_factory=dict, description="工具參數")
    raw_result: str = Field(description="工具回傳的原始結果 (JSON 字串)")
    parsed_status: str = Field(default="", description="從結果中解析的狀態 (pass/fail/error/skip)")
    reasoning: str = Field(default="", description="排查方式: 為什麼執行此測試、預期什麼結果")
    causal_analysis: Optional[CausalAnalysis] = Field(default=None, description="因果推理分析")
    execution_context: dict = Field(default_factory=dict, description="執行上下文 (phase, mode 等)")


class ObserverReport(BaseModel):
    """完整的觀察員報告。"""
    report_id: str = Field(description="報告 ID")
    range_id: str = Field(description="靶場 ID")
    mode: str = Field(description="QA 模式 (blackbox/whitebox)")
    generated_at: str = Field(description="報告生成時間")
    observer_backend: str = Field(default="", description="觀察員使用的 LLM 後端")
    observer_model: str = Field(default="", description="觀察員使用的 LLM 模型")
    total_steps: int = Field(default=0, description="總步驟數")
    infra_ok: bool = Field(default=True, description="基礎設施是否正常")
    records: list[ObservationRecord] = Field(default_factory=list, description="所有步驟的觀察紀錄")
    causal_chain_summary: str = Field(default="", description="因果鏈總結 (LLM 生成)")
    overall_summary: str = Field(default="", description="整體觀察總結 (LLM 生成)")


# ═══════════════════════════════════════════════════════════════════════════
# Observer LLM System Prompt
# ═══════════════════════════════════════════════════════════════════════════
OBSERVER_SYSTEM = """你是靶場 QA 的觀察員 AI。你不執行任何操作,只負責觀察、記錄與分析。

對每一步 QA 工具的執行,你必須輸出嚴格的 JSON 格式分析:

{
  "reasoning": "排查方式: 為什麼選擇這個測試？預期結果是什麼？這個測試在整體驗證流程中的位置？",
  "causal_analysis": {
    "observation": "觀察到的現象: 工具回傳了什麼？狀態是 pass/fail/error/skip？",
    "root_cause": "推測的根因: 為什麼出現這個結果？如果失敗,最可能的原因是什麼？",
    "impact": "影響: 這個結果對靶場驗證有什麼影響？會影響其他測試嗎？",
    "next_step": "建議的下一步: 基於此結果,接下來應該做什麼？"
  }
}

分析原則:
1. 基礎設施測試 (DNS/LDAP/SMB/Kerberos) 是前置條件,失敗會影響所有後續弱點測試
2. 弱點測試的 pass 代表「弱點已成功種植且可被利用」,fail 代表「種了但打不通」
3. skip 通常代表沒有直接測試方法或缺少前置條件 (如 QA 帳號)
4. error 代表工具層面出問題 (不是弱點本身的問題)
5. 注意測試之間的依賴關係和因果鏈

你的回覆必須是純 JSON,不要加任何其他文字或 markdown 格式。"""

SUMMARY_SYSTEM = """你是靶場 QA 觀察員。基於所有步驟的觀察紀錄,產出兩段總結:

1. **因果鏈總結** (causal_chain_summary): 串連所有步驟的因果關係,形成完整的推理鏈。
   例如: "DNS 解析正常 → LDAP 可連 → SMB 可連 → AS-REP roast 成功取得 hash (弱點已種植) → Kerberoast 失敗 (SPN 未設定) → 建議重跑 Ansible task..."

2. **整體觀察總結** (overall_summary): 用人類可讀的語言總結整個 QA 過程,包含:
   - 基礎設施狀態
   - 各弱點驗證結果
   - 失敗項的根因分析
   - 整體建議

回覆格式 (純 JSON):
{
  "causal_chain_summary": "...",
  "overall_summary": "..."
}"""


# ═══════════════════════════════════════════════════════════════════════════
# Observer LLM 建構
# ═══════════════════════════════════════════════════════════════════════════
def _build_observer_llm(backend: str, model: Optional[str] = None):
    """為觀察員建構 LLM。支援與 QA Agent 相同的後端選項。"""
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=model or os.environ.get("LLM_MODEL", "gpt-4o"),
            temperature=0,
        )
    if backend == "local-openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
            model=model or os.environ.get("LLM_MODEL", "llama4:scout"),
            temperature=0,
        )
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or os.environ.get("LLM_MODEL", "llama4:scout"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    from langchain.chat_models import init_chat_model
    return init_chat_model(
        os.environ.get("CYBERRANGE_MODEL", "anthropic:claude-sonnet-4-6"),
        temperature=0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ObserverAgent 主類別
# ═══════════════════════════════════════════════════════════════════════════
class ObserverAgent:
    """
    被動記錄型 Agent,觀察並分析 QA Agent 的每步操作。

    使用方式:
      1. LLM Agent 模式: observer.wrap_qa_tools(QA_TOOLS) 包裹工具
      2. Pipeline 模式: observer.record_step(...) 手動記錄

    最終: observer.generate_report(...) 產出 JSON + Markdown 報告
    """

    def __init__(self, backend: str = "ollama", model: Optional[str] = None):
        self.backend = backend
        self.model_name = model or os.environ.get("LLM_MODEL", "")
        self._llm = None  # lazy init
        self._records: list[ObservationRecord] = []
        self._step_counter = 0

    @property
    def llm(self):
        """延遲初始化 LLM (只在需要推理時建構)。"""
        if self._llm is None:
            self._llm = _build_observer_llm(self.backend, self.model_name)
        return self._llm

    # ── 核心: 記錄一步 ────────────────────────────────────────────────────
    def record_step(
        self,
        tool_name: str,
        tool_args: dict,
        raw_result: str,
        context: Optional[dict] = None,
    ) -> ObservationRecord:
        """
        記錄一步 QA 工具呼叫,用 LLM 做因果推理。

        Args:
            tool_name: 工具名稱 (如 test_vuln, run_validation)
            tool_args: 工具參數
            raw_result: 工具回傳的原始 JSON 字串
            context: 額外上下文 (phase, mode 等)

        Returns:
            ObservationRecord: 結構化觀察紀錄
        """
        self._step_counter += 1
        now = datetime.now(timezone.utc).isoformat()

        # 解析 status
        parsed_status = ""
        try:
            result_obj = json.loads(raw_result)
            if isinstance(result_obj, dict):
                parsed_status = result_obj.get("status", "")
            elif isinstance(result_obj, list):
                # run_validation 回傳 list
                statuses = [r.get("status", "") for r in result_obj if isinstance(r, dict)]
                if all(s == "pass" for s in statuses):
                    parsed_status = "all_pass"
                elif any(s == "fail" for s in statuses):
                    parsed_status = "has_failures"
                else:
                    parsed_status = "mixed"
        except (json.JSONDecodeError, AttributeError):
            parsed_status = "parse_error"

        # 用 LLM 做因果推理
        reasoning = ""
        causal = None
        try:
            reasoning, causal = self._llm_analyze(
                tool_name, tool_args, raw_result, parsed_status, context or {}
            )
        except Exception as e:
            reasoning = f"[Observer LLM error: {e}] — 使用規則式備援分析"
            causal = self._rule_based_analysis(tool_name, parsed_status, raw_result)

        record = ObservationRecord(
            step_id=self._step_counter,
            timestamp=now,
            tool_name=tool_name,
            tool_args=tool_args,
            raw_result=raw_result[:2000],  # 截斷過長結果
            parsed_status=parsed_status,
            reasoning=reasoning,
            causal_analysis=causal,
            execution_context=context or {},
        )
        self._records.append(record)
        self._print_step(record)
        return record

    # ── LLM 分析 ─────────────────────────────────────────────────────────
    def _llm_analyze(
        self,
        tool_name: str,
        tool_args: dict,
        raw_result: str,
        parsed_status: str,
        context: dict,
    ) -> tuple[str, CausalAnalysis]:
        """用 LLM 對單步做因果推理分析。"""
        # 構建先前步驟的摘要 (給 LLM 上下文)
        prev_summary = ""
        if self._records:
            prev_lines = []
            for r in self._records[-5:]:  # 最近 5 步
                prev_lines.append(
                    f"  Step {r.step_id}: {r.tool_name} → {r.parsed_status}"
                )
            prev_summary = "先前步驟:\n" + "\n".join(prev_lines)

        prompt = f"""分析以下 QA 工具呼叫:

工具: {tool_name}
參數: {json.dumps(tool_args, ensure_ascii=False)}
狀態: {parsed_status}
結果 (截斷): {raw_result[:1000]}
執行階段: {context.get('phase', 'unknown')}
QA 模式: {context.get('mode', 'unknown')}

{prev_summary}

請輸出 JSON 格式的分析。"""

        messages = [
            {"role": "system", "content": OBSERVER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        resp = self.llm.invoke(messages)
        content = resp.content if hasattr(resp, "content") else str(resp)

        # 解析 LLM 回覆
        try:
            # 嘗試清理可能的 markdown 包裹
            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean
                clean = clean.rsplit("```", 1)[0] if "```" in clean else clean
            data = json.loads(clean)
            reasoning = data.get("reasoning", "")
            ca_data = data.get("causal_analysis", {})
            causal = CausalAnalysis(
                observation=ca_data.get("observation", ""),
                root_cause=ca_data.get("root_cause", ""),
                impact=ca_data.get("impact", ""),
                next_step=ca_data.get("next_step", ""),
            )
            return reasoning, causal
        except (json.JSONDecodeError, KeyError, TypeError):
            # LLM 回覆格式不正確,用純文字
            return content[:500], self._rule_based_analysis(tool_name, parsed_status, raw_result)

    # ── 規則式備援分析 ────────────────────────────────────────────────────
    def _rule_based_analysis(
        self, tool_name: str, parsed_status: str, raw_result: str
    ) -> CausalAnalysis:
        """LLM 不可用時的規則式備援分析。"""
        templates = {
            "run_validation": {
                "all_pass": CausalAnalysis(
                    observation="所有基礎設施檢查通過",
                    root_cause="DNS/LDAP/SMB/Kerberos 服務正常運作",
                    impact="可以繼續進行弱點測試",
                    next_step="逐一測試已種植的弱點",
                ),
                "has_failures": CausalAnalysis(
                    observation="部分基礎設施檢查失敗",
                    root_cause="某些服務可能未啟動或網路不通",
                    impact="失敗的服務相關弱點測試可能也會失敗",
                    next_step="先修復基礎設施問題再測弱點",
                ),
            },
            "test_vuln": {
                "pass": CausalAnalysis(
                    observation="弱點驗證通過,可被利用",
                    root_cause="弱點已正確種植到靶場",
                    impact="攻防演練中此弱點可作為攻擊路徑",
                    next_step="繼續測試下一個弱點",
                ),
                "fail": CausalAnalysis(
                    observation="弱點驗證失敗,無法利用",
                    root_cause="弱點可能未正確種植,或 Ansible task 執行失敗",
                    impact="攻防演練中此攻擊路徑不可用",
                    next_step="使用 whitebox_check_setting 或 suggest_fix 排查根因",
                ),
                "skip": CausalAnalysis(
                    observation="弱點測試被跳過",
                    root_cause="無直接測試方法或缺少前置條件",
                    impact="需人工或 BloodHound 驗證",
                    next_step="考慮使用 bloodhound_verify 或手動驗證",
                ),
            },
            "bloodhound_verify": {
                "collected": CausalAnalysis(
                    observation="BloodHound 資料採集成功",
                    root_cause="QA 帳號有足夠權限進行 LDAP 查詢",
                    impact="可在 BloodHound CE 中執行 Cypher 確認攻擊路徑",
                    next_step="在 BloodHound CE 執行 cypher 驗證邊是否存在",
                ),
                "collect_failed": CausalAnalysis(
                    observation="BloodHound 資料採集失敗",
                    root_cause="QA 帳號權限不足或 LDAP 連線問題",
                    impact="無法驗證 ACL 類弱點的攻擊路徑",
                    next_step="檢查 QA 帳號權限和 LDAP 連線狀態",
                ),
            },
        }

        tool_templates = templates.get(tool_name, {})
        if parsed_status in tool_templates:
            return tool_templates[parsed_status]

        return CausalAnalysis(
            observation=f"工具 {tool_name} 回傳狀態: {parsed_status}",
            root_cause="需進一步分析",
            impact="待確認",
            next_step="檢查原始結果詳細內容",
        )

    # ── 即時輸出 ──────────────────────────────────────────────────────────
    def _print_step(self, record: ObservationRecord) -> None:
        """即時印出觀察紀錄。"""
        ca = record.causal_analysis
        print(f"\n{'─' * 60}")
        print(f"[Observer] Step {record.step_id}: {record.tool_name} "
              f"→ {record.parsed_status}")
        if record.tool_args:
            print(f"  args: {json.dumps(record.tool_args, ensure_ascii=False)}")
        print(f"  排查方式: {record.reasoning[:200]}")
        if ca:
            print(f"  觀察: {ca.observation[:150]}")
            print(f"  根因: {ca.root_cause[:150]}")
            print(f"  影響: {ca.impact[:150]}")
            print(f"  下一步: {ca.next_step[:150]}")
        print(f"{'─' * 60}")

    # ── 包裹 QA 工具 (LLM Agent 模式) ────────────────────────────────────
    def wrap_qa_tools(self, tools: list) -> list:
        """
        包裹 QA_TOOLS,在每次工具呼叫前後自動記錄觀察。
        回傳新的工具列表,保持原始工具的 name/description 不變。
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            from langchain.tools import StructuredTool

        wrapped = []
        for t in tools:
            original_func = t.func if hasattr(t, "func") else t
            original_name = t.name if hasattr(t, "name") else str(t)
            original_desc = t.description if hasattr(t, "description") else ""
            original_schema = t.args_schema if hasattr(t, "args_schema") else None

            observer_ref = self  # closure capture

            @wraps(original_func)
            def make_wrapper(orig_fn, orig_name):
                def wrapper(*args, **kwargs):
                    # 執行原始工具
                    result = orig_fn(*args, **kwargs)
                    # 記錄觀察
                    tool_args = kwargs.copy()
                    if args:
                        tool_args["_positional"] = list(args)
                    observer_ref.record_step(
                        tool_name=orig_name,
                        tool_args=tool_args,
                        raw_result=result if isinstance(result, str) else json.dumps(result),
                        context={"phase": "agent_mode", "mode": "auto"},
                    )
                    return result
                return wrapper

            wrapped_fn = make_wrapper(original_func, original_name)

            # 建構新的 StructuredTool 保持相同介面
            new_tool = StructuredTool.from_function(
                func=wrapped_fn,
                name=original_name,
                description=original_desc,
                args_schema=original_schema,
            )
            wrapped.append(new_tool)
        return wrapped

    # ── 產出報告 ──────────────────────────────────────────────────────────
    def generate_report(
        self,
        range_id: str,
        mode: str,
        infra_ok: bool,
        output_dir: str = ".",
    ) -> tuple[str, str]:
        """
        產出完整的觀察報告 (JSON + Markdown)。

        Returns:
            (json_path, md_path): 報告檔案路徑
        """
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        report_id = f"obs_{range_id}_{timestamp_str}"

        # 用 LLM 產出總結
        causal_chain = ""
        overall = ""
        try:
            causal_chain, overall = self._llm_summarize()
        except Exception as e:
            causal_chain = self._rule_based_chain_summary()
            overall = f"[Observer LLM 總結失敗: {e}] — 請參閱各步驟的個別分析。"

        report = ObserverReport(
            report_id=report_id,
            range_id=range_id,
            mode=mode,
            generated_at=now.isoformat(),
            observer_backend=self.backend,
            observer_model=self.model_name,
            total_steps=len(self._records),
            infra_ok=infra_ok,
            records=self._records,
            causal_chain_summary=causal_chain,
            overall_summary=overall,
        )

        # 寫 JSON
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"qa_observation_{range_id}_{timestamp_str}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2, exclude_none=True))

        # 寫 Markdown
        md_path = os.path.join(output_dir, f"qa_observation_{range_id}_{timestamp_str}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._render_markdown(report))

        print(f"\n[Observer] 報告已儲存:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")
        return json_path, md_path

    def _llm_summarize(self) -> tuple[str, str]:
        """用 LLM 彙整所有步驟產出總結。"""
        steps_summary = []
        for r in self._records:
            ca = r.causal_analysis
            steps_summary.append({
                "step": r.step_id,
                "tool": r.tool_name,
                "args": r.tool_args,
                "status": r.parsed_status,
                "reasoning": r.reasoning[:200],
                "observation": ca.observation if ca else "",
                "root_cause": ca.root_cause if ca else "",
            })

        prompt = f"""以下是 QA 過程中所有步驟的觀察紀錄:

{json.dumps(steps_summary, ensure_ascii=False, indent=2)}

請產出 JSON 格式的總結。"""

        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        resp = self.llm.invoke(messages)
        content = resp.content if hasattr(resp, "content") else str(resp)

        # 解析
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean
            clean = clean.rsplit("```", 1)[0] if "```" in clean else clean
        data = json.loads(clean)
        return data.get("causal_chain_summary", ""), data.get("overall_summary", "")

    def _rule_based_chain_summary(self) -> str:
        """LLM 不可用時的規則式因果鏈總結。"""
        parts = []
        for r in self._records:
            status_emoji = {"pass": "✅", "all_pass": "✅", "fail": "❌",
                            "error": "⚠️", "skip": "⏭️",
                            "has_failures": "❌"}.get(r.parsed_status, "❓")
            parts.append(f"{status_emoji} {r.tool_name}({r.parsed_status})")
        return " → ".join(parts)

    def _render_markdown(self, report: ObserverReport) -> str:
        """將報告渲染為 Markdown 格式。"""
        lines = [
            f"# 🔍 QA 觀察報告",
            f"",
            f"| 欄位 | 值 |",
            f"|------|-----|",
            f"| 報告 ID | `{report.report_id}` |",
            f"| 靶場 ID | `{report.range_id}` |",
            f"| QA 模式 | `{report.mode}` |",
            f"| 生成時間 | {report.generated_at} |",
            f"| 觀察員後端 | `{report.observer_backend}` |",
            f"| 觀察員模型 | `{report.observer_model}` |",
            f"| 總步驟數 | {report.total_steps} |",
            f"| 基礎設施 | {'✅ 正常' if report.infra_ok else '❌ 異常'} |",
            f"",
            f"---",
            f"",
            f"## 📋 因果鏈總結",
            f"",
            f"{report.causal_chain_summary}",
            f"",
            f"---",
            f"",
            f"## 📊 整體觀察總結",
            f"",
            f"{report.overall_summary}",
            f"",
            f"---",
            f"",
            f"## 📝 逐步觀察紀錄",
            f"",
        ]

        for r in report.records:
            lines.append(f"### Step {r.step_id}: `{r.tool_name}` → {r.parsed_status}")
            lines.append(f"")
            lines.append(f"- **時間**: {r.timestamp}")
            if r.tool_args:
                lines.append(f"- **參數**: `{json.dumps(r.tool_args, ensure_ascii=False)}`")
            lines.append(f"- **狀態**: `{r.parsed_status}`")
            lines.append(f"")

            lines.append(f"#### 排查方式")
            lines.append(f"{r.reasoning}")
            lines.append(f"")

            if r.causal_analysis:
                ca = r.causal_analysis
                lines.append(f"#### 因果推理")
                lines.append(f"| 維度 | 分析 |")
                lines.append(f"|------|------|")
                lines.append(f"| 觀察 | {ca.observation} |")
                lines.append(f"| 根因 | {ca.root_cause} |")
                lines.append(f"| 影響 | {ca.impact} |")
                lines.append(f"| 下一步 | {ca.next_step} |")
                lines.append(f"")

            lines.append(f"<details>")
            lines.append(f"<summary>原始結果 (點擊展開)</summary>")
            lines.append(f"")
            lines.append(f"```json")
            # 嘗試格式化 JSON
            try:
                pretty = json.dumps(json.loads(r.raw_result), ensure_ascii=False, indent=2)
                lines.append(pretty)
            except (json.JSONDecodeError, TypeError):
                lines.append(r.raw_result)
            lines.append(f"```")
            lines.append(f"</details>")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI (獨立測試用)
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="observer_agent",
                                description="觀察員 Agent — 獨立測試",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--backend", choices=["openai", "ollama", "local-openai", "cloud"],
                   default="ollama")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    observer = ObserverAgent(backend=args.backend, model=args.model)

    # 模擬記錄
    observer.record_step(
        tool_name="run_validation", tool_args={},
        raw_result=json.dumps([
            {"check": "DNS resolution", "status": "pass", "evidence": "10.3.10.10"},
            {"check": "LDAP reachable", "status": "pass", "evidence": "LDAP OK"},
        ]),
        context={"phase": "infrastructure", "mode": "blackbox"},
    )
    observer.record_step(
        tool_name="test_vuln",
        tool_args={"vuln_id": 1, "vuln_name": "AS-REP Roasting"},
        raw_result=json.dumps({"status": "pass", "tool": "impacket",
                               "evidence": "$krb5asrep$..."}),
        context={"phase": "vuln_test", "mode": "blackbox"},
    )
    observer.record_step(
        tool_name="test_vuln",
        tool_args={"vuln_id": 2, "vuln_name": "Kerberoasting"},
        raw_result=json.dumps({"status": "fail", "tool": "impacket",
                               "evidence": "No SPN found"}),
        context={"phase": "vuln_test", "mode": "blackbox"},
    )

    observer.generate_report(
        range_id="demo-001", mode="blackbox",
        infra_ok=True, output_dir=".",
    )
    print("\n[OK] 觀察員 Agent 測試完成")
