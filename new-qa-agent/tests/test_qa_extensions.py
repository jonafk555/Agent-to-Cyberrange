from __future__ import annotations

import json
from types import SimpleNamespace

from cyberqa.qa_extensions import (
    load_range_specification,
    write_qa_assessment,
    write_red_team_report,
)


def test_specification_is_generic_context_and_not_a_fixed_schema(tmp_path):
    path = tmp_path / "range.json"
    path.write_text(
        json.dumps({
            "name": "arbitrary-range",
            "hosts": [{"address": "192.0.2.10", "service": "custom"}],
            "checks": [{"id": "custom-check", "description": "custom service is reachable"}],
        }),
        encoding="utf-8",
    )

    specification = load_range_specification(path)

    assert specification is not None
    assert specification.title == "arbitrary-range"
    assert specification.assertions[0]["id"] == "custom-check"
    assert "custom service" in specification.prompt_context()
    assert "command list" in specification.prompt_context()


def test_qa_appendix_is_post_run_and_report_keeps_knowledge(tmp_path):
    class KnowledgeStub:
        def get_knowledge(self):
            return "alice password=P@ssw0rd!"

    planner = SimpleNamespace(
        knowledge=KnowledgeStub(),
        history=[{"role": "assistant", "content": "initial red-team plan"}],
    )
    path = tmp_path / "range.json"
    path.write_text(
        json.dumps({"assertions": [{"id": "account", "statement": "alice account exists"}]}),
        encoding="utf-8",
    )
    specification = load_range_specification(path)

    qa_path = write_qa_assessment(tmp_path, specification, planner)
    report_path = write_red_team_report(
        tmp_path,
        "red-team scenario",
        planner,
        status="completed",
        specification=specification,
    )

    assert qa_path is not None
    assert "post-run QA triage" in (tmp_path / "qa-assessment.md").read_text(encoding="utf-8")
    assert "P@ssw0rd!" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report_path.endswith("report.md")
