from __future__ import annotations

from types import SimpleNamespace

from rich.console import Console

from cyberqa.executor import looks_like_missing_artifact
from cyberqa.knowledge import Knowledge
from cyberqa.logger import Logger


def test_cochise_knowledge_keeps_recorded_secret():
    logger = SimpleNamespace(console=Console())
    knowledge = Knowledge(logger)

    import asyncio

    asyncio.run(knowledge.add_compromised_account("alice", "P@ssw0rd!", "SMB login"))
    rendered = knowledge.get_knowledge()

    assert "alice" in rendered
    assert "P@ssw0rd!" in rendered
    assert "SMB login" in rendered


def test_missing_artifact_detection_is_used_for_human_recovery():
    assert looks_like_missing_artifact(
        "cat /tmp/expected.txt",
        "cat: /tmp/expected.txt: No such file or directory",
    )
    assert not looks_like_missing_artifact("whoami", "root")


def test_logger_places_raw_json_log_inside_run_directory(tmp_path):
    logger = Logger(Console(), run_directory=tmp_path)
    logger.log_data("test-event", {"value": "observed"}, output=False)

    assert logger.log_path is not None
    assert logger.log_path.parent == tmp_path / "logs"
    assert logger.log_path.is_file()
    assert logger.log_path.read_text(encoding="utf-8").strip()
