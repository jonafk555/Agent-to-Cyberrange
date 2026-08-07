"""Cochise-compatible autonomous cyber-range red-team entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import litellm
from dotenv import load_dotenv
from rich.console import Console

from cyberqa.common import get_llm_config_from_env
from cyberqa.executor import ExecutorFactory
from cyberqa.human_interaction import HumanInteraction, is_stop_response
from cyberqa.logger import Logger
from cyberqa.planner import Planner
from cyberqa.qa_extensions import (
    RangeSpecification,
    load_range_specification,
    safe_component,
    write_qa_assessment,
    write_red_team_report,
    write_run_metadata,
)
from cyberqa.ssh_connection import get_ssh_connection_from_env


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"
DEFAULT_SCENARIO = (TEMPLATE_DIR / "scenario.md").read_text(encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cochise-compatible autonomous red-team agent for an authorized cyber range"
    )
    parser.add_argument(
        "--scenario-id",
        default=os.getenv("CYBERQA_SCENARIO_ID", "cyberqa-run"),
        help="Label used for the per-run artifact directory",
    )
    parser.add_argument(
        "--spec",
        default=os.getenv("CYBERQA_SPEC_PATH", ""),
        help="Optional JSON/YAML range QA reference; it is context, not an execution plan",
    )
    parser.add_argument(
        "--scenario-file",
        default=os.getenv("CYBERQA_SCENARIO_FILE", ""),
        help="Optional additional scenario/rules file",
    )
    parser.add_argument(
        "--runs-dir",
        default=os.getenv("CYBERQA_RUNS_DIR", "runs"),
        help="Directory under which each run receives its own artifacts",
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=int(os.getenv("MAX_RUN_TIME", "0")),
        help="Maximum runtime in seconds; 0 means no runtime limit",
    )
    parser.add_argument(
        "--planner-max-context-size",
        type=int,
        default=int(os.getenv("PLANNER_MAX_CONTEXT_SIZE", "250000")),
    )
    parser.add_argument(
        "--planner-max-interactions",
        type=int,
        default=int(os.getenv("PLANNER_MAX_INTERACTIONS", "0")),
    )
    return parser.parse_args(argv)


def create_run_directory(root: str | Path, scenario_id: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    directory = Path(root).expanduser() / (
        f"{safe_component(scenario_id, 'cyberqa-run')}_{timestamp}_{uuid4().hex[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _read_optional_file(
    reference: str | None,
    human_interaction: HumanInteraction,
) -> str | None:
    if not reference:
        return ""
    path = Path(reference).expanduser()
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, TypeError, ValueError):
        return None


async def async_main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    run_directory = create_run_directory(args.runs_dir, args.scenario_id)
    console = Console()
    logger = Logger(console, run_directory=run_directory)
    human_interaction = HumanInteraction(console)

    scenario = DEFAULT_SCENARIO
    planner: Planner | None = None
    llm_config = None
    connection = None
    specification: RangeSpecification | None = None
    status = "failed"
    error_text: str | None = None
    human_guidance: str | None = None

    logger.log_data("starting test-run")
    logger.log_data("run_directory", str(run_directory))
    write_run_metadata(run_directory, {
        "scenario_id": args.scenario_id,
        "started_at": datetime.now(UTC).isoformat(),
        "run_directory": str(run_directory.resolve()),
        "log_directory": str((run_directory / "logs").resolve()),
    })

    try:
        additional_scenario = _read_optional_file(args.scenario_file, human_interaction)
        if additional_scenario is None:
            human_guidance = await human_interaction.ask_human(
                f"The scenario file '{args.scenario_file}' is unavailable. "
                "Provide a valid path or type continue/stop.",
                "A requested scenario artifact could not be read.",
            )
            if is_stop_response(human_guidance):
                status = "stopped"
                return
            if human_guidance and Path(human_guidance).expanduser().is_file():
                additional_scenario = _read_optional_file(human_guidance, human_interaction)
            else:
                additional_scenario = f"Human guidance about the missing scenario file: {human_guidance}"
        if additional_scenario:
            scenario += "\n\n# Operator-provided scenario context\n\n" + additional_scenario

        if args.spec:
            specification = load_range_specification(args.spec)
            if specification is None:
                human_guidance = await human_interaction.ask_human(
                    f"The range specification '{args.spec}' is unavailable or invalid. "
                    "Provide a valid path, type continue to run without it, or type stop.",
                    "A requested Cyber Range QA reference could not be loaded.",
                )
                if is_stop_response(human_guidance):
                    status = "stopped"
                    return
                candidate = Path(human_guidance).expanduser() if human_guidance else None
                specification = load_range_specification(candidate)
                if specification is None:
                    logger.log_data("configuration-warning", {
                        "missing-specification": args.spec,
                        "human-guidance": human_guidance or "continue",
                    })
            if specification is not None:
                scenario += "\n\n# Cyber Range QA reference\n\n" + specification.prompt_context()

        llm_config = get_llm_config_from_env()
        connection = get_ssh_connection_from_env()
        litellm.suppress_debug_info = True
        scenario += (
            "\n\n# Runtime target context\n\n"
            f"SSH target host: {connection.host}\n"
            f"SSH target username: {connection.username}\n"
            "Use the runtime target and observed scope; do not substitute an "
            "example range from memory."
        )

        logger.log_data("configuration", {
            **llm_config.to_log_dict(),
            "ssh-host": connection.host,
            "ssh-user": connection.username,
            "scenario-id": args.scenario_id,
            "scenario": scenario,
            "max_runtime": args.max_runtime,
            "planner_max_context_size": args.planner_max_context_size,
            "planner_max_interactions": args.planner_max_interactions,
            "qa_specification": specification.reference if specification else None,
        }, output=False)
        write_run_metadata(run_directory, {
            "scenario_id": args.scenario_id,
            "started_at": datetime.now(UTC).isoformat(),
            "run_directory": str(run_directory.resolve()),
            "log_directory": str((run_directory / "logs").resolve()),
            "target_host": connection.host,
            "target_username": connection.username,
            "llm": llm_config.to_log_dict(),
            "specification": specification.reference if specification else args.spec or None,
        })

        await connection.connect()
        executor_factory = ExecutorFactory(
            llm_config,
            None,
            scenario,
            [connection.execute_command],
            logger,
            human_interaction,
        )
        planner = Planner(
            llm_config,
            None,
            scenario,
            executor_factory,
            logger,
            args.max_runtime,
            args.planner_max_context_size,
            args.planner_max_interactions,
            human_interaction,
        )
        await planner.engage()
        status = "stopped" if planner.human_stop_requested else "completed"
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        logger.log_data("run-error", error_text)
        raise
    finally:
        report_path = write_red_team_report(
            run_directory,
            scenario,
            planner,
            status=status,
            specification=specification,
            error=error_text,
        )
        qa_path = write_qa_assessment(run_directory, specification, planner) if specification else None
        write_run_metadata(run_directory, {
            "scenario_id": args.scenario_id,
            "finished_at": datetime.now(UTC).isoformat(),
            "status": status,
            "report": report_path,
            "qa_assessment": qa_path,
            "log": str(logger.log_path) if logger.log_path else None,
        })
        console.print(f"[Cochise run directory] {run_directory}")
        console.print(f"[Cochise report] {report_path}")
        if qa_path:
            console.print(f"[QA appendix] {qa_path}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
