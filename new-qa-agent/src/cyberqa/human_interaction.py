import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


STOP_RESPONSES = frozenset({
    "stop",
    "quit",
    "exit",
    "abort",
    "cancel",
    "停止",
    "離開",
    "取消",
})


def is_stop_response(response: str) -> bool:
    return response.strip().lower() in STOP_RESPONSES


class HumanInteraction:
    """Interactive human-in-the-loop input shared by planner and executors."""

    def __init__(self, console: Console):
        self.console = console
        self._input_lock = asyncio.Lock()

    async def ask_human(self, question: str, reason: str) -> str:
        """Ask a human for guidance when the agent is blocked or missing an artifact.

        Parameters
        ----------
        question : str
            The concrete question that should be answered by the human. Include
            the missing file path, expected artifact, or decision that is needed.
        reason : str
            Why the agent cannot continue, such as a missing file, missing
            credential, inaccessible host, or an unsuccessful approach.

        Returns
        -------
        str
            The human's guidance. Reply ``stop`` to stop the current run.
        """

        async with self._input_lock:
            self.console.print(
                Panel(
                    Text(f"Reason: {reason}\n\n{question}"),
                    title="Human input required",
                )
            )
            try:
                response = await asyncio.to_thread(input, "Human response (or 'stop'): ")
            except (EOFError, KeyboardInterrupt):
                response = "stop"

            response = response.strip()
            if not response:
                self.console.print(
                    "[yellow]No response received; the agent will treat this as a stop request.[/yellow]"
                )
                return "stop"
            return response

