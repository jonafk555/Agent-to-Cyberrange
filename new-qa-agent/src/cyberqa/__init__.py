"""Cochise-compatible autonomous cyber-range red-team agent.

The package keeps the original Cochise Planner/Executor/SSH/Knowledge loop.
Cyber Range QA helpers are optional and observational; they live in the
cyberqa.qa_extensions module and do not replace the execution core.
"""

from .executor import Executor, ExecutorFactory
from .knowledge import Knowledge
from .logger import Logger
from .planner import Planner
from .ssh_connection import SSHConnection

__all__ = [
    "Executor",
    "ExecutorFactory",
    "Knowledge",
    "Logger",
    "Planner",
    "SSHConnection",
]
