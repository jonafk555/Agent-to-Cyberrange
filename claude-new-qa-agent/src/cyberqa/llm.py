"""LLM provider construction kept separate from graph and agent logic."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel


def build_llm() -> BaseChatModel | None:
    """Create the configured LangChain chat model, or None for safe offline mode."""
    provider = os.getenv("CYBERQA_LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("CYBERQA_LLM_MODEL", "gpt-4.1-mini"),
            temperature=0,
            max_retries=2,
        )
    raise ValueError(f"Unsupported CYBERQA_LLM_PROVIDER={provider!r}")
