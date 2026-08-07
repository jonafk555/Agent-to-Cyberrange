import datetime
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import litellm


@dataclass(frozen=True)
class LLMConfig:
    """Connection details for a LiteLLM-compatible chat model.

    ``model`` is stored in LiteLLM's provider/model format (for example
    ``anthropic/claude-sonnet-4-5`` or ``ollama/llama3.1``).  The API key is
    deliberately excluded from the dataclass representation so accidentally
    logging this object does not expose credentials.
    """

    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    api_base: str | None = None
    local_backend: str | None = None

    def completion_kwargs(self) -> dict[str, str]:
        """Return the provider-specific arguments for ``litellm.completion``."""

        kwargs = {"model": self.model}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs

    def to_log_dict(self) -> dict[str, str]:
        """Return safe connection details for structured logs."""

        result = {"provider": self.provider, "model": self.model}
        if self.local_backend:
            result["local_backend"] = self.local_backend
        if self.api_base:
            result["api_base"] = self.api_base
        return result


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _normalise_provider(value: str) -> str:
    provider = value.strip().lower().replace("_", "-")
    aliases = {
        "openai": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "gemini": "gemini",
        "google": "gemini",
        "local": "local",
        "ollama": "local",
        "lmstudio": "local",
        "lm-studio": "local",
        "openai-compatible": "local",
        "vllm": "local",
        "llamacpp": "local",
        "llama-cpp": "local",
    }
    if provider not in aliases:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{value}'. Choose openai, claude, gemini, "
            "or local."
        )
    return aliases[provider]


def _local_backend(provider_value: str, configured_backend: str | None) -> str:
    backend = (configured_backend or "").strip().lower().replace("_", "-")
    provider_value = provider_value.strip().lower().replace("_", "-")

    if not backend:
        backend = (
            "openai-compatible"
            if provider_value
            in {"lmstudio", "lm-studio", "openai-compatible", "vllm", "llamacpp", "llama-cpp"}
            else "ollama"
        )

    if backend in {"ollama"}:
        return "ollama"
    if backend in {
        "openai",
        "openai-compatible",
        "lmstudio",
        "lm-studio",
        "vllm",
        "llamacpp",
        "llama-cpp",
    }:
        return "openai-compatible"
    raise ValueError(
        f"Unsupported LOCAL_LLM_BACKEND '{configured_backend}'. Choose ollama "
        "or openai-compatible."
    )


def _with_provider_prefix(model: str, prefix: str, aliases: set[str]) -> str:
    model = model.strip()
    model_prefix, separator, model_name = model.partition("/")
    if separator and model_prefix.lower() in aliases:
        return f"{prefix}/{model_name}"
    return f"{prefix}/{model}"


def _provider_from_model(model: str) -> str:
    prefix = model.partition("/")[0].lower()
    if prefix in {"anthropic", "claude"}:
        return "anthropic"
    if prefix in {"gemini", "google", "vertex-ai", "vertex_ai"}:
        return "gemini"
    if prefix in {"ollama"}:
        return "local"
    if prefix in {"openai"}:
        return "openai"
    return "litellm"


def get_llm_config_from_env() -> LLMConfig:
    """Build the active LLM connection from environment variables.

    The explicit configuration uses ``LLM_PROVIDER`` and ``LLM_MODEL``.  The
    previous ``LITELLM_MODEL``/``LITELLM_API_KEY`` configuration remains
    supported so existing OpenRouter and other LiteLLM setups keep working.
    """

    provider_value = _env_first("LLM_PROVIDER")
    if provider_value is None:
        model = _env_first("LLM_MODEL", "LITELLM_MODEL")
        if model is None:
            raise ValueError(
                "No LLM configured. Set LLM_PROVIDER and LLM_MODEL, or use the "
                "legacy LITELLM_MODEL setting."
            )
        return LLMConfig(
            provider=_provider_from_model(model),
            model=model,
            api_key=_env_first("LLM_API_KEY", "LITELLM_API_KEY"),
            api_base=_env_first("LLM_BASE_URL", "LITELLM_API_BASE"),
        )

    provider = _normalise_provider(provider_value)
    model_names = {
        "openai": ("OPENAI_MODEL",),
        "anthropic": ("ANTHROPIC_MODEL", "CLAUDE_MODEL"),
        "gemini": ("GEMINI_MODEL", "GOOGLE_MODEL"),
        "local": ("LOCAL_LLM_MODEL", "OLLAMA_MODEL"),
    }
    model = _env_first("LLM_MODEL", *model_names[provider])
    if model is None:
        raise ValueError(
            "LLM_MODEL is required when LLM_PROVIDER is set. You can also use the "
            "provider-specific model variable, such as OPENAI_MODEL or GEMINI_MODEL."
        )

    key_names = {
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "local": ("LOCAL_LLM_API_KEY", "OLLAMA_API_KEY"),
    }
    api_key = _env_first("LLM_API_KEY", *key_names[provider], "LITELLM_API_KEY")

    if provider in {"openai", "anthropic", "gemini"} and api_key is None:
        expected = ", ".join(key_names[provider])
        raise ValueError(f"{provider} requires an API key. Set LLM_API_KEY or {expected}.")

    if provider != "local":
        base_names = {
            "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
            "anthropic": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE"),
            "gemini": ("GEMINI_BASE_URL", "GEMINI_API_BASE"),
        }
        api_base = _env_first("LLM_BASE_URL", *base_names[provider])
        prefix = provider
        aliases = {provider}
        if provider == "anthropic":
            aliases.add("claude")
        if provider == "gemini":
            aliases.update({"gemini", "google", "vertex-ai", "vertex_ai"})
        return LLMConfig(
            provider=provider,
            model=_with_provider_prefix(model, prefix, aliases),
            api_key=api_key,
            api_base=api_base,
        )

    backend = _local_backend(provider_value, _env_first("LOCAL_LLM_BACKEND"))
    if backend == "ollama":
        api_base = _env_first("LLM_BASE_URL", "LOCAL_LLM_BASE_URL", "OLLAMA_API_BASE")
        api_base = api_base or "http://127.0.0.1:11434"
        return LLMConfig(
            provider="local",
            model=_with_provider_prefix(model, "ollama", {"ollama"}),
            api_key=api_key,
            api_base=api_base,
            local_backend=backend,
        )

    api_base = _env_first(
        "LLM_BASE_URL", "LOCAL_LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"
    )
    api_base = api_base or "http://127.0.0.1:1234/v1"
    return LLMConfig(
        provider="local",
        model=_with_provider_prefix(model, "openai", {"openai"}),
        api_key=api_key or "local",
        api_base=api_base,
        local_backend=backend,
    )


def get_or_fail(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"Environment variable {name} not set")
    return value


def is_tool_call(msg) -> bool:
    return hasattr(msg, "tool_calls") and msg.tool_calls is not None and len(msg.tool_calls) > 0


class LLMFunctionMapping:
    def __init__(self, tool_functions: list[Callable]):
        self.tools = []
        self.mapping = {}

        for i in tool_functions:
            tool = litellm.utils.function_to_dict(i)

            self.tools.append({"type": "function", "function": tool})
            self.mapping[tool["name"]] = i

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return self.tools

    def get_function(self, value) -> Callable:
        return self.mapping[value]


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return _to_jsonable(value.__dict__)
    return str(value)


def _as_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def convert_costs_to_json(costs, response=None) -> dict:
    """Normalize provider usage objects to the shape expected by the logger."""

    result = _to_jsonable(costs)
    if not isinstance(result, dict):
        result = {}

    result["prompt_tokens"] = _as_int(result.get("prompt_tokens"))
    result["completion_tokens"] = _as_int(result.get("completion_tokens"))
    result["total_tokens"] = _as_int(
        result.get("total_tokens")
        or result["prompt_tokens"] + result["completion_tokens"]
    )
    prompt_details = result.get("prompt_tokens_details")
    completion_details = result.get("completion_tokens_details")
    result["prompt_tokens_details"] = (
        prompt_details if isinstance(prompt_details, dict) else {}
    )
    result["completion_tokens_details"] = (
        completion_details if isinstance(completion_details, dict) else {}
    )
    result["prompt_tokens_details"]["cached_tokens"] = _as_int(
        result["prompt_tokens_details"].get("cached_tokens")
    )
    result["completion_tokens_details"]["reasoning_tokens"] = _as_int(
        result["completion_tokens_details"].get("reasoning_tokens")
    )

    cost = result.get("cost")
    if cost is None and response is not None:
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        cost = hidden_params.get("response_cost")
    try:
        result["cost"] = float(cost or 0)
    except (TypeError, ValueError):
        result["cost"] = 0.0
    return result


def _completion_kwargs(model: str | LLMConfig, api_key: str | None) -> dict[str, str]:
    if isinstance(model, LLMConfig):
        kwargs = model.completion_kwargs()
        if api_key:
            kwargs["api_key"] = api_key
        return kwargs

    kwargs = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def llm_tool_call(
    model: str | LLMConfig,
    api_key: str | None,
    tools: LLMFunctionMapping,
    messages: list[dict[str, Any]]):

    tik = datetime.datetime.now()
    response = litellm.completion(
        messages=messages,
        tools=tools.get_tool_definitions(),
        **_completion_kwargs(model, api_key),
    )
    tok = datetime.datetime.now()

    if len(response.choices) != 1:
        raise RuntimeError(f"Expected exactly one LLM choice, but got {len(response.choices)}.")

    response_message = response.choices[0].message
    costs = convert_costs_to_json(response.usage, response)
    duration = (tok - tik).total_seconds()

    return response_message, costs, duration


def message_to_json(message):
    result = {"role": message.role}

    if message.content:
        result["content"] = message.content
    else:
        result["content"] = ""

    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]

    return result


# only used by ptt for now, but could be used by executor in the future as well
def llm_call(
    model: str | LLMConfig,
    api_key: str | None,
    messages: list[dict[str, Any]],
):
    """make a simple LLM call without any response format parsing"""

    tik = datetime.datetime.now()
    response = litellm.completion(
        messages=messages,
        **_completion_kwargs(model, api_key),
    )
    tok = datetime.datetime.now()

    if len(response.choices) != 1:
        raise RuntimeError(f"Expected exactly one LLM choice, but got {len(response.choices)}.")

    # output tokens costs
    costs = convert_costs_to_json(response.usage, response)
    duration = (tok - tik).total_seconds()

    result = response.choices[0].message
    content = {
        "content": result.content,
        "reasoning_content": result.reasoning_content
        if hasattr(result, "reasoning_content")
        else None,
    }
    return content, duration, costs

