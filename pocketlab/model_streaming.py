from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

from pocketlab.model_run_control import (
    record_model_reasoning_activity,
    record_model_stream_delta,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamedChatCompletion:
    """Provider-neutral result assembled from visible Chat Completions deltas."""

    content: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    reasoning_characters: int = 0


def _usage_value(usage: object | None, name: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    return int(value) if value is not None else None


def _visible_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


async def consume_chat_completion(response_or_stream: Any) -> StreamedChatCompletion:
    """Drain a real provider stream and publish only visible model output.

    The helper deliberately counts, but never stores or publishes,
    ``reasoning_content``. A non-stream response is accepted for compatibility
    with local provider fakes and older OpenAI-compatible test doubles.
    """

    content_parts: list[str] = []
    finish_reason: str | None = None
    usage: object | None = None
    reasoning_characters = 0

    if hasattr(response_or_stream, "__aiter__"):
        try:
            async for chunk in response_or_stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                choice_finish = getattr(choice, "finish_reason", None)
                if choice_finish is not None:
                    finish_reason = str(choice_finish)
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                reasoning = _visible_text(getattr(delta, "reasoning_content", None))
                if not reasoning:
                    reasoning = _visible_text(getattr(delta, "reasoning", None))
                if reasoning:
                    reasoning_characters += len(reasoning)
                    record_model_reasoning_activity()
                visible = _visible_text(getattr(delta, "content", None))
                if visible:
                    content_parts.append(visible)
                    record_model_stream_delta(visible)
                for tool_call in getattr(delta, "tool_calls", None) or []:
                    function = getattr(tool_call, "function", None)
                    arguments = _visible_text(getattr(function, "arguments", None))
                    if arguments:
                        record_model_stream_delta(arguments, kind="tool")
        finally:
            close = getattr(response_or_stream, "close", None)
            if close is None:
                close = getattr(response_or_stream, "aclose", None)
            if callable(close):
                try:
                    close_result = close()
                    if inspect.isawaitable(close_result):
                        await close_result
                except Exception:
                    # Stream cleanup is best effort and must not replace the
                    # completed model result with a transport-specific close error.
                    _LOGGER.debug("Ignored model stream close failure", exc_info=True)
    else:
        usage = getattr(response_or_stream, "usage", None)
        choices = getattr(response_or_stream, "choices", None) or []
        if choices:
            choice = choices[0]
            choice_finish = getattr(choice, "finish_reason", None)
            finish_reason = str(choice_finish) if choice_finish is not None else None
            message = getattr(choice, "message", None)
            visible = _visible_text(getattr(message, "content", None))
            if visible:
                content_parts.append(visible)
                record_model_stream_delta(visible)
            reasoning = _visible_text(getattr(message, "reasoning_content", None))
            reasoning_characters = len(reasoning)
            if reasoning:
                record_model_reasoning_activity()

    return StreamedChatCompletion(
        content="".join(content_parts),
        finish_reason=finish_reason,
        prompt_tokens=_usage_value(usage, "prompt_tokens"),
        completion_tokens=_usage_value(usage, "completion_tokens"),
        total_tokens=_usage_value(usage, "total_tokens"),
        reasoning_characters=reasoning_characters,
    )
