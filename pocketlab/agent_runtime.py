from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from agents import Agent, Runner
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError, ToolTimeoutError
from agents.items import ToolCallItem, ToolCallOutputItem
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from pocketlab.model_run_control import (
    ModelFallbackRequested,
    await_model_with_user_control,
    current_model_run_id,
    current_model_run_reasoning_mode,
    record_model_reasoning_activity,
    record_model_stream_delta,
    record_model_stream_stage,
)
from pocketlab.provider_compat import reasoning_metadata_from_model_settings

RunnerCallable = Callable[..., Awaitable[Any]]
TraceStatus = Literal["completed", "failed", "cancelled"]


@dataclass(frozen=True)
class AgentRuntimePolicy:
    """Offline run bounds plus optional accounting metadata.

    ``token_budget`` is an evaluation gate, not a mid-request cancellation mechanism. The active
    offline safety bounds are ``timeout_s`` and ``max_turns``. Browser requests with a signed-in
    model-run control channel deliberately do not inherit ``timeout_s``: after two minutes the user
    can continue waiting or explicitly choose the registered fallback. A whole mutation run is
    never retried because a disconnected caller may not know whether its commit tool succeeded.
    """

    timeout_s: float = 180.0
    max_turns: int = 6
    read_only_retries: int = 1
    retry_backoff_s: float = 0.5
    token_budget: int = 32_000
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    concurrency_per_user: int = 2


@dataclass(frozen=True)
class AgentAttemptTrace:
    attempt: int
    status: Literal["completed", "failed"]
    elapsed_s: float
    error_kind: str | None = None
    error_type: str | None = None
    retryable: bool = False


@dataclass(frozen=True)
class AgentRunTrace:
    run_id: str
    operation: str
    model: str
    started_at: str
    finished_at: str
    status: TraceStatus
    elapsed_s: float
    timeout_s: float
    max_turns: int
    retry_limit: int
    reasoning_mode: Literal["fast", "deep", "provider_default"] | None = None
    reasoning_effort: str | None = None
    attempts: list[AgentAttemptTrace] = field(default_factory=list)
    model_requests: int = 0
    tool_calls: int = 0
    tool_events: list[dict[str, str]] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    token_budget: int = 0
    token_budget_exceeded: bool = False
    estimated_cost: float | None = None
    error_kind: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentRuntimeError(RuntimeError):
    def __init__(self, kind: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


def agent_runtime_http_status(error_kind: str) -> int:
    if error_kind in {"concurrency_limit", "rate_limit"}:
        return 429
    if error_kind == "timeout":
        return 504
    return 503


_TRACE_HISTORY: ContextVar[tuple[AgentRunTrace, ...]] = ContextVar(
    "pocketlab_agent_trace_history",
    default=(),
)
logger = logging.getLogger(__name__)
_ACTIVE_RUN_LOCK = Lock()
_ACTIVE_RUNS_BY_USER: dict[str, int] = {}


def _number_from_env(
    values: Mapping[str, str],
    name: str,
    default: float | None,
    *,
    minimum: float,
    maximum: float,
    integer: bool = False,
) -> float | int | None:
    raw = values.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw) if integer else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是数字。") from exc
    if not minimum <= parsed <= maximum:
        raise RuntimeError(f"{name} 必须位于 {minimum:g} 到 {maximum:g} 之间。")
    return parsed


def load_agent_runtime_policy(
    env: Mapping[str, str] | None = None,
) -> AgentRuntimePolicy:
    values = os.environ if env is None else env
    policy = AgentRuntimePolicy(
        timeout_s=float(
            _number_from_env(values, "AGENT_TIMEOUT_S", 180.0, minimum=1, maximum=600)
        ),
        max_turns=int(
            _number_from_env(
                values,
                "AGENT_MAX_TURNS",
                6,
                minimum=1,
                maximum=20,
                integer=True,
            )
        ),
        read_only_retries=int(
            _number_from_env(
                values,
                "AGENT_READ_ONLY_RETRIES",
                1,
                minimum=0,
                maximum=3,
                integer=True,
            )
        ),
        retry_backoff_s=float(
            _number_from_env(
                values,
                "AGENT_RETRY_BACKOFF_S",
                0.5,
                minimum=0,
                maximum=10,
            )
        ),
        token_budget=int(
            _number_from_env(
                values,
                "AGENT_TOKEN_BUDGET",
                32_000,
                minimum=100,
                maximum=1_000_000,
                integer=True,
            )
        ),
        input_cost_per_million=_number_from_env(
            values,
            "LLM_INPUT_COST_PER_MILLION",
            None,
            minimum=0,
            maximum=100_000,
        ),
        output_cost_per_million=_number_from_env(
            values,
            "LLM_OUTPUT_COST_PER_MILLION",
            None,
            minimum=0,
            maximum=100_000,
        ),
        concurrency_per_user=int(
            _number_from_env(
                values,
                "AGENT_CONCURRENCY_PER_USER",
                2,
                minimum=1,
                maximum=8,
                integer=True,
            )
        ),
    )
    if env is not None:
        return policy
    try:
        from pocketlab.model_profiles import ENVIRONMENT_PROFILE_ID, model_profile_store

        catalog = model_profile_store.catalog()
        if catalog.active_profile_id and catalog.active_profile_id != ENVIRONMENT_PROFILE_ID:
            active = next(
                item
                for item in catalog.profiles
                if item.profile_id == catalog.active_profile_id
            )
            return replace(
                policy,
                input_cost_per_million=active.input_cost_per_million,
                output_cost_per_million=active.output_cost_per_million,
            )
    except (RuntimeError, StopIteration):
        pass
    return policy


def clear_agent_run_traces() -> None:
    _TRACE_HISTORY.set(())


def get_agent_run_traces(*, clear: bool = False) -> list[dict[str, Any]]:
    traces = [item.to_dict() for item in _TRACE_HISTORY.get()]
    if clear:
        _TRACE_HISTORY.set(())
    return traces


def _append_trace(trace: AgentRunTrace) -> None:
    _TRACE_HISTORY.set((*_TRACE_HISTORY.get(), trace))
    try:
        from pocketlab.runtime_audit import agent_run_audit_store

        agent_run_audit_store.save_trace(trace.to_dict())
    except Exception as exc:  # noqa: BLE001 - observability must never break an Agent result
        logger.warning("Agent trace persistence failed: %s", type(exc).__name__)


def _try_acquire_run_slot(user_id: str, limit: int) -> bool:
    with _ACTIVE_RUN_LOCK:
        active = _ACTIVE_RUNS_BY_USER.get(user_id, 0)
        if active >= limit:
            return False
        _ACTIVE_RUNS_BY_USER[user_id] = active + 1
        return True


def _release_run_slot(user_id: str) -> None:
    with _ACTIVE_RUN_LOCK:
        active = _ACTIVE_RUNS_BY_USER.get(user_id, 0)
        if active <= 1:
            _ACTIVE_RUNS_BY_USER.pop(user_id, None)
        else:
            _ACTIVE_RUNS_BY_USER[user_id] = active - 1


def is_committed_tool_output(output: object) -> bool:
    """Trust the structured status field, never success-looking prose."""

    try:
        payload = json.loads(str(output))
    except (TypeError, json.JSONDecodeError):
        return False
    return _is_consistent_committed_payload(payload)


def _is_consistent_committed_payload(payload: object) -> bool:
    if not isinstance(payload, dict) or payload.get("status") != "committed":
        return False
    if payload.get("success") is False:
        return False
    return payload.get("error") in (None, "", False)


def _safe_error(error: BaseException) -> str:
    text = str(error)
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED_SECRET]", text)
    text = re.sub(
        r"(?i)(authorization|api[_-]?key|bearer|access[_-]?token|refresh[_-]?token|"
        r"client[_-]?secret)(\s*[:=]\s*|\s+)[^\s,;]+",
        r"\1=[REDACTED_SECRET]",
        text,
    )
    text = re.sub(
        r"(?i)(token|secret)(\s*[:=]\s*)[^\s,;]+",
        r"\1=[REDACTED_SECRET]",
        text,
    )
    text = re.sub(r"https?://[^\s,;]+", "[REDACTED_PROVIDER_URL]", text)
    return text[:500]


def _classify_error(error: BaseException) -> tuple[str, bool]:
    if isinstance(error, ModelFallbackRequested):
        return "user_fallback", False
    if isinstance(error, (TimeoutError, APITimeoutError)):
        return "timeout", True
    if isinstance(error, RateLimitError):
        return "rate_limit", True
    if isinstance(error, APIConnectionError):
        return "connection", True
    if isinstance(error, InternalServerError):
        return "provider_5xx", True
    if isinstance(error, MaxTurnsExceeded):
        return "max_turns", False
    if isinstance(error, ModelBehaviorError):
        return "malformed_model_output", False
    if isinstance(error, ToolTimeoutError):
        return "tool_timeout", False
    return "runtime_error", False


def _usage_from_result(result: Any) -> tuple[int | None, int | None, int | None]:
    context = getattr(result, "context_wrapper", None)
    usage = getattr(context, "usage", None)
    if usage is None:
        return None, None, None
    return (
        int(getattr(usage, "input_tokens", 0)),
        int(getattr(usage, "output_tokens", 0)),
        int(getattr(usage, "total_tokens", 0)),
    )


def _tool_events(result: Any) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    calls_by_id: dict[str, int] = {}
    for item in getattr(result, "new_items", []):
        if isinstance(item, ToolCallItem):
            event = {"name": item.tool_name or "unknown", "status": "called"}
            events.append(event)
            if item.call_id:
                calls_by_id[item.call_id] = len(events) - 1
        elif isinstance(item, ToolCallOutputItem):
            status = "returned"
            preview = str(item.output)[:2048]
            try:
                payload = json.loads(preview)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                if payload.get("success") is False or payload.get("error") not in (
                    None,
                    "",
                    False,
                ):
                    status = "error"
                elif _is_consistent_committed_payload(payload):
                    status = "committed"
            if item.call_id and item.call_id in calls_by_id:
                events[calls_by_id[item.call_id]]["status"] = status
            else:
                events.append({"name": "unknown", "status": status})
    return events


def _estimated_cost(
    policy: AgentRuntimePolicy,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if (
        input_tokens is None
        or output_tokens is None
        or policy.input_cost_per_million is None
        or policy.output_cost_per_million is None
    ):
        return None
    value = (
        input_tokens * policy.input_cost_per_million
        + output_tokens * policy.output_cost_per_million
    ) / 1_000_000
    return round(value, 8)


async def _run_streamed_agent(
    agent: Agent[Any],
    input_payload: str,
    **runner_kwargs: Any,
) -> Any:
    """Drain Agents SDK events while exposing only safe, visible progress."""

    result = Runner.run_streamed(agent, input_payload, **runner_kwargs)
    try:
        async for event in result.stream_events():
            event_type = getattr(event, "type", "")
            if event_type == "raw_response_event":
                data = getattr(event, "data", None)
                data_type = str(getattr(data, "type", ""))
                delta = str(getattr(data, "delta", "") or "")
                if data_type in {"response.output_text.delta", "response.refusal.delta"}:
                    record_model_stream_delta(delta)
                elif data_type == "response.function_call_arguments.delta":
                    record_model_stream_delta(delta, kind="tool")
                elif data_type in {
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                }:
                    # Hidden reasoning contents are intentionally discarded.
                    record_model_reasoning_activity()
            elif event_type == "run_item_stream_event":
                name = str(getattr(event, "name", ""))
                item = getattr(event, "item", None)
                if name == "tool_called":
                    record_model_stream_stage(
                        "TOOL CALL",
                        f"正在执行 {getattr(item, 'tool_name', None) or '受控工具'}",
                    )
                elif name == "tool_output":
                    record_model_stream_stage(
                        "TOOL RESULT",
                        "工具结果已返回，基模将据此继续组织答案",
                    )
                elif name == "message_output_created":
                    record_model_stream_stage(
                        "MODEL OUTPUT",
                        "本轮可见输出已完成，正在进入服务端整理",
                    )
            elif event_type == "agent_updated_stream_event":
                record_model_stream_stage("AGENT ROUTE", "Agent 已进入下一处理阶段")
        return result
    finally:
        if not result.is_complete:
            result.cancel()


def _clone_agent_for_fast_mode(agent: Agent[Any] | Any) -> Agent[Any] | Any:
    settings = getattr(agent, "model_settings", None)
    clone = getattr(agent, "clone", None)
    if settings is None or not callable(clone):
        return agent
    extra_body = dict(getattr(settings, "extra_body", None) or {})
    thinking = extra_body.get("thinking")
    if isinstance(thinking, Mapping):
        extra_body["thinking"] = {**thinking, "type": "disabled"}
    if isinstance(extra_body.get("enable_thinking"), bool):
        extra_body["enable_thinking"] = False
    updates: dict[str, Any] = {
        "extra_body": extra_body or None,
        "temperature": (
            getattr(settings, "temperature", None)
            if getattr(settings, "temperature", None) is not None
            else 0.1
        ),
    }
    if getattr(settings, "reasoning", None) is not None:
        updates["reasoning"] = {"effort": "low"}
    return clone(model_settings=replace(settings, **updates))


async def run_bounded_agent(
    agent: Agent[Any] | Any,
    input_payload: str,
    *,
    operation: str,
    model_name: str,
    allow_retry: bool,
    policy: AgentRuntimePolicy | None = None,
    runner: RunnerCallable | None = None,
    context: object | None = None,
) -> Any:
    """Apply a per-user admission gate before entering the bounded Agent runtime."""

    from pocketlab.auth import get_current_user_id

    active_policy = policy or load_agent_runtime_policy()
    user_id = get_current_user_id()
    reasoning_mode, reasoning_effort = reasoning_metadata_from_model_settings(
        getattr(agent, "model_settings", None)
    )
    if not _try_acquire_run_slot(user_id, active_policy.concurrency_per_user):
        now = datetime.now(UTC).isoformat()
        trace = AgentRunTrace(
            run_id=f"run-{uuid4().hex}",
            operation=operation,
            model=model_name,
            started_at=now,
            finished_at=now,
            status="failed",
            elapsed_s=0,
            timeout_s=active_policy.timeout_s,
            max_turns=active_policy.max_turns,
            retry_limit=0,
            reasoning_mode=reasoning_mode,
            reasoning_effort=reasoning_effort,
            token_budget=active_policy.token_budget,
            error_kind="concurrency_limit",
            error_type="AgentConcurrencyLimit",
            error_message="当前账号已有过多 Agent 运行，请等待其中一项完成后重试。",
        )
        _append_trace(trace)
        raise AgentRuntimeError(
            "concurrency_limit",
            trace.error_message or "Agent concurrency limit reached",
            retryable=True,
        )
    try:
        return await _run_bounded_agent_unlimited(
            agent,
            input_payload,
            operation=operation,
            model_name=model_name,
            allow_retry=allow_retry,
            policy=active_policy,
            runner=runner,
            context=context,
        )
    finally:
        _release_run_slot(user_id)


async def _run_bounded_agent_unlimited(
    agent: Agent[Any] | Any,
    input_payload: str,
    *,
    operation: str,
    model_name: str,
    allow_retry: bool,
    policy: AgentRuntimePolicy | None = None,
    runner: RunnerCallable | None = None,
    context: object | None = None,
) -> Any:
    """Run an agent with bounded turns/time and privacy-safe local tracing."""

    active_policy = policy or load_agent_runtime_policy()
    active_runner = runner or _run_streamed_agent
    retry_limit = active_policy.read_only_retries if allow_retry else 0
    run_id = f"run-{uuid4().hex}"
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    interactive_wait = current_model_run_id() is not None
    deadline = None if interactive_wait else started + active_policy.timeout_s
    effective_timeout_s = 0.0 if interactive_wait else active_policy.timeout_s
    attempts: list[AgentAttemptTrace] = []
    runner_kwargs: dict[str, Any] = {"max_turns": active_policy.max_turns}
    interactive_runner_attempt = 0
    last_max_turns = active_policy.max_turns
    reasoning_mode, reasoning_effort = reasoning_metadata_from_model_settings(
        getattr(agent, "model_settings", None)
    )
    configured_run_mode: Literal["fast", "high", "provider_default"] = (
        "high"
        if reasoning_mode == "deep"
        else reasoning_mode
        if reasoning_mode == "fast"
        else "provider_default"
    )
    supports_fast_switch = runner is None and configured_run_mode == "high"
    fast_agent: Agent[Any] | Any | None = None
    if context is not None:
        runner_kwargs["context"] = context

    def build_runner_awaitable() -> Awaitable[Any]:
        nonlocal fast_agent, interactive_runner_attempt, last_max_turns
        call_kwargs = dict(runner_kwargs)
        if interactive_wait and interactive_runner_attempt:
            call_kwargs["max_turns"] = min(
                20,
                active_policy.max_turns + 4 * interactive_runner_attempt,
            )
        last_max_turns = int(call_kwargs["max_turns"])
        interactive_runner_attempt += 1
        active_agent = agent
        if (
            supports_fast_switch
            and current_model_run_reasoning_mode(configured_run_mode) == "fast"
        ):
            if fast_agent is None:
                fast_agent = _clone_agent_for_fast_mode(agent)
            active_agent = fast_agent
        return active_runner(active_agent, input_payload, **call_kwargs)

    for attempt_number in range(1, retry_limit + 2):
        attempt_started = time.perf_counter()
        try:
            remaining_s = None if deadline is None else deadline - time.perf_counter()
            if remaining_s is not None and remaining_s <= 0:
                raise TimeoutError("Agent operation deadline exhausted before retry.")
            result = await await_model_with_user_control(
                operation=operation,
                model=model_name,
                noninteractive_timeout_s=remaining_s,
                awaitable_factory=build_runner_awaitable,
                reasoning_mode=configured_run_mode,
                supports_fast_switch=supports_fast_switch,
            )
        except asyncio.CancelledError:
            elapsed = round(time.perf_counter() - started, 4)
            trace = AgentRunTrace(
                run_id=run_id,
                operation=operation,
                model=model_name,
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
                status="cancelled",
                elapsed_s=elapsed,
                timeout_s=effective_timeout_s,
                max_turns=last_max_turns,
                retry_limit=retry_limit,
                reasoning_mode=reasoning_mode,
                reasoning_effort=reasoning_effort,
                attempts=attempts,
                token_budget=active_policy.token_budget,
                error_kind="cancelled",
                error_type="CancelledError",
                error_message="Agent run cancelled by caller.",
            )
            _append_trace(trace)
            raise
        except Exception as exc:
            error_kind, retryable = _classify_error(exc)
            attempts.append(
                AgentAttemptTrace(
                    attempt=attempt_number,
                    status="failed",
                    elapsed_s=round(time.perf_counter() - attempt_started, 4),
                    error_kind=error_kind,
                    error_type=type(exc).__name__,
                    retryable=retryable,
                )
            )
            backoff_s = active_policy.retry_backoff_s * (2 ** (attempt_number - 1))
            remaining_s = None if deadline is None else deadline - time.perf_counter()
            should_retry = (
                retryable
                and attempt_number <= retry_limit
                and (remaining_s is None or remaining_s > backoff_s)
            )
            if should_retry:
                await asyncio.sleep(backoff_s)
                continue

            safe_message = _safe_error(exc)
            trace = AgentRunTrace(
                run_id=run_id,
                operation=operation,
                model=model_name,
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
                status="failed",
                elapsed_s=round(time.perf_counter() - started, 4),
                timeout_s=effective_timeout_s,
                max_turns=last_max_turns,
                retry_limit=retry_limit,
                reasoning_mode=reasoning_mode,
                reasoning_effort=reasoning_effort,
                attempts=attempts,
                token_budget=active_policy.token_budget,
                error_kind=error_kind,
                error_type=type(exc).__name__,
                error_message=safe_message,
            )
            _append_trace(trace)
            raise AgentRuntimeError(
                error_kind,
                f"Agent 运行失败（{error_kind}）：{safe_message}",
                retryable=retryable,
            ) from exc

        effective_reasoning_mode = (
            "fast"
            if current_model_run_reasoning_mode(configured_run_mode) == "fast"
            else reasoning_mode
        )
        effective_reasoning_effort = (
            "low" if effective_reasoning_mode == "fast" else reasoning_effort
        )
        attempts.append(
            AgentAttemptTrace(
                attempt=attempt_number,
                status="completed",
                elapsed_s=round(time.perf_counter() - attempt_started, 4),
            )
        )
        input_tokens, output_tokens, total_tokens = _usage_from_result(result)
        tool_events = _tool_events(result)
        trace = AgentRunTrace(
            run_id=run_id,
            operation=operation,
            model=model_name,
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            status="completed",
            elapsed_s=round(time.perf_counter() - started, 4),
            timeout_s=effective_timeout_s,
            max_turns=last_max_turns,
            retry_limit=retry_limit,
            reasoning_mode=effective_reasoning_mode,
            reasoning_effort=effective_reasoning_effort,
            attempts=attempts,
            model_requests=len(getattr(result, "raw_responses", [])),
            tool_calls=len(tool_events),
            tool_events=tool_events,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_budget=active_policy.token_budget,
            token_budget_exceeded=(
                total_tokens is not None and total_tokens > active_policy.token_budget
            ),
            estimated_cost=_estimated_cost(active_policy, input_tokens, output_tokens),
        )
        _append_trace(trace)
        return result

    raise AssertionError("Agent retry loop exited unexpectedly.")
