from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Literal, TypeVar

from pydantic import BaseModel

from pocketlab.auth import get_current_user_id

ModelRunPhase = Literal[
    "connecting",
    "thinking",
    "streaming",
    "validating",
    "completed",
    "failed",
    "fallback_requested",
]
ModelRunDecision = Literal["continue", "fast", "fallback"]
ModelRunReasoningMode = Literal["fast", "high", "provider_default"]
ModelValidationRecoveryDecision = Literal[
    "retry",
    "retry_fast",
    "user_fallback",
    "noninteractive_error",
]

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,96}$")
_current_model_run_id: ContextVar[str | None] = ContextVar(
    "pocketlab_current_model_run_id",
    default=None,
)
_current_model_operation: ContextVar[str | None] = ContextVar(
    "pocketlab_current_model_operation",
    default=None,
)
_RECORD_TTL_S = 60 * 60
_THINKING_AFTER_S = 4.0
_USER_DECISION_AFTER_S = 120.0

T = TypeVar("T")


class ModelFallbackRequested(RuntimeError):
    """The signed-in user explicitly chose the deterministic safety path."""


class ModelRunDecisionRequest(BaseModel):
    decision: ModelRunDecision


@dataclass
class _ModelRunRecord:
    run_id: str
    user_id: str
    operation: str
    model: str
    started_at: str
    started_perf: float
    step_started_perf: float
    reasoning_mode: ModelRunReasoningMode = "provider_default"
    supports_fast_switch: bool = False
    fast_locked: bool = False
    phase: ModelRunPhase = "connecting"
    detail: str = "正在连接基模服务"
    decision_after_s: float = _USER_DECISION_AFTER_S
    fallback_event: asyncio.Event = field(default_factory=asyncio.Event)
    fast_event: asyncio.Event = field(default_factory=asyncio.Event)
    retry_event: asyncio.Event = field(default_factory=asyncio.Event)
    retry_count: int = 0
    error_kind: str | None = None
    stream_buffer: str = ""
    stream_characters: int = 0
    stream_kind: Literal["output", "tool"] | None = None
    first_stream_perf: float | None = None
    reasoning_stream_chunks: int = 0
    stage_events: list[dict[str, object]] = field(default_factory=list)
    updated_perf: float = field(default_factory=time.perf_counter)

    def snapshot(self) -> dict[str, object]:
        now = time.perf_counter()
        elapsed_s = max(0.0, now - self.started_perf)
        step_elapsed_s = max(0.0, now - self.step_started_perf)
        phase = self.phase
        detail = self.detail
        if phase == "connecting" and step_elapsed_s >= _THINKING_AFTER_S:
            phase = "thinking"
            detail = (
                "请求已发出；High 模式正在推理，尚未产生可展示正文"
                if self.reasoning_mode == "high"
                else "请求已发出；正在等待基模返回首个可展示片段"
            )
        decision_available = phase == "failed" or (
            phase in {"connecting", "thinking", "streaming"} and elapsed_s >= self.decision_after_s
        )
        allowed_decisions: list[ModelRunDecision] = []
        if decision_available:
            allowed_decisions.append("continue")
            if self.reasoning_mode == "high" and self.supports_fast_switch:
                allowed_decisions.append("fast")
            allowed_decisions.append("fallback")
        return {
            "run_id": self.run_id,
            "operation": self.operation,
            "model": self.model,
            "phase": phase,
            "detail": detail,
            "started_at": self.started_at,
            "elapsed_s": round(elapsed_s, 1),
            "step_elapsed_s": round(step_elapsed_s, 1),
            "decision_available": decision_available,
            "allowed_decisions": allowed_decisions,
            "next_decision_at_s": round(self.decision_after_s, 1),
            "fallback_requested": self.fallback_event.is_set(),
            "reasoning_mode": self.reasoning_mode,
            "fast_locked": self.fast_locked,
            "retry_count": self.retry_count,
            "error_kind": self.error_kind,
            "streaming": phase == "streaming",
            "stream_kind": self.stream_kind,
            "stream_characters": self.stream_characters,
            "stream_preview": _presentable_stream_preview(self.stream_buffer),
            "first_stream_elapsed_s": (
                round(max(0.0, self.first_stream_perf - self.started_perf), 1)
                if self.first_stream_perf is not None
                else None
            ),
            "reasoning_stream_chunks": self.reasoning_stream_chunks,
            "stage_events": list(self.stage_events),
        }


_registry_lock = RLock()
_runs: dict[tuple[str, str], _ModelRunRecord] = {}


def _presentable_stream_preview(value: str) -> str:
    """Return visible model output without exposing hidden reasoning text."""

    text = value.strip()
    if not text:
        return ""
    if text.startswith(("{", "[", "```json")):
        visible_values: list[str] = []
        for match in re.finditer(r'"((?:\\.|[^"\\])*)"\s*(?=[,}\]])', text):
            try:
                decoded = json.loads(f'"{match.group(1)}"')
            except (json.JSONDecodeError, TypeError):
                continue
            normalized = re.sub(r"\s+", " ", str(decoded)).strip()
            if len(normalized) >= 3 and normalized not in visible_values:
                visible_values.append(normalized)
        if visible_values:
            return " · ".join(visible_values[-4:])[-900:]
        return ""
    return re.sub(r"\s+", " ", text)[-900:]


def validate_model_run_id(value: str | None) -> str | None:
    run_id = (value or "").strip()
    if not run_id:
        return None
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("invalid PocketLab model run ID")
    return run_id


@contextmanager
def model_run_context(run_id: str | None, operation: str | None = None) -> Iterator[None]:
    normalized = validate_model_run_id(run_id)
    run_token = _current_model_run_id.set(normalized)
    operation_token = _current_model_operation.set(operation)
    try:
        yield
    finally:
        _current_model_operation.reset(operation_token)
        _current_model_run_id.reset(run_token)


def current_model_run_id() -> str | None:
    return _current_model_run_id.get()


def _prune_locked(now: float) -> None:
    expired = [key for key, record in _runs.items() if now - record.updated_perf > _RECORD_TTL_S]
    for key in expired:
        _runs.pop(key, None)


def _start_or_update_record(
    operation: str,
    model: str,
    *,
    reasoning_mode: ModelRunReasoningMode,
    supports_fast_switch: bool,
) -> _ModelRunRecord | None:
    run_id = current_model_run_id()
    if run_id is None:
        return None
    user_id = get_current_user_id()
    now = time.perf_counter()
    key = (user_id, run_id)
    with _registry_lock:
        _prune_locked(now)
        record = _runs.get(key)
        if record is None:
            record = _ModelRunRecord(
                run_id=run_id,
                user_id=user_id,
                operation=operation,
                model=model,
                started_at=datetime.now(UTC).isoformat(),
                started_perf=now,
                step_started_perf=now,
                reasoning_mode=reasoning_mode,
                supports_fast_switch=(supports_fast_switch and reasoning_mode == "high"),
            )
            _runs[key] = record
        else:
            record.operation = operation
            record.model = model
            record.step_started_perf = now
            if not record.fast_locked:
                record.reasoning_mode = reasoning_mode
                record.supports_fast_switch = supports_fast_switch and reasoning_mode == "high"
            record.phase = "connecting"
            record.detail = "正在连接基模服务"
            record.error_kind = None
            record.decision_after_s = max(0.0, now - record.started_perf) + _USER_DECISION_AFTER_S
            record.retry_event.clear()
            record.fast_event.clear()
            record.stream_buffer = ""
            record.stream_characters = 0
            record.stream_kind = None
            record.first_stream_perf = None
            record.reasoning_stream_chunks = 0
            record.stage_events = []
            record.updated_perf = now
        return record


def _current_record() -> _ModelRunRecord | None:
    run_id = current_model_run_id()
    if run_id is None:
        return None
    user_id = get_current_user_id()
    with _registry_lock:
        return _runs.get((user_id, run_id))


def current_model_run_reasoning_mode(
    default: ModelRunReasoningMode,
) -> ModelRunReasoningMode:
    record = _current_record()
    return record.reasoning_mode if record is not None else default


def record_model_stream_delta(
    delta: str,
    *,
    kind: Literal["output", "tool"] = "output",
) -> None:
    """Publish only user-visible output/tool JSON, never hidden reasoning content."""

    if not delta:
        return
    record = _current_record()
    if record is None:
        return
    now = time.perf_counter()
    with _registry_lock:
        record.stream_characters += len(delta)
        record.stream_buffer = (record.stream_buffer + delta)[-12_000:]
        record.stream_kind = kind
        if record.first_stream_perf is None:
            record.first_stream_perf = now
        record.phase = "streaming"
        record.detail = (
            "正在流式生成工具草案；完成后仍会由服务端校验"
            if kind == "tool"
            else "正在流式生成可见结果；完成后仍会由服务端校验"
        )
        record.updated_perf = now


def record_model_reasoning_activity() -> None:
    """Record a heartbeat for hidden reasoning without retaining its contents."""

    record = _current_record()
    if record is None:
        return
    with _registry_lock:
        record.reasoning_stream_chunks += 1
        if record.phase == "connecting":
            record.phase = "thinking"
        record.detail = "High 模式推理流持续返回；隐藏思维内容不会显示或保存"
        record.updated_perf = time.perf_counter()


def record_model_stream_stage(label: str, detail: str) -> None:
    record = _current_record()
    if record is None:
        return
    now = time.perf_counter()
    with _registry_lock:
        record.stage_events.append(
            {
                "label": label[:48],
                "detail": detail[:160],
                "at_s": round(max(0.0, now - record.started_perf), 1),
            }
        )
        record.stage_events = record.stage_events[-6:]
        record.updated_perf = now


def get_model_run_status(run_id: str) -> dict[str, object]:
    normalized = validate_model_run_id(run_id)
    assert normalized is not None
    user_id = get_current_user_id()
    with _registry_lock:
        _prune_locked(time.perf_counter())
        record = _runs.get((user_id, normalized))
        if record is None:
            raise KeyError(f"Unknown model run: {normalized}")
        return record.snapshot()


def decide_model_run(run_id: str, decision: ModelRunDecision) -> dict[str, object]:
    normalized = validate_model_run_id(run_id)
    assert normalized is not None
    user_id = get_current_user_id()
    now = time.perf_counter()
    with _registry_lock:
        record = _runs.get((user_id, normalized))
        if record is None:
            raise KeyError(f"Unknown model run: {normalized}")
        if record.phase in {"completed", "fallback_requested"}:
            return record.snapshot()
        snapshot = record.snapshot()
        if decision not in snapshot["allowed_decisions"]:
            raise ValueError("当前阶段不允许这个模型运行选择。")
        if decision == "fallback":
            record.phase = "fallback_requested"
            record.detail = "用户已选择立即使用明确标记的安全兜底"
            record.fallback_event.set()
        elif decision == "fast":
            record.reasoning_mode = "fast"
            record.supports_fast_switch = False
            record.fast_locked = True
            record.phase = "connecting"
            record.detail = "用户已停止本轮 High 生成；正在以 Fast 模式重新请求基模"
            record.step_started_perf = now
            record.decision_after_s = max(0.0, now - record.started_perf) + _USER_DECISION_AFTER_S
            record.error_kind = None
            record.stream_buffer = ""
            record.stream_characters = 0
            record.stream_kind = None
            record.first_stream_perf = None
            record.reasoning_stream_chunks = 0
            record.fast_event.set()
        elif decision == "continue":
            elapsed_s = max(0.0, now - record.started_perf)
            record.decision_after_s = elapsed_s + _USER_DECISION_AFTER_S
            if record.phase == "failed":
                record.retry_count += 1
                record.phase = "connecting"
                record.step_started_perf = now
                record.detail = "用户选择重试基模；正在重新建立连接"
                record.error_kind = None
                record.retry_event.set()
            else:
                record.detail = (
                    "用户选择继续等待 High 模式完成推理"
                    if record.reasoning_mode == "high"
                    else "用户选择继续等待 Fast 模式完成生成"
                )
        else:
            raise ValueError("model run decision must be continue, fast or fallback")
        record.updated_perf = now
        return record.snapshot()


def _mark_record(
    record: _ModelRunRecord | None,
    *,
    phase: ModelRunPhase,
    detail: str,
    error_kind: str | None = None,
) -> None:
    if record is None:
        return
    with _registry_lock:
        record.phase = phase
        record.detail = detail
        record.error_kind = error_kind
        record.updated_perf = time.perf_counter()


async def await_model_with_user_control(
    awaitable: Awaitable[T] | None = None,
    *,
    operation: str,
    model: str,
    noninteractive_timeout_s: float | None,
    awaitable_factory: Callable[[], Awaitable[T]] | None = None,
    reasoning_mode: ModelRunReasoningMode = "provider_default",
    supports_fast_switch: bool = False,
) -> T:
    """Await a provider call without a PocketLab hard cap for interactive runs.

    Browser requests supply a run ID and can therefore keep waiting, switch a High
    run to Fast, or explicitly choose a deterministic fallback. Non-interactive API
    clients and automated Harnesses retain their explicit timeout so a forgotten
    task cannot hang CI forever.
    """

    if awaitable is None and awaitable_factory is None:
        raise ValueError("a model awaitable or awaitable_factory is required")
    if supports_fast_switch and awaitable_factory is None:
        raise ValueError("Fast mode switching requires an awaitable_factory")

    record = _start_or_update_record(
        operation,
        model,
        reasoning_mode=reasoning_mode,
        supports_fast_switch=supports_fast_switch,
    )
    if record is None:
        active_awaitable = awaitable if awaitable is not None else awaitable_factory()
        if noninteractive_timeout_s is None:
            return await active_awaitable
        return await asyncio.wait_for(active_awaitable, timeout=noninteractive_timeout_s)

    first_awaitable = awaitable
    while True:
        if record.fallback_event.is_set():
            if first_awaitable is not None and hasattr(first_awaitable, "close"):
                first_awaitable.close()  # type: ignore[attr-defined]
                first_awaitable = None
            raise ModelFallbackRequested("user-requested-deterministic-fallback")
        active_awaitable = (
            first_awaitable
            if first_awaitable is not None
            else awaitable_factory()
            if awaitable_factory is not None
            else None
        )
        first_awaitable = None
        if active_awaitable is None:
            raise ValueError("interactive model retry requires awaitable_factory")
        provider_task = asyncio.create_task(active_awaitable)
        fallback_task = asyncio.create_task(record.fallback_event.wait())
        fast_task = asyncio.create_task(record.fast_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {provider_task, fallback_task, fast_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if fallback_task in done and record.fallback_event.is_set():
                provider_task.cancel()
                await asyncio.gather(provider_task, return_exceptions=True)
                _mark_record(
                    record,
                    phase="fallback_requested",
                    detail="用户已停止等待；正在进入明确标记的安全兜底",
                )
                raise ModelFallbackRequested("user-requested-deterministic-fallback")
            if fast_task in done and record.fast_event.is_set():
                provider_task.cancel()
                await asyncio.gather(provider_task, return_exceptions=True)
                record.fast_event.clear()
                _mark_record(
                    record,
                    phase="connecting",
                    detail="正在按用户选择以 Fast 模式重新请求基模",
                )
                continue
            result = await provider_task
        except ModelFallbackRequested:
            raise
        except Exception as exc:
            if awaitable_factory is None:
                _mark_record(
                    record,
                    phase="failed",
                    detail="基模请求失败；当前调用没有可安全重试的请求工厂",
                    error_kind=type(exc).__name__,
                )
                raise
            _mark_record(
                record,
                phase="failed",
                detail="本轮基模调用未完成；请选择重试基模，或明确接受安全兜底",
                error_kind=type(exc).__name__,
            )
            retry_task = asyncio.create_task(record.retry_event.wait())
            fallback_after_failure = asyncio.create_task(record.fallback_event.wait())
            fast_after_failure = asyncio.create_task(record.fast_event.wait())
            try:
                recovered, _pending = await asyncio.wait(
                    {retry_task, fallback_after_failure, fast_after_failure},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if fallback_after_failure in recovered and record.fallback_event.is_set():
                    raise ModelFallbackRequested("user-requested-deterministic-fallback") from exc
                if fast_after_failure in recovered and record.fast_event.is_set():
                    record.fast_event.clear()
                    _mark_record(
                        record,
                        phase="connecting",
                        detail="正在按用户选择以 Fast 模式重新请求基模",
                    )
                    continue
                record.retry_event.clear()
                _mark_record(
                    record,
                    phase="connecting",
                    detail="正在按用户选择重试基模",
                )
                continue
            finally:
                retry_task.cancel()
                fallback_after_failure.cancel()
                fast_after_failure.cancel()
        finally:
            fallback_task.cancel()
            fast_task.cancel()
        _mark_record(
            record,
            phase="validating",
            detail="流式生成已结束；正在进行服务端验证、整理与写入",
        )
        return result


async def await_model_validation_recovery_decision(
    *,
    detail: str,
    error_kind: str = "model_output_validation",
) -> ModelValidationRecoveryDecision:
    """Let an interactive user choose what happens after model output is rejected.

    Provider completion and application acceptance are different events. A model can
    return successfully while its JSON/tool proposal still fails the server contract.
    Interactive browser runs must not silently convert that failure into deterministic
    output. Non-interactive callers receive an error decision and cannot opt into a
    deterministic fallback on behalf of the user.
    """

    run_id = current_model_run_id()
    if run_id is None:
        return "noninteractive_error"
    user_id = get_current_user_id()
    with _registry_lock:
        record = _runs.get((user_id, run_id))
        if record is None:
            return "noninteractive_error"
        record.retry_event.clear()
        record.fast_event.clear()
        record.phase = "failed"
        record.detail = detail
        record.error_kind = error_kind
        record.updated_perf = time.perf_counter()

    retry_task = asyncio.create_task(record.retry_event.wait())
    fallback_task = asyncio.create_task(record.fallback_event.wait())
    fast_task = asyncio.create_task(record.fast_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {retry_task, fallback_task, fast_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if fallback_task in done and record.fallback_event.is_set():
            return "user_fallback"
        if fast_task in done and record.fast_event.is_set():
            record.fast_event.clear()
            return "retry_fast"
        record.retry_event.clear()
        return "retry"
    finally:
        retry_task.cancel()
        fallback_task.cancel()
        fast_task.cancel()


def mark_model_run_finished(*, detail: str = "模型结果已通过服务端处理") -> None:
    run_id = current_model_run_id()
    if run_id is None:
        return
    user_id = get_current_user_id()
    with _registry_lock:
        record = _runs.get((user_id, run_id))
    _mark_record(record, phase="completed", detail=detail)


def clear_model_runs_for_tests() -> None:
    with _registry_lock:
        _runs.clear()
