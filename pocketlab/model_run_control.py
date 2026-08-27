from __future__ import annotations

import asyncio
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
    "completed",
    "failed",
    "fallback_requested",
]
ModelRunDecision = Literal["continue", "fallback"]
ModelValidationRecoveryDecision = Literal[
    "retry",
    "user_fallback",
    "noninteractive_fallback",
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
    phase: ModelRunPhase = "connecting"
    detail: str = "正在连接基模服务"
    decision_after_s: float = _USER_DECISION_AFTER_S
    fallback_event: asyncio.Event = field(default_factory=asyncio.Event)
    retry_event: asyncio.Event = field(default_factory=asyncio.Event)
    retry_count: int = 0
    error_kind: str | None = None
    updated_perf: float = field(default_factory=time.perf_counter)

    def snapshot(self) -> dict[str, object]:
        now = time.perf_counter()
        elapsed_s = max(0.0, now - self.started_perf)
        step_elapsed_s = max(0.0, now - self.step_started_perf)
        phase = self.phase
        detail = self.detail
        if phase == "connecting" and step_elapsed_s >= _THINKING_AFTER_S:
            phase = "thinking"
            detail = "请求已发出；正在等待基模生成，可能处于网络传输或深度推理阶段"
        return {
            "run_id": self.run_id,
            "operation": self.operation,
            "model": self.model,
            "phase": phase,
            "detail": detail,
            "started_at": self.started_at,
            "elapsed_s": round(elapsed_s, 1),
            "decision_available": (
                phase == "failed"
                or (
                    phase in {"connecting", "thinking"}
                    and elapsed_s >= self.decision_after_s
                )
            ),
            "next_decision_at_s": round(self.decision_after_s, 1),
            "fallback_requested": self.fallback_event.is_set(),
            "retry_count": self.retry_count,
            "error_kind": self.error_kind,
        }


_registry_lock = RLock()
_runs: dict[tuple[str, str], _ModelRunRecord] = {}


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
    expired = [
        key
        for key, record in _runs.items()
        if now - record.updated_perf > _RECORD_TTL_S
    ]
    for key in expired:
        _runs.pop(key, None)


def _start_or_update_record(operation: str, model: str) -> _ModelRunRecord | None:
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
            )
            _runs[key] = record
        else:
            record.operation = operation
            record.model = model
            record.step_started_perf = now
            record.phase = "connecting"
            record.detail = "正在连接基模服务"
            record.error_kind = None
            record.updated_perf = now
        return record


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
        if decision == "fallback":
            record.phase = "fallback_requested"
            record.detail = "用户已选择立即使用明确标记的安全兜底"
            record.fallback_event.set()
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
                record.detail = "用户选择继续等待基模完成深度推理"
        else:
            raise ValueError("model run decision must be continue or fallback")
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
) -> T:
    """Await a provider call without a PocketLab hard cap for interactive runs.

    Browser requests supply a run ID and can therefore keep waiting or cancel into
    a deterministic fallback. Non-interactive API clients and automated Harnesses
    retain their explicit timeout so a forgotten task cannot hang CI forever.
    """

    if awaitable is None and awaitable_factory is None:
        raise ValueError("a model awaitable or awaitable_factory is required")

    record = _start_or_update_record(operation, model)
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
        try:
            done, _pending = await asyncio.wait(
                {provider_task, fallback_task},
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
            try:
                recovered, _pending = await asyncio.wait(
                    {retry_task, fallback_after_failure},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if (
                    fallback_after_failure in recovered
                    and record.fallback_event.is_set()
                ):
                    raise ModelFallbackRequested(
                        "user-requested-deterministic-fallback"
                    ) from exc
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
        finally:
            fallback_task.cancel()
        _mark_record(
            record,
            phase="completed",
            detail="基模已返回结果，正在进行服务端验证与写入",
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
    output. Offline Harnesses remain bounded and report the deterministic fallback.
    """

    run_id = current_model_run_id()
    if run_id is None:
        return "noninteractive_fallback"
    user_id = get_current_user_id()
    with _registry_lock:
        record = _runs.get((user_id, run_id))
        if record is None:
            return "noninteractive_fallback"
        record.retry_event.clear()
        record.phase = "failed"
        record.detail = detail
        record.error_kind = error_kind
        record.updated_perf = time.perf_counter()

    retry_task = asyncio.create_task(record.retry_event.wait())
    fallback_task = asyncio.create_task(record.fallback_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {retry_task, fallback_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if fallback_task in done and record.fallback_event.is_set():
            return "user_fallback"
        record.retry_event.clear()
        return "retry"
    finally:
        retry_task.cancel()
        fallback_task.cancel()


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
