from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pocketlab.auth import get_current_user_id
from pocketlab.persistence import DEFAULT_USER_ID, SQLiteDatabase, database


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentRunAuditItem(_StrictModel):
    run_id: str
    operation: str
    model: str
    reasoning_mode: Literal["fast", "deep", "provider_default"] | None = None
    reasoning_effort: str | None = None
    status: Literal["completed", "failed", "cancelled"]
    started_at: str
    finished_at: str
    elapsed_s: float = Field(ge=0)
    attempts: int = Field(ge=0)
    model_requests: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    error_kind: str | None = None
    retryable: bool = False


class AgentRunAuditSummary(_StrictModel):
    run_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    completion_rate: float = Field(ge=0, le=1)
    average_elapsed_s: float = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    latest_error_kind: str | None = None


class AgentRunAuditCatalog(_StrictModel):
    summary: AgentRunAuditSummary
    runs: list[AgentRunAuditItem]
    privacy_boundary: str


def _audit_item(trace: dict[str, Any]) -> AgentRunAuditItem:
    attempts = trace.get("attempts") or []
    retryable = any(bool(item.get("retryable")) for item in attempts)
    return AgentRunAuditItem(
        run_id=str(trace["run_id"]),
        operation=str(trace["operation"]),
        model=str(trace["model"]),
        reasoning_mode=trace.get("reasoning_mode"),
        reasoning_effort=trace.get("reasoning_effort"),
        status=trace["status"],
        started_at=str(trace["started_at"]),
        finished_at=str(trace["finished_at"]),
        elapsed_s=max(0.0, float(trace.get("elapsed_s") or 0)),
        attempts=len(attempts),
        model_requests=max(0, int(trace.get("model_requests") or 0)),
        tool_calls=max(0, int(trace.get("tool_calls") or 0)),
        input_tokens=trace.get("input_tokens"),
        output_tokens=trace.get("output_tokens"),
        total_tokens=trace.get("total_tokens"),
        estimated_cost=trace.get("estimated_cost"),
        error_kind=trace.get("error_kind"),
        retryable=retryable,
    )


class AgentRunAuditStore:
    def __init__(
        self,
        storage: SQLiteDatabase | None = None,
        *,
        user_id: str | None = DEFAULT_USER_ID,
    ) -> None:
        self._database = storage or SQLiteDatabase(":memory:")
        self._user_id = user_id

    @property
    def _active_user_id(self) -> str:
        return self._user_id or get_current_user_id()

    def save_trace(self, trace: dict[str, Any]) -> None:
        item = _audit_item(trace)
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO agent_run_audits(
                    run_id, user_id, audit_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    item.run_id,
                    self._active_user_id,
                    item.model_dump_json(),
                    item.finished_at,
                ),
            )
            connection.execute(
                """
                DELETE FROM agent_run_audits
                WHERE user_id = ? AND run_id NOT IN (
                    SELECT run_id FROM agent_run_audits
                    WHERE user_id = ? ORDER BY created_at DESC LIMIT 500
                )
                """,
                (self._active_user_id, self._active_user_id),
            )

    def list(self, *, limit: int = 30) -> list[AgentRunAuditItem]:
        bounded = min(max(limit, 1), 100)
        rows = self._database.fetch_all(
            "SELECT audit_json FROM agent_run_audits WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (self._active_user_id, bounded),
        )
        return [AgentRunAuditItem.model_validate_json(row["audit_json"]) for row in rows]

    def catalog(self, *, limit: int = 30) -> AgentRunAuditCatalog:
        runs = self.list(limit=limit)
        completed = sum(item.status == "completed" for item in runs)
        failed = sum(item.status == "failed" for item in runs)
        cancelled = sum(item.status == "cancelled" for item in runs)
        elapsed = sum(item.elapsed_s for item in runs)
        token_values = [item.total_tokens for item in runs if item.total_tokens is not None]
        cost_values = [item.estimated_cost for item in runs if item.estimated_cost is not None]
        latest_error = next((item.error_kind for item in runs if item.error_kind), None)
        return AgentRunAuditCatalog(
            summary=AgentRunAuditSummary(
                run_count=len(runs),
                completed_count=completed,
                failed_count=failed,
                cancelled_count=cancelled,
                completion_rate=completed / len(runs) if runs else 0,
                average_elapsed_s=elapsed / len(runs) if runs else 0,
                total_tokens=sum(token_values) if token_values else None,
                estimated_cost=round(sum(cost_values), 8) if cost_values else None,
                latest_error_kind=latest_error,
            ),
            runs=runs,
            privacy_boundary=(
                "只保存运行类别、模型名、耗时、重试、工具计数、token/成本与安全错误分类；"
                "同时记录快速/深度推理模式，"
                "不保存完整提示词、模型回答、API Key 或原始传感器数据。"
            ),
        )

    def clear(self) -> None:
        self._database.execute(
            "DELETE FROM agent_run_audits WHERE user_id = ?",
            (self._active_user_id,),
        )


agent_run_audit_store = AgentRunAuditStore(database, user_id=None)
