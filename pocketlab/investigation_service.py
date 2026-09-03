from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from pocketlab.agent_runtime import AgentRuntimeError
from pocketlab.investigation_models import (
    InvestigationCase,
    InvestigationMeasurementSubmit,
    LightPlannerRequest,
)
from pocketlab.investigation_planner import (
    LightPlannerRunResult,
    LightPlannerUnavailable,
    run_light_investigation_planner,
)
from pocketlab.investigations import InvestigationStore, InvestigationValidation
from pocketlab.model_run_control import await_model_validation_recovery_decision

PlannerRunner = Callable[[LightPlannerRequest], Awaitable[LightPlannerRunResult]]


@dataclass(frozen=True)
class InvestigationAdvanceOutcome:
    case: InvestigationCase
    planner_status: Literal["not_invoked", "accepted", "fallback"]
    fallback_reason: str | None = None
    planner_runtime_trace: dict[str, object] | None = None


async def advance_investigation(
    store: InvestigationStore,
    case_id: str,
    request: InvestigationMeasurementSubmit,
    *,
    planner_runner: PlannerRunner | None = None,
) -> InvestigationAdvanceOutcome:
    """Advance one measurement through the same prepare → plan → one-CAS path."""

    prepared = store.prepare_measurement_transition(case_id, request)
    planner_request = prepared.planner_request
    if planner_request is None or prepared.provisional_case.planning_policy != "bounded_agent":
        return InvestigationAdvanceOutcome(
            case=store.commit_prepared_transition(prepared),
            planner_status="not_invoked",
        )
    injected_harness = planner_runner is not None
    active_planner_runner = planner_runner or run_light_investigation_planner

    def commit_fallback(
        reason: str,
        runtime_trace: dict[str, object] | None,
    ) -> InvestigationAdvanceOutcome:
        return InvestigationAdvanceOutcome(
            case=store.commit_prepared_transition(
                prepared,
                runtime_trace=runtime_trace,
                fallback_reason=reason,
            ),
            planner_status="fallback",
            fallback_reason=reason,
            planner_runtime_trace=runtime_trace,
        )

    while True:
        try:
            result = await active_planner_runner(planner_request)
        except LightPlannerUnavailable as exc:
            if exc.reason == "user-fallback":
                return commit_fallback("user-requested-fallback", exc.runtime_trace)
            if injected_harness:
                return commit_fallback(exc.reason, exc.runtime_trace)
            recovery = await await_model_validation_recovery_decision(
                detail=(
                    "基模未能产生可采纳的照度实验计划。请选择重试基模、切换 Fast，"
                    "或明确接受标记为兜底的协议默认步骤。"
                ),
                error_kind=exc.reason,
            )
            if recovery in {"retry", "retry_fast"}:
                continue
            if recovery == "user_fallback":
                return commit_fallback("user-requested-fallback", exc.runtime_trace)
            raise AgentRuntimeError(
                exc.reason.replace("-", "_"),
                "照度实验计划未完成；PocketLab 未替用户自动启用确定性兜底。",
                retryable=True,
            ) from exc

        try:
            case = store.commit_prepared_transition(
                prepared,
                decision=result.decision,
                runtime_trace=result.runtime_trace,
            )
        except InvestigationValidation as exc:
            if injected_harness:
                return commit_fallback("decision-rejected", result.runtime_trace)
            recovery = await await_model_validation_recovery_decision(
                detail=(
                    "基模计划已生成，但未通过当前实验状态契约。请选择重试基模、"
                    "切换 Fast，或明确接受标记为兜底的协议默认步骤。"
                ),
                error_kind="decision-rejected",
            )
            if recovery in {"retry", "retry_fast"}:
                continue
            if recovery == "user_fallback":
                return commit_fallback("user-requested-fallback", result.runtime_trace)
            raise AgentRuntimeError(
                "malformed_model_output",
                "照度实验计划未通过服务端契约；PocketLab 未替用户自动启用兜底。",
                retryable=True,
            ) from exc
        return InvestigationAdvanceOutcome(
            case=case,
            planner_status="accepted",
            planner_runtime_trace=result.runtime_trace,
        )
