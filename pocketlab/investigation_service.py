from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

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
    planner_runner: PlannerRunner = run_light_investigation_planner,
) -> InvestigationAdvanceOutcome:
    """Advance one measurement through the same prepare → plan → one-CAS path."""

    prepared = store.prepare_measurement_transition(case_id, request)
    planner_request = prepared.planner_request
    if (
        planner_request is None
        or prepared.provisional_case.planning_policy != "bounded_agent"
    ):
        return InvestigationAdvanceOutcome(
            case=store.commit_prepared_transition(prepared),
            planner_status="not_invoked",
        )
    try:
        result = await planner_runner(planner_request)
    except LightPlannerUnavailable as exc:
        return InvestigationAdvanceOutcome(
            case=store.commit_prepared_transition(
                prepared,
                runtime_trace=exc.runtime_trace,
                fallback_reason=exc.reason,
            ),
            planner_status="fallback",
            fallback_reason=exc.reason,
            planner_runtime_trace=exc.runtime_trace,
        )
    try:
        case = store.commit_prepared_transition(
            prepared,
            decision=result.decision,
            runtime_trace=result.runtime_trace,
        )
    except InvestigationValidation:
        reason = "decision-rejected"
        return InvestigationAdvanceOutcome(
            case=store.commit_prepared_transition(
                prepared,
                runtime_trace=result.runtime_trace,
                fallback_reason=reason,
            ),
            planner_status="fallback",
            fallback_reason=reason,
            planner_runtime_trace=result.runtime_trace,
        )
    return InvestigationAdvanceOutcome(
        case=case,
        planner_status="accepted",
        planner_runtime_trace=result.runtime_trace,
    )
