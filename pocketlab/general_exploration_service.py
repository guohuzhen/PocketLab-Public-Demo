from __future__ import annotations

from typing import Any

from agents import Agent

from pocketlab.agent_runtime import AgentRuntimePolicy
from pocketlab.general_exploration_engine import (
    commit_general_measurement,
    commit_general_reasoning_checkpoint,
    complete_general_reasoning_checkpoint,
    continue_general_reasoning_checkpoint,
    prepare_reasoned_report,
    prepare_reasoning_continuation,
)
from pocketlab.general_exploration_planner import (
    GeneralPlannerRunContext,
    commit_with_general_exploration_planner,
)
from pocketlab.general_exploration_reasoner import (
    render_reasoned_general_report,
    render_reasoned_general_report_with_labels,
    run_general_evidence_reasoner,
)
from pocketlab.general_exploration_state import (
    GeneralExperimentCase,
    GeneralReasoningCheckpointDecision,
    GeneralReasoningReceipt,
)
from pocketlab.general_exploration_store import (
    GeneralExplorationStore,
    GeneralRecordingMeasurementSubmit,
)
from pocketlab.general_simulation import GeneralSimulationMeasurementRequest


async def _commit_prepared_general_exploration(
    store: GeneralExplorationStore,
    prepared: Any,
    *,
    agent: Agent[GeneralPlannerRunContext] | Any | None = None,
    runner: Any | None = None,
    policy: AgentRuntimePolicy | None = None,
    reasoner_agent: Agent | Any | None = None,
    reasoner_runner: Any | None = None,
    reasoner_policy: AgentRuntimePolicy | None = None,
) -> GeneralExperimentCase:
    """Run the bounded Planner or evidence reasoner, then persist one CAS transition."""

    if prepared.report is not None and prepared.report.outcome == "completed_descriptive":
        reasoning = await run_general_evidence_reasoner(
            prepared,
            agent=reasoner_agent,
            runner=reasoner_runner,
            policy=reasoner_policy,
        )
        if reasoning.receipt.decision == "continue":
            task_count = reasoning.request.completed_measurement_task_count
            checkpoint_at = general_reasoning_checkpoint_task_count(
                prepared.base_case
            )
            if task_count >= checkpoint_at:
                updated = commit_general_reasoning_checkpoint(
                    prepared,
                    reasoning.receipt,
                )
            else:
                continued = prepare_reasoning_continuation(prepared)
                updated = commit_general_measurement(
                    continued,
                    selected_candidate_id=reasoning.receipt.selected_candidate_id,
                    selection_source="reasoning_agent",
                    reasoning_receipt=reasoning.receipt,
                )
        else:
            report = render_reasoned_general_report(
                prepared.report,
                reasoning.request,
                reasoning.receipt,
            )
            reasoned = prepare_reasoned_report(prepared, report)
            updated = commit_general_measurement(
                reasoned,
                reasoning_receipt=reasoning.receipt,
            )
        store.save_committed(updated, expected_revision=prepared.base_case.revision)
        return updated

    updated = await commit_with_general_exploration_planner(
        prepared,
        agent=agent,
        runner=runner,
        policy=policy,
    )
    store.save_committed(
        updated,
        expected_revision=prepared.base_case.revision,
    )
    return updated


def general_reasoning_checkpoint_task_count(case: GeneralExperimentCase) -> int:
    """Return the next soft user checkpoint; initial prompt is exactly task 20."""

    policy = case.protocol.evidence_policy
    return min(
        policy.hard_task_count,
        policy.user_checkpoint_task_count + case.reasoning_checkpoint_count * 5,
    )


async def advance_general_exploration(
    store: GeneralExplorationStore,
    case_id: str,
    request: GeneralRecordingMeasurementSubmit,
    *,
    agent: Agent[GeneralPlannerRunContext] | Any | None = None,
    runner: Any | None = None,
    policy: AgentRuntimePolicy | None = None,
    reasoner_agent: Agent | Any | None = None,
    reasoner_runner: Any | None = None,
    reasoner_policy: AgentRuntimePolicy | None = None,
) -> GeneralExperimentCase:
    """Prepare without mutation, run a read-only Planner, then commit once via CAS."""

    prepared = store.prepare_recording_submission(case_id, request)
    return await _commit_prepared_general_exploration(
        store,
        prepared,
        agent=agent,
        runner=runner,
        policy=policy,
        reasoner_agent=reasoner_agent,
        reasoner_runner=reasoner_runner,
        reasoner_policy=reasoner_policy,
    )


def decide_general_reasoning_checkpoint(
    store: GeneralExplorationStore,
    case_id: str,
    request: GeneralReasoningCheckpointDecision,
) -> GeneralExperimentCase:
    case = store.get(case_id)
    checkpoint = case.reasoning_checkpoint
    if checkpoint is None:
        raise ValueError("该 Exploration 当前没有待处理的继续/收手检查点。")
    if request.action == "continue":
        updated = continue_general_reasoning_checkpoint(case, request)
    else:
        offered = checkpoint.reasoning
        confidence_score = min(offered.confidence_score, 0.74)
        confidence = "medium" if confidence_score >= 0.60 else "low"
        stopped = GeneralReasoningReceipt.model_validate(
            offered.model_copy(
                update={
                    "decision": "user_stop",
                    "confidence": confidence,
                    "confidence_score": confidence_score,
                }
            ).model_dump(mode="python")
        )
        labels = {item.condition_id: item.label for item in case.protocol.conditions}
        report = render_reasoned_general_report_with_labels(
            checkpoint.provisional_report,
            labels,
            stopped,
        )
        updated = complete_general_reasoning_checkpoint(case, request, report)
    store.save_committed(updated, expected_revision=case.revision)
    return updated


async def advance_general_simulation(
    store: GeneralExplorationStore,
    case_id: str,
    request: GeneralSimulationMeasurementRequest,
    *,
    agent: Agent[GeneralPlannerRunContext] | Any | None = None,
    runner: Any | None = None,
    policy: AgentRuntimePolicy | None = None,
    reasoner_agent: Agent | Any | None = None,
    reasoner_runner: Any | None = None,
    reasoner_policy: AgentRuntimePolicy | None = None,
) -> GeneralExperimentCase:
    """Advance an explicit software rehearsal without creating physical evidence."""

    prepared = store.prepare_simulated_submission(case_id, request)
    return await _commit_prepared_general_exploration(
        store,
        prepared,
        agent=agent,
        runner=runner,
        policy=policy,
        reasoner_agent=reasoner_agent,
        reasoner_runner=reasoner_runner,
        reasoner_policy=reasoner_policy,
    )
