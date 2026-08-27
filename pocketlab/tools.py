import json
import os

from agents import function_tool

from pocketlab.diagnostics import diagnostic_case_store
from pocketlab.schemas import (
    AgentMeasurementTaskDraft,
    DiagnosticActionId,
    DiagnosticHypothesisDraft,
    DiagnosticReasoningReceipt,
    DiagnosticSensorPlanDraft,
    HypothesisAssessmentDraft,
    MeasurementTaskDraft,
)
from pocketlab.store import session_store


@function_tool
def analyze_vibration_session(session_id: str) -> str:
    """Return deterministic vibration metrics for one uploaded sensor session.

    Args:
        session_id: The exact session identifier supplied by the application.
    """

    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    return json.dumps(
        {
            "session_id": session.session_id,
            "label": session.upload.label,
            "notes": session.upload.notes,
            "analysis": session.analysis.model_dump(),
        },
        ensure_ascii=False,
    )


@function_tool
def compare_vibration_sessions(primary_session_id: str, control_session_id: str) -> str:
    """Compare a primary vibration recording with a control recording.

    Args:
        primary_session_id: Session recorded under the suspected cause.
        control_session_id: Session recorded after changing one experimental condition.
    """

    try:
        primary = session_store.get(primary_session_id)
        control = session_store.get(control_session_id)
    except KeyError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    p = primary.analysis
    c = control.analysis
    frequency_shift = c.dominant_frequency_hz - p.dominant_frequency_hz
    rms_ratio = c.rms_acceleration_m_s2 / max(p.rms_acceleration_m_s2, 1e-12)
    return json.dumps(
        {
            "primary": {
                "session_id": primary.session_id,
                "label": primary.upload.label,
                "analysis": p.model_dump(),
            },
            "control": {
                "session_id": control.session_id,
                "label": control.upload.label,
                "analysis": c.model_dump(),
            },
            "comparison": {
                "dominant_frequency_shift_hz": round(frequency_shift, 4),
                "control_to_primary_rms_ratio": round(rms_ratio, 4),
                "amplitude_change_percent": round((rms_ratio - 1.0) * 100.0, 2),
            },
        },
        ensure_ascii=False,
    )


@function_tool
def inspect_diagnostic_case(case_id: str) -> str:
    """Inspect the current hypotheses, evidence, and measurement task for a case.

    Args:
        case_id: The exact diagnostic case identifier supplied by the application.
    """

    try:
        case = diagnostic_case_store.get(case_id)
    except KeyError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    return case.model_dump_json()


@function_tool
def commit_initial_diagnostic_plan(
    case_id: str,
    hypotheses: list[DiagnosticHypothesisDraft],
    sensor_plan: list[DiagnosticSensorPlanDraft],
    first_task: AgentMeasurementTaskDraft,
) -> str:
    """Commit 2-3 testable hypotheses and the first single-variable measurement task.

    Args:
        case_id: The exact diagnostic case identifier supplied by the application.
        hypotheses: Two or three mutually distinguishable explanations, each with one
            observable critical prediction.
        sensor_plan: One primary and up to three supporting or optional phone sensors.
        first_task: The baseline measurement task. The backend targets all hypotheses.
    """

    try:
        case = diagnostic_case_store.commit_initial_plan(
            case_id=case_id,
            hypotheses=hypotheses,
            sensor_plan=sensor_plan,
            task=MeasurementTaskDraft(
                **first_task.model_dump(),
                task_kind="baseline",
                target_hypothesis_ids=[f"h{index}" for index in range(1, len(hypotheses) + 1)],
            ),
        )
    except (KeyError, ValueError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    return json.dumps(
        {
            "status": "committed",
            "case_id": case.case_id,
            "hypothesis_ids": [item.hypothesis_id for item in case.hypotheses],
            "current_task_id": case.current_task.task_id if case.current_task else None,
        },
        ensure_ascii=False,
    )


@function_tool
def commit_diagnostic_measurement(
    case_id: str,
    task_id: str,
    session_id: str,
    observation_notes: str,
    evidence_summary: str,
    assessments: list[HypothesisAssessmentDraft],
    next_task: AgentMeasurementTaskDraft,
    next_task_kind: str,
    next_target_hypothesis_ids: list[str],
    next_expected_effect: str,
    next_effect_metric: str,
    answer_headline: str,
    mechanism_explanation: str,
    reasoning_confidence: str,
    ranked_hypothesis_ids: list[str],
    source_fact_ids: list[str],
    next_measurement_reason: str,
    solution_rationale: str,
    recommended_action_ids: list[DiagnosticActionId],
) -> str:
    """Bind evidence and let the deterministic termination vector continue or finish.

    Args:
        case_id: The exact diagnostic case identifier supplied by the application.
        task_id: The current pending task identifier.
        session_id: The exact uploaded Session used as evidence.
        observation_notes: The user's visible observations during the measurement.
        evidence_summary: A concise evidence summary based on deterministic metrics.
        assessments: Exactly one assessment for every hypothesis; mark a critical prediction
            tested only when the current task actually targeted it.
        next_task: A fallback single-variable task used only when more evidence is required.
        next_task_kind: One of control, replication, correction, or exploration.
        next_target_hypothesis_ids: Hypotheses whose critical predictions the task will test.
        next_expected_effect: One of increase, decrease, change, no_change, or unknown. Control
            and replication tasks may not use unknown.
        next_effect_metric: One of rms, frequency, or either.
        answer_headline: Direct current answer to the household problem, not a chart summary.
        mechanism_explanation: Physical explanation connecting conditions, sensor facts, and cause.
        reasoning_confidence: One of low, medium, or high.
        ranked_hypothesis_ids: Every hypothesis ID ordered from most to least plausible.
        source_fact_ids: Exact deterministic fact IDs used for the explanation.
        next_measurement_reason: Why the next task has the highest information value.
        solution_rationale: Evidence-grounded rationale for the safest reversible response path.
        recommended_action_ids: One to three server-allowlisted safe action IDs, ordered.
            Sensor-specific choices include reduce-magnetic-interference, clear-sensor-path,
            verify-environmental-context, and isolate-operating-source. The server expands
            each choice into preparation, steps, verification, failure branches, and limits.
    """

    try:
        if next_task_kind not in {"control", "replication", "correction", "exploration"}:
            raise ValueError("next_task_kind 无效。")
        if next_expected_effect not in {"increase", "decrease", "change", "no_change", "unknown"}:
            raise ValueError("next_expected_effect 无效。")
        if next_effect_metric not in {"rms", "frequency", "either"}:
            raise ValueError("next_effect_metric 无效。")
        if reasoning_confidence not in {"low", "medium", "high"}:
            raise ValueError("reasoning_confidence 无效。")
        comparison_task_id = task_id if next_task_kind in {"control", "replication"} else None
        enriched_next_task = MeasurementTaskDraft(
            **next_task.model_dump(),
            task_kind=next_task_kind,
            comparison_task_id=comparison_task_id,
            target_hypothesis_ids=next_target_hypothesis_ids,
            expected_effect=next_expected_effect,
            effect_metric=next_effect_metric,
        )
        case = diagnostic_case_store.commit_measurement(
            case_id=case_id,
            task_id=task_id,
            session_id=session_id,
            observation_notes=observation_notes,
            evidence_summary=evidence_summary,
            assessments=assessments,
            next_task=enriched_next_task,
            reasoning_receipt=DiagnosticReasoningReceipt(
                model_name=(
                    os.getenv("LLM_MODEL", "").strip()
                    or os.getenv("PPIO_MODEL", "").strip()
                    or "configured-provider-model"
                ),
                answer_headline=answer_headline,
                mechanism_explanation=mechanism_explanation,
                confidence=reasoning_confidence,
                ranked_hypothesis_ids=ranked_hypothesis_ids,
                source_fact_ids=source_fact_ids,
                next_measurement_reason=next_measurement_reason,
                solution_rationale=solution_rationale,
                recommended_action_ids=recommended_action_ids,
            ),
        )
    except (KeyError, ValueError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    return json.dumps(
        {
            "status": "committed",
            "case_id": case.case_id,
            "evidence_id": case.evidence[-1].evidence_id,
            "next_task_id": case.current_task.task_id if case.current_task else None,
            "case_status": case.status,
            "termination_vector": case.termination_vector.model_dump(),
            "final_report": case.final_report.model_dump() if case.final_report else None,
        },
        ensure_ascii=False,
    )
